use gstreamer::ClockTime;
use std::{collections::HashMap, time::Duration};

use anyhow::{anyhow, Context, Result};
use gstreamer::{prelude::*, Bin, Caps, Element, ElementFactory};
use gstreamer_app::{AppSink, AppSrc, AppSrcCallbacks, AppStreamType};
use std::sync::Arc;
use neolink_core::{
    bc_protocol::StreamKind,
    bcmedia::model::{
        AacDurationInfo, BcMedia, BcMediaIframe, BcMediaInfoV1, BcMediaInfoV2, BcMediaPframe,
        VideoType,
    },
};
use once_cell::sync::Lazy;
use std::sync::mpsc as std_mpsc;
use tokio::sync::mpsc as tokio_mpsc;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use std::future::Future;

use crate::{common::NeoInstance, rtsp::gst::NeoMediaFactory, AnyResult};

/// Spawn a detached task and log (instead of silently dropping) an error result.
fn spawn_logged<F>(label: String, fut: F)
where
    F: Future<Output = AnyResult<()>> + Send + 'static,
{
    tokio::task::spawn(async move {
        if let Err(e) = fut.await {
            log::warn!("{label}: task ended with error: {e:?}");
        }
    });
}

/// `spawn_logged` for a blocking task.
fn spawn_blocking_logged<F>(label: String, f: F)
where
    F: FnOnce() -> AnyResult<()> + Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        if let Err(e) = f() {
            log::warn!("{label}: blocking task ended with error: {e:?}");
        }
    });
}

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{RecvTimeoutError, TrySendError};

#[derive(Clone, Debug)]
pub enum AudioType {
    Aac,
    Adpcm(u32),
}

#[derive(Clone, Debug)]
struct StreamTypeCache {
    vid_type: Option<VideoType>,
    aud_type: Option<AudioType>,
    // Sizing learned alongside the codec types, so the fast path can build a
    // correctly-sized pipeline from cache without re-querying the camera.
    resolution: [u32; 2],
    bitrate: u32,
    fps: u32,
    fps_table: Vec<u32>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct StreamCacheKey {
    camera_id: String,
    stream: StreamKind,
}

static STREAM_TYPE_CACHE: Lazy<RwLock<HashMap<StreamCacheKey, StreamTypeCache>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

async fn cached_stream_types(key: &StreamCacheKey) -> Option<StreamTypeCache> {
    let cache = STREAM_TYPE_CACHE.read().await;
    cache.get(key).cloned()
}

async fn store_stream_types(key: StreamCacheKey, types: StreamTypeCache) {
    let mut cache = STREAM_TYPE_CACHE.write().await;
    cache.insert(key, types);
}

#[derive(Clone, Debug)]
struct StreamConfig {
    #[allow(dead_code)]
    resolution: [u32; 2],
    bitrate: u32,
    fps: u32,
    fps_table: Vec<u32>,
    vid_type: Option<VideoType>,
    aud_type: Option<AudioType>,
    buffer_duration_ms: u64,
}
impl StreamConfig {
    /// Build a config from cached stream types + sizing, without contacting the
    /// camera. Used by the fast path so a previously-seen stream can be served
    /// immediately even while the camera is offline.
    fn from_cache(cache: &StreamTypeCache, buffer_duration_ms: u64) -> Self {
        StreamConfig {
            resolution: cache.resolution,
            bitrate: cache.bitrate,
            fps: cache.fps,
            fps_table: cache.fps_table.clone(),
            vid_type: cache.vid_type.clone(),
            aud_type: cache.aud_type.clone(),
            buffer_duration_ms,
        }
    }

    async fn new(
        instance: &NeoInstance,
        name: StreamKind,
        buffer_duration_ms: u64,
    ) -> AnyResult<Self> {
        let (resolution, bitrate, fps, fps_table) = instance
            .run_passive_task(|cam| {
                Box::pin(async move {
                    let infos = cam
                        .get_stream_info()
                        .await?
                        .stream_infos
                        .iter()
                        .flat_map(|info| info.encode_tables.clone())
                        .collect::<Vec<_>>();
                    if let Some(encode) =
                        infos.iter().find(|encode| encode.name == name.to_string())
                    {
                        let bitrate_table = encode
                            .bitrate_table
                            .split(',')
                            .filter_map(|c| {
                                let i: Result<u32, _> = c.parse();
                                i.ok()
                            })
                            .collect::<Vec<u32>>();
                        let framerate_table = encode
                            .framerate_table
                            .split(',')
                            .filter_map(|c| {
                                let i: Result<u32, _> = c.parse();
                                i.ok()
                            })
                            .collect::<Vec<u32>>();

                        Ok((
                            [encode.resolution.width, encode.resolution.height],
                            bitrate_table
                                .get(encode.default_bitrate as usize)
                                .copied()
                                .unwrap_or(encode.default_bitrate)
                                * 1024,
                            framerate_table
                                .get(encode.default_framerate as usize)
                                .copied()
                                .unwrap_or(encode.default_framerate),
                            framerate_table.clone(),
                        ))
                    } else {
                        Ok(([0, 0], 0, 0, vec![]))
                    }
                })
            })
            .await?;

        Ok(StreamConfig {
            resolution,
            bitrate,
            fps,
            fps_table,
            vid_type: None,
            aud_type: None,
            buffer_duration_ms,
        })
    }

    fn update_fps(&mut self, fps: u32) {
        let new_fps = self.fps_table.get(fps as usize).copied().unwrap_or(fps);
        self.fps = new_fps;
    }

    fn update_from_media(&mut self, media: &BcMedia) {
        match media {
            BcMedia::InfoV1(BcMediaInfoV1 { fps, .. })
            | BcMedia::InfoV2(BcMediaInfoV2 { fps, .. }) => self.update_fps(*fps as u32),
            BcMedia::Aac(_) => {
                self.aud_type = Some(AudioType::Aac);
            }
            BcMedia::Adpcm(adpcm) => {
                if let Some(block_size) = adpcm.block_size() {
                    self.aud_type = Some(AudioType::Adpcm(block_size));
                } else {
                    log::warn!("Ignoring malformed ADPCM frame shorter than its header");
                }
            }
            BcMedia::Iframe(BcMediaIframe { video_type, .. })
            | BcMedia::Pframe(BcMediaPframe { video_type, .. }) => {
                self.vid_type = Some(*video_type);
            }
        }
    }
}

pub(super) async fn make_dummy_factory(
    use_splash: bool,
    pattern: String,
) -> AnyResult<NeoMediaFactory> {
    NeoMediaFactory::new_with_callback(move |element| {
        clear_bin(&element)?;
        if !use_splash {
            Ok(None)
        } else {
            build_unknown(&element, &pattern)?;
            Ok(Some(element))
        }
    })
    .await
}

/// What the per-client setup task hands back to the gst factory callback. Using an
/// explicit outcome (rather than dropping the reply on failure) lets the callback
/// tell an expected camera-unavailable result apart from the setup task dying — the
/// former is logged quietly, only the latter is a real error.
enum SetupOutcome {
    /// The pipeline is built and ready to serve.
    Pipeline(Element),
    /// The camera could not be set up (offline / timeout / generation cancelled).
    /// Expected; the client's SETUP fails cleanly and it can retry.
    Unavailable,
}

enum ClientMsg {
    NewClient {
        element: Element,
        reply: tokio::sync::oneshot::Sender<SetupOutcome>,
    },
}

struct TimedMedia {
    recv_at: std::time::Instant,
    media: BcMedia,
}

pub(super) async fn make_factory(
    camera: NeoInstance,
    stream: StreamKind,
    cancel: CancellationToken,
) -> AnyResult<(NeoMediaFactory, JoinHandle<AnyResult<()>>)> {
    log::debug!("make_factory called for stream {:?}", stream);

    let (client_tx, mut client_rx) = tokio_mpsc::channel(100);

    log::debug!("Creating factory for stream {:?}", stream);

    // Create the task that creates the pipelines
    let thread = tokio::task::spawn(async move {
        // Child token handed to each per-client task (forwarder + blocking sender)
        // so a generation cancel (config-change restart) tears those down too, not
        // just the message-handler loop.
        let child_cancel = cancel.clone();
        let r: AnyResult<()> = tokio::select! {
            // Cancelled when its `stream_main` generation is dropped (e.g. a config
            // change), so the handler task doesn't leak.
            _ = cancel.cancelled() => {
                log::debug!("RTSP factory message-handler cancelled");
                Ok(())
            }
            v = async move {
        let name = camera.config().await?.borrow().name.clone();
        log::info!("{name}::{stream}: Message handler task started, waiting for messages");

        while let Some(msg) = client_rx.recv().await {
            log::debug!("{name}::{stream}: Received message in handler");
            match msg {
                ClientMsg::NewClient { element, reply } => {
                    log::debug!("NewClient message received for {name}::{stream}");
                    let camera = camera.clone();
                    let name = name.clone();
                    let cancel_child = child_cancel.clone();
                    let task_label = format!("{name}::{stream} client");
                    spawn_logged(task_label, async move {
                        clear_bin(&element)?;
                        log::info!(
                            "{name}::{stream}: Factory received new client, setting up pipeline"
                        );

                        // Camera config is local (it does not require the camera to be
                        // online), so fetch it up front for the cache key and splash.
                        let config = camera.config().await?.borrow().clone();
                        let camera_id = config
                            .camera_uid
                            .clone()
                            .unwrap_or_else(|| config.name.clone());
                        let cache_key = StreamCacheKey { camera_id, stream };
                        let cached = cached_stream_types(&cache_key).await;

                        let (stream_config, mut buffer, permit, mut media_rx, vid_src, aud_src, use_keepalive) =
                            if let Some(cached) = cached.filter(|c| c.vid_type.is_some()) {
                                // ===== Fast path: stream type is cached =====
                                // Build the pipeline from cached caps and return it to
                                // gstreamer immediately, so the RTSP callback unblocks and
                                // the client can reach PLAY without the camera being online.
                                // Then connect in the background and feed the appsrc. The
                                // stream layer (run_passive_task) keeps media_rx open across
                                // camera drops, so once an offline camera (re)connects the
                                // frames flow into the existing session with no client
                                // reconnect.
                                log::info!(
                                    "{name}::{stream}: Using cached stream types (serving immediately): video={:?}, audio={:?}",
                                    cached.vid_type, cached.aud_type
                                );
                                let stream_config =
                                    StreamConfig::from_cache(&cached, config.buffer_duration);
                                let (vid_src, aud_src) = build_sources(
                                    &element,
                                    &stream_config,
                                    &config.splash_pattern.to_string(),
                                )?;
                                let _ = reply.send(SetupOutcome::Pipeline(element));

                                // permit() + stream_while_live() return promptly even while
                                // the camera is mid-outage; media_rx stays empty until the
                                // camera is reachable.
                                let connect = async {
                                    let permit = camera.permit().await?;
                                    let media_rx = camera.stream_while_live(stream).await?;
                                    log::info!("{name}::{stream}: Camera relay established");
                                    AnyResult::Ok((permit, media_rx))
                                };
                                let (permit, media_rx) = tokio::select! {
                                    _ = cancel_child.cancelled() => return Ok(()),
                                    r = connect => r?,
                                };
                                (stream_config, Vec::new(), permit, media_rx, vid_src, aud_src, true)
                            } else {
                                // ===== Slow path: stream type not cached =====
                                // We must reach the camera to learn the codec before the
                                // pipeline can be built, so the gst callback stays blocked on
                                // the reply. Bound it and make it cancellation-aware; on
                                // timeout/cancel we report Unavailable (rare: only a first-ever
                                // connect to a camera that is currently offline).
                                let connect = async {
                                    let permit = camera.permit().await?;
                                    let media_rx = camera.stream_while_live(stream).await?;
                                    log::info!("{name}::{stream}: Camera relay established");
                                    let stream_config =
                                        StreamConfig::new(&camera, stream, config.buffer_duration)
                                            .await?;
                                    AnyResult::Ok((permit, media_rx, stream_config))
                                };
                                let (permit, mut media_rx, mut stream_config) = tokio::select! {
                                    _ = cancel_child.cancelled() => {
                                        log::debug!("{name}::{stream}: client setup cancelled before the camera connected");
                                        let _ = reply.send(SetupOutcome::Unavailable);
                                        return Ok(());
                                    }
                                    r = tokio::time::timeout(Duration::from_secs(20), connect) => match r {
                                        Ok(Ok(v)) => v,
                                        Ok(Err(e)) => {
                                            log::warn!("{name}::{stream}: camera setup failed: {e:?}");
                                            let _ = reply.send(SetupOutcome::Unavailable);
                                            return Ok(());
                                        }
                                        Err(_) => {
                                            log::warn!("{name}::{stream}: camera setup did not complete within 20s (camera likely unavailable)");
                                            let _ = reply.send(SetupOutcome::Unavailable);
                                            return Ok(());
                                        }
                                    }
                                };

                                log::info!("{name}::{stream}: Learning camera stream type");
                                let mut buffer = vec![];
                                let mut frame_count = 0usize;
                                let deadline = tokio::time::sleep(Duration::from_secs(15));
                                tokio::pin!(deadline);
                                loop {
                                    tokio::select! {
                                        _ = cancel_child.cancelled() => {
                                            log::debug!("{name}::{stream}: client setup cancelled while learning stream type");
                                            let _ = reply.send(SetupOutcome::Unavailable);
                                            return Ok(());
                                        }
                                        _ = &mut deadline => {
                                            log::info!("{name}::{stream}: Stream type timeout, building fallback pipeline");
                                            break;
                                        }
                                        media = media_rx.recv() => {
                                            let Some(media) = media else { break; };
                                            stream_config.update_from_media(&media);
                                            buffer.push(media);
                                            frame_count += 1;
                                            if frame_count >= 15
                                                || (frame_count >= 10
                                                    && stream_config.vid_type.is_some()
                                                    && stream_config.aud_type.is_some())
                                            {
                                                log::info!("{name}::{stream}: Stream type learned: video={:?}, audio={:?}",
                                                    stream_config.vid_type, stream_config.aud_type);
                                                break;
                                            }
                                        }
                                    }
                                }

                                if stream_config.vid_type.is_none()
                                    && stream_config.aud_type.is_none()
                                {
                                    log::warn!("{name}::{stream}: No media received from camera, building fallback pipeline");
                                } else {
                                    store_stream_types(
                                        cache_key.clone(),
                                        StreamTypeCache {
                                            vid_type: stream_config.vid_type.clone(),
                                            aud_type: stream_config.aud_type.clone(),
                                            resolution: stream_config.resolution,
                                            bitrate: stream_config.bitrate,
                                            fps: stream_config.fps,
                                            fps_table: stream_config.fps_table.clone(),
                                        },
                                    )
                                    .await;
                                }

                                let sizing_bytes = buffer_size_bytes(&stream_config);
                                log::info!(
                                    "{name}::{stream}: Stream sizing: bitrate={}bps fps={} buffer_ms={} -> buffer_bytes={}",
                                    stream_config.bitrate,
                                    stream_config.fps,
                                    stream_config.buffer_duration_ms,
                                    sizing_bytes
                                );
                                let (vid_src, aud_src) = build_sources(
                                    &element,
                                    &stream_config,
                                    &config.splash_pattern.to_string(),
                                )?;
                                let _ = reply.send(SetupOutcome::Pipeline(element));
                                (stream_config, buffer, permit, media_rx, vid_src, aud_src, false)
                            };

                        // ---- Clean fix: keep tokio receiver async; forward into bounded std channel ----
                        let queue_capacity = media_queue_capacity(&stream_config);
                        let stream_label = format!("{name}::{stream}");
                        log::info!(
                            "{stream_label}: Media queue capacity set to {}",
                            queue_capacity
                        );
                        let (std_tx, std_rx): (
                            std_mpsc::SyncSender<TimedMedia>,
                            std_mpsc::Receiver<TimedMedia>,
                        ) = std_mpsc::sync_channel(queue_capacity);

                        let in_audio = std::sync::Arc::new(AtomicU64::new(0));
                        let in_video = std::sync::Arc::new(AtomicU64::new(0));
                        let in_other = std::sync::Arc::new(AtomicU64::new(0));
                        let drop_audio = std::sync::Arc::new(AtomicU64::new(0));
                        let drop_video = std::sync::Arc::new(AtomicU64::new(0));
                        let drop_other = std::sync::Arc::new(AtomicU64::new(0));

                        let in_audio_fwd = in_audio.clone();
                        let in_video_fwd = in_video.clone();
                        let in_other_fwd = in_other.clone();
                        let drop_audio_fwd = drop_audio.clone();
                        let drop_video_fwd = drop_video.clone();
                        let drop_other_fwd = drop_other.clone();
                        let label_fwd = stream_label.clone();
                        let cancel_fwd = cancel_child.clone();

                        // Forwarder task: runs on tokio, owns media_rx. Stops when the
                        // camera stream ends or the generation is cancelled; dropping
                        // std_tx then disconnects the blocking sender too.
                        tokio::spawn(async move {
                            let mut last_drop_log = std::time::Instant::now();
                            let mut dropped = 0u64;
                            loop {
                                let m = tokio::select! {
                                    _ = cancel_fwd.cancelled() => break,
                                    m = media_rx.recv() => match m {
                                        Some(m) => m,
                                        None => break,
                                    },
                                };
                                let kind = match &m {
                                    BcMedia::Aac(_) | BcMedia::Adpcm(_) => 0,
                                    BcMedia::Iframe(_) | BcMedia::Pframe(_) => 1,
                                    _ => 2,
                                };
                                match kind {
                                    0 => in_audio_fwd.fetch_add(1, Ordering::Relaxed),
                                    1 => in_video_fwd.fetch_add(1, Ordering::Relaxed),
                                    _ => in_other_fwd.fetch_add(1, Ordering::Relaxed),
                                };

                                let timed = TimedMedia {
                                    recv_at: std::time::Instant::now(),
                                    media: m,
                                };
                                match std_tx.try_send(timed) {
                                    Ok(()) => {}
                                    Err(TrySendError::Full(_)) => {
                                        match kind {
                                            0 => drop_audio_fwd.fetch_add(1, Ordering::Relaxed),
                                            1 => drop_video_fwd.fetch_add(1, Ordering::Relaxed),
                                            _ => drop_other_fwd.fetch_add(1, Ordering::Relaxed),
                                        };
                                        dropped += 1;
                                        let now = std::time::Instant::now();
                                        if now.duration_since(last_drop_log)
                                            >= Duration::from_secs(1)
                                        {
                                            log::warn!(
                                                "{label_fwd}: Media queue full; dropped {} frames in last {:?}",
                                                dropped,
                                                now.duration_since(last_drop_log)
                                            );
                                            dropped = 0;
                                            last_drop_log = now;
                                        }
                                    }
                                    Err(TrySendError::Disconnected(_)) => break,
                                }
                            }
                            // Dropping std_tx will cause std_rx.recv_timeout to return Disconnected
                        });

                        // Run blocking code in tokio's blocking thread pool
                        // This maintains the tokio runtime context needed for permit drop
                        // Move permit into this thread to keep it alive for the session duration
                        let sender_label = stream_label.clone();
                        let cancel_send = cancel_child.clone();
                        spawn_blocking_logged(sender_label, move || {
                            use std::time::{Duration, Instant};

                            let start = Instant::now();
                            let _permit = permit; // hold for lifetime

                            // Wait for the RTSP server to link the pads (client reaches PLAY).
                            let link_deadline = Instant::now() + Duration::from_secs(2);
                            loop {
                                let vid_linked =
                                    vid_src.as_ref().map(|s| is_linked(s)).unwrap_or(true);
                                let aud_linked =
                                    aud_src.as_ref().map(|s| is_linked(s)).unwrap_or(true);

                                if vid_linked && aud_linked {
                                    break;
                                }
                                if Instant::now() >= link_deadline {
                                    break;
                                }
                                std::thread::sleep(Duration::from_millis(10));
                            }

                            let mut aud_ts: u64 = 0;
                            let mut vid_ts: u64 = 0;

                            let mut vid_pacer = Some(PacerState::new(start, Duration::ZERO));
                            let mut aud_pacer = Some(PacerState::new(start, Duration::ZERO));

                            let mut pools: HashMap<usize, gstreamer::BufferPool> =
                                Default::default();
                            let mut stats = StreamStats::new(Instant::now());

                            let frame_step_us: u64 = {
                                let fps = stream_config.fps.max(1) as u64;
                                1_000_000u64 / fps
                            };

                            let mut drop_until_keyframe: bool = false;
                            let high_bitrate = stream_config.bitrate >= 2_000_000;
                            let backpressure_threshold = if high_bitrate { 0.92 } else { 0.85 };
                            let lag_enter_threshold = if high_bitrate {
                                Duration::from_millis(900)
                            } else {
                                Duration::from_millis(500)
                            };
                            let lag_exit_threshold = if high_bitrate {
                                Duration::from_millis(300)
                            } else {
                                Duration::from_millis(150)
                            };
                            log::info!(
                                "{stream_label}: Backpressure thresholds: fill>{:.2}, lag_enter={:?}, lag_exit={:?}",
                                backpressure_threshold,
                                lag_enter_threshold,
                                lag_exit_threshold
                            );
                            let mut client_active = false;
                            let mut last_linked = Instant::now();
                            let client_active_deadline = Instant::now() + Duration::from_secs(10);
                            let disconnect_grace = Duration::from_secs(2);
                            let mut play_logged = false;

                            // Keepalive (cached fast path only): push a low-rate black
                            // placeholder keyframe until the camera sends its first real
                            // keyframe, so a client that connected before the camera
                            // produced frames doesn't time out. Encoded once at the cached
                            // resolution, so there's no resolution change at the handoff.
                            let keepalive = if use_keepalive {
                                match stream_config.vid_type {
                                    Some(vt) => keepalive_keyframe(vt, stream_config.resolution),
                                    None => None,
                                }
                            } else {
                                None
                            };
                            // If keepalive is disabled (slow path, no video, or encode
                            // failed), treat the stream as already live so real frames flow
                            // immediately with no gating.
                            let mut seen_real_keyframe = keepalive.is_none();
                            let mut next_keepalive = Instant::now();
                            let keepalive_step = Duration::from_secs(1);
                            let mut keepalive_pacer: Option<PacerState> = None;

                            // Send buffered frames (no pacing)
                            for buffered in buffer.drain(..) {
                                match send_to_sources(
                                    buffered,
                                    &stream_label,
                                    &mut pools,
                                    &vid_src,
                                    &aud_src,
                                    &mut vid_ts,
                                    &mut aud_ts,
                                    &stream_config,
                                    false,
                                    &mut vid_pacer,
                                    &mut aud_pacer,
                                    &mut stats,
                                    None,
                                )? {
                                    PushOutcome::Gone => {
                                        log::info!("{stream_label}: Client disconnected, stopping camera relay");
                                        return AnyResult::Ok(());
                                    }
                                    // Pre-PLAY buffered frames: NotLinked is expected, tolerate it.
                                    PushOutcome::Pushed | PushOutcome::NotLinked => {}
                                }
                            }

                            // Live loop: recv_timeout so heartbeat + disconnect checks keep running
                            loop {
                                if cancel_send.is_cancelled() {
                                    log::info!("{stream_label}: Generation cancelled, stopping camera relay");
                                    break;
                                }

                                // Keepalive: until the first real camera keyframe, push the
                                // placeholder at a low rate so the client keeps receiving RTP
                                // and doesn't time out while the camera is still connecting.
                                if !seen_real_keyframe {
                                    if let (Some(bytes), Some(app)) =
                                        (keepalive.as_ref(), vid_src.as_ref())
                                    {
                                        let now = Instant::now();
                                        if now >= next_keepalive {
                                            // Advance the shared video clock so the eventual
                                            // real frame continues monotonically from here.
                                            stats.vid_pts_us = stats
                                                .vid_pts_us
                                                .saturating_add(keepalive_step.as_micros() as u64);
                                            vid_ts = stats.vid_pts_us;
                                            match send_to_appsrc(
                                                app,
                                                &stream_label,
                                                bytes.as_ref().clone(),
                                                Duration::from_micros(vid_ts),
                                                Some(keepalive_step),
                                                &mut pools,
                                                false,
                                                &mut keepalive_pacer,
                                            )? {
                                                PushOutcome::Gone => {
                                                    log::info!("{stream_label}: Client disconnected during keepalive, stopping camera relay");
                                                    break;
                                                }
                                                PushOutcome::Pushed | PushOutcome::NotLinked => {}
                                            }
                                            next_keepalive = now + keepalive_step;
                                        }
                                    }
                                }
                                let vid_closed =
                                    vid_src.as_ref().map(|src| is_closed(src)).unwrap_or(true);
                                let aud_closed =
                                    aud_src.as_ref().map(|src| is_closed(src)).unwrap_or(true);
                                let vid_linked =
                                    vid_src.as_ref().map(|s| is_linked(s)).unwrap_or(false);
                                let aud_linked =
                                    aud_src.as_ref().map(|s| is_linked(s)).unwrap_or(false);

                                if vid_linked || aud_linked {
                                    if !play_logged {
                                        log::info!(
                                            "{stream_label}: RTSP PLAY reached after {:?} (vid_linked={}, aud_linked={})",
                                            start.elapsed(),
                                            vid_linked,
                                            aud_linked
                                        );
                                        play_logged = true;
                                    }
                                    client_active = true;
                                    last_linked = Instant::now();
                                }

                                if client_active {
                                    if vid_closed && aud_closed {
                                        log::info!("{stream_label}: Client disconnected, stopping camera relay");
                                        break;
                                    }
                                    if !(vid_linked || aud_linked)
                                        && Instant::now().duration_since(last_linked)
                                            >= disconnect_grace
                                    {
                                        log::info!("{stream_label}: Client disconnected, stopping camera relay");
                                        break;
                                    }
                                } else if Instant::now() >= client_active_deadline {
                                    log::info!(
                                        "{stream_label}: Client never reached PLAY after {:?}, stopping camera relay",
                                        start.elapsed()
                                    );
                                    break;
                                }

                                // once-per-second heartbeat
                                let now = Instant::now();
                                if now.duration_since(stats.last_report) >= Duration::from_secs(1) {
                                    let elapsed = start.elapsed();

                                    let vid_level = vid_src
                                        .as_ref()
                                        .map(|s| s.current_level_bytes())
                                        .unwrap_or(0);
                                    let vid_max =
                                        vid_src.as_ref().map(|s| s.max_bytes()).unwrap_or(0);
                                    let aud_level = aud_src
                                        .as_ref()
                                        .map(|s| s.current_level_bytes())
                                        .unwrap_or(0);
                                    let aud_max =
                                        aud_src.as_ref().map(|s| s.max_bytes()).unwrap_or(0);

                                    let since_last_media =
                                        now.duration_since(stats.last_media_instant);
                                    let av_drift_ms = (vid_ts as i64 - aud_ts as i64) / 1000;
                                    let aud_snap = stats.aud_stats_snapshot();
                                    let report_interval = now.duration_since(stats.last_report);
                                    let report_interval_us =
                                        report_interval.as_micros().max(1) as u64;
                                    let aud_total_ms = aud_snap.total_us / 1000;
                                    let aud_ratio_pct =
                                        aud_snap.total_us.saturating_mul(100) / report_interval_us;
                                    let in_audio = in_audio.swap(0, Ordering::Relaxed);
                                    let in_video = in_video.swap(0, Ordering::Relaxed);
                                    let in_other = in_other.swap(0, Ordering::Relaxed);
                                    let drop_audio = drop_audio.swap(0, Ordering::Relaxed);
                                    let drop_video = drop_video.swap(0, Ordering::Relaxed);
                                    let drop_other = drop_other.swap(0, Ordering::Relaxed);

                                    log::debug!(
                                        "{stream_label}: HB elapsed={:?} vid_ts={:?} aud_ts={:?} \
                                        vid_buf={}/{} aud_buf={}/{} last_media={:?} \
                                        vid_push={} aud_push={} dropP_pressure={} dropP_backpressure={} \
                                        fps={} frame_us={} av_drift_ms={} \
                                        aud_avg_us={} aud_min_us={} aud_max_us={} aud_last_us={} aud_rate={} \
                                        aud_adts_frames={} aud_raw_blocks={} aud_profile={} aud_samp_idx={} aud_ch={} \
                                        aud_frame_len={} aud_hdr_len={} aud_payload={} aud_parsed={} aud_bytes={} \
                                        aud_pkt={} aud_total_ms={} aud_ratio_pct={} \
                                        aud_gap_avg_us={} aud_gap_min_us={} aud_gap_max_us={} \
                                        in_audio={} in_video={} in_other={} drop_audio={} drop_video={} drop_other={}",
                                        elapsed,
                                        Duration::from_micros(vid_ts),
                                        Duration::from_micros(aud_ts),
                                        vid_level, vid_max,
                                        aud_level, aud_max,
                                        since_last_media,
                                        stats.vid_pushed, stats.aud_pushed,
                                        stats.p_dropped_pressure, stats.p_dropped_backpressure,
                                        stream_config.fps,
                                        frame_step_us,
                                        av_drift_ms,
                                        aud_snap.avg_us,
                                        aud_snap.min_us,
                                        aud_snap.max_us,
                                        stats.aud_last_us,
                                        stats.aud_last_rate,
                                        stats.aud_last_adts_frames,
                                        stats.aud_last_raw_blocks,
                                        stats.aud_last_profile,
                                        stats.aud_last_sampling_index,
                                        stats.aud_last_channel_config,
                                        stats.aud_last_frame_len,
                                        stats.aud_last_header_len,
                                        stats.aud_last_payload_len,
                                        stats.aud_last_parsed_len,
                                        stats.aud_last_bytes,
                                        aud_snap.packets,
                                        aud_total_ms,
                                        aud_ratio_pct,
                                        aud_snap.gap_avg_us,
                                        aud_snap.gap_min_us,
                                        aud_snap.gap_max_us,
                                        in_audio,
                                        in_video,
                                        in_other,
                                        drop_audio,
                                        drop_video,
                                        drop_other,
                                    );

                                    stats.last_report = now;
                                    stats.reset_aud_stats();
                                }

                                // Receive next frame (or timeout to keep loop responsive)
                                let timed = match std_rx.recv_timeout(Duration::from_millis(200)) {
                                    Ok(d) => d,
                                    Err(RecvTimeoutError::Timeout) => {
                                        // no frame right now; loop continues (heartbeat/disconnect still works)
                                        continue;
                                    }
                                    Err(RecvTimeoutError::Disconnected) => {
                                        log::info!(
                                            "{stream_label}: Camera stream ended, stopping relay"
                                        );
                                        break;
                                    }
                                };

                                stats.tryrecv_ok += 1;
                                stats.last_media_instant = Instant::now();

                                // Keepalive handoff: until the first real camera keyframe, drop
                                // real frames (P-frames / audio) and keep serving the placeholder.
                                // On the first keyframe, step one frame past the last placeholder
                                // PTS so the real frame's timestamp is strictly greater, then let
                                // the normal path timestamp it (vid_cam_last is still None, so
                                // video_ts_from_camera returns this accumulated value).
                                if !seen_real_keyframe {
                                    if is_video(&timed.media) && is_video_keyframe(&timed.media) {
                                        stats.vid_pts_us =
                                            stats.vid_pts_us.saturating_add(frame_step_us);
                                        seen_real_keyframe = true;
                                        log::info!("{stream_label}: first camera keyframe received; switching from keepalive to live");
                                    } else {
                                        continue;
                                    }
                                }

                                // ---- Backpressure + lag handling ----
                                let vid_fill = vid_src
                                    .as_ref()
                                    .map(|s| {
                                        s.current_level_bytes() as f32 / s.max_bytes().max(1) as f32
                                    })
                                    .unwrap_or(0.0);

                                let in_backpressure = vid_fill > backpressure_threshold;

                                let elapsed = start.elapsed();
                                let vid_ts_dur = Duration::from_micros(vid_ts);
                                let lag = elapsed.checked_sub(vid_ts_dur).unwrap_or(Duration::ZERO);

                                if in_backpressure
                                    && lag >= lag_enter_threshold
                                    && !drop_until_keyframe
                                {
                                    log::warn!("{stream_label}: CATCHUP enter (lag={:?}), drop until keyframe", lag);
                                    drop_until_keyframe = true;
                                }

                                if drop_until_keyframe
                                    && lag <= lag_exit_threshold
                                    && !in_backpressure
                                {
                                    log::warn!("{stream_label}: CATCHUP exit (lag={:?})", lag);
                                    drop_until_keyframe = false;
                                }

                                if drop_until_keyframe
                                    && is_video(&timed.media)
                                    && !is_video_keyframe(&timed.media)
                                {
                                    // Keep the camera-clock advancing even for dropped frames.
                                    if let Some(us) = video_microseconds(&timed.media) {
                                        vid_ts = stats.video_ts_from_camera(us);
                                    }
                                    stats.p_dropped_backpressure += 1;
                                    continue;
                                }

                                let should_drop_p =
                                    in_backpressure && matches!(timed.media, BcMedia::Pframe(_));
                                if should_drop_p {
                                    stats.p_dropped_backpressure += 1;
                                    if let Some(us) = video_microseconds(&timed.media) {
                                        vid_ts = stats.video_ts_from_camera(us);
                                    }
                                    continue;
                                }

                                match send_to_sources(
                                    timed.media,
                                    &stream_label,
                                    &mut pools,
                                    &vid_src,
                                    &aud_src,
                                    &mut vid_ts,
                                    &mut aud_ts,
                                    &stream_config,
                                    true,
                                    &mut vid_pacer,
                                    &mut aud_pacer,
                                    &mut stats,
                                    Some(timed.recv_at),
                                )? {
                                    PushOutcome::Gone => {
                                        log::info!("{stream_label}: Client disconnected, stopping camera relay");
                                        break;
                                    }
                                    // NotLinked after PLAY is handled by the link /
                                    // disconnect_grace checks at the top of the loop.
                                    PushOutcome::Pushed | PushOutcome::NotLinked => {}
                                }
                            }

                            log::info!("{stream_label}: Camera relay disconnected");
                            AnyResult::Ok(())
                        });
                        AnyResult::Ok(())
                    });
                }
            }
        }
        AnyResult::Ok(())
            } => v,
        };
        if let Err(e) = &r {
            log::warn!("RTSP factory message-handler task ended with error: {e:?}");
        }
        r
    });

    log::debug!("Setting up factory with custom callback");

    // Now setup the factory
    let factory = NeoMediaFactory::new_with_callback(move |element| {
        log::debug!("Factory callback invoked for new client");
        let (reply, new_element) = tokio::sync::oneshot::channel();

        log::debug!("Sending NewClient message via channel");
        if let Err(e) = client_tx.blocking_send(ClientMsg::NewClient { element, reply }) {
            log::error!("Failed to send NewClient message: {:?}", e);
            return Err(anyhow::anyhow!("Failed to send NewClient message: {:?}", e));
        }
        log::debug!("NewClient message sent successfully");

        log::debug!("Waiting for pipeline element response...");
        match new_element.blocking_recv() {
            Ok(SetupOutcome::Pipeline(element)) => {
                log::debug!("Factory callback received pipeline element successfully");
                Ok(Some(element))
            }
            Ok(SetupOutcome::Unavailable) => {
                // Expected when the camera is offline; the client's SETUP fails and
                // it retries. Not an error.
                log::debug!("Factory callback: camera unavailable, no pipeline served");
                Ok(None)
            }
            Err(e) => {
                // The setup task ended without replying — a genuine internal fault.
                log::error!("Factory callback: setup task ended without replying: {e:?}");
                Err(anyhow::anyhow!("setup task ended without replying: {e:?}"))
            }
        }
    })
    .await?;

    log::debug!("Factory created successfully with callback registered");
    Ok((factory, thread))
}

fn is_video_keyframe(m: &BcMedia) -> bool {
    matches!(m, BcMedia::Iframe(_))
}

fn is_video(m: &BcMedia) -> bool {
    matches!(m, BcMedia::Iframe(_) | BcMedia::Pframe(_))
}

/// The camera's capture timestamp (microseconds) for a video frame, if present.
fn video_microseconds(m: &BcMedia) -> Option<u32> {
    match m {
        BcMedia::Iframe(f) => Some(f.microseconds),
        BcMedia::Pframe(f) => Some(f.microseconds),
        _ => None,
    }
}

/// Outcome of pushing a buffer to an appsrc / sending a frame to the sources.
enum PushOutcome {
    /// Buffer accepted (or nothing to push for this frame).
    Pushed,
    /// Pad not linked yet. This is normal before the RTSP client reaches PLAY, so
    /// it is tolerated; the live loop's link/`disconnect_grace`/activation-deadline
    /// checks decide when an unlinked client is actually gone.
    NotLinked,
    /// The appsrc/pipeline is gone (flushing, EOS, closed, or another hard push
    /// error): stop this client's relay.
    Gone,
}

fn send_to_sources(
    data: BcMedia,
    stream_label: &str,
    pools: &mut HashMap<usize, gstreamer::BufferPool>,
    vid_src: &Option<AppSrc>,
    aud_src: &Option<AppSrc>,
    vid_ts: &mut u64,
    aud_ts: &mut u64,
    stream_config: &StreamConfig,
    pace: bool,
    vid_pacer: &mut Option<PacerState>,
    aud_pacer: &mut Option<PacerState>,
    stats: &mut StreamStats,
    recv_at: Option<std::time::Instant>,
) -> AnyResult<PushOutcome> {
    // Update TS
    match data {
        BcMedia::Aac(aac) => {
            let Some(info) = aac.duration_info() else {
                log::warn!("{stream_label}: dropping AAC frame with unparseable duration");
                return Ok(PushOutcome::Pushed);
            };
            let aac_len = aac.data.len();
            if let Some(recv_at) = recv_at {
                stats.record_aac_gap(recv_at);
            }
            let dur = Duration::from_micros(info.duration_us as u64);
            let outcome = if let Some(aud_src) = aud_src.as_ref() {
                // Audio carries no camera timestamp: ride a content-clock (advance by
                // the frame's own duration), anchored once to the video camera-clock so
                // it stays aligned with video without following bursty network arrival.
                if !stats.aud_anchored {
                    stats.aud_anchored = true;
                    *aud_ts = stats.last_vid_pts_us;
                }
                let ts_us = *aud_ts;
                log::debug!("Sending AAC: {:?}", Duration::from_micros(ts_us));
                let outcome = send_to_appsrc(
                    aud_src,
                    stream_label,
                    aac.data,
                    Duration::from_micros(ts_us),
                    Some(dur),
                    pools,
                    pace,
                    aud_pacer,
                )?;
                if matches!(outcome, PushOutcome::Pushed) {
                    stats.aud_pushed += 1;
                }
                *aud_ts = ts_us + info.duration_us as u64;
                outcome
            } else {
                PushOutcome::Pushed
            };
            stats.record_aac(&info, aac_len);
            Ok(outcome)
        }
        BcMedia::Adpcm(adpcm) => {
            let Some(duration) = adpcm.duration() else {
                log::warn!("{stream_label}: dropping ADPCM frame with unparseable duration");
                return Ok(PushOutcome::Pushed);
            };
            let dur = Duration::from_micros(duration as u64);
            if let Some(aud_src) = aud_src.as_ref() {
                if !stats.aud_anchored {
                    stats.aud_anchored = true;
                    *aud_ts = stats.last_vid_pts_us;
                }
                let ts_us = *aud_ts;
                log::trace!("Sending ADPCM: {:?}", Duration::from_micros(ts_us));
                let outcome = send_to_appsrc(
                    aud_src,
                    stream_label,
                    adpcm.data,
                    Duration::from_micros(ts_us),
                    Some(dur),
                    pools,
                    pace,
                    aud_pacer,
                )?;
                if matches!(outcome, PushOutcome::Pushed) {
                    stats.aud_pushed += 1;
                }
                *aud_ts = ts_us + duration as u64;
                return Ok(outcome);
            }
            Ok(PushOutcome::Pushed)
        }
        BcMedia::Iframe(BcMediaIframe {
            data, microseconds, ..
        }) => {
            if let Some(vid_src) = vid_src.as_ref() {
                if let Some(recv_at) = recv_at {
                    stats.record_video_gap(recv_at);
                }
                let ts_us = stats.video_ts_from_camera(microseconds);
                let frame_dur =
                    Duration::from_micros(1_000_000u64 / stream_config.fps.max(1) as u64);
                log::trace!("Sending I-frame: {:?}", Duration::from_micros(ts_us));
                let outcome = send_to_appsrc(
                    vid_src,
                    stream_label,
                    data,
                    Duration::from_micros(ts_us),
                    Some(frame_dur),
                    pools,
                    pace,
                    vid_pacer,
                )?;
                if matches!(outcome, PushOutcome::Pushed) {
                    stats.vid_pushed += 1;
                }
                *vid_ts = ts_us;
                return Ok(outcome);
            }
            Ok(PushOutcome::Pushed)
        }
        BcMedia::Pframe(BcMediaPframe {
            data, microseconds, ..
        }) => {
            if let Some(vid_src) = vid_src.as_ref() {
                // Intelligent frame dropping: drop P-frames when buffer is getting full
                // to prioritize I-frames (keyframes) which can restart the stream
                let level = vid_src.current_level_bytes();
                let max = vid_src.max_bytes();
                if let Some(recv_at) = recv_at {
                    stats.record_video_gap(recv_at);
                }
                let ts_us = stats.video_ts_from_camera(microseconds);
                let frame_dur =
                    Duration::from_micros(1_000_000u64 / stream_config.fps.max(1) as u64);

                let outcome = if max > 0 && level >= max * 80 / 100 {
                    stats.p_dropped_pressure += 1;
                    log::trace!(
                        "Dropping P-frame due to buffer pressure ({}/{} bytes, {}%)",
                        level,
                        max,
                        level * 100 / max
                    );
                    PushOutcome::Pushed
                } else {
                    log::trace!("Sending P-frame: {:?}", Duration::from_micros(ts_us));
                    let outcome = send_to_appsrc(
                        vid_src,
                        stream_label,
                        data,
                        Duration::from_micros(ts_us),
                        Some(frame_dur),
                        pools,
                        pace,
                        vid_pacer,
                    )?;
                    if matches!(outcome, PushOutcome::Pushed) {
                        stats.vid_pushed += 1;
                    }
                    outcome
                };
                *vid_ts = ts_us;
                return Ok(outcome);
            }
            Ok(PushOutcome::Pushed)
        }
        _ => Ok(PushOutcome::Pushed),
    }
}

fn bucket_size_for(n: usize) -> Option<usize> {
    const MIN_BUCKET: usize = 256;
    const MAX_BUCKET: usize = 64 * 1024;
    if n == 0 {
        return Some(MIN_BUCKET);
    }
    if n > MAX_BUCKET {
        return None;
    }
    let mut b = n.next_power_of_two();
    if b < MIN_BUCKET {
        b = MIN_BUCKET;
    }
    Some(b)
}

fn acquire_pooled_buffer(
    pools: &mut std::collections::HashMap<usize, gstreamer::BufferPool>,
    data: &[u8],
    timestamp: gstreamer::ClockTime,
) -> AnyResult<gstreamer::Buffer> {
    let needed = data.len();
    if let Some(bucket) = bucket_size_for(needed) {
        let pool = pools.entry(bucket).or_insert_with(|| {
            let pool = gstreamer::BufferPool::new();
            let mut cfg = pool.config();
            // caps=None, size=bucket, min=8, max=64
            cfg.set_params(None, bucket as u32, 8, 64);
            pool.set_config(cfg).expect("pool config failed");
            pool.set_active(true).expect("activate pool");
            log::info!("New BufferPool (Bucket) allocated: size={bucket}");
            pool
        });

        let mut buf = pool.acquire_buffer(None)?;
        {
            let buf_ref = buf.get_mut().unwrap();
            buf_ref.set_dts(timestamp);
            buf_ref.set_pts(timestamp);
            {
                let mut map = buf_ref.map_writable().unwrap();
                map[..needed].copy_from_slice(data);
            }
            if bucket > needed {
                let _ = buf_ref.set_size(needed);
            }
        }
        Ok(buf)
    } else {
        // Fallback without pooling
        let mut buf =
            gstreamer::Buffer::with_size(needed).context("allocate large non-pooled buffer")?;
        {
            let buf_ref = buf.get_mut().unwrap();
            buf_ref.set_dts(timestamp);
            buf_ref.set_pts(timestamp);
            let mut map = buf_ref.map_writable().unwrap();
            map.copy_from_slice(data);
        }
        Ok(buf)
    }
}

fn send_to_appsrc(
    appsrc: &gstreamer_app::AppSrc,
    stream_label: &str,
    data: Vec<u8>,
    ts: std::time::Duration,
    duration: Option<std::time::Duration>,
    pools: &mut std::collections::HashMap<usize, gstreamer::BufferPool>,
    pace: bool,
    pacer: &mut Option<PacerState>,
) -> AnyResult<PushOutcome> {
    // Treat shutdown as clean stop
    // if is_closed(appsrc) {
    //     return Ok(());
    // }

    // (don’t early-return on !linked — push_buffer will return NotLinked and we already ignore it)

    if pace {
        let now = std::time::Instant::now();
        if let Some(st) = pacer.as_ref() {
            let target = st.target_instant(ts);
            if target > now {
                std::thread::sleep(target - now);
            }
        }
    }

    let timestamp = ClockTime::from_nseconds(ts.as_nanos() as u64);
    let mut buf = acquire_pooled_buffer(pools, &data, timestamp)?;

    if let Some(dur) = duration {
        let dur_ct = ClockTime::from_nseconds(dur.as_nanos() as u64);
        if let Some(b) = buf.get_mut() {
            b.set_duration(dur_ct);
        }
    }

    let push_start = std::time::Instant::now();
    let result = appsrc.push_buffer(buf);
    let push_elapsed = push_start.elapsed();
    if push_elapsed >= Duration::from_millis(20) {
        let level = appsrc.current_level_bytes();
        let max = appsrc.max_bytes();
        let app_name = appsrc.name();
        if push_elapsed >= Duration::from_millis(100) {
            log::warn!(
                "{stream_label}: appsrc {} push_buffer took {:?} (level {}/{} bytes)",
                app_name,
                push_elapsed,
                level,
                max
            );
        } else {
            log::debug!(
                "{stream_label}: appsrc {} push_buffer took {:?} (level {}/{} bytes)",
                app_name,
                push_elapsed,
                level,
                max
            );
        }
    }

    Ok(match result {
        Ok(_) => PushOutcome::Pushed,
        Err(gstreamer::FlowError::NotLinked) => {
            // Normal before the client reaches PLAY (pads not linked yet); tolerate.
            log::debug!("{stream_label}: push_buffer => NOT_LINKED");
            PushOutcome::NotLinked
        }
        Err(gstreamer::FlowError::Flushing) => {
            log::debug!("{stream_label}: push_buffer => FLUSHING");
            PushOutcome::Gone
        }
        Err(e) => {
            log::warn!("{stream_label}: push_buffer failed: {e:?}");
            PushOutcome::Gone
        }
    })
}

#[derive(Clone, Debug)]
struct PacerState {
    base_instant: std::time::Instant,
    base_ts: std::time::Duration,
}

impl PacerState {
    fn new(now: std::time::Instant, ts: std::time::Duration) -> Self {
        Self {
            base_instant: now,
            base_ts: ts,
        }
    }

    fn target_instant(&self, ts: std::time::Duration) -> std::time::Instant {
        if ts >= self.base_ts {
            self.base_instant + (ts - self.base_ts)
        } else {
            // If timestamps go backwards, clamp to "now-ish" rather than panicking.
            self.base_instant
        }
    }
}

fn is_closed(app: &AppSrc) -> bool {
    let (_res, current, pending) = app.state(None);
    current == gstreamer::State::Null && pending == gstreamer::State::Null
}

fn is_linked(app: &AppSrc) -> bool {
    app.static_pad("src")
        .map(|p| p.is_linked())
        .unwrap_or(false)
}

fn clear_bin(bin: &Element) -> Result<()> {
    let bin = bin
        .clone()
        .dynamic_cast::<Bin>()
        .map_err(|_| anyhow!("Media source's element should be a bin"))?;
    // Clear the autogenerated ones
    for element in bin.iterate_elements().into_iter().flatten() {
        bin.remove(&element)?;
    }

    Ok(())
}

static KEEPALIVE_CACHE: Lazy<std::sync::Mutex<HashMap<(u8, u32, u32), Arc<Vec<u8>>>>> =
    Lazy::new(|| std::sync::Mutex::new(HashMap::new()));

/// A black keyframe (Annex-B byte-stream access unit) in the given codec at the
/// given resolution, used as low-rate keepalive RTP so a client that connected
/// before the camera produced frames doesn't time out. Encoded once per
/// (codec, resolution) via gstreamer and cached for the process; this is a single
/// encode, not a running encoder. Returns `None` (keepalive disabled) if the
/// resolution is unknown or encoding/validation fails.
fn keepalive_keyframe(vid_type: VideoType, resolution: [u32; 2]) -> Option<Arc<Vec<u8>>> {
    let (w, h) = (resolution[0], resolution[1]);
    if w == 0 || h == 0 {
        return None;
    }
    let codec_id = match vid_type {
        VideoType::H264 => 0u8,
        VideoType::H265 => 1u8,
    };
    let key = (codec_id, w, h);
    if let Ok(cache) = KEEPALIVE_CACHE.lock() {
        if let Some(bytes) = cache.get(&key) {
            return Some(bytes.clone());
        }
    }
    match encode_black_keyframe(vid_type, w, h) {
        Ok(bytes) => {
            let arc = Arc::new(bytes);
            if let Ok(mut cache) = KEEPALIVE_CACHE.lock() {
                cache.insert(key, arc.clone());
            }
            log::info!("keepalive: encoded {vid_type:?} {w}x{h} placeholder keyframe");
            Some(arc)
        }
        Err(e) => {
            log::warn!(
                "keepalive: could not build a {vid_type:?} {w}x{h} placeholder keyframe ({e:?}); serving without keepalive"
            );
            None
        }
    }
}

/// One-shot encode of a black keyframe with gstreamer (openh264enc / x265enc),
/// returning the first validated keyframe access unit as Annex-B byte-stream bytes.
fn encode_black_keyframe(vid_type: VideoType, width: u32, height: u32) -> AnyResult<Vec<u8>> {
    let (enc, parse, kind) = match vid_type {
        VideoType::H264 => ("openh264enc", "h264parse", "h264"),
        VideoType::H265 => ("x265enc", "h265parse", "h265"),
    };
    // I420 input + several frames so the encoder reliably emits an IDR with
    // parameter sets (a single non-I420 frame can fail to initialise).
    let desc = format!(
        "videotestsrc num-buffers=5 pattern=black ! \
         video/x-raw,format=I420,width={width},height={height},framerate=5/1 ! \
         {enc} ! {parse} config-interval=-1 ! \
         video/x-{kind},stream-format=byte-stream,alignment=au ! \
         appsink name=sink"
    );
    let pipeline = gstreamer::parse::launch(&desc)
        .context("keepalive: parse encode pipeline")?
        .downcast::<gstreamer::Pipeline>()
        .map_err(|_| anyhow!("keepalive: encode pipeline is not a Pipeline"))?;
    let sink = pipeline
        .by_name("sink")
        .context("keepalive: appsink missing")?
        .downcast::<AppSink>()
        .map_err(|_| anyhow!("keepalive: sink is not an AppSink"))?;
    pipeline.set_state(gstreamer::State::Playing)?;

    let mut found: Option<Vec<u8>> = None;
    for _ in 0..8 {
        let sample = match sink.pull_sample() {
            Ok(s) => s,
            Err(_) => break, // EOS / error
        };
        if let Some(buf) = sample.buffer() {
            let is_keyframe = !buf.flags().contains(gstreamer::BufferFlags::DELTA_UNIT);
            if is_keyframe {
                if let Ok(map) = buf.map_readable() {
                    let bytes = map.as_slice().to_vec();
                    if contains_keyframe_nals(&bytes, vid_type) {
                        found = Some(bytes);
                        break;
                    }
                }
            }
        }
    }
    let _ = pipeline.set_state(gstreamer::State::Null);
    found.ok_or_else(|| anyhow!("keepalive: no valid keyframe produced"))
}

/// Validate that an Annex-B access unit carries both a parameter set (SPS) and an
/// IDR slice for the codec, so a client can decode it standalone.
fn contains_keyframe_nals(bytes: &[u8], vid_type: VideoType) -> bool {
    let (mut has_idr, mut has_sps) = (false, false);
    let mut i = 0usize;
    while i + 3 < bytes.len() {
        if bytes[i] == 0 && bytes[i + 1] == 0 && bytes[i + 2] == 1 {
            let hdr = bytes[i + 3];
            match vid_type {
                VideoType::H264 => match hdr & 0x1f {
                    5 => has_idr = true,  // IDR slice
                    7 => has_sps = true,  // SPS
                    _ => {}
                },
                VideoType::H265 => match (hdr >> 1) & 0x3f {
                    19 | 20 => has_idr = true, // IDR_W_RADL / IDR_N_LP
                    33 => has_sps = true,      // SPS
                    _ => {}
                },
            }
            i += 3;
        } else {
            i += 1;
        }
    }
    has_idr && has_sps
}

/// Build the video + audio appsrc pipelines into `bin` from a (possibly cached)
/// StreamConfig, returning the appsrc handles to feed. A `None` video type builds
/// the splash / "unknown" pipeline instead and yields no video appsrc.
fn build_sources(
    bin: &Element,
    stream_config: &StreamConfig,
    splash_pattern: &str,
) -> AnyResult<(Option<AppSrc>, Option<AppSrc>)> {
    let vid_src = match stream_config.vid_type.as_ref() {
        Some(VideoType::H264) => Some(build_h264(bin, stream_config)?),
        Some(VideoType::H265) => Some(build_h265(bin, stream_config)?),
        None => {
            build_unknown(bin, splash_pattern)?;
            None
        }
    };
    let aud_src = match stream_config.aud_type.as_ref() {
        Some(AudioType::Aac) => Some(build_aac(bin, stream_config)?),
        Some(AudioType::Adpcm(block_size)) => Some(build_adpcm(bin, *block_size, stream_config)?),
        None => None,
    };
    if let Some(app) = vid_src.as_ref() {
        app.set_callbacks(
            AppSrcCallbacks::builder()
                .seek_data(move |_, _seek_pos| true)
                .build(),
        );
    }
    if let Some(app) = aud_src.as_ref() {
        app.set_callbacks(
            AppSrcCallbacks::builder()
                .seek_data(move |_, _seek_pos| true)
                .build(),
        );
    }
    Ok((vid_src, aud_src))
}

fn build_unknown(bin: &Element, pattern: &str) -> Result<()> {
    let bin = bin
        .clone()
        .dynamic_cast::<Bin>()
        .map_err(|_| anyhow!("Media source's element should be a bin"))?;
    log::debug!("Building Unknown Pipeline");
    let source = make_element("videotestsrc", "testvidsrc")?;
    source.set_property_from_str("pattern", pattern);
    source.set_property("num-buffers", 500i32); // Send buffers then EOS
    let queue = make_queue("queue0", 1024 * 1024 * 4)?;

    let overlay = make_element("textoverlay", "overlay")?;
    overlay.set_property("text", "Stream not Ready");
    overlay.set_property_from_str("valignment", "top");
    overlay.set_property_from_str("halignment", "left");
    overlay.set_property("font-desc", "Sans, 16");
    let encoder = make_element("jpegenc", "encoder")?;
    let payload = make_element("rtpjpegpay", "pay0")?;

    bin.add_many([&source, &queue, &overlay, &encoder, &payload])?;
    source.link_filtered(
        &queue,
        &Caps::builder("video/x-raw")
            .field("format", "YUY2")
            .field("width", 896i32)
            .field("height", 512i32)
            .field("framerate", gstreamer::Fraction::new(25, 1))
            .build(),
    )?;
    Element::link_many([&queue, &overlay, &encoder, &payload])?;

    Ok(())
}

struct Linked {
    appsrc: AppSrc,
    output: Element,
}

fn pipe_h264(bin: &Element, stream_config: &StreamConfig) -> Result<Linked> {
    let buffer_size = buffer_size_bytes(stream_config);
    log::debug!(
        "buffer_size: {buffer_size}, bitrate: {}",
        stream_config.bitrate
    );
    let bin = bin
        .clone()
        .dynamic_cast::<Bin>()
        .map_err(|_| anyhow!("Media source's element should be a bin"))?;
    log::debug!("Building H264 Pipeline");
    let source = make_element("appsrc", "vidsrc")?
        .dynamic_cast::<AppSrc>()
        .map_err(|_| anyhow!("Cannot cast to appsrc."))?;

    source.set_is_live(true);
    source.set_property("format", &gstreamer::Format::Time);
    source.set_do_timestamp(false); // ✅ was true
    source.set_block(false); // ok if you choose dropping
    source.set_property("emit-signals", false);
    source.set_stream_type(AppStreamType::Stream);
    source.set_max_bytes(buffer_size as u64);

    // Set caps so RTSP server can build SDP before data flows
    let caps = Caps::builder("video/x-h264")
        .field("stream-format", "byte-stream")
        .build();
    source.set_caps(Some(&caps));

    let source = source
        .dynamic_cast::<Element>()
        .map_err(|_| anyhow!("Cannot cast back"))?;
    let queue = make_queue("source_queue", buffer_size)?;
    let parser = make_element("h264parse", "parser")?;
    parser.set_property("disable-passthrough", true);
    // Tell parser to be more aggressive about fixing stream errors
    parser.set_property("config-interval", -1i32); // Force SPS/PPS insertion at parser level
                                                   // let stamper = make_element("h264timestamper", "stamper")?;

    bin.add_many([&source, &queue, &parser])?;
    Element::link_many([&source, &queue, &parser])?;

    let source = source
        .dynamic_cast::<AppSrc>()
        .map_err(|_| anyhow!("Cannot convert appsrc"))?;
    Ok(Linked {
        appsrc: source,
        output: parser,
    })
}

fn build_h264(bin: &Element, stream_config: &StreamConfig) -> Result<AppSrc> {
    let linked = pipe_h264(bin, stream_config)?;

    let bin = bin
        .clone()
        .dynamic_cast::<Bin>()
        .map_err(|_| anyhow!("Media source's element should be a bin"))?;

    let payload = make_element("rtph264pay", "pay0")?;
    // Configure payload for better timing and client compatibility
    payload.set_property("config-interval", -1i32); // Send SPS/PPS with every IDR frame
    payload.set_property_from_str("aggregate-mode", "zero-latency");

    bin.add_many([&payload])?;
    Element::link_many([&linked.output, &payload])?;
    Ok(linked.appsrc)
}

fn pipe_h265(bin: &Element, stream_config: &StreamConfig) -> Result<Linked> {
    let buffer_size = buffer_size_bytes(stream_config);
    let bin = bin
        .clone()
        .dynamic_cast::<Bin>()
        .map_err(|_| anyhow!("Media source's element should be a bin"))?;
    log::debug!("Building H265 Pipeline");
    let source = make_element("appsrc", "vidsrc")?
        .dynamic_cast::<AppSrc>()
        .map_err(|_| anyhow!("Cannot cast to appsrc."))?;

    source.set_is_live(true);
    source.set_property("format", &gstreamer::Format::Time);
    source.set_do_timestamp(false); // ✅ was true
    source.set_block(false); // ok if you choose dropping
    source.set_property("emit-signals", false);
    source.set_stream_type(AppStreamType::Stream);
    source.set_max_bytes(buffer_size as u64);

    // Set caps so RTSP server can build SDP before data flows
    let caps = Caps::builder("video/x-h265")
        .field("stream-format", "byte-stream")
        .build();
    source.set_caps(Some(&caps));

    let source = source
        .dynamic_cast::<Element>()
        .map_err(|_| anyhow!("Cannot cast back"))?;
    let queue = make_queue("source_queue", buffer_size)?;
    let parser = make_element("h265parse", "parser")?;
    parser.set_property("disable-passthrough", true);
    // Tell parser to be more aggressive about fixing stream errors
    parser.set_property("config-interval", -1i32); // Force VPS/SPS/PPS insertion at parser level
                                                   // let stamper = make_element("h265timestamper", "stamper")?;

    bin.add_many([&source, &queue, &parser])?;
    Element::link_many([&source, &queue, &parser])?;

    let source = source
        .dynamic_cast::<AppSrc>()
        .map_err(|_| anyhow!("Cannot convert appsrc"))?;
    Ok(Linked {
        appsrc: source,
        output: parser,
    })
}

fn build_h265(bin: &Element, stream_config: &StreamConfig) -> Result<AppSrc> {
    let linked = pipe_h265(bin, stream_config)?;

    let bin = bin
        .clone()
        .dynamic_cast::<Bin>()
        .map_err(|_| anyhow!("Media source's element should be a bin"))?;

    let payload = make_element("rtph265pay", "pay0")?;
    // Configure payload for better timing and client compatibility
    payload.set_property("config-interval", -1i32); // Send VPS/SPS/PPS with every IDR frame
    payload.set_property_from_str("aggregate-mode", "zero-latency");

    bin.add_many([&payload])?;
    Element::link_many([&linked.output, &payload])?;
    Ok(linked.appsrc)
}

fn pipe_aac(bin: &Element, _stream_config: &StreamConfig) -> Result<Linked> {
    // Audio seems to run at about 800kbs
    let buffer_size = 512 * 1416;
    let bin = bin
        .clone()
        .dynamic_cast::<Bin>()
        .map_err(|_| anyhow!("Media source's element should be a bin"))?;
    log::debug!("Building Aac pipeline");
    let source = make_element("appsrc", "audsrc")?
        .dynamic_cast::<AppSrc>()
        .map_err(|_| anyhow!("Cannot cast to appsrc."))?;

    source.set_is_live(true);
    source.set_property("format", &gstreamer::Format::Time);
    source.set_do_timestamp(false); // ✅ was true
    source.set_block(false); // ok if you choose dropping
    source.set_property("emit-signals", false);
    source.set_stream_type(AppStreamType::Stream);
    source.set_max_bytes(buffer_size as u64);

    // Set caps so RTSP server can build SDP before data flows
    let caps = Caps::builder("audio/mpeg")
        .field("mpegversion", 4i32)
        .build();
    source.set_caps(Some(&caps));

    let source = source
        .dynamic_cast::<Element>()
        .map_err(|_| anyhow!("Cannot cast back"))?;

    let queue = make_queue("audqueue", buffer_size)?;
    let parser = make_element("aacparse", "audparser")?;
    let decoder = match make_element("faad", "auddecoder_faad") {
        Ok(ele) => Ok(ele),
        Err(_) => make_element("avdec_aac", "auddecoder_avdec_aac"),
    }?;

    // The fallback
    let silence = make_element("audiotestsrc", "audsilence")?;
    silence.set_property_from_str("wave", "silence");
    let fallback_switch = make_element("fallbackswitch", "audfallbackswitch");
    if let Ok(fallback_switch) = fallback_switch.as_ref() {
        fallback_switch.set_property("timeout", 3u64 * 1_000_000_000u64);
        fallback_switch.set_property("immediate-fallback", true);
    }

    let audiorate = make_element("audiorate", "audrate")?;
    let encoder = make_element("audioconvert", "audencoder")?;

    bin.add_many([&source, &queue, &parser, &decoder, &audiorate, &encoder])?;
    if let Ok(fallback_switch) = fallback_switch.as_ref() {
        bin.add_many([&silence, fallback_switch])?;
        Element::link_many([
            &source,
            &queue,
            &parser,
            &decoder,
            fallback_switch,
            &audiorate,
            &encoder,
        ])?;
        Element::link_many([&silence, fallback_switch])?;
    } else {
        Element::link_many([&source, &queue, &parser, &decoder, &audiorate, &encoder])?;
    }

    let source = source
        .dynamic_cast::<AppSrc>()
        .map_err(|_| anyhow!("Cannot convert appsrc"))?;
    Ok(Linked {
        appsrc: source,
        output: encoder,
    })
}

fn build_aac(bin: &Element, stream_config: &StreamConfig) -> Result<AppSrc> {
    let linked = pipe_aac(bin, stream_config)?;

    let bin = bin
        .clone()
        .dynamic_cast::<Bin>()
        .map_err(|_| anyhow!("Media source's element should be a bin"))?;

    let payload = make_element("rtpL16pay", "pay1")?;
    bin.add_many([&payload])?;
    Element::link_many([&linked.output, &payload])?;
    Ok(linked.appsrc)
}

fn pipe_adpcm(bin: &Element, block_size: u32, _stream_config: &StreamConfig) -> Result<Linked> {
    let buffer_size = 512 * 1416;
    let bin = bin
        .clone()
        .dynamic_cast::<Bin>()
        .map_err(|_| anyhow!("Media source's element should be a bin"))?;
    log::debug!("Building Adpcm pipeline");
    // Original command line
    // caps=audio/x-adpcm,layout=dvi,block_align={},channels=1,rate=8000
    // ! queue silent=true max-size-bytes=10485760 min-threshold-bytes=1024
    // ! adpcmdec
    // ! audioconvert
    // ! rtpL16pay name=pay1

    let source = make_element("appsrc", "audsrc")?
        .dynamic_cast::<AppSrc>()
        .map_err(|_| anyhow!("Cannot cast to appsrc."))?;

    source.set_is_live(true);
    source.set_property("format", &gstreamer::Format::Time);
    source.set_do_timestamp(false); // ✅ was true
    source.set_block(false); // ok if you choose dropping
    source.set_property("emit-signals", false);
    source.set_stream_type(AppStreamType::Stream);
    source.set_max_bytes(buffer_size as u64);

    source.set_caps(Some(
        &Caps::builder("audio/x-adpcm")
            .field("layout", "div")
            .field("block_align", block_size as i32)
            .field("channels", 1i32)
            .field("rate", 8000i32)
            .build(),
    ));

    let source = source
        .dynamic_cast::<Element>()
        .map_err(|_| anyhow!("Cannot cast back"))?;

    let queue = make_queue("audqueue", buffer_size)?;
    let decoder = make_element("decodebin", "auddecoder")?;
    let audiorate = make_element("audiorate", "audrate")?;
    let encoder = make_element("audioconvert", "audencoder")?;
    let encoder_out = encoder.clone();
    let audiorate_for_pad = audiorate.clone();

    bin.add_many([&source, &queue, &decoder, &audiorate, &encoder])?;
    Element::link_many([&source, &queue, &decoder])?;
    decoder.connect_pad_added(move |_element, pad| {
        let sink_pad = audiorate_for_pad
            .static_pad("sink")
            .expect("Encoder is missing its pad");
        pad.link(&sink_pad)
            .expect("Failed to link ADPCM decoder to audiorate");
    });
    Element::link_many([&audiorate, &encoder])?;

    let source = source
        .dynamic_cast::<AppSrc>()
        .map_err(|_| anyhow!("Cannot convert appsrc"))?;
    Ok(Linked {
        appsrc: source,
        output: encoder_out,
    })
}

fn build_adpcm(bin: &Element, block_size: u32, stream_config: &StreamConfig) -> Result<AppSrc> {
    let linked = pipe_adpcm(bin, block_size, stream_config)?;

    let bin = bin
        .clone()
        .dynamic_cast::<Bin>()
        .map_err(|_| anyhow!("Media source's element should be a bin"))?;

    let payload = make_element("rtpL16pay", "pay1")?;
    bin.add_many([&payload])?;
    Element::link_many([&linked.output, &payload])?;
    Ok(linked.appsrc)
}

// Convenice funcion to make an element or provide a message
// about what plugin is missing
fn make_element(kind: &str, name: &str) -> AnyResult<Element> {
    ElementFactory::make_with_name(kind, Some(name)).with_context(|| {
        let plugin = match kind {
            "appsrc" => "app (gst-plugins-base)",
            "audioconvert" => "audioconvert (gst-plugins-base)",
            "audiorate" => "audiorate (gst-plugins-base)",
            "adpcmdec" => "Required for audio",
            "h264parse" => "videoparsersbad (gst-plugins-bad)",
            "h265parse" => "videoparsersbad (gst-plugins-bad)",
            "rtph264pay" => "rtp (gst-plugins-good)",
            "rtph265pay" => "rtp (gst-plugins-good)",
            "rtpjitterbuffer" => "rtp (gst-plugins-good)",
            "aacparse" => "audioparsers (gst-plugins-good)",
            "rtpL16pay" => "rtp (gst-plugins-good)",
            "x264enc" => "x264 (gst-plugins-ugly)",
            "x265enc" => "x265 (gst-plugins-bad)",
            "avdec_h264" => "libav (gst-libav)",
            "avdec_h265" => "libav (gst-libav)",
            "videotestsrc" => "videotestsrc (gst-plugins-base)",
            "imagefreeze" => "imagefreeze (gst-plugins-good)",
            "audiotestsrc" => "audiotestsrc (gst-plugins-base)",
            "decodebin" => "playback (gst-plugins-good)",
            "textoverlay" => "pango (gst-plugins-base)",
            "jpegenc" => "jpeg (gst-plugins-good)",
            "rtpjpegpay" => "rtp (gst-plugins-good)",
            "faad" => "faad (gst-plugins-bad)",
            "avdec_aac" => "libav (gst-libav)",
            "fallbackswitch" => "fallbackswitch (gst-plugins-rs / gst-plugins-bad)",
            "queue" => "coreelements (gstreamer)",
            _ => "Unknown",
        };
        format!(
            "Missing required gstreamer plugin `{}` for `{}` element",
            plugin, kind
        )
    })
}

struct StreamStats {
    last_report: std::time::Instant,

    vid_pushed: u64,
    aud_pushed: u64,
    p_dropped_pressure: u64,
    p_dropped_backpressure: u64,

    tryrecv_ok: u64,

    last_media_instant: std::time::Instant,

    aud_last_us: u32,
    aud_last_rate: u32,
    aud_last_adts_frames: u32,
    aud_last_raw_blocks: u32,
    aud_last_profile: u8,
    aud_last_sampling_index: u8,
    aud_last_channel_config: u8,
    aud_last_frame_len: u16,
    aud_last_header_len: u8,
    aud_last_payload_len: u16,
    aud_last_parsed_len: u16,
    aud_last_bytes: usize,
    aud_packets: u64,
    aud_dur_total_us: u64,
    aud_dur_min_us: u32,
    aud_dur_max_us: u32,

    aud_gap_last_instant: Option<std::time::Instant>,
    aud_gap_samples: u64,
    aud_gap_total_us: u64,
    aud_gap_min_us: u64,
    aud_gap_max_us: u64,

    vid_gap_last_instant: Option<std::time::Instant>,

    // Camera-clock timestamping. Video frames carry the camera's own capture
    // timestamp (microseconds, u32, wraps ~every 71 min); we rebase it to a
    // monotonic PTS. Audio carries no timestamp, so it rides a content-clock
    // (+frame duration) anchored to the video camera-clock on the first audio frame.
    vid_cam_last: Option<u32>,
    vid_pts_us: u64,
    last_vid_pts_us: u64,
    aud_anchored: bool,
}

impl StreamStats {
    fn new(now: std::time::Instant) -> Self {
        Self {
            last_report: now,
            last_media_instant: now,
            vid_pushed: 0,
            aud_pushed: 0,
            p_dropped_pressure: 0,
            p_dropped_backpressure: 0,
            tryrecv_ok: 0,
            aud_last_us: 0,
            aud_last_rate: 0,
            aud_last_adts_frames: 0,
            aud_last_raw_blocks: 0,
            aud_last_profile: 0,
            aud_last_sampling_index: 0,
            aud_last_channel_config: 0,
            aud_last_frame_len: 0,
            aud_last_header_len: 0,
            aud_last_payload_len: 0,
            aud_last_parsed_len: 0,
            aud_last_bytes: 0,
            aud_packets: 0,
            aud_dur_total_us: 0,
            aud_dur_min_us: u32::MAX,
            aud_dur_max_us: 0,

            aud_gap_last_instant: None,
            aud_gap_samples: 0,
            aud_gap_total_us: 0,
            aud_gap_min_us: u64::MAX,
            aud_gap_max_us: 0,

            vid_gap_last_instant: None,
            vid_cam_last: None,
            vid_pts_us: 0,
            last_vid_pts_us: 0,
            aud_anchored: false,
        }
    }

    fn record_aac(&mut self, info: &AacDurationInfo, bytes: usize) {
        self.aud_last_us = info.duration_us;
        self.aud_last_rate = info.sample_rate;
        self.aud_last_adts_frames = info.adts_frames;
        self.aud_last_raw_blocks = info.raw_blocks;
        self.aud_last_profile = info.profile;
        self.aud_last_sampling_index = info.sampling_index;
        self.aud_last_channel_config = info.channel_config;
        self.aud_last_frame_len = info.frame_length;
        self.aud_last_header_len = info.header_len;
        self.aud_last_payload_len = info.payload_len;
        self.aud_last_parsed_len = info.parsed_len;
        self.aud_last_bytes = bytes;
        self.aud_packets += 1;
        self.aud_dur_total_us += info.duration_us as u64;
        self.aud_dur_min_us = self.aud_dur_min_us.min(info.duration_us);
        self.aud_dur_max_us = self.aud_dur_max_us.max(info.duration_us);
    }

    fn record_aac_gap(&mut self, now: std::time::Instant) -> Option<u64> {
        if let Some(prev) = self.aud_gap_last_instant {
            let gap_us = now.duration_since(prev).as_micros() as u64;
            self.aud_gap_samples += 1;
            self.aud_gap_total_us += gap_us;
            self.aud_gap_min_us = self.aud_gap_min_us.min(gap_us);
            self.aud_gap_max_us = self.aud_gap_max_us.max(gap_us);
            self.aud_gap_last_instant = Some(now);
            return Some(gap_us);
        }
        self.aud_gap_last_instant = Some(now);
        None
    }

    fn record_video_gap(&mut self, now: std::time::Instant) -> Option<u64> {
        if let Some(prev) = self.vid_gap_last_instant {
            let gap_us = now.duration_since(prev).as_micros() as u64;
            self.vid_gap_last_instant = Some(now);
            return Some(gap_us);
        }
        self.vid_gap_last_instant = Some(now);
        None
    }

    /// Convert the camera's per-frame microsecond timestamp into a monotonic PTS
    /// rebased so the first video frame is 0. The camera value is a `u32` that wraps
    /// (~every 71 min), so we accumulate per-frame deltas (wrap-safe via
    /// `wrapping_sub`) rather than subtracting an absolute base. Implausibly large
    /// deltas (> 10 s) are treated as a camera clock reset and skipped, holding the
    /// PTS steady rather than jumping. Also records the value as the anchor point for
    /// the audio content-clock.
    fn video_ts_from_camera(&mut self, micros: u32) -> u64 {
        if let Some(last) = self.vid_cam_last {
            let delta = micros.wrapping_sub(last) as u64;
            if delta < 10_000_000 {
                self.vid_pts_us = self.vid_pts_us.saturating_add(delta);
            }
            // else: implausible jump/reset -> hold PTS, just re-sync the reference below
        }
        self.vid_cam_last = Some(micros);
        self.last_vid_pts_us = self.vid_pts_us;
        self.vid_pts_us
    }

    fn aud_stats_snapshot(&self) -> AudStatsSnapshot {
        if self.aud_packets == 0 {
            return AudStatsSnapshot::empty();
        }
        let avg = self.aud_dur_total_us / self.aud_packets;
        let min = self.aud_dur_min_us;
        let max = self.aud_dur_max_us;
        let gap_avg = if self.aud_gap_samples == 0 {
            0
        } else {
            self.aud_gap_total_us / self.aud_gap_samples
        };
        let gap_min = if self.aud_gap_samples == 0 {
            0
        } else {
            self.aud_gap_min_us
        };
        let gap_max = if self.aud_gap_samples == 0 {
            0
        } else {
            self.aud_gap_max_us
        };
        AudStatsSnapshot {
            avg_us: avg,
            min_us: min,
            max_us: max,
            total_us: self.aud_dur_total_us,
            packets: self.aud_packets,
            gap_avg_us: gap_avg,
            gap_min_us: gap_min,
            gap_max_us: gap_max,
        }
    }

    fn reset_aud_stats(&mut self) {
        self.aud_packets = 0;
        self.aud_dur_total_us = 0;
        self.aud_dur_min_us = u32::MAX;
        self.aud_dur_max_us = 0;
        self.aud_gap_samples = 0;
        self.aud_gap_total_us = 0;
        self.aud_gap_min_us = u64::MAX;
        self.aud_gap_max_us = 0;
    }
}

#[derive(Debug, Clone, Copy)]
struct AudStatsSnapshot {
    avg_us: u64,
    min_us: u32,
    max_us: u32,
    total_us: u64,
    packets: u64,
    gap_avg_us: u64,
    gap_min_us: u64,
    gap_max_us: u64,
}

impl AudStatsSnapshot {
    fn empty() -> Self {
        Self {
            avg_us: 0,
            min_us: 0,
            max_us: 0,
            total_us: 0,
            packets: 0,
            gap_avg_us: 0,
            gap_min_us: 0,
            gap_max_us: 0,
        }
    }
}

fn make_queue(name: &str, buffer_size: u32) -> AnyResult<Element> {
    let queue = make_element("queue", &format!("queue1_{}", name))?;
    queue.set_property("max-size-bytes", buffer_size);
    queue.set_property("max-size-buffers", 0u32);
    queue.set_property("max-size-time", 0u64);

    // Alternatives:
    // "downstream" → drop oldest buffers (best for live)
    // "upstream"   → block upstream instead (higher latency)
    //queue.set_property_from_str("leaky", "downstream");

    Ok(queue)
}

fn buffer_size_bytes(stream_config: &StreamConfig) -> u32 {
    let bitrate = stream_config.bitrate.max(1) as u64;
    let fps = stream_config.fps.max(1) as u64;
    let bytes_per_sec = (bitrate + 7) / 8;
    let buffer_ms = stream_config.buffer_duration_ms.max(1000);
    let base = bytes_per_sec.saturating_mul(buffer_ms) / 1000;
    let max_frame_guess = max_frame_guess_bytes(bytes_per_sec, fps);
    let target = base.max(max_frame_guess.saturating_mul(4)).max(128 * 1024);
    target.min(u32::MAX as u64) as u32
}

fn max_frame_guess_bytes(bytes_per_sec: u64, fps: u64) -> u64 {
    let avg_frame = bytes_per_sec / fps.max(1);
    // Headroom for bursty I-frames (~6x the average frame size).
    const FACTOR: u64 = 6;
    avg_frame.saturating_mul(FACTOR)
}

fn media_queue_capacity(stream_config: &StreamConfig) -> usize {
    let fps = stream_config.fps.max(1) as u64;
    let buffer_ms = stream_config.buffer_duration_ms.max(1000);
    let base = (fps * buffer_ms) / 1000;
    let min_capacity = if stream_config.bitrate >= 2_000_000 {
        1000
    } else {
        300
    };
    let target = base.saturating_mul(4) as usize;
    target.max(min_capacity).min(5000)
}
