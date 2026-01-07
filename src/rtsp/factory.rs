use gstreamer::ClockTime;
use std::{collections::HashMap, time::Duration};

use anyhow::{anyhow, Context, Result};
use gstreamer::{prelude::*, Bin, Caps, Element, ElementFactory, GhostPad};
use gstreamer_app::{AppSrc, AppSrcCallbacks, AppStreamType};
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

use crate::{common::NeoInstance, rtsp::gst::NeoMediaFactory, AnyResult};

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
    bitrate_table: Vec<u32>,
    fps_table: Vec<u32>,
    vid_type: Option<VideoType>,
    aud_type: Option<AudioType>,
}
impl StreamConfig {
    async fn new(instance: &NeoInstance, name: StreamKind) -> AnyResult<Self> {
        let (resolution, bitrate, fps, fps_table, bitrate_table) = instance
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
                            bitrate_table.clone(),
                        ))
                    } else {
                        Ok(([0, 0], 0, 0, vec![], vec![]))
                    }
                })
            })
            .await?;

        Ok(StreamConfig {
            resolution,
            bitrate,
            fps,
            fps_table,
            bitrate_table,
            vid_type: None,
            aud_type: None,
        })
    }

    fn update_fps(&mut self, fps: u32) {
        let new_fps = self.fps_table.get(fps as usize).copied().unwrap_or(fps);
        self.fps = new_fps;
    }
    #[allow(dead_code)]
    fn update_bitrate(&mut self, bitrate: u32) {
        let new_bitrate = self
            .bitrate_table
            .get(bitrate as usize)
            .copied()
            .unwrap_or(bitrate);
        self.bitrate = new_bitrate;
    }

    fn update_from_media(&mut self, media: &BcMedia) {
        match media {
            BcMedia::InfoV1(BcMediaInfoV1 { fps, .. })
            | BcMedia::InfoV2(BcMediaInfoV2 { fps, .. }) => self.update_fps(*fps as u32),
            BcMedia::Aac(_) => {
                self.aud_type = Some(AudioType::Aac);
            }
            BcMedia::Adpcm(adpcm) => {
                self.aud_type = Some(AudioType::Adpcm(adpcm.block_size()));
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

enum ClientMsg {
    NewClient {
        element: Element,
        reply: tokio::sync::oneshot::Sender<Element>,
    },
}

struct TimedMedia {
    recv_at: std::time::Instant,
    media: BcMedia,
}

pub(super) async fn make_factory(
    camera: NeoInstance,
    stream: StreamKind,
) -> AnyResult<(NeoMediaFactory, JoinHandle<AnyResult<()>>)> {
    log::debug!("make_factory called for stream {:?}", stream);

    let (client_tx, mut client_rx) = tokio_mpsc::channel(100);

    log::debug!("Creating factory for stream {:?}", stream);

    // Create the task that creates the pipelines
    let thread = tokio::task::spawn(async move {
        let name = camera.config().await?.borrow().name.clone();
        log::info!("{name}::{stream}: Message handler task started, waiting for messages");

        while let Some(msg) = client_rx.recv().await {
            log::debug!("{name}::{stream}: Received message in handler");
            match msg {
                ClientMsg::NewClient { element, reply } => {
                    log::debug!("NewClient message received for {name}::{stream}");
                    let camera = camera.clone();
                    let name = name.clone();
                    tokio::task::spawn(async move {
                        clear_bin(&element)?;
                        log::info!("{name}::{stream}: Factory received new client, setting up pipeline");

                        // Acquire permit - this triggers camera relay connection
                        // IMPORTANT: Must be moved into blocking thread to keep it alive
                        log::debug!("{name}::{stream}: Acquiring permit to trigger camera connection");
                        let permit = camera.permit().await?;
                        log::debug!("{name}::{stream}: Permit acquired successfully");

                        // Start the camera relay connection
                        let config = camera.config().await?.borrow().clone();
                        let mut media_rx = camera.stream_while_live(stream).await?;

                        log::info!("{name}::{stream}: Camera relay established");

                        log::info!("{name}::{stream}: Learning camera stream type");
                        // Learn the camera data type
                        let mut buffer = vec![];
                        let mut frame_count = 0usize;

                        let camera_id = config
                            .camera_uid
                            .clone()
                            .unwrap_or_else(|| config.name.clone());
                        let cache_key = StreamCacheKey {
                            camera_id,
                            stream,
                        };

                        let mut stream_config = StreamConfig::new(&camera, stream).await?;
                        if let Some(cached) = cached_stream_types(&cache_key).await {
                            if cached.vid_type.is_some() || cached.aud_type.is_some() {
                                log::info!(
                                    "{name}::{stream}: Using cached stream types: video={:?}, audio={:?}",
                                    cached.vid_type,
                                    cached.aud_type
                                );
                            }
                            stream_config.vid_type = cached.vid_type;
                            stream_config.aud_type = cached.aud_type;
                        }
                        let cached_both =
                            stream_config.vid_type.is_some() && stream_config.aud_type.is_some();
                        let cached_any =
                            stream_config.vid_type.is_some() || stream_config.aud_type.is_some();
                        let buffer_target = if cached_both {
                            3
                        } else if cached_any {
                            8
                        } else {
                            15
                        };

                        log::info!("{name}::{stream}: Waiting for media frames from camera");
                        while let Some(media) = media_rx.recv().await {
                            log::debug!("{name}::{stream}: Received media frame #{}", frame_count);
                            stream_config.update_from_media(&media);
                            buffer.push(media);
                            frame_count += 1;
                            // Buffer a few frames before building the pipeline (shorter when cached types exist).
                            if frame_count >= buffer_target
                                || (frame_count >= 10
                                    && stream_config.vid_type.is_some()
                                    && stream_config.aud_type.is_some())
                            {
                                log::info!("{name}::{stream}: Stream type learned: video={:?}, audio={:?}",
                                    stream_config.vid_type, stream_config.aud_type);
                                break;
                            }
                        }

                        if stream_config.vid_type.is_none() && stream_config.aud_type.is_none() {
                            log::warn!("{name}::{stream}: No media received from camera, building fallback pipeline");
                        } else {
                            store_stream_types(
                                cache_key.clone(),
                                StreamTypeCache {
                                    vid_type: stream_config.vid_type.clone(),
                                    aud_type: stream_config.aud_type.clone(),
                                },
                            )
                            .await;
                        }

                        log::trace!("{name}::{stream}: Building the pipeline");
                        // Build the right video pipeline
                        let vid_src = match stream_config.vid_type.as_ref() {
                            Some(VideoType::H264) => {
                                let src = build_h264(&element, &stream_config)?;
                                AnyResult::Ok(Some(src))
                            }
                            Some(VideoType::H265) => {
                                let src = build_h265(&element, &stream_config)?;
                                AnyResult::Ok(Some(src))
                            }
                            None => {
                                build_unknown(&element, &config.splash_pattern.to_string())?;
                                AnyResult::Ok(None)
                            }
                        }?;

                        // Build the right audio pipeline
                        let aud_src = match stream_config.aud_type.as_ref() {
                            Some(AudioType::Aac) => {
                                let src = build_aac(&element, &stream_config)?;
                                AnyResult::Ok(Some(src))
                            }
                            Some(AudioType::Adpcm(block_size)) => {
                                let src = build_adpcm(&element, *block_size, &stream_config)?;
                                AnyResult::Ok(Some(src))
                            }
                            None => AnyResult::Ok(None),
                        }?;

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

                        log::trace!("{name}::{stream}: Sending pipeline to gstreamer");
                        // Send the pipeline back to the fac
                        // tory so it can start
                        let _ = reply.send(element);


                        // ---- Clean fix: keep tokio receiver async; forward into bounded std channel ----
                        let (std_tx, std_rx): (
                            std_mpsc::SyncSender<TimedMedia>,
                            std_mpsc::Receiver<TimedMedia>,
                        ) = std_mpsc::sync_channel(200);

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

                        // Forwarder task: runs on tokio, owns media_rx
                        tokio::spawn(async move {
                            while let Some(m) = media_rx.recv().await {
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
                                    }
                                    Err(TrySendError::Disconnected(_)) => break,
                                }
                            }
                            // Dropping std_tx will cause std_rx.recv_timeout to return Disconnected
                        });

                        // Run blocking code in tokio's blocking thread pool
                        // This maintains the tokio runtime context needed for permit drop
                        // Move permit into this thread to keep it alive for the session duration
                        tokio::task::spawn_blocking(move || {
                            use std::time::{Duration, Instant};

                            let start = Instant::now();
                            let _permit = permit; // hold for lifetime

                            // Wait for the RTSP server to link the pads (client reaches PLAY).
                            let link_deadline = Instant::now() + Duration::from_secs(2);
                            loop {
                                let vid_linked = vid_src.as_ref().map(|s| is_linked(s)).unwrap_or(true);
                                let aud_linked = aud_src.as_ref().map(|s| is_linked(s)).unwrap_or(true);

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

                            let mut pools: HashMap<usize, gstreamer::BufferPool> = Default::default();
                            let mut stats = StreamStats::new(Instant::now());

                            let frame_step_us: u64 = {
                                let fps = stream_config.fps.max(1) as u64;
                                1_000_000u64 / fps
                            };

                            let mut drop_until_keyframe: bool = false;
                            let lag_enter_threshold = Duration::from_millis(500);
                            let lag_exit_threshold  = Duration::from_millis(150);
                            let mut client_active = false;
                            let mut last_linked = Instant::now();
                            let client_active_deadline = Instant::now() + Duration::from_secs(10);
                            let disconnect_grace = Duration::from_secs(2);

                            // Send buffered frames (no pacing)
                            for buffered in buffer.drain(..) {
                                if let Err(e) = send_to_sources(
                                    buffered,
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
                                ) {
                                    if e.to_string().contains("App source is closed") {
                                        log::info!("{name}::{stream}: Client disconnected, stopping camera relay");
                                        return AnyResult::Ok(());
                                    }
                                    return Err(e);
                                }
                            }

                            // Live loop: recv_timeout so heartbeat + disconnect checks keep running
                            loop {
                                let vid_closed = vid_src.as_ref().map(|src| is_closed(src)).unwrap_or(true);
                                let aud_closed = aud_src.as_ref().map(|src| is_closed(src)).unwrap_or(true);
                                let vid_linked = vid_src.as_ref().map(|s| is_linked(s)).unwrap_or(false);
                                let aud_linked = aud_src.as_ref().map(|s| is_linked(s)).unwrap_or(false);

                                if vid_linked || aud_linked {
                                    client_active = true;
                                    last_linked = Instant::now();
                                }

                                if client_active {
                                    if vid_closed && aud_closed {
                                        log::info!("{name}::{stream}: Client disconnected, stopping camera relay");
                                        break;
                                    }
                                    if !(vid_linked || aud_linked)
                                        && Instant::now().duration_since(last_linked) >= disconnect_grace
                                    {
                                        log::info!("{name}::{stream}: Client disconnected, stopping camera relay");
                                        break;
                                    }
                                } else if Instant::now() >= client_active_deadline {
                                    log::info!("{name}::{stream}: Client never reached PLAY, stopping camera relay");
                                    break;
                                }

                                // once-per-second heartbeat
                                let now = Instant::now();
                                if now.duration_since(stats.last_report) >= Duration::from_secs(1) {
                                    let elapsed = start.elapsed();

                                    let vid_level = vid_src.as_ref().map(|s| s.current_level_bytes()).unwrap_or(0);
                                    let vid_max   = vid_src.as_ref().map(|s| s.max_bytes()).unwrap_or(0);
                                    let aud_level = aud_src.as_ref().map(|s| s.current_level_bytes()).unwrap_or(0);
                                    let aud_max   = aud_src.as_ref().map(|s| s.max_bytes()).unwrap_or(0);

                                    let since_last_media = now.duration_since(stats.last_media_instant);
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

                                    log::info!(
                                        "{name}::{stream}: HB elapsed={:?} vid_ts={:?} aud_ts={:?} \
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
                                        log::info!("{name}::{stream}: Camera stream ended, stopping relay");
                                        break;
                                    }
                                };

                                stats.tryrecv_ok += 1;
                                stats.last_media_instant = Instant::now();

                                // ---- Backpressure + lag handling ----
                                let vid_fill = vid_src
                                    .as_ref()
                                    .map(|s| s.current_level_bytes() as f32 / s.max_bytes().max(1) as f32)
                                    .unwrap_or(0.0);

                                let in_backpressure = vid_fill > 0.85;

                                let elapsed = start.elapsed();
                                let vid_ts_dur = Duration::from_micros(vid_ts);
                                let lag = elapsed.checked_sub(vid_ts_dur).unwrap_or(Duration::ZERO);

                                if in_backpressure && lag >= lag_enter_threshold && !drop_until_keyframe {
                                    log::warn!("{name}::{stream}: CATCHUP enter (lag={:?}), drop until keyframe", lag);
                                    drop_until_keyframe = true;
                                }

                                if drop_until_keyframe && lag <= lag_exit_threshold && !in_backpressure {
                                    log::warn!("{name}::{stream}: CATCHUP exit (lag={:?})", lag);
                                    drop_until_keyframe = false;
                                }

                                if drop_until_keyframe
                                    && is_video(&timed.media)
                                    && !is_video_keyframe(&timed.media)
                                {
                                    vid_ts = stats.video_ts_from_recv(timed.recv_at, vid_ts);
                                    stats.p_dropped_backpressure += 1;
                                    continue;
                                }

                                let should_drop_p =
                                    in_backpressure && matches!(timed.media, BcMedia::Pframe(_));
                                if should_drop_p {
                                    stats.p_dropped_backpressure += 1;
                                    vid_ts = stats.video_ts_from_recv(timed.recv_at, vid_ts);
                                    continue;
                                }

                                if let Err(e) = send_to_sources(
                                    timed.media,
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
                                ) {
                                    if e.to_string().contains("App source is closed")
                                        || e.to_string().contains("App source is not linked")
                                    {
                                        log::info!("{name}::{stream}: Client disconnected, stopping camera relay");
                                        break;
                                    }
                                    log::warn!("Failed to send to source: {e:?}");
                                    return Err(e);
                                }
                            }

                            log::info!("{name}::{stream}: Camera relay disconnected");
                            AnyResult::Ok(())
                        });
                        AnyResult::Ok(())
                    });
                }
            }
        }
        AnyResult::Ok(())
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
            Ok(element) => {
                log::debug!("Factory callback received pipeline element successfully");
                Ok(Some(element))
            }
            Err(e) => {
                log::error!("Failed to receive pipeline element: {:?}", e);
                Err(anyhow::anyhow!("Failed to receive pipeline element: {:?}", e))
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

fn send_to_sources(
    data: BcMedia,
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
) -> AnyResult<()> {
    // Update TS
    match data {
        BcMedia::Aac(aac) => {
            let info = aac
                .duration_info()
                .expect("Could not calculate AAC duration");
            let aac_len = aac.data.len();
            if let Some(recv_at) = recv_at {
                stats.record_aac_gap(recv_at);
            }
            let ts_us = recv_at
                .map(|t| stats.audio_ts_from_recv(t, *aud_ts))
                .unwrap_or(*aud_ts);
            let dur = Duration::from_micros(info.duration_us as u64);
            if let Some(aud_src) = aud_src.as_ref() {
                log::debug!("Sending AAC: {:?}", Duration::from_micros(ts_us));
                send_to_appsrc(
                    aud_src,
                    aac.data,
                    Duration::from_micros(ts_us),
                    Some(dur),
                    pools,
                    pace,
                    aud_pacer,
                )?;
                stats.aud_pushed += 1;
            }
            stats.record_aac(&info, aac_len);
            *aud_ts = ts_us + info.duration_us as u64;
        }
        BcMedia::Adpcm(adpcm) => {
            let duration = adpcm
                .duration()
                .expect("Could not calculate ADPCM duration");
            let dur = Duration::from_micros(duration as u64);
            if let Some(aud_src) = aud_src.as_ref() {
                let ts_us = recv_at
                    .map(|t| stats.audio_ts_from_recv(t, *aud_ts))
                    .unwrap_or(*aud_ts);
                log::trace!("Sending ADPCM: {:?}", Duration::from_micros(ts_us));
                send_to_appsrc(
                    aud_src,
                    adpcm.data,
                    Duration::from_micros(ts_us),
                    Some(dur),
                    pools,
                    pace,
                    aud_pacer,
                )?;
                stats.aud_pushed += 1;
                *aud_ts = ts_us + duration as u64;
                return Ok(());
            }
        }
        BcMedia::Iframe(BcMediaIframe { data, .. }) => {
            if let Some(vid_src) = vid_src.as_ref() {
                if let Some(recv_at) = recv_at {
                    stats.record_video_gap(recv_at);
                }
                let ts_us = recv_at
                    .map(|t| stats.video_ts_from_recv(t, *vid_ts))
                    .unwrap_or(*vid_ts);
                let frame_dur =
                    Duration::from_micros(1_000_000u64 / stream_config.fps.max(1) as u64);
                log::trace!("Sending I-frame: {:?}", Duration::from_micros(ts_us));
                send_to_appsrc(
                    vid_src,
                    data,
                    Duration::from_micros(ts_us),
                    Some(frame_dur),
                    pools,
                    pace,
                    vid_pacer,
                )?;
                stats.vid_pushed += 1;
                *vid_ts = ts_us;
                return Ok(());
            }
        }
        BcMedia::Pframe(BcMediaPframe { data, .. }) => {
            if let Some(vid_src) = vid_src.as_ref() {
                // Intelligent frame dropping: drop P-frames when buffer is getting full
                // to prioritize I-frames (keyframes) which can restart the stream
                let level = vid_src.current_level_bytes();
                let max = vid_src.max_bytes();
                if let Some(recv_at) = recv_at {
                    stats.record_video_gap(recv_at);
                }
                let ts_us = recv_at
                    .map(|t| stats.video_ts_from_recv(t, *vid_ts))
                    .unwrap_or(*vid_ts);
                let frame_dur =
                    Duration::from_micros(1_000_000u64 / stream_config.fps.max(1) as u64);

                if max > 0 && level >= max * 80 / 100 {
                    stats.p_dropped_pressure += 1;
                    log::trace!(
                        "Dropping P-frame due to buffer pressure ({}/{} bytes, {}%)",
                        level,
                        max,
                        level * 100 / max
                    );
                } else {
                    log::trace!("Sending P-frame: {:?}", Duration::from_micros(ts_us));
                    send_to_appsrc(
                        vid_src,
                        data,
                        Duration::from_micros(ts_us),
                        Some(frame_dur),
                        pools,
                        pace,
                        vid_pacer,
                    )?;
                    stats.vid_pushed += 1;
                }
                *vid_ts = ts_us;
                return Ok(());
            }
        }
        _ => {}
    }
    Ok(())
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
        let mut buf = gstreamer::Buffer::with_size(needed)
            .context("allocate large non-pooled buffer")?;
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
    data: Vec<u8>,
    ts: std::time::Duration,
    duration: Option<std::time::Duration>,
    pools: &mut std::collections::HashMap<usize, gstreamer::BufferPool>,
    pace: bool,
    pacer: &mut Option<PacerState>,
) -> AnyResult<()> {
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

    match appsrc.push_buffer(buf) {
        Ok(_) => Ok(()),
        Err(gstreamer::FlowError::Flushing) => {
            log::debug!("push_buffer => FLUSHING");
            Ok(())
        }
        Err(gstreamer::FlowError::NotLinked) => {
            log::debug!("push_buffer => NOT_LINKED");
            Ok(())
        }
        Err(e) => Err(anyhow::anyhow!("Error in streaming: {e:?}")),
    }
}

#[derive(Clone, Debug)]
struct PacerState {
    base_instant: std::time::Instant,
    base_ts: std::time::Duration,
}

impl PacerState {
    fn new(now: std::time::Instant, ts: std::time::Duration) -> Self {
        Self { base_instant: now, base_ts: ts }
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
    let buffer_size = buffer_size(stream_config.bitrate);
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
    source.set_do_timestamp(false);        // ✅ was true
    source.set_block(false);               // ok if you choose dropping
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
    let buffer_size = buffer_size(stream_config.bitrate);
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
    source.set_do_timestamp(false);        // ✅ was true
    source.set_block(false);               // ok if you choose dropping
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
    source.set_do_timestamp(false);        // ✅ was true
    source.set_block(false);               // ok if you choose dropping
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
    source.set_do_timestamp(false);        // ✅ was true
    source.set_block(false);               // ok if you choose dropping
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

#[allow(dead_code)]
fn pipe_silence(bin: &Element, stream_config: &StreamConfig) -> Result<Linked> {
    // Audio seems to run at about 800kbs
    let buffer_size = 512 * 1416;
    let bin = bin
        .clone()
        .dynamic_cast::<Bin>()
        .map_err(|_| anyhow!("Media source's element should be a bin"))?;
    log::debug!("Building Silence pipeline");
    let source = make_element("appsrc", "audsrc")?
        .dynamic_cast::<AppSrc>()
        .map_err(|_| anyhow!("Cannot cast to appsrc."))?;

    source.set_is_live(true);
    source.set_property("format", &gstreamer::Format::Time);
    source.set_block(false);
    source.set_min_latency(1000 / (stream_config.fps as i64));
    source.set_property("emit-signals", false);
    source.set_do_timestamp(false);
    source.set_stream_type(AppStreamType::Stream);
    source.set_max_bytes(buffer_size as u64);

    let source = source
        .dynamic_cast::<Element>()
        .map_err(|_| anyhow!("Cannot cast back"))?;

    let sink_queue = make_queue("audsinkqueue", buffer_size)?;
    let sink = make_element("fakesink", "silence_sink")?;

    let silence = make_element("audiotestsrc", "audsilence")?;
    silence.set_property_from_str("wave", "silence");
    let src_queue = make_queue("audsinkqueue", buffer_size)?;
    let encoder = make_element("audioconvert", "audencoder")?;

    bin.add_many([&source, &sink_queue, &sink, &silence, &src_queue, &encoder])?;

    Element::link_many([&source, &sink_queue, &sink])?;

    Element::link_many([&silence, &src_queue, &encoder])?;

    let source = source
        .dynamic_cast::<AppSrc>()
        .map_err(|_| anyhow!("Cannot convert appsrc"))?;
    Ok(Linked {
        appsrc: source,
        output: encoder,
    })
}

#[allow(dead_code)]
struct AppSrcPair {
    vid: AppSrc,
    aud: Option<AppSrc>,
}

// #[allow(dead_code)]
// /// Experimental build a stream of MPEGTS
// fn build_mpegts(bin: &Element, stream_config: &StreamConfig) -> Result<AppSrcPair> {
//     let buffer_size = buffer_size(stream_config.bitrate);
//     log::debug!(
//         "buffer_size: {buffer_size}, bitrate: {}",
//         stream_config.bitrate
//     );

//     // VID
//     let vid_link = match stream_config.vid_format {
//         VidFormat::H264 => pipe_h264(bin, stream_config)?,
//         VidFormat::H265 => pipe_h265(bin, stream_config)?,
//         VidFormat::None => unreachable!(),
//     };

//     // AUD
//     let aud_link = match stream_config.aud_format {
//         AudFormat::Aac => pipe_aac(bin, stream_config)?,
//         AudFormat::Adpcm(block) => pipe_adpcm(bin, block, stream_config)?,
//         AudFormat::None => pipe_silence(bin, stream_config)?,
//     };

//     let bin = bin
//         .clone()
//         .dynamic_cast::<Bin>()
//         .map_err(|_| anyhow!("Media source's element should be a bin"))?;

//     // MUX
//     let muxer = make_element("mpegtsmux", "mpeg_muxer")?;
//     let rtp = make_element("rtpmp2tpay", "pay0")?;

//     bin.add_many([&muxer, &rtp])?;
//     Element::link_many([&vid_link.output, &muxer, &rtp])?;
//     Element::link_many([&aud_link.output, &muxer])?;

//     Ok(AppSrcPair {
//         vid: vid_link.appsrc,
//         aud: Some(aud_link.appsrc),
//     })
// }

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
    aud_base_instant: Option<std::time::Instant>,
    vid_base_instant: Option<std::time::Instant>,
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
            aud_base_instant: None,
            vid_base_instant: None,
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

    fn audio_ts_from_recv(&mut self, recv_at: std::time::Instant, current_ts: u64) -> u64 {
        let base = match self.aud_base_instant {
            Some(base) => base,
            None => {
                let base = recv_at
                    .checked_sub(std::time::Duration::from_micros(current_ts))
                    .unwrap_or(recv_at);
                self.aud_base_instant = Some(base);
                base
            }
        };
        let ts_us = recv_at.duration_since(base).as_micros() as u64;
        ts_us.max(current_ts)
    }

    fn video_ts_from_recv(&mut self, recv_at: std::time::Instant, current_ts: u64) -> u64 {
        let base = match self.vid_base_instant {
            Some(base) => base,
            None => {
                let base = recv_at
                    .checked_sub(std::time::Duration::from_micros(current_ts))
                    .unwrap_or(recv_at);
                self.vid_base_instant = Some(base);
                base
            }
        };
        let ts_us = recv_at.duration_since(base).as_micros() as u64;
        ts_us.max(current_ts)
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

#[allow(dead_code)]
fn make_dbl_queue(name: &str, buffer_size: u32) -> AnyResult<Element> {
    let queue = make_element("queue", &format!("queue1_{}", name))?;
    queue.set_property("max-size-bytes", buffer_size);
    queue.set_property("max-size-buffers", 0u32);
    queue.set_property("max-size-time", 0u64);
    //queue.set_property_from_str("leaky", "downstream");

    let queue2 = make_element("queue2", &format!("queue2_{}", name))?;
    queue2.set_property("max-size-bytes", buffer_size * 2u32 / 3u32);
    queue2.set_property("max-size-buffers", 0u32);
    queue2.set_property("max-size-time", 0u64);
    queue2.set_property("use-buffering", false);
    //queue2.set_property_from_str("leaky", "downstream");

    let bin = gstreamer::Bin::builder().name(name).build();
    bin.add_many([&queue, &queue2])?;
    Element::link_many([&queue, &queue2])?;

    let pad = queue
        .static_pad("sink")
        .expect("Failed to get a static pad from queue.");
    let ghost_pad = GhostPad::builder_with_target(&pad).unwrap().build();
    ghost_pad.set_active(true)?;
    bin.add_pad(&ghost_pad)?;

    let pad = queue2
        .static_pad("src")
        .expect("Failed to get a static pad from queue2.");
    let ghost_pad = GhostPad::builder_with_target(&pad).unwrap().build();
    ghost_pad.set_active(true)?;
    bin.add_pad(&ghost_pad)?;

    let bin = bin
        .dynamic_cast::<Element>()
        .map_err(|_| anyhow!("Cannot convert bin"))?;
    Ok(bin)
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

fn buffer_size(bitrate: u32) -> u32 {
    // bitrate is bits/sec
    let bytes_per_sec = bitrate / 8;
    let target_ms = 1000u32;
    let bytes = bytes_per_sec * target_ms / 1000;
    std::cmp::max(bytes, 64 * 1024)
}
