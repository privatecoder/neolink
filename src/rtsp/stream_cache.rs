//! Persistent stream-type cache.
//!
//! Neolink learns each camera stream's codec + sizing the first time a client
//! connects, and caches it so a later client can be served the offline
//! placeholder ("stream not ready" / keepalive) immediately — even while the
//! camera is unreachable — without first re-learning the codec from the camera.
//!
//! That cache used to be in-memory only, so a fresh process start lost it: an
//! RTSP client connecting before the camera was reachable could not have a
//! placeholder built and got nothing. This module persists the cache to disk so
//! a known camera survives a restart.
//!
//! Design notes:
//! - The on-disk format ([`CacheFileDto`]) is intentionally decoupled from the
//!   in-memory types so internal enums can evolve without changing the file
//!   format. A top-level `version` guards against incompatible formats.
//! - Disk entries are a *hint*, never the truth: they are reconciled against the
//!   live camera on first connect (see [`reconcile`]).
//! - All disk I/O is behind small pure functions ([`serialize_map`],
//!   [`parse_file`], [`atomic_write`]) and the [`StreamCache`] struct, so it can
//!   be unit-tested with a temp dir, without gstreamer or the RTSP server.

use neolink_core::bc_protocol::StreamKind;
use neolink_core::bcmedia::model::VideoType;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Current on-disk schema version. Bump only on an incompatible format change.
const SCHEMA_VERSION: u32 = 1;

/// Conservative fps substituted when a cached entry has `fps == 0` (rather than
/// rejecting the otherwise-usable entry).
const DEFAULT_FPS: u32 = 25;

/// Audio codec of a stream. Carries the ADPCM block size, which is needed to
/// build the decoder, and is distinct enough from AAC that a mismatch is
/// caps-breaking (the audio RTP clock-rate is fixed in the SDP).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum AudioType {
    Aac,
    Adpcm(u32),
}

/// Identifies one cached stream. Only *stable* identifiers — values that do not
/// change run-to-run for the same physical stream. No codec/sizing here (those
/// are the cached payload, not identity).
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct StreamCacheKey {
    /// Stable camera identity: UID if set, else the configured address, else the
    /// camera name. (UID is the most stable; name is the last-resort fallback.)
    pub(crate) camera_id: String,
    /// NVR channel (0 for a standalone camera). Two channels of one NVR share a
    /// UID but are different streams, so this belongs in the key.
    pub(crate) channel_id: u8,
    pub(crate) stream: StreamKind,
}

impl StreamCacheKey {
    /// Build a key from a camera's stable identifiers. `camera_id` resolves to
    /// `uid ?? address ?? name`.
    pub(crate) fn new(
        uid: Option<&str>,
        address: Option<&str>,
        name: &str,
        channel_id: u8,
        stream: StreamKind,
    ) -> Self {
        let camera_id = uid
            .filter(|s| !s.is_empty())
            .or(address.filter(|s| !s.is_empty()))
            .unwrap_or(name)
            .to_string();
        Self {
            camera_id,
            channel_id,
            stream,
        }
    }
}

/// The cached codec + sizing for one stream.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct StreamTypeCache {
    pub(crate) vid_type: Option<VideoType>,
    pub(crate) aud_type: Option<AudioType>,
    pub(crate) resolution: [u32; 2],
    pub(crate) bitrate: u32,
    pub(crate) fps: u32,
    pub(crate) fps_table: Vec<u32>,
    pub(crate) aud_rate: u32,
    pub(crate) aud_channels: u32,
}

/// Outcome of comparing the cached types used to build a session against the
/// types learned from the live camera once it connected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reconcile {
    /// Learned types match what we built from — nothing to do.
    Identical,
    /// Only sizing drifted (bitrate / fps). Refresh the cache silently; the live
    /// pipeline keeps playing.
    DriftOnly,
    /// A field that is baked into the SDP / caps changed (codec, audio format,
    /// audio sample-rate or channel count, or — conservatively — resolution).
    /// The already-built pipeline is wrong: refresh the cache and tear this
    /// session down so the client reconnects to a correct pipeline.
    CapsBreaking,
}

// ===================== Pure serialization layer (DTOs) =====================

#[derive(Serialize, Deserialize)]
struct CacheFileDto {
    version: u32,
    entries: Vec<CacheEntryDto>,
}

#[derive(Serialize, Deserialize)]
struct CacheEntryDto {
    camera_id: String,
    channel_id: u8,
    stream: String,
    #[serde(default)]
    video: Option<String>,
    #[serde(default)]
    audio: Option<AudioDto>,
    resolution: [u32; 2],
    bitrate: u32,
    fps: u32,
    #[serde(default)]
    fps_table: Vec<u32>,
    aud_rate: u32,
    aud_channels: u32,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum AudioDto {
    Aac,
    Adpcm(u32),
}

fn video_to_str(v: VideoType) -> &'static str {
    match v {
        VideoType::H264 => "h264",
        VideoType::H265 => "h265",
    }
}

fn str_to_video(s: &str) -> Option<VideoType> {
    match s.to_ascii_lowercase().as_str() {
        "h264" => Some(VideoType::H264),
        "h265" => Some(VideoType::H265),
        _ => None,
    }
}

fn stream_to_str(s: StreamKind) -> &'static str {
    match s {
        StreamKind::Main => "main",
        StreamKind::Sub => "sub",
        StreamKind::Extern => "extern",
    }
}

fn str_to_stream(s: &str) -> Option<StreamKind> {
    match s.to_ascii_lowercase().as_str() {
        "main" => Some(StreamKind::Main),
        "sub" => Some(StreamKind::Sub),
        "extern" => Some(StreamKind::Extern),
        _ => None,
    }
}

fn audio_to_dto(a: &AudioType) -> AudioDto {
    match a {
        AudioType::Aac => AudioDto::Aac,
        AudioType::Adpcm(n) => AudioDto::Adpcm(*n),
    }
}

fn audio_from_dto(a: AudioDto) -> AudioType {
    match a {
        AudioDto::Aac => AudioType::Aac,
        AudioDto::Adpcm(n) => AudioType::Adpcm(n),
    }
}

/// Validate and convert a parsed DTO entry into a `(key, value)` pair. Returns
/// `None` (entry dropped) if it can't be used: unknown stream, missing/unknown
/// video codec, or zero resolution. `fps == 0` is repaired to [`DEFAULT_FPS`];
/// AAC audio missing its rate/channels has the audio dropped (video kept).
fn dto_to_entry(dto: CacheEntryDto) -> Option<(StreamCacheKey, StreamTypeCache)> {
    let Some(stream) = str_to_stream(&dto.stream) else {
        log::warn!(
            "stream cache: dropping entry with unknown stream {:?}",
            dto.stream
        );
        return None;
    };
    let vid_type = match dto.video.as_deref() {
        Some(v) => match str_to_video(v) {
            Some(vt) => vt,
            None => {
                log::warn!("stream cache: dropping entry with unknown video codec {v:?}");
                return None;
            }
        },
        None => {
            log::warn!("stream cache: dropping entry without a video codec");
            return None;
        }
    };
    if dto.resolution[0] == 0 || dto.resolution[1] == 0 {
        log::warn!("stream cache: dropping entry with zero resolution");
        return None;
    }
    let fps = if dto.fps == 0 { DEFAULT_FPS } else { dto.fps };

    let mut aud_type = dto.audio.map(audio_from_dto);
    let mut aud_rate = dto.aud_rate;
    let mut aud_channels = dto.aud_channels;
    if matches!(aud_type, Some(AudioType::Aac)) && (aud_rate == 0 || aud_channels == 0) {
        log::warn!("stream cache: dropping AAC audio with zero rate/channels (keeping video)");
        aud_type = None;
        aud_rate = 0;
        aud_channels = 0;
    }

    Some((
        StreamCacheKey {
            camera_id: dto.camera_id,
            channel_id: dto.channel_id,
            stream,
        },
        StreamTypeCache {
            vid_type: Some(vid_type),
            aud_type,
            resolution: dto.resolution,
            bitrate: dto.bitrate,
            fps,
            fps_table: dto.fps_table,
            aud_rate,
            aud_channels,
        },
    ))
}

/// Serialize the in-memory map to the on-disk JSON form. Entries are sorted by
/// key so output is deterministic (stable diffs, golden tests).
fn serialize_map(map: &HashMap<StreamCacheKey, StreamTypeCache>) -> String {
    let mut keys: Vec<&StreamCacheKey> = map.keys().collect();
    keys.sort_by(|a, b| {
        (a.camera_id.as_str(), a.channel_id, stream_to_str(a.stream)).cmp(&(
            b.camera_id.as_str(),
            b.channel_id,
            stream_to_str(b.stream),
        ))
    });
    let entries = keys
        .into_iter()
        .map(|k| {
            let v = &map[k];
            CacheEntryDto {
                camera_id: k.camera_id.clone(),
                channel_id: k.channel_id,
                stream: stream_to_str(k.stream).to_string(),
                video: v.vid_type.map(|vt| video_to_str(vt).to_string()),
                audio: v.aud_type.as_ref().map(audio_to_dto),
                resolution: v.resolution,
                bitrate: v.bitrate,
                fps: v.fps,
                fps_table: v.fps_table.clone(),
                aud_rate: v.aud_rate,
                aud_channels: v.aud_channels,
            }
        })
        .collect();
    let file = CacheFileDto {
        version: SCHEMA_VERSION,
        entries,
    };
    // A plain data structure with String/number fields cannot fail to serialize.
    serde_json::to_string_pretty(&file).expect("stream cache serialization is infallible")
}

/// Parse the on-disk JSON form into the in-memory map. Lenient and total: any
/// failure (unreadable, corrupt, wrong/missing version) yields an empty map
/// (logged), and a single malformed entry is skipped without discarding the
/// rest. Each surviving entry is validated (see [`dto_to_entry`]).
fn parse_file(contents: &str) -> HashMap<StreamCacheKey, StreamTypeCache> {
    let mut out = HashMap::new();
    let value: serde_json::Value = match serde_json::from_str(contents) {
        Ok(v) => v,
        Err(e) => {
            log::warn!("stream cache: ignoring unparseable file: {e}");
            return out;
        }
    };
    match value.get("version").and_then(|v| v.as_u64()) {
        Some(v) if v == SCHEMA_VERSION as u64 => {}
        Some(v) => {
            log::warn!("stream cache: ignoring file with unsupported version {v}");
            return out;
        }
        None => {
            log::warn!("stream cache: ignoring file with no version field");
            return out;
        }
    }
    let entries = value
        .get("entries")
        .and_then(|e| e.as_array())
        .cloned()
        .unwrap_or_default();
    for entry in entries {
        match serde_json::from_value::<CacheEntryDto>(entry) {
            Ok(dto) => {
                if let Some((k, v)) = dto_to_entry(dto) {
                    out.insert(k, v);
                }
            }
            Err(e) => log::warn!("stream cache: skipping malformed entry: {e}"),
        }
    }
    out
}

/// Atomically replace the file at `path` with `contents`: write a sibling
/// `.tmp`, fsync it, then rename over the target. On any error the previous file
/// is left intact.
fn atomic_write(path: &Path, contents: &str) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    // Sibling temp file in the same directory (same filesystem → atomic rename).
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(contents.as_bytes())?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)
}

/// Compare the cached types used to build a session (`used`) with the types
/// learned from the live camera (`learned`).
///
/// Caps-breaking covers everything baked into the SDP/caps: video codec, audio
/// format, audio sample-rate and channel count, and — conservatively —
/// resolution. (The spec names codec/format/rate explicitly; channels and
/// resolution are added conservatively, since a mismatch there also produces
/// wrong caps. Flagged for review.)
pub(crate) fn reconcile(used: &StreamTypeCache, learned: &StreamTypeCache) -> Reconcile {
    if used.vid_type != learned.vid_type
        || used.aud_type != learned.aud_type
        || used.aud_rate != learned.aud_rate
        || used.aud_channels != learned.aud_channels
        || used.resolution != learned.resolution
    {
        return Reconcile::CapsBreaking;
    }
    if used.bitrate != learned.bitrate
        || used.fps != learned.fps
        || used.fps_table != learned.fps_table
    {
        return Reconcile::DriftOnly;
    }
    Reconcile::Identical
}

// ===================== Testable cache (in-memory + disk) =====================

/// In-memory cache plus its optional backing file. This is the unit-testable
/// core; the global async layer below is a thin wrapper.
pub(crate) struct StreamCache {
    map: HashMap<StreamCacheKey, StreamTypeCache>,
    path: Option<PathBuf>,
}

impl StreamCache {
    /// Build a cache, loading the backing file if a path is given. A missing or
    /// unreadable file starts empty (never an error).
    fn load(path: Option<PathBuf>) -> Self {
        let map = match &path {
            Some(p) => match std::fs::read_to_string(p) {
                Ok(contents) => parse_file(&contents),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => HashMap::new(),
                Err(e) => {
                    log::warn!("stream cache: could not read {p:?}: {e}");
                    HashMap::new()
                }
            },
            None => HashMap::new(),
        };
        StreamCache { map, path }
    }

    fn get(&self, key: &StreamCacheKey) -> Option<StreamTypeCache> {
        self.map.get(key).cloned()
    }

    /// Insert/replace `key`'s value in memory only. Returns `true` if the value
    /// actually changed (new key or different value).
    fn update_in_memory(&mut self, key: StreamCacheKey, value: StreamTypeCache) -> bool {
        if self.map.get(&key) == Some(&value) {
            return false;
        }
        self.map.insert(key, value);
        true
    }

    /// Update in memory and, if changed and a path is configured, persist the
    /// whole file atomically. Returns `true` if a disk write was attempted.
    fn store(&mut self, key: StreamCacheKey, value: StreamTypeCache) -> bool {
        if !self.update_in_memory(key, value) {
            return false;
        }
        if let Some(path) = &self.path {
            if let Err(e) = atomic_write(path, &serialize_map(&self.map)) {
                log::warn!("stream cache: failed to persist to {path:?}: {e}");
            }
            return true;
        }
        false
    }
}

// ===================== Global layer =====================
//
// Synchronous (std Mutex): stream-type writes are rare (≈once per camera/stream
// lifetime, plus on a codec change) and the file is tiny, so a brief lock around
// the in-memory update + atomic write is fine. A sync API also lets the blocking
// streaming loop refresh the cache during reconciliation without bridging
// async/blocking.

static CACHE: Lazy<std::sync::Mutex<StreamCache>> =
    Lazy::new(|| std::sync::Mutex::new(StreamCache::load(None)));

/// Lock the global cache, recovering the inner value if a previous holder
/// panicked (a poisoned cache mutex should never take down streaming).
fn cache() -> std::sync::MutexGuard<'static, StreamCache> {
    CACHE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Initialise the global cache once at startup. `path = None` keeps persistence
/// disabled (in-memory only); `Some(path)` loads it and enables persist-on-change.
pub(crate) fn init(path: Option<PathBuf>) {
    *cache() = StreamCache::load(path);
}

pub(crate) fn get(key: &StreamCacheKey) -> Option<StreamTypeCache> {
    cache().get(key)
}

pub(crate) fn store(key: StreamCacheKey, value: StreamTypeCache) {
    cache().store(key, value);
}

/// Resolve the configured cache path, applying the `NEOLINK_STREAM_CACHE_PATH`
/// env override. An empty value (env or config) explicitly disables persistence.
pub(crate) fn resolve_path(config_value: Option<&str>) -> Option<PathBuf> {
    let env = std::env::var("NEOLINK_STREAM_CACHE_PATH").ok();
    resolve_path_from(env.as_deref(), config_value)
}

/// Pure precedence logic for [`resolve_path`]: env over config, empty = disabled.
fn resolve_path_from(env: Option<&str>, config_value: Option<&str>) -> Option<PathBuf> {
    if let Some(env) = env {
        return if env.is_empty() {
            None
        } else {
            Some(PathBuf::from(env))
        };
    }
    config_value.filter(|s| !s.is_empty()).map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(camera_id: &str, channel_id: u8, stream: StreamKind) -> StreamCacheKey {
        StreamCacheKey {
            camera_id: camera_id.to_string(),
            channel_id,
            stream,
        }
    }

    fn sample() -> StreamTypeCache {
        StreamTypeCache {
            vid_type: Some(VideoType::H265),
            aud_type: Some(AudioType::Aac),
            resolution: [2560, 1920],
            bitrate: 4096000,
            fps: 20,
            fps_table: vec![10, 15, 20],
            aud_rate: 16000,
            aud_channels: 1,
        }
    }

    /// A unique scratch dir under the system temp dir — avoids adding a
    /// `tempfile` dependency for a couple of file round-trips.
    fn scratch_dir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!(
            "neolink-stream-cache-test-{tag}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn dto_round_trip_every_codec_and_audio_variant() {
        let mut map = HashMap::new();
        // H264 + AAC
        map.insert(
            key("camA", 0, StreamKind::Main),
            StreamTypeCache {
                vid_type: Some(VideoType::H264),
                aud_type: Some(AudioType::Aac),
                resolution: [1920, 1080],
                bitrate: 2048000,
                fps: 25,
                fps_table: vec![5, 25],
                aud_rate: 44100,
                aud_channels: 2,
            },
        );
        // H265 + ADPCM(block_size) — block size must survive the round-trip
        map.insert(
            key("camB", 1, StreamKind::Sub),
            StreamTypeCache {
                vid_type: Some(VideoType::H265),
                aud_type: Some(AudioType::Adpcm(1024)),
                resolution: [640, 480],
                bitrate: 512000,
                fps: 15,
                fps_table: vec![15],
                aud_rate: 0,
                aud_channels: 0,
            },
        );
        // H264 + no audio
        map.insert(
            key("camC", 0, StreamKind::Extern),
            StreamTypeCache {
                vid_type: Some(VideoType::H264),
                aud_type: None,
                resolution: [1280, 720],
                bitrate: 1024000,
                fps: 30,
                fps_table: vec![30],
                aud_rate: 0,
                aud_channels: 0,
            },
        );

        let json = serialize_map(&map);
        let back = parse_file(&json);
        assert_eq!(back, map, "round-trip must preserve every entry exactly");
    }

    #[test]
    fn golden_v1_json_is_stable() {
        let mut map = HashMap::new();
        map.insert(key("cam-uid-1", 0, StreamKind::Main), sample());
        let json = serialize_map(&map);
        let expected = r#"{
  "version": 1,
  "entries": [
    {
      "camera_id": "cam-uid-1",
      "channel_id": 0,
      "stream": "main",
      "video": "h265",
      "audio": "aac",
      "resolution": [
        2560,
        1920
      ],
      "bitrate": 4096000,
      "fps": 20,
      "fps_table": [
        10,
        15,
        20
      ],
      "aud_rate": 16000,
      "aud_channels": 1
    }
  ]
}"#;
        assert_eq!(json, expected);
    }

    #[test]
    fn loader_missing_file_is_empty() {
        let dir = scratch_dir("missing");
        let path = dir.join("does-not-exist.json");
        let cache = StreamCache::load(Some(path));
        assert!(cache.get(&key("camA", 0, StreamKind::Main)).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn loader_corrupt_json_is_empty_no_panic() {
        let map = parse_file("this is { not json");
        assert!(map.is_empty());
    }

    #[test]
    fn loader_wrong_version_is_ignored() {
        let json = r#"{"version": 999, "entries": [
            {"camera_id":"camA","channel_id":0,"stream":"main","video":"h264",
             "audio":"aac","resolution":[1920,1080],"bitrate":1,"fps":25,
             "fps_table":[25],"aud_rate":44100,"aud_channels":2}]}"#;
        assert!(parse_file(json).is_empty());
    }

    #[test]
    fn loader_missing_version_is_ignored() {
        let json = r#"{"entries": []}"#;
        assert!(parse_file(json).is_empty());
    }

    #[test]
    fn loader_one_bad_entry_others_survive() {
        // First entry has an unknown stream value (malformed); second is valid.
        let json = r#"{"version":1,"entries":[
            {"camera_id":"bad","channel_id":0,"stream":"nonsense","video":"h264",
             "audio":null,"resolution":[1920,1080],"bitrate":1,"fps":25,
             "fps_table":[],"aud_rate":0,"aud_channels":0},
            {"camera_id":"good","channel_id":0,"stream":"sub","video":"h265",
             "audio":null,"resolution":[640,480],"bitrate":1,"fps":15,
             "fps_table":[],"aud_rate":0,"aud_channels":0}]}"#;
        let map = parse_file(json);
        assert_eq!(map.len(), 1);
        assert!(map.contains_key(&key("good", 0, StreamKind::Sub)));
    }

    #[test]
    fn loader_unknown_field_tolerated() {
        let json = r#"{"version":1,"entries":[
            {"camera_id":"camA","channel_id":0,"stream":"main","video":"h264",
             "audio":null,"resolution":[1920,1080],"bitrate":1,"fps":25,
             "fps_table":[],"aud_rate":0,"aud_channels":0,
             "some_future_field":"ignored"}]}"#;
        let map = parse_file(json);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn loader_rejects_entry_without_video() {
        // vid_type must be present — a video-less entry can't unblock the
        // offline placeholder, so it's dropped.
        let json = r#"{"version":1,"entries":[
            {"camera_id":"camA","channel_id":0,"stream":"main","video":null,
             "audio":null,"resolution":[1920,1080],"bitrate":1,"fps":25,
             "fps_table":[],"aud_rate":0,"aud_channels":0}]}"#;
        assert!(parse_file(json).is_empty());
    }

    #[test]
    fn loader_rejects_zero_resolution() {
        let json = r#"{"version":1,"entries":[
            {"camera_id":"camA","channel_id":0,"stream":"main","video":"h264",
             "audio":null,"resolution":[0,0],"bitrate":1,"fps":25,
             "fps_table":[],"aud_rate":0,"aud_channels":0}]}"#;
        assert!(parse_file(json).is_empty());
    }

    #[test]
    fn loader_substitutes_default_fps_when_zero() {
        let json = r#"{"version":1,"entries":[
            {"camera_id":"camA","channel_id":0,"stream":"main","video":"h264",
             "audio":null,"resolution":[1920,1080],"bitrate":1,"fps":0,
             "fps_table":[],"aud_rate":0,"aud_channels":0}]}"#;
        let map = parse_file(json);
        let entry = map.get(&key("camA", 0, StreamKind::Main)).unwrap();
        assert_eq!(entry.fps, DEFAULT_FPS);
    }

    #[test]
    fn loader_drops_aac_audio_with_zero_rate() {
        // AAC with no sample rate / channels is unusable (clock-rate fixed in
        // SDP); drop the audio but keep the usable video entry.
        let json = r#"{"version":1,"entries":[
            {"camera_id":"camA","channel_id":0,"stream":"main","video":"h264",
             "audio":"aac","resolution":[1920,1080],"bitrate":1,"fps":25,
             "fps_table":[],"aud_rate":0,"aud_channels":0}]}"#;
        let map = parse_file(json);
        let entry = map.get(&key("camA", 0, StreamKind::Main)).unwrap();
        assert_eq!(entry.aud_type, None);
    }

    #[test]
    fn atomic_write_then_reload_equals_original() {
        let dir = scratch_dir("atomic");
        let path = dir.join("cache.json");
        let mut written = HashMap::new();
        written.insert(key("cam-uid-1", 0, StreamKind::Main), sample());
        atomic_write(&path, &serialize_map(&written)).unwrap();

        let reloaded = StreamCache::load(Some(path.clone()));
        assert_eq!(
            reloaded.get(&key("cam-uid-1", 0, StreamKind::Main)),
            Some(sample())
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn atomic_write_ignores_stray_tmp_on_load() {
        let dir = scratch_dir("stray-tmp");
        let path = dir.join("cache.json");
        let mut written = HashMap::new();
        written.insert(key("cam-uid-1", 0, StreamKind::Main), sample());
        atomic_write(&path, &serialize_map(&written)).unwrap();
        // A crash mid-write could leave a sibling .tmp; it must be ignored.
        std::fs::write(dir.join("cache.json.tmp"), "garbage not json").unwrap();

        let reloaded = StreamCache::load(Some(path));
        assert_eq!(
            reloaded.get(&key("cam-uid-1", 0, StreamKind::Main)),
            Some(sample())
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn good_file_preserved_when_write_fails() {
        let dir = scratch_dir("write-fail");
        let path = dir.join("cache.json");
        let mut written = HashMap::new();
        written.insert(key("cam-uid-1", 0, StreamKind::Main), sample());
        atomic_write(&path, &serialize_map(&written)).unwrap();

        // Force a write failure: target a path whose parent is a file.
        let not_a_dir = dir.join("cache.json"); // existing file
        let bad = not_a_dir.join("nested/cache.json");
        assert!(atomic_write(&bad, "{}").is_err());

        // Original file must be untouched and still valid.
        let reloaded = StreamCache::load(Some(path));
        assert_eq!(
            reloaded.get(&key("cam-uid-1", 0, StreamKind::Main)),
            Some(sample())
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn store_skips_write_when_unchanged() {
        let dir = scratch_dir("unchanged");
        let path = dir.join("cache.json");
        let mut cache = StreamCache::load(Some(path));

        let k = key("cam-uid-1", 0, StreamKind::Main);
        assert!(cache.store(k.clone(), sample()), "first store writes");
        assert!(
            !cache.store(k.clone(), sample()),
            "storing identical value must not write"
        );
        assert!(
            cache.store(k, {
                let mut v = sample();
                v.bitrate = 999;
                v
            }),
            "storing a changed value writes again"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reconcile_identical() {
        assert_eq!(reconcile(&sample(), &sample()), Reconcile::Identical);
    }

    #[test]
    fn reconcile_bitrate_or_fps_drift_is_non_breaking() {
        let mut learned = sample();
        learned.bitrate = 8_000_000;
        learned.fps = 25;
        assert_eq!(reconcile(&sample(), &learned), Reconcile::DriftOnly);
    }

    #[test]
    fn reconcile_video_codec_change_is_caps_breaking() {
        let mut learned = sample();
        learned.vid_type = Some(VideoType::H264);
        assert_eq!(reconcile(&sample(), &learned), Reconcile::CapsBreaking);
    }

    #[test]
    fn reconcile_audio_format_change_is_caps_breaking() {
        let mut learned = sample();
        learned.aud_type = Some(AudioType::Adpcm(512));
        assert_eq!(reconcile(&sample(), &learned), Reconcile::CapsBreaking);
    }

    #[test]
    fn reconcile_audio_rate_change_is_caps_breaking() {
        let mut learned = sample();
        learned.aud_rate = 8000;
        assert_eq!(reconcile(&sample(), &learned), Reconcile::CapsBreaking);
    }

    #[test]
    fn reconcile_resolution_change_is_caps_breaking_conservative() {
        let mut learned = sample();
        learned.resolution = [1920, 1080];
        assert_eq!(reconcile(&sample(), &learned), Reconcile::CapsBreaking);
    }

    #[test]
    fn resolve_path_precedence_env_over_config_empty_disables() {
        // env wins over config
        assert_eq!(
            resolve_path_from(Some("/data/c.json"), Some("/etc/c.json")),
            Some(PathBuf::from("/data/c.json"))
        );
        // empty env explicitly disables, even with a config value
        assert_eq!(resolve_path_from(Some(""), Some("/etc/c.json")), None);
        // no env falls back to config
        assert_eq!(
            resolve_path_from(None, Some("/etc/c.json")),
            Some(PathBuf::from("/etc/c.json"))
        );
        // empty config disables
        assert_eq!(resolve_path_from(None, Some("")), None);
        // nothing set disables
        assert_eq!(resolve_path_from(None, None), None);
    }

    #[test]
    fn key_prefers_uid_then_address_then_name() {
        assert_eq!(
            StreamCacheKey::new(Some("uid1"), Some("1.2.3.4"), "name", 0, StreamKind::Main)
                .camera_id,
            "uid1"
        );
        assert_eq!(
            StreamCacheKey::new(None, Some("1.2.3.4"), "name", 0, StreamKind::Main).camera_id,
            "1.2.3.4"
        );
        assert_eq!(
            StreamCacheKey::new(None, None, "name", 0, StreamKind::Main).camera_id,
            "name"
        );
    }
}
