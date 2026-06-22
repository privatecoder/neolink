use crate::mqtt::Discoveries;
use neolink_core::bc_protocol::DiscoveryMethods;
#[cfg(feature = "gstreamer")]
use neolink_core::bc_protocol::StreamKind;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::clone::Clone;
use std::collections::HashSet;
use validator::Validate;
use validator::ValidationError;

// Accept both the bare (`request`/`require`) and the GStreamer-style trailing-d
// spellings (`requested`/`required`) — the latter is what sample_config.toml has
// long documented and what users naturally copy. The consumer (`set_up_tls`)
// maps both spellings, so the regex and the match stay in agreement.
static RE_TLS_CLIENT_AUTH: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^(none|request|requested|require|required)$").unwrap());
static RE_PAUSE_MODE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(black|still|test|none)$").unwrap());
static RE_MAXENC_SRC: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^([nN]one|[Aa][Ee][Ss]|[Bb][Cc][Ee][Nn][Cc][Rr][Yy][Pp][Tt])$").unwrap()
});

#[derive(Debug, Deserialize, Serialize, Validate, Clone, PartialEq)]
pub(crate) struct Config {
    #[validate(nested)]
    pub(crate) cameras: Vec<CameraConfig>,

    #[serde(rename = "bind", default = "default_bind_addr")]
    pub(crate) bind_addr: String,

    #[validate(range(min = 0, max = 65535, message = "Invalid port", code = "bind_port"))]
    #[serde(default = "default_bind_port")]
    pub(crate) bind_port: u16,

    #[serde(default = "default_certificate")]
    pub(crate) certificate: Option<String>,

    #[serde(default = "Default::default")]
    pub(crate) mqtt: Option<MqttServerConfig>,

    #[validate(regex(
        path = *RE_TLS_CLIENT_AUTH,
        message = "Incorrect tls auth",
        code = "tls_client_auth"
    ))]
    #[serde(default = "default_tls_client_auth")]
    pub(crate) tls_client_auth: String,

    #[validate(range(
        min = 0,
        max = 86400,
        message = "Invalid offline timeout seconds",
        code = "offline_timeout_secs"
    ))]
    /// Global default for how many seconds an RTSP viewer's keepalive placeholder is
    /// served while the camera is offline before that session is torn down. `0` = never
    /// (the default: the placeholder is held indefinitely). A per-camera
    /// `offline_timeout_secs` overrides this. Resolved per camera at load.
    #[serde(default)]
    pub(crate) offline_timeout_secs: Option<u32>,

    #[validate(nested)]
    #[serde(default)]
    pub(crate) users: Vec<UserConfig>,

    /// Where to persist the learned stream-type cache so a known camera's offline
    /// placeholder can still be built after a restart. `None` (default) disables
    /// persistence (in-memory only). Overridable at runtime by the
    /// `NEOLINK_STREAM_CACHE_PATH` environment variable.
    #[serde(default)]
    pub(crate) stream_cache_path: Option<String>,
}

impl Config {
    /// Resolve each camera's effective `offline_timeout_secs` from the
    /// per-camera ?? global ?? 0 precedence, and enforce the >=60s safety floor
    /// (a value below it would tear a viewer down mid camera-reboot, the exact
    /// failure the keepalive avoids). After this every camera holds a concrete
    /// resolved value, so consumers read a single effective number.
    ///
    /// This MUST be applied identically at startup (`main.rs`) and on every
    /// runtime config reload (the reactor's `update_config`); otherwise a
    /// reloaded sub-60 value would violate the floor and an unset value would
    /// flip from inherit-global to 0.
    pub(crate) fn resolve_offline_timeouts(&mut self) {
        let global_offline_timeout = self.offline_timeout_secs;
        for cam in self.cameras.iter_mut() {
            let mut secs = cam
                .offline_timeout_secs
                .or(global_offline_timeout)
                .unwrap_or(0);
            if secs > 0 && secs < 60 {
                log::warn!(
                    "{}: offline_timeout_secs={secs} is below the 60s floor; clamping to 60 (must exceed your camera's reboot time)",
                    cam.name
                );
                secs = 60;
            }
            cam.offline_timeout_secs = Some(secs.min(86400));
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, Validate, PartialEq, Eq)]
#[validate(schema(function = "validate_mqtt_server", skip_on_field_errors = true))]
pub(crate) struct MqttServerConfig {
    #[serde(alias = "server")]
    pub(crate) broker_addr: String,

    pub(crate) port: u16,

    #[serde(default, skip_serializing)]
    pub(crate) credentials: Option<(String, String)>,

    #[serde(default, skip_serializing)]
    pub(crate) ca: Option<std::path::PathBuf>,

    #[serde(default, skip_serializing)]
    pub(crate) client_auth: Option<(std::path::PathBuf, std::path::PathBuf)>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, Eq, PartialEq)]
pub(crate) enum StreamConfig {
    #[serde(alias = "none")]
    None,
    #[serde(alias = "all")]
    All,
    #[serde(alias = "both")]
    Both,
    #[serde(
        alias = "main",
        alias = "mainStream",
        alias = "mainstream",
        alias = "MainStream"
    )]
    Main,
    #[serde(
        alias = "sub",
        alias = "subStream",
        alias = "substream",
        alias = "SubStream"
    )]
    Sub,
    #[serde(
        alias = "extern",
        alias = "externStream",
        alias = "externstream",
        alias = "ExternStream"
    )]
    Extern,
}

impl StreamConfig {
    #[cfg(feature = "gstreamer")]
    pub(crate) fn as_stream_kinds(&self) -> Vec<StreamKind> {
        match self {
            StreamConfig::All => {
                vec![StreamKind::Main, StreamKind::Extern, StreamKind::Sub]
            }
            StreamConfig::Both => {
                vec![StreamKind::Main, StreamKind::Sub]
            }
            StreamConfig::Main => {
                vec![StreamKind::Main]
            }
            StreamConfig::Sub => {
                vec![StreamKind::Sub]
            }
            StreamConfig::Extern => {
                vec![StreamKind::Extern]
            }
            StreamConfig::None => {
                vec![]
            }
        }
    }
}

/// How a camera maintains its connection to Neolink.
#[derive(Debug, Deserialize, Serialize, Clone, Copy, Eq, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ConnectMode {
    /// Connect at startup and stay connected, reconnecting on drops. Idle
    /// disconnection is governed by `idle_timeout_secs` (0 = never disconnect).
    /// This is the regular, always-available behaviour.
    #[default]
    #[serde(alias = "always", alias = "connected", alias = "on")]
    Always,
    /// Connect only when needed (an RTSP client connects, an MQTT command runs,
    /// motion is active) and disconnect when idle. Idle linger is governed by
    /// `relay_warm_seconds`. Useful for battery-powered cameras.
    #[serde(alias = "ondemand", alias = "demand", alias = "lazy")]
    OnDemand,
}

#[derive(Debug, Deserialize, Serialize, Validate, Clone, PartialEq)]
#[validate(schema(function = "validate_camera_config"))]
pub(crate) struct CameraConfig {
    pub(crate) name: String,

    #[serde(rename = "address")]
    pub(crate) camera_addr: Option<String>,

    #[serde(rename = "uid")]
    pub(crate) camera_uid: Option<String>,

    pub(crate) username: String,

    #[serde(alias = "pass", skip_serializing, default)]
    pub(crate) password: Option<String>,

    #[serde(default = "default_stream")]
    pub(crate) stream: StreamConfig,

    pub(crate) permitted_users: Option<Vec<String>>,

    #[validate(range(min = 0, max = 31, message = "Invalid channel", code = "channel_id"))]
    #[serde(default = "default_channel_id", alias = "channel")]
    pub(crate) channel_id: u8,

    #[validate(nested)]
    #[serde(default = "default_mqtt")]
    pub(crate) mqtt: MqttConfig,

    #[validate(nested)]
    #[serde(default = "default_pause")]
    pub(crate) pause: PauseConfig,

    #[serde(default = "default_discovery")]
    pub(crate) discovery: DiscoveryMethods,

    #[serde(default, alias = "relay_region")]
    pub(crate) relay_server_region: Option<String>,

    #[validate(range(
        min = 0,
        max = 5000,
        message = "Invalid UDP gap skip wait (ms)",
        code = "udp_gap_skip_ms"
    ))]
    /// How long to wait for missing UDP packets before skipping (ms).
    #[serde(default, alias = "udp_gap_skip_ms")]
    pub(crate) udp_gap_skip_ms: Option<u64>,

    #[validate(range(
        min = 0,
        max = 3600,
        message = "Invalid relay warm seconds",
        code = "relay_warm_seconds"
    ))]
    /// Keep the camera connection warm after last client disconnects (0 disables).
    #[serde(
        default = "default_relay_warm_seconds",
        alias = "relay_warm",
        alias = "relay_warm_secs",
        alias = "relay_warm_seconds"
    )]
    pub(crate) relay_warm_seconds: u64,

    /// How the camera maintains its connection: `always` (default) connects at
    /// startup and stays connected; `on_demand` connects only when needed.
    #[serde(default, alias = "connect", alias = "connection_mode")]
    pub(crate) connect_mode: ConnectMode,

    #[validate(range(
        min = 0,
        max = 86400,
        message = "Invalid idle timeout seconds",
        code = "idle_timeout_secs"
    ))]
    /// In `connect_mode = always`: seconds the camera may stay idle (no active
    /// use) before disconnecting; 0 = never disconnect. Ignored in `on_demand`
    /// mode (which uses `relay_warm_seconds`).
    #[serde(default, alias = "idle_timeout", alias = "idle_secs")]
    pub(crate) idle_timeout_secs: u64,

    #[validate(range(
        min = 0,
        max = 86400,
        message = "Invalid offline timeout seconds",
        code = "offline_timeout_secs"
    ))]
    /// Seconds an RTSP viewer's keepalive placeholder is served while this camera is
    /// offline before that session is torn down. Unset = inherit the global
    /// `offline_timeout_secs`. `0` = never; values 1-59 are clamped up to the 60s floor.
    #[serde(default, alias = "offline_timeout")]
    pub(crate) offline_timeout_secs: Option<u32>,

    #[serde(default = "default_maxenc")]
    #[validate(regex(
        path = *RE_MAXENC_SRC,
        message = "Invalid maximum encryption method",
        code = "max_encryption"
    ))]
    pub(crate) max_encryption: String,

    #[serde(default = "default_strict")]
    /// If strict then the media stream will error in the event that the media packets are not as expected
    pub(crate) strict: bool,

    #[serde(default = "default_update_time", alias = "time")]
    pub(crate) update_time: bool,

    #[validate(range(
        min = 1,
        max = 15000,
        message = "Invalid buffer duration (it's in ms)",
        code = "buffer_duration"
    ))]
    /// Buffer duration in ms
    #[serde(
        default = "default_buffer_duration",
        alias = "duration",
        alias = "buffer"
    )]
    pub(crate) buffer_duration: u64,

    #[serde(default = "default_true", alias = "enable")]
    pub(crate) enabled: bool,

    #[serde(default = "default_false", alias = "verbose")]
    pub(crate) debug: bool,

    #[serde(default = "default_true", alias = "splash")]
    pub(crate) use_splash: bool,

    #[serde(default = "default_splash", alias = "pattern")]
    pub(crate) splash_pattern: SplashPattern,

    #[serde(
        default = "default_max_discovery_retries",
        alias = "retries",
        alias = "max_retries"
    )]
    pub(crate) max_discovery_retries: usize,
}

#[derive(Debug, Deserialize, Serialize, Validate, Clone, PartialEq, Eq, Hash)]
pub(crate) struct UserConfig {
    #[validate(custom(function = "validate_username"))]
    #[serde(alias = "username")]
    pub(crate) name: String,

    #[serde(alias = "password", skip_serializing, default)]
    pub(crate) pass: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Validate, PartialEq, Eq)]
pub(crate) struct MqttConfig {
    #[serde(default = "default_true")]
    pub(crate) enable_motion: bool,
    #[serde(default = "default_true")]
    pub(crate) enable_light: bool,
    #[serde(default = "default_true")]
    pub(crate) enable_battery: bool,
    /// Update time in ms
    #[serde(default = "default_2000")]
    #[validate(range(
        min = 500,
        message = "Update ms should be > 500",
        code = "battery_update"
    ))]
    pub(crate) battery_update: u64,
    #[serde(default = "default_true")]
    pub(crate) enable_preview: bool,
    /// Update time in ms
    #[validate(range(
        min = 500,
        message = "Update ms should be > 500",
        code = "preview_update"
    ))]
    #[serde(default = "default_2000")]
    pub(crate) preview_update: u64,

    /// Enable the flood light tasks status
    /// Will not do anything if no floodlight
    /// is detected
    #[serde(default = "default_true")]
    pub(crate) enable_floodlight: bool,
    /// Update time in ms
    #[validate(range(
        min = 500,
        message = "Update ms should be > 500",
        code = "floodlight_update"
    ))]
    #[serde(default = "default_2000")]
    pub(crate) floodlight_update: u64,

    /// Publish doorbell ("visitor") presses as discrete events.
    ///
    /// Explicit opt-in: defaults to off. Cameras without a doorbell simply never
    /// emit a "visitor" status, so enabling this is a graceful no-op for them.
    #[serde(default = "default_false")]
    pub(crate) enable_doorbell: bool,

    #[serde(default)]
    pub(crate) discovery: Option<MqttDiscoveryConfig>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Validate, PartialEq, Eq)]
pub(crate) struct MqttDiscoveryConfig {
    pub(crate) topic: String,

    pub(crate) features: HashSet<Discoveries>,
}

fn validate_mqtt_server(config: &MqttServerConfig) -> Result<(), ValidationError> {
    if config.ca.is_some() && config.client_auth.is_some() {
        Err(ValidationError::new(
            "Cannot have both ca and client_auth set",
        ))
    } else {
        Ok(())
    }
}

const fn default_true() -> bool {
    true
}

const fn default_false() -> bool {
    false
}

fn default_mqtt() -> MqttConfig {
    MqttConfig {
        enable_motion: true,
        enable_light: true,
        enable_battery: true,
        battery_update: 2000,
        enable_preview: true,
        preview_update: 2000,
        enable_floodlight: true,
        floodlight_update: 2000,
        enable_doorbell: false,
        discovery: Default::default(),
    }
}

fn default_discovery() -> DiscoveryMethods {
    DiscoveryMethods::Relay
}

fn default_relay_warm_seconds() -> u64 {
    60
}

fn default_maxenc() -> String {
    "Aes".to_string()
}

#[derive(Debug, Deserialize, Serialize, Validate, Clone, PartialEq)]
pub(crate) struct PauseConfig {
    #[serde(default = "default_on_motion")]
    pub(crate) on_motion: bool,

    #[serde(default = "default_on_disconnect", alias = "on_client")]
    pub(crate) on_disconnect: bool,

    #[serde(default = "default_motion_timeout", alias = "timeout")]
    pub(crate) motion_timeout: f64,

    #[serde(default = "default_pause_mode")]
    #[validate(regex(
        path = *RE_PAUSE_MODE,
        message = "Incorrect pause mode",
        code = "mode"
    ))]
    pub(crate) mode: String,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, Eq, PartialEq)]
pub(crate) enum SplashPattern {
    #[serde(alias = "smpte")]
    Smpte,
    #[serde(alias = "snow")]
    Snow,
    #[serde(alias = "black")]
    Black,
    #[serde(alias = "white")]
    White,
    #[serde(alias = "red")]
    Red,
    #[serde(alias = "green")]
    Green,
    #[serde(alias = "blue")]
    Blue,
    #[serde(alias = "checkers-1")]
    Checkers1,
    #[serde(alias = "checkers-2")]
    Checkers2,
    #[serde(alias = "checkers-4")]
    Checkers4,
    #[serde(alias = "checkers-8")]
    Checkers8,
    #[serde(alias = "circular")]
    Circular,
    #[serde(alias = "blink")]
    Blink,
    #[serde(alias = "smpte75")]
    Smpte75,
    #[serde(alias = "zone-plate")]
    ZonePlate,
    #[serde(alias = "gamut")]
    Gamut,
    #[serde(alias = "chroma-zone-plate")]
    ChromaZonePlate,
    #[serde(alias = "solid-color")]
    SolidColor,
    #[serde(alias = "ball")]
    Ball,
    #[serde(alias = "smpte100")]
    Smpte100,
    #[serde(alias = "bar")]
    Bar,
    #[serde(alias = "pinwheel")]
    Pinwheel,
    #[serde(alias = "spokes")]
    Spokes,
    #[serde(alias = "gradient")]
    Gradient,
    #[serde(alias = "colors")]
    Colors,
    #[serde(alias = "smpte-rp-219")]
    SmpteRp219,
}

impl std::fmt::Display for SplashPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let s = match self {
            SplashPattern::Smpte => "smpte",
            SplashPattern::Snow => "snow",
            SplashPattern::Black => "black",
            SplashPattern::White => "white",
            SplashPattern::Red => "red",
            SplashPattern::Green => "green",
            SplashPattern::Blue => "blue",
            SplashPattern::Checkers1 => "checkers-1",
            SplashPattern::Checkers2 => "checkers-2",
            SplashPattern::Checkers4 => "checkers-4",
            SplashPattern::Checkers8 => "checkers-8",
            SplashPattern::Circular => "circular",
            SplashPattern::Blink => "blink",
            SplashPattern::Smpte75 => "smpte75",
            SplashPattern::ZonePlate => "zone-plate",
            SplashPattern::Gamut => "gamut",
            SplashPattern::ChromaZonePlate => "chroma-zone-plate",
            SplashPattern::SolidColor => "solid-color",
            SplashPattern::Ball => "ball",
            SplashPattern::Smpte100 => "smpte100",
            SplashPattern::Bar => "bar",
            SplashPattern::Pinwheel => "pinwheel",
            SplashPattern::Spokes => "spokes",
            SplashPattern::Gradient => "gradient",
            SplashPattern::Colors => "colors",
            SplashPattern::SmpteRp219 => "smpte-rp-219",
        }
        .to_string();
        write!(f, "{}", s)
    }
}

fn default_bind_addr() -> String {
    "0.0.0.0".to_string()
}

fn default_bind_port() -> u16 {
    8554
}

fn default_stream() -> StreamConfig {
    StreamConfig::All
}

fn default_certificate() -> Option<String> {
    None
}

fn default_tls_client_auth() -> String {
    "none".to_string()
}

fn default_channel_id() -> u8 {
    0
}

fn default_update_time() -> bool {
    false
}

fn default_motion_timeout() -> f64 {
    1.
}

fn default_on_disconnect() -> bool {
    false
}

fn default_on_motion() -> bool {
    false
}

fn default_pause_mode() -> String {
    "none".to_string()
}

fn default_strict() -> bool {
    false
}

fn default_pause() -> PauseConfig {
    PauseConfig {
        on_motion: default_on_motion(),
        on_disconnect: default_on_disconnect(),
        motion_timeout: default_motion_timeout(),
        mode: default_pause_mode(),
    }
}

fn default_buffer_duration() -> u64 {
    3000
}

fn default_max_discovery_retries() -> usize {
    10
}

fn default_2000() -> u64 {
    2000
}

fn default_splash() -> SplashPattern {
    SplashPattern::Snow
}

pub(crate) static RESERVED_NAMES: &[&str] = &["anyone", "anonymous"];
fn validate_username(name: &str) -> Result<(), ValidationError> {
    if name.trim().is_empty() {
        return Err(ValidationError::new("username cannot be empty"));
    }
    if RESERVED_NAMES.contains(&name) {
        return Err(ValidationError::new("This is a reserved username"));
    }
    Ok(())
}

fn validate_camera_config(camera_config: &CameraConfig) -> Result<(), ValidationError> {
    match (&camera_config.camera_addr, &camera_config.camera_uid) {
        (None, None) => Err(ValidationError::new(
            "Either camera address or uid must be given",
        )),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enable_doorbell_defaults_to_false() {
        // Doorbell is explicit opt-in: absent from config means disabled.
        let config: MqttConfig = toml::from_str("").unwrap();
        assert!(!config.enable_doorbell);
    }

    #[test]
    fn enable_doorbell_can_be_enabled() {
        let config: MqttConfig = toml::from_str("enable_doorbell = true").unwrap();
        assert!(config.enable_doorbell);
    }

    #[test]
    fn enable_motion_still_defaults_to_true() {
        // Guard the existing default while adding the new opt-in flag next to it.
        let config: MqttConfig = toml::from_str("").unwrap();
        assert!(config.enable_motion);
    }
}
