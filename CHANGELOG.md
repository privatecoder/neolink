# Changelog

All notable changes to this fork are documented here.
The format is loosely based on [Keep a Changelog](https://keepachangelog.com/).

## [0.6.4-beta.14] - 2026-06-04

### Added

- **Per-camera `connect_mode`** so a single build serves both styles:
  - **`always`** (default) — connect at startup and stay connected, reconnecting
    on drops (the "regular" always-available behaviour). Optional
    **`idle_timeout_secs`** (default `0` = never) disconnects after that many
    seconds idle and reconnects on demand.
  - **`on_demand`** — connect only when needed and disconnect when idle (best for
    battery cameras). Uses `relay_warm_seconds` for the idle linger.

  Previously the branch was hardcoded on-demand; the default is now always-connected.
  Existing configs need no change (default = always); set `connect_mode = "on_demand"`
  per camera for the old behaviour. The removed `idle_disconnect` flag is superseded
  by `connect_mode` + `idle_timeout_secs`.

### Removed

- **Push notifications (FCM) — removed entirely.** Google shut down the API this
  relied on, so it had been dead. Removed the `pushnoti` cargo feature and its
  `fcm-push-listener` / `dirs` / `md5` dependencies, the standalone `crates/pushnoti`
  tool, all the in-binary push plumbing, and the `push_notifications` config option.
  Configs that still set `push_notifications` are simply ignored. To wake a
  fully-disconnected `on_demand` camera, use an external trigger (an RTSP client,
  an MQTT command, or `/control/wakeup`).
- **Dead config options `print_format` and `tokio_console`** — both were
  deserialized but never read by any logic. Removed; configs that still set them
  are ignored.
- **Internal dead code cleanup** — removed an orphaned (never-compiled) `adpcm.rs`
  module, a commented-out experimental `build_mpegts` pipeline and its unused
  helpers, and several unused functions/fields/commands in the binary.
- **`stream_tuning` (`bitrate_kbps` / `interframe_speed`)** — these only resized
  the internal video buffer and didn't change the camera's encode or skip stream
  detection; `buffer_duration` already covers buffer sizing more directly. Removed
  to cut config surface and a footgun (it was a per-camera key often mistakenly
  placed at the document root, where it was silently ignored). Existing configs
  that still contain it are simply ignored — no change needed.

## [0.6.4-beta.13] - 2026-06-04

### Fixed

- **Audio/video drift eliminated.** Audio ran progressively/persistently ahead of
  video (~0.4 s on the main stream) because every media frame was timestamped by
  its *arrival* time, and the audio path additionally accumulated frame durations,
  so during bursty network delivery the audio PTS raced ahead of video. Timestamps
  are now derived from a single media clock:
  - **Video** uses the camera's own per-frame capture timestamp
    (`BcMediaIframe`/`BcMediaPframe.microseconds`), rebased to a monotonic PTS
    (wrap-safe; a camera clock reset is detected and skipped). The catch-up/drop
    paths advance this clock too, so it stays continuous across dropped frames.
  - **Audio** (which carries no timestamp) rides a content-clock (advancing by each
    frame's own duration) that is anchored once to the video camera-clock, keeping
    audio smooth *and* aligned with video regardless of bursty arrival.

  A/V drift now stays near zero with no playback stalls.

## [0.6.4-beta.12] - 2026-06-04

### Fixed

- **Direct P2P streams are no longer throttled to ~340 kbps.** High-bitrate
  streams (e.g. a 4 MBit/s 4K main stream) would start, stutter, and stall
  because the camera's adaptive bitrate downshifted to a low "fluent" floor.
  The cause was the `maybe_latency` value neolink reports back to the camera in
  every UDP ACK: it was computed from the inter-arrival gap between the camera's
  ACKs (effectively "how often the camera acks our near-empty uplink") rather
  than real link latency, producing a steady ~21 ms that the camera read as a
  poor link. neolink now reports `0` ("healthy link") so the camera serves the
  full configured bitrate over direct P2P. Genuine packet loss is still handled
  by the reliable-UDP layer (resend + gap-skip). See
  [docs/connection-and-bandwidth.md](docs/connection-and-bandwidth.md).

### Added

- **`MEDIA PATH` log line** at connection time stating whether media flows over
  **direct P2P** (LAN / device / dmap hole-punch) or is **routed through Reolink
  relay servers**. Previously the relay-assisted path's P2P success and its
  relay-server fallback were indistinguishable in the logs.
- **Effective connection config log line** printed when a camera connects
  (`discovery=`, `relay_region=`, `udp_gap_skip_ms=`, `max_encryption=`,
  `max_discovery_retries=`, `stream=`) so the active settings are unambiguous.
- **UDP transport heartbeat** (debug level) in the payload reassembler: per-second
  `in_pkts`, `in_kbps`, `delivered`, `resends`, `packets_want`, `sent_unacked`,
  `recieved_pending`, and `ack_latency_us`. Enable with
  `RUST_LOG=...connection::udpsource=debug`. Invaluable for diagnosing throughput
  and stall behaviour (note: `trace!`-level logs are compiled out of release
  builds via `release_max_level_debug`, so these diagnostics use `debug!`).

### Documentation

- Added a **Connection / Discovery Methods** section to the README explaining
  what each `discovery` option does and its bandwidth characteristics.
- Added [docs/connection-and-bandwidth.md](docs/connection-and-bandwidth.md) with
  the full discovery-method reference, the bitrate findings, and how to read the
  new diagnostics.
- Added [docs/reolink.md](docs/reolink.md) — reverse-engineering notes on
  Reolink's SongP2P / relay behaviour.
