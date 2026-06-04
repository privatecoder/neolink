# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Neolink is a proxy/bridge between Reolink IP cameras (which speak a proprietary
reverse-engineered "Baichuan" / BC protocol over port 9000 instead of ONVIF/RTSP)
and standard clients. Its primary mode runs an RTSP server so NVR software (Blue
Iris, Shinobi) can consume the camera streams; it also exposes MQTT control,
push-notification wakeups, and one-shot CLI commands (image, battery, ptz, talk, etc.).

This is the `privatecoder` fork of `QuantumEntangledAndy/neolink`, focused on
stability and on-demand connection behavior.

## Build / test / lint

```bash
cargo build --release                 # default build (includes `gstreamer` feature)
cargo build --no-default-features     # core/CLI only, no RTSP/gstreamer/talk/image

cargo test                            # whole workspace; most tests live in crates/core
cargo test -p neolink_core            # just the protocol crate
cargo test -p neolink_core bcudp::de  # a single module's tests by path filter

cargo fmt                             # rustfmt, edition 2018 (see rustfmt.toml)
cargo clippy --all-targets
```

- **GStreamer is a build/runtime dependency** for the default feature set. `build.rs`
  wires up link paths for macOS (`/Library/Frameworks/GStreamer.framework`) and Windows;
  on Linux the system gstreamer dev packages must be present. Building with
  `--no-default-features` avoids this but drops `rtsp`, `talk`, and `image` subcommands.
- The crate compiles with `--cfg tokio_unstable` (set globally in `.cargo/config.toml`)
  to enable tokio tracing / console support.
- `pre-commit` runs `cargo fmt`, whitespace/merge-conflict checks, and markdownlint on
  README.md. Install hooks with `pre-commit install`.
- Version string shown at startup comes from `git describe --tags` (or `GITHUB_SHA` in CI)
  via `build.rs`, not from Cargo.toml directly.

Run locally with a config file (required):
```bash
cargo run -- rtsp --config=neolink.toml          # default RTSP server
cargo run -- mqtt-rtsp --config=neolink.toml     # RTSP + MQTT control together
```

## Workspace layout

- Root crate (`src/`) — the `neolink` binary: CLI, config, RTSP/MQTT servers, and the
  per-camera orchestration layer (`src/common/`).
- `crates/core` (`neolink_core`) — the reverse-engineered BC protocol library. The
  `BcCamera` type and all camera operations live here; everything else depends on it.
- `crates/decoder`, `crates/mailnoti` — supporting crates (media decoding,
  email-based notifications). (FCM push notifications were removed — the API is dead.)
- `dissector/` — Wireshark Lua dissector for the BC protocol (debugging aid, not built).

## Architecture: the camera ownership model

The central design is a **channel-actor pattern** with a clonable handle, so many
subsystems (RTSP, MQTT, motion detection) can share one camera
connection without locking. Understanding this requires reading several files in
`src/common/`:

- **`NeoReactor`** (`reactor.rs`) — top-level owner of all cameras. Holds the live
  `Config` in a `watch` channel and a `HashMap<name, NeoCam>`. Cameras are created
  **lazily** on first `get(name)`. `update_config` diffs the new config against running
  cameras (drops removed ones, pushes new config to survivors). All access goes through
  an mpsc command channel — never touch the map directly.

- **`NeoCam`** (`neocam.rs`) — owns one physical camera. Spawns a `JoinSet` of long-lived
  tasks: a command dispatcher, the connection loop (`NeoCamThread`), the motion-detection
  loop (`NeoCamMdThread`), motion/push-notification permit watchers, and the **on-demand
  connect loop**. Communicates outward only by handing out `NeoInstance`s.

- **`NeoInstance`** (`instance.rs`) — the cheap, `Clone`able handle every subsystem holds.
  It does not own the camera; it talks to `NeoCam` over channels. Key helpers:
  - `run_task(f)` — runs `f(&BcCamera)`, **taking a use-permit** (keeps the camera
    connected for the duration), and auto-retries `f` if the camera reconnects mid-call.
  - `run_passive_task(f)` — same retry behavior but **no permit**, so the camera may
    disconnect for inactivity during the call. Streams and motion detection use this.
  - The retry/reconnect logic keys off a `watch<Weak<BcCamera>>`: when the camera object
    is swapped (reconnect), in-flight tasks observe the change and re-run.

- **Connection lifecycle** via **`UseCounter` / `Permit`** (`usecounter.rs`): a permit is
  an RAII token. RTSP clients, MQTT commands, and motion events each take a permit. The
  connect loop in `neocam.rs` branches on the per-camera **`connect_mode`** (`config.rs`,
  `ConnectMode`):
  - **`Always`** (default) → `connect()` at startup and stay connected (camthread
    reconnects on drops). If `idle_timeout_secs` > 0, it disconnects after that many
    seconds with no permits and reconnects on demand; `0` = never disconnect.
  - **`OnDemand`** → waits for `aquired_users()` > 0 to `connect()`, and `dropped_users()`
    == 0 to `disconnect()` (with an optional `relay_warm_seconds` grace window). The
    camera stays dark until something needs it — this is why the startup info-queries were
    removed (see the "Camera info reporting removed" comment) so `OnDemand` truly never
    connects at boot.
  The loop re-reads the config each cycle, so `connect_mode` can change at runtime.

Subcommands (`src/{rtsp,mqtt,ptz,battery,...}/`) each have a `cmdline.rs` (clap `Opt`) and
a `mod.rs` `main(opts, reactor)`. `src/main.rs` parses, loads+validates config, builds the
`NeoReactor`, and dispatches. `mqtt-rtsp` runs the MQTT and RTSP servers concurrently via
`tokio::select!`.

## Core protocol crate (`crates/core`)

- `bc_protocol/` — high-level camera operations. `BcCamera` (in `bc_protocol.rs`) is the
  public entry point; each file (`login.rs`, `ptz.rs`, `motion.rs`, `talk.rs`, `snap.rs`,
  `battery.rs`, …) implements one feature area as methods on it.
- `bc_protocol/connection/` — transport layer: `discovery.rs` (how the camera is located —
  `Local`, `Remote`, `Map`, `Relay`, plus relay-assisted P2P), `tcpsource.rs`,
  `udpsource.rs`, `bcconn.rs` (connection multiplexing), `bcsub.rs` (per-message subscriber).
- `bc/`, `bcudp/`, `bcmedia/` — wire format (de)serialization for the control protocol,
  the UDP transport, and the media substream respectively. These have the bulk of the
  unit tests (`de.rs`/`ser.rs`/`xml.rs` pairs).

`DiscoveryMethods` and the relay region/UDP-gap tuning are surfaced up through
`CameraConfig` in `src/config.rs` and passed into `BcCameraOpt`.

`docs/reolink.md` holds reverse-engineering findings on Reolink's SongP2P/relay
behavior — the DMAP relay-map phase, direct-vs-relay path selection (direct P2P
often carries the stream even when relay is available), and observed frame-size /
reassembly characteristics (main-stream I-frame bursts up to ~240 KB). Read it
before touching the discovery/connection/reassembly code.
`docs/connection-and-bandwidth.md` documents the `discovery` methods and the
transport diagnostics.

## Transport / bandwidth gotchas (UDP path)

These bit hard during the P2P-bitrate investigation; keep them in mind when
touching `crates/core/src/bc_protocol/connection/udpsource.rs`:

- **`maybe_latency` drives the camera's adaptive bitrate.** Every UDP ACK Neolink
  sends carries a `maybe_latency` field the camera reads as a link-quality signal.
  A non-zero value makes the camera downshift hard (a 4 Mbit/s stream collapses to
  ~340 kbps even over a loss-free direct-P2P link). Neolink reports `0`
  ("healthy"); do not "restore" a computed value here without understanding this.
  The legacy `AckLatency` was computed wrong (ACK inter-arrival, not RTT) and is
  retained only for the heartbeat log / a future loss-aware signal.
- **`release_max_level_debug`** (in root `Cargo.toml`'s `log` dependency) compiles
  **all `log::trace!` out of release builds** — including Docker. Diagnostics that
  must survive release use `log::debug!`. The per-second `UDP HB:` heartbeat in
  `udpsource.rs::run()` and the `MEDIA PATH:` line are `debug!`/`info!` for this
  reason; enable with `RUST_LOG=...connection::udpsource=debug`.
- The reliable-UDP reassembler advances `packets_want` only over contiguous
  packets; `recieved_pending > 0` means a gap is blocking delivery, and
  `udp_gap_skip_ms` bounds how long it waits before skipping. A frozen
  `packets_want` with `in_pkts=0` means the camera stopped sending (not loss).
- The displayed version comes from `git describe --tags` (build.rs), not just the
  `Cargo.toml` version — tag releases for the version string to be meaningful.

## A/V timestamping (RTSP factory)

In `src/rtsp/factory.rs::send_to_sources`, PTS must come from a **single media
clock**, never arrival time. Video uses the camera's per-frame `microseconds`
(`video_ts_from_camera`, wrap-safe); audio (no timestamp) uses a `+ duration`
content-clock anchored once to the video clock (`aud_anchored`). Do NOT switch
audio to arrival-based timestamps — audio is constant-rate but arrives bursty, so
arrival-timed PTS destabilizes the pipeline and causes periodic stalls (tried, fails).
The catch-up/backpressure drop paths must keep advancing the video clock
(`video_microseconds` helper) so it stays continuous across dropped frames.

## Config

A TOML `--config` file is mandatory for every subcommand. The `Config`/`CameraConfig`
structs (`src/config.rs`) use serde defaults + the `validator` crate; `main.rs` calls
`.validate()` before doing anything. The live config is held in the reactor's `watch`
channel, and publishing to MQTT `neolink/config` updates it at runtime (see README MQTT
section). See `sample_config.toml` for the full set of options and README.md for usage of
each subcommand and the MQTT topic surface.
