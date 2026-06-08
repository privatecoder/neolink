# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Neolink is a proxy/bridge between Reolink IP cameras (which speak a proprietary
reverse-engineered "Baichuan" / BC protocol over port 9000 instead of ONVIF/RTSP)
and standard clients. Its primary mode runs an RTSP server so NVR software (Blue
Iris, Shinobi) can consume the camera streams; it also exposes MQTT control and
one-shot CLI commands (image, battery, ptz, talk, etc.).

This is the `privatecoder/neolink` fork, focused on stability and on-demand
connection behavior.

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
  `BcCamera` type and all camera operations live here; it's the only workspace member
  besides the root binary, which depends on it.
- `dissector/` — Wireshark Lua dissector for the BC protocol (debugging aid, not built).
  (The `pushnoti`, `mailnoti`, and `decoder` auxiliary crates were removed — push
  notifications are dead, the others were unused experiments/dev tools.)

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
  loop (`NeoCamMdThread`), the motion permit watcher, and the **on-demand
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

The `docs/` directory holds the full technical reference for how Neolink talks to
Reolink cameras — start at [`docs/README.md`](docs/README.md) (index + end-to-end
overview). Read the relevant page before touching that area:

- [`docs/architecture.md`](docs/architecture.md) — the camera-ownership model
  (reactor / actors / permits).
- [`docs/connection-modes.md`](docs/connection-modes.md) — `connect_mode`, the
  connect/disconnect lifecycle, reconnect + backoff.
- [`docs/connection-and-bandwidth.md`](docs/connection-and-bandwidth.md) —
  `discovery` methods, regions, the UDP/CUBIC flow control that governs bitrate, and
  the transport diagnostics.
- [`docs/discovery-handshake.md`](docs/discovery-handshake.md) — the UDP P2P
  negotiation message catalog and per-method sequences.
- [`docs/bc-protocol.md`](docs/bc-protocol.md) — BC control framing, binary mode,
  encryption, message IDs.
- [`docs/media-streams.md`](docs/media-streams.md) — the BcMedia substream format,
  codecs, stream kinds, and frame-size / reassembly characteristics.
- [`docs/av-timestamping.md`](docs/av-timestamping.md) — RTSP-output PTS.
- [`docs/two-way-audio.md`](docs/two-way-audio.md) — talk (ADPCM DVI-4).
- [`docs/reolink.md`](docs/reolink.md) — empirical SongP2P / RDT notes (direct-vs-relay
  selection, the CUBIC governor).

## Transport / bandwidth gotchas (UDP path)

These bit hard during the P2P-bitrate investigation; keep them in mind when
touching `crates/core/src/bc_protocol/connection/udpsource.rs`:

- **`maybe_latency` is the camera's bitrate lever — report your measured
  received-bytes/second there, not `0`.** Despite the name it is NOT latency: it is
  the receiver's measured throughput (bytes/s) that the camera's CUBIC uses as its
  bandwidth estimate. History: a miscomputed ACK-inter-arrival value originally
  pinned the bitrate to ~340 kbps; reporting a constant `0` cleared that floor but
  left the camera on a conservative default that capped some models / remote paths
  at ~2 Mbit/s of a 4 Mbit/s feed; **Neolink now reports the real
  received-bytes-per-~1s** (`ack_recv_rate`, latched ~1 Hz in `build_send_ack`),
  letting CUBIC ramp to full rate (an affected camera: 1.9→4.7 Mbit/s; others
  unchanged). Pitfalls if you touch this: the unit is bytes/**second** (a
  bytes/100 ms value is 10× too small and throttles); `0` or any constant
  under-reports and caps the rate; verified against an official-client packet
  capture (ratio ≈ 1.0). The legacy `AckLatency` (ACK inter-arrival, not RTT) is now
  heartbeat-log-only. Full writeup: `docs/connection-and-bandwidth.md`
  "UDP transport flow control" and `docs/reolink.md` "RDT / p2p_udt flow control".
- **`release_max_level_debug`** (in root `Cargo.toml`'s `log` dependency) compiles
  **all `log::trace!` out of release builds** — including Docker. Diagnostics that
  must survive release use `log::debug!`. The per-second `UDP HB:` heartbeat in
  `udpsource.rs::run()` and the `MEDIA PATH:` line are `debug!`/`info!` for this
  reason; enable with `RUST_LOG=...connection::udpsource=debug`.
- The reliable-UDP reassembler advances `packets_want` only over contiguous
  packets; `recieved_pending > 0` means a gap is blocking delivery, and
  `udp_gap_skip_ms` bounds how long it waits before skipping (default 500 ms). A
  frozen `packets_want` with `in_pkts=0` means the camera stopped sending (not
  loss).
- **A skipped packet desyncs the framing, so both codecs resync instead of
  dropping the connection.** Skipping punches a hole in the byte stream; the next
  BC header then fails its magic check. `BcCodex` (control) and `BcMediaCodex`
  (media) both scan forward to the next magic and resume (`bc::codex` /
  `bcmedia::codex`), losing only the one corrupted message. Before this fix
  `BcCodex` errored fatally on `Magic invalid` and forced a reconnect + re-login
  on every skip — the symptom was a `Connection Lost: ... Magic invalid` every
  few minutes on high-bitrate remote streams. Do NOT make `BcCodex` strict again:
  the BC header is plaintext + length-prefixed and each message decrypts
  independently, so resync is safe. The search starts at offset 1 to guarantee
  forward progress (`next_bc_magic_offset`).
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
