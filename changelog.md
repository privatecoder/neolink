# Changelog — since fork

This is a curated summary of every notable improvement and fix in this fork
(`privatecoder/neolink`) since it diverged from upstream at commit **`f607d44`**
("create workflow to create and push container image").

It is organised by theme rather than commit-by-commit: intermediate experiments,
reverts, and iteration noise have been collapsed into their **net outcomes**.
Format loosely based on [Keep a Changelog](https://keepachangelog.com/).

---

## 0.7.15

Live-view reliability fix for Home Assistant / go2rtc: the RTSP keepalive
placeholder now preserves the camera's own codec parameter sets, and the
forwarder's backpressure queue self-corrects latency instead of pinning it.
Together these fix a reconnect loop where a stream would connect and then
immediately drop.

### Fixes — RTSP / live view

- **Keepalive no longer breaks the placeholder→live handoff.** When a client
  connected before the camera produced its first frame, Neolink served a
  locally-encoded black keyframe whose H264/H265 parameter sets (SPS/PPS, plus
  VPS for H265) differed from the camera's. The RTSP/SDP parameters the client
  negotiated were inferred from that synthetic frame, so when the real camera
  bitstream arrived a strict client (go2rtc / Home Assistant) rejected it and
  reconnected — a tight reconnect loop where the stream never held, made worse
  once a camera fell behind and clients lingered on the placeholder. Neolink now
  caches each stream's first valid real camera keyframe (one carrying complete
  parameter sets) and replays *that* as the keepalive placeholder, so the
  negotiated parameters match the live stream exactly. The synthetic black frame
  remains only as a cold-start fallback, before any real keyframe has been seen
  for that stream.
- **Backpressure now drops the oldest queued frame, not the newest.** The
  forwarder→sender bridge queue previously discarded the incoming frame when full
  and kept the stale backlog, so a brief camera lag pinned latency at the backlog
  depth (the `CATCHUP enter (lag=…)` path) and starved recovery. It now evicts the
  oldest queued frame and enqueues the newest, so latency self-corrects; an
  evicted video frame's clock advance is carried forward to the next delivered
  video frame, keeping the media clock continuous (no PTS jump / A-V drift).

No configuration changes are required.

## 0.7.14

Security and robustness release: hardens the protocol parsers against malformed
camera/network input, fixes several MQTT / Home Assistant integration bugs,
improves connection recovery and RTSP cleanup, fixes a handful of CLI issues,
and includes an internal cleanup pass. From a multi-round audit, independently
reviewed.

### Hardening (untrusted input)

- The BC, BC-media and BC-UDP parsers now bounds-check every wire-supplied
  length/offset before slicing, subtracting, or buffering: oversized declared
  bodies/frames are rejected (16 MiB cap) instead of buffered toward a huge
  allocation, length underflows return a parse error, and a malformed
  UDP-discovery checksum is a skippable parse error rather than a panic. A
  malformed packet can no longer crash neolink or exhaust memory.

### Fixes — MQTT / Home Assistant

- Command acknowledgements for **reboot**, **ptz/preset** and **ptz/assign**
  went to `control/ir`; they now reply on their own topics. The
  `query/ptz/preset` data and ack now use `status/ptz/preset` / `query/ptz/preset`.
- Removing a camera from the config now clears its retained Home Assistant
  discovery entities (previously they lingered as ghosts), and re-adding a
  removed camera now restarts it (a stale internal entry used to block it).
- The floodlight light entity now reflects real state in Home Assistant, and
  `control/wakeup` validates/caps its input.

### Fixes — connection / RTSP lifecycle

- A camera that fails login now recovers on a config reload (e.g. after fixing a
  wrong password) instead of staying dead until a restart.
- Runtime config reloads apply the offline-timeout safety floor and inheritance
  identically to startup.
- RTSP per-client cleanup is prompt (no leaked forwarder/stream subscription when
  a client disconnects while its camera is quiet), gstreamer buffer pools are
  deactivated on every exit path, a pipeline-setup failure serves the splash
  placeholder instead of a hard error, and the video clock keeps advancing across
  forwarder-queue frame drops (avoiding a PTS jump / A-V drift).

### Fixes — CLI

- `ptz control … <speed 0>` no longer panics; non-UTF-8 / spaced snapshot paths
  work; the `image` snapshot no longer leaves a 0-byte file on failure;
  `services … port` is validated as a 16-bit port; on/off arguments are
  case-insensitive; `tls_client_auth` accepts the `requested`/`required`
  spellings the sample config documents.

### Notes

- No configuration changes are required.
- Internal cleanup only (no behaviour change): named constants for duplicated
  literals, clippy cleanup, and a spelling pass over log/error messages.
- Verified: `cargo build`, `cargo test`, `cargo clippy`, `cargo fmt` all clean.
- The video-clock change touches the single-media-clock path; a quick A/V-sync
  sanity check after deploying is worthwhile.

---

## 0.7.13

Robustness fixes and a code-quality pass.

### Fixes

- **Discovery no longer aborts on a stray datagram.** During relay-assisted P2P
  hole-punching the discovery UDP socket is exposed to the network and can
  receive foreign, non-Baichuan datagrams (other protocols, NAT keepalives,
  scanners). A single such packet previously logged an error and tore down the
  discovery attempt; it is now skipped (logged at `debug`) and discovery keeps
  reading. Relay/camera terminate signals and genuine socket errors remain fatal.
- **`talk` no longer blocks the async runtime.** Playback pacing used a blocking
  `std::thread::sleep` inside the async `talk` path, stalling other tasks on that
  worker thread; it now uses `tokio::time::sleep`.
- **Several panics replaced with clean errors.** `neolink talk <camera>` with no
  audio source, a non-UTF-8 `--file-path`, a camera that omits `user_id` in the
  `users` listing, and re-serializing an unknown legacy message now return errors
  instead of panicking.

### Notes

- Internal cleanup (no behavioural change): removed dead code — an unused
  `BcCamera` field, an unused enum variant, and an unused field/parameter — and
  promoted duplicated timeout/capacity literals to named constants.
- No configuration changes are required.
- Verified: `cargo build`, `cargo test`, `cargo clippy`, `cargo fmt` all clean.

### References

- Full history: [`changelog.md`](changelog.md).

### Docker image

```bash
docker pull ghcr.io/privatecoder/neolink:0.7.13
```

---

## 0.7.12

Hardens camera snapshot reliability.

- Snapshots are validated as structurally complete JPEGs before they are published or written: the assembled image must start with the SOI marker (`FF D8`) and contain an EOI marker (`FF D9`), and any bytes after the final EOI are trimmed. An incomplete capture is retried up to three times; if it is still incomplete the MQTT `status/preview` keeps the last good frame and the `image` CLI fails cleanly, rather than emitting a partial JPEG.

### Notes
- No configuration changes are required.

---

## 0.7.11

Removes a dead MQTT topic and adds snapshot completeness checking.

### Removed the dead `status/notification` MQTT topic

- `status/notification` was a vestige of the push-notification feature removed in
  0.7.0 (Google FCM shutdown). Nothing has updated it since, yet it published a
  retained `"unknown"` that persisted on the broker indefinitely. Neolink now
  publishes a retained empty payload to that topic once at startup, which clears
  the stale retained value, and never publishes it again. No other behaviour
  depended on the topic.

### Snapshot completeness checking

- `get_snapshot` assembles the snapshot from binary chunks the camera streams. It
  now validates the assembled result and, if it is incomplete (for example a chunk
  lost to a UDP gap-skip during the one-shot fetch), retries the whole snap up to
  three times before returning a new `IncompleteSnapshot` error rather than the
  partial bytes. The MQTT preview loop treats `IncompleteSnapshot` as a transient
  skip: it keeps the last retained good frame and tries again on the next interval
  rather than tearing down the preview task.

### Notes

- No configuration changes are required.

---

## 0.7.10

Doorbell (visitor) button-press detection, reported separately from motion.

### Doorbell events split out of the motion/alarm stream

- Reolink cameras report a doorbell press as a `visitor` status on the same
  alarm stream that carries motion. The alarm decoder now distinguishes these:
  `visitor` is decoded as a doorbell press, motion statuses (`MD`, `PIR`, AI
  types) as motion start, and `none` as motion stop. Statuses that arrive
  comma-separated in a single message (for example `visitor,MD`) are decoded
  independently rather than collapsed, so a press that coincides with motion
  surfaces both. Existing motion behaviour for RTSP gating is unchanged.
- Doorbell presses are routed out of the existing motion-detection thread on a
  separate channel, so no additional camera subscription is opened.

### Per-camera `enable_doorbell` toggle (opt-in)

- New `[cameras.mqtt]` option `enable_doorbell`, default `false`. When enabled,
  each press publishes `{"event_type":"press"}` to
  `neolink/{camera}/status/doorbell`. It is a discrete event, not a state: the
  message is not retained and no idle/clear value is published. Cameras without a
  doorbell never emit a `visitor` status, so enabling the option is a no-op for
  them.

### Home Assistant discovery for doorbell

- New `doorbell` discovery feature (aliases `db`, `visitor`) published as a Home
  Assistant `event` entity with `device_class` `doorbell` and `event_types`
  `["press"]`, added via `features = ["doorbell"]` under
  `[cameras.mqtt.discovery]`.

---

## 0.7.9

Stability fix for long-running streams: a single slow RTSP client could stall the
camera's control channel and get the whole session dropped. Backported and adapted
from upstream #399.

### Poller no longer blocks on a full subscriber channel

- The per-connection poll loop (`crates/core/.../connection/bcconn.rs`) routes every
  inbound camera message to its subscriber. It used to **`await`** that delivery, so
  one subscriber whose channel was full — e.g. an overwhelmed RTSP client that had
  stopped draining its video frames — blocked the entire poll loop. That starved the
  camera's keepalive/control traffic, the camera dropped the session, and the stream
  fell into a reconnect cycle (the classic "streams die after hours of running").
- Delivery is now **non-blocking** (`try_send`): if a subscriber's channel is full,
  that one message is dropped for that subscriber (with a rate-limited warning) and
  the loop moves on. Keepalive and control are always processed promptly; the poll
  loop never waits on a single consumer. Frame dropping here is intentionally not
  keyframe-aware — that already happens downstream in the RTSP relay.
- `poll_commander` channel capacity raised `200 → 1000` for headroom, since the loop
  no longer back-pressures on slow subscribers.

### Audio buffer overflow no longer tears down the relay

- An audio appsrc that filled up returned `FlowError::Flushing`, which the relay
  treated as a hard disconnect and tore the whole session down. The relay now applies
  the same high-watermark drop strategy it already uses for video: at/above 80% of the
  audio buffer it drops the audio frame (while still advancing the audio clock) instead
  of overflowing, and a residual `Flushing` is treated as a dropped audio frame rather
  than a teardown whenever video is still flowing.

## 0.7.8

### Persistent stream-type cache

- Neolink learns each stream's codec and sizing on the first client connection and
  caches it so a later client can be served the offline placeholder immediately,
  without first re-learning the codec from the camera. That cache can now be
  persisted to disk.
- New top-level `stream_cache_path` option. Default is unset = in-memory only
  (unchanged behaviour). When set, the cache is loaded at startup and written
  on-change (atomic temp-file + rename) to that path. The
  `NEOLINK_STREAM_CACHE_PATH` environment variable overrides the config value; an
  empty value disables persistence.
- Effect: after a restart, a client that connects while a known camera is still
  offline gets the "stream not ready" placeholder built from the persisted types,
  instead of nothing. (The Home Assistant add-on points this at its persistent
  `/data` volume.)
- The on-disk format is versioned (`version = 1`) and decoupled from internal
  types; an unreadable, corrupt, or unknown-version file is ignored and rebuilt,
  and a single malformed entry is skipped rather than discarding the file.
- Disk entries are treated as a hint, not the truth: on the first connection the
  cached types are reconciled against the live stream. A caps-breaking change
  (video codec, audio format, audio sample-rate / channel count, or — treated
  conservatively — resolution) refreshes the cache and tears the session down so
  the client reconnects to a correctly-built pipeline; a sizing-only drift
  (bitrate / fps) refreshes the cache silently.

### Defensive splash fallback

- When a client connected before any codec had been learned and the camera was
  unreachable, the per-client pipeline callback could return nothing, which left
  GStreamer with an empty element and produced CRITICALs
  (`g_object_force_floating`, "could not create element") — the client received no
  stream.
- The from-scratch pipeline path now always falls back to the existing "Stream not
  Ready" splash pipeline instead of returning nothing, so the client gets a valid
  placeholder and the session reconciles when the camera comes back.

---

## Highlights

- **On-demand connections** — cameras can now stay completely dark until something
  actually needs them, then connect on demand (ideal for battery / relay cameras),
  while still supporting the classic always-connected mode.
- **Full-rate high-bitrate streaming** — 4K main streams that previously stuttered
  and "constantly buffered" now stream at their full configured bitrate.
- **Stable audio/video sync** — the persistent audio-ahead-of-video drift is gone.
- **Long-run stability** — the file-descriptor / memory leaks that crashed Neolink
  after hours of running are fixed, and the RTSP server now survives unlimited
  camera disconnect/reconnect cycles.
- **Modernised** — all dependencies updated to current releases and both crates
  bumped to a synced `0.7.0`.

---

## Features

- **Per-camera `connect_mode` (`always` | `on_demand`)** — a single build serves
  both styles:
  - **`always`** (default) connects at startup and stays connected, reconnecting
    on drops. Optional **`idle_timeout_secs`** (default `0` = never) disconnects
    after that many idle seconds and reconnects on demand.
  - **`on_demand`** connects only when a client/MQTT/motion needs the camera and
    disconnects when idle (best for battery cameras), with **`relay_warm_seconds`**
    controlling the idle linger.

  This grew out of an extended on-demand relay-connect investigation and replaces
  the earlier hardcoded on-demand behaviour and the removed `idle_disconnect` flag.

- **Configurable relay server region** — new `relay_server_region` setting (alias
  `relay_region`) so remote/relay connections can target the right Reolink region.

- **Relay-assisted P2P via the DMAP hole-punch** — discovery prefers a direct P2P
  path (LAN / device / dmap) and only forwards through Reolink relay servers as a
  fallback. The enabled `discovery` options were re-defined to a clear set
  (`local` / `remote` / `map` / `relay` / `cellular` / `debug`).

- **Two-way audio (talk) improvements**
  - Support for cameras that advertise **multiple `audioStreamMode` entries** in
    their TalkAbility XML (previously such cameras failed).
  - Optional **noise suppression** (and an echo-cancel toggle) for the talk input via
    a WebRTC DSP stage, with CLI flags `--noise-suppression`, `--echo-cancel`,
    `--noise-suppression-level`, and `--echo-suppression-level`. Talk audio is
    captured as PCM and encoded to ADPCM (DVI-4) for the camera.

- **Connection diagnostics**
  - **`MEDIA PATH` log line** — states at connect time whether media flows over
    **direct P2P** or is **routed through Reolink relay servers** (previously
    indistinguishable in the logs).
  - **Effective-connection-config log line** — prints the *resolved* values
    applied for a camera (`discovery`, `relay_region`, `connect_mode`,
    `udp_gap_skip_ms`, `buffer_duration`, `max_encryption`,
    `max_discovery_retries`, `strict`, `stream`). Defaulted/optional settings
    are shown with their effective value (e.g. `udp_gap_skip_ms=500 ms
    (default)`, `relay_region=none (all relay servers)`) instead of a raw
    `None`.
  - **Per-second UDP transport heartbeat** — `in_pkts`, `in_kbps`, `delivered`,
    `resends`, `packets_want`, `sent_unacked`, `recieved_pending`, plus
    reassembly/health counters `reorder_events`, `max_reorder_depth`,
    `max_pending`, and `ack_recv_rate`. Enable with
    `RUST_LOG='info,neolink_core::bc_protocol::connection::udpsource=debug'`
    (these are `debug!`-level because `release_max_level_debug` strips `trace!`
    from release builds).

---

## Streaming & transport fixes

- **High-bitrate streams no longer under-deliver / "constantly buffer".**
  High-bitrate streams (e.g. a 4 Mbit/s 4K main stream) would start, stutter, and
  stall because the camera's adaptive bitrate downshifted. Root cause: the UDP ACK
  field originally reverse-engineered upstream as `maybe_latency` is **not latency** —
  it is the **receiver's measured throughput in bytes/second**, which the camera's
  CUBIC rate-controller uses as its bandwidth estimate.
  - A miscomputed ACK-inter-arrival value originally pinned the bitrate to a
    ~340 kbps floor; an interim fix reported a constant `0`, which cleared that
    floor but left some models/remote paths capped at ~2 Mbit/s of a 4 Mbit/s feed.
  - **Final fix:** Neolink now reports its **actual received bytes over the trailing
    ~1 s** (latched ~1 Hz), so the camera tracks the real link and ramps to full
    rate. Measured: an affected camera went from 1.9 Mbit/s / 0.38× to
    ~4.7 Mbit/s / 0.93× real-time; already-fine cameras are unchanged. Verified
    against a packet capture of the official client (ratio ≈ 1.0). See
    [docs/connection-and-bandwidth.md](docs/connection-and-bandwidth.md) and
    [docs/reolink.md](docs/reolink.md).
  - **Renamed:** now that its purpose is known, the `UdpAck` field is called
    `recv_bytes_per_sec` in the code (it was the upstream guess `maybe_latency`). This
    is a source-only rename — the on-the-wire packet and behaviour are unchanged.

- **Audio/video drift eliminated.** Audio ran progressively ahead of video
  (~0.4 s on the main stream) because every frame was timestamped by its *arrival*
  time and audio additionally accumulated durations, so bursty delivery raced the
  audio PTS ahead. Timestamps now come from a single media clock: **video** uses the
  camera's own per-frame capture timestamp (wrap-safe, continuous across dropped
  frames), and **audio** rides a content-clock anchored once to the video clock. A/V
  drift now stays near zero with no playback stalls.

- **A single lost UDP packet no longer drops the whole connection.** When the
  reliable-UDP reassembler skips a missing packet (after `udp_gap_skip_ms`), the
  hole desynced the control-protocol framing and the `BcCodex` decoder errored
  fatally on the next header (`Magic invalid`), tearing down the connection and
  forcing a reconnect + re-login — observed every few minutes on high-bitrate
  remote/relay streams. The control codec now **resyncs** to the next BC header
  magic (the media codec already did), losing only the one corrupted message
  instead of the connection. Verified live: a real 2-packet loss on a 4K stream
  resynced in place (dropped ~13 KB, one I-frame chunk) with zero reconnect and
  no playback stall. The default `udp_gap_skip_ms` was also raised 120 → 500 ms
  so a retransmit has more time to fill the gap before a skip is attempted.

- **Lower latency.** Removed unnecessary internal sleeps and tuned the camera
  wake-up delay to reduce end-to-end latency.

---

## Stability & resource fixes

- **File-descriptor exhaustion and memory fragmentation fixed.** A new GStreamer
  `BufferPool` was being allocated for every received frame-sized packet, which
  (a) leaked a socketpair per allocation, steadily exhausting file descriptors, and
  (b) fragmented memory badly with H.264's large frame-size variance — on some
  cameras memory climbed past 7–8 GiB over a few hours and eventually crashed.
  Buffer pools are now reused (sized to the next power of two), stabilising
  long-running instances.

- **RTSP server stays operational indefinitely.** The server now survives unlimited
  camera disconnect/reconnect cycles, with **exponential backoff** on reconnection
  to give the network time to recover and reduce load.

- **No more 100% CPU / wedged runtime after a camera drop.** When a camera dropped,
  the process could peg one or more CPU cores and go silent right after
  `Connection Lost … Attempt reconnect` — the camera never reconnected and RTSP
  clients connected but were never served. The trigger was the per-camera MQTT
  handler being restarted in a tight loop with no backoff on a connection drop:
  each restart re-ran the full handler setup (retained publishes, last-will
  connections, resubscribes), which flooded the broker and drove `rumqttc`'s
  `EventLoop::poll()` — a single `select!` that returns on every call — into a
  busy-spin in our `loop { poll().await }`. With only a couple of worker threads,
  the spin starved the runtime so the reconnect task never ran. Fixes:
  - the per-camera handler now waits a **cancellation-aware 5 s backoff** before
    restarting, so a drop can no longer drive the restart/flood loop;
  - a spinning event loop is **detected and the connection dropped so it
    reconnects**, and a `PollRateLimiter` caps each poll loop to ~1000 iterations/s
    as a structural failsafe (logged, rate-limited) regardless of which internal
    rumqttc path returns fast; `pending_throttle` is also set non-zero to pace the
    pending-replay path.

  Two further non-yielding loops were fixed along the way: the connection's
  message-router (`bcconn`) re-polling a closed command channel, and the motion
  listener re-polling a dropped subscription.

- **RTSP client setup is bounded, self-cleaning, and serves known streams
  instantly.** The factory callback that builds a client's pipeline runs on a
  GStreamer thread that blocks until the pipeline is ready, so a slow or offline
  camera could hold that thread. Setup is now bounded by a timeout and is
  cancellation-aware, a stream generation owns and tears down its per-client tasks
  and RTSP mounts on reconfiguration (no leaked handlers or stale mounts), detached
  per-client tasks log their errors instead of dropping them, and appsrc pushes use
  typed outcomes (the pre-PLAY "not linked" state is tolerated rather than treated
  as a disconnect). Once a stream's codec has been seen it is cached, so a later
  client gets its pipeline built and served **immediately** — without waiting on
  the camera — while the camera connection is made in the background; a stream that
  drops resumes into the same session without the client reconnecting.

- **Live view survives a camera reboot / connects while the camera is offline.** When
  a client opens a cached stream while the camera is down (or the camera reboots while
  a card is open), a low-rate placeholder is generated and streamed — a black video
  keyframe plus matching silent audio, encoded once at the stream's cached resolution
  and AAC rate — so the negotiated video *and* audio tracks keep producing RTP and the
  viewer (e.g. go2rtc/Home Assistant) doesn't time out. When the camera returns, the
  session hands off to live video on its own, with no reconnect. Without the audio
  placeholder, players that negotiated an audio track dropped the session after ~20 s
  even though video was flowing. If the placeholder can't be built, the stream falls
  back to its previous behaviour.

- **Optional offline timeout (`offline_timeout_secs`).** By default the offline
  placeholder is held indefinitely (`0` = never), which is ideal for always-on
  dashboards. For operators who would rather a long outage *end* the stream — so the
  viewer (e.g. Home Assistant) can mark the camera unavailable / trigger an automation
  — an optional per-camera or global `offline_timeout_secs` tears the session down
  after N seconds with no real camera frames. It's per-session (the shared camera
  connection keeps reconnecting for other viewers), the clock counts only genuine
  offline-placeholder time, and values 1-59 are raised to a 60 s floor (it must exceed
  your camera's reboot time). Precedence: per-camera, else the global default, else 0.

- **Hardened media/control parsing against malformed input.** The control-codec
  resync is now bounded (it gives up after a sane byte budget instead of scanning
  indefinitely), AAC duration parsing validates frame length before counting
  frames, and a short/truncated ADPCM frame can no longer underflow its block-size
  computation. Initial UDP packet gaps at stream start are recovered instead of
  dropping the connection.

- **Pipeline torn down on client disconnect.** When an RTSP client disconnects, its
  GStreamer pipeline is now killed, preventing writes to a closed socket.

- **Intelligent frame dropping under backpressure.** Frames are now dropped *before*
  they fill the BC protocol channel, and the AppSrc / internal message buffers were
  enlarged (100 → 500 messages) to absorb temporary bursts without stalling.

- **Reduced log spam** — quieter handling of H.264/H.265 parser warnings caused by
  camera stream errors. The per-second RTSP factory heartbeat (`… HB elapsed=…
  vid_ts=… aud_buf=…`) is now `debug!` rather than `info!`, so default `info`
  logs are no longer flooded; enable it with
  `RUST_LOG='info,neolink::rtsp::factory=debug'`.

- **Unsupported camera features are no longer re-probed.** Cameras without a
  battery or floodlight rejected those queries with a confusing `Task Error` on
  every check. Neolink now caches the camera's "unsupported" reply for the
  connection, skips re-probing, and logs a single clear `no battery, skipping` /
  `floodlight tasks not supported, skipping` line instead of an error.

- **RTSP server now fails loudly if the bind port is already in use.** The
  socket bind result was previously ignored, so a port clash (most commonly
  Home Assistant's built-in go2rtc, which also uses RTSP port `8554`) left
  Neolink logging `Starting RTSP Server …` while serving nothing. It now returns
  a clear error naming the address/port and the likely go2rtc conflict instead
  of silently running dead.

---

## Removed

- **Push notifications (FCM) — removed entirely.** Google shut down the API this
  relied on. Removed the `pushnoti` feature and its `fcm-push-listener` / `dirs` /
  `md5` deps, the standalone `crates/pushnoti` tool, all in-binary push plumbing,
  and the `push_notifications` config option. To wake a fully-disconnected
  `on_demand` camera, use an external trigger (an RTSP client, an MQTT command, or
  `/control/wakeup`).
- **`stream_tuning` (`bitrate_kbps` / `interframe_speed`)** — only resized the
  internal video buffer (which `buffer_duration` already covers) and was a footgun
  when mistakenly placed at the document root. Removed.
- **Dead config options** — `print_format`, `tokio_console`, `idle_disconnect`, and
  `push_notifications` were deserialized but unused. Configs that still set any of
  these are simply ignored.
- **Dead code and unused crates** — removed an orphaned `adpcm.rs`, a commented-out
  `build_mpegts` experiment, the dead FCM `pushinfo` module and various unused
  functions/fields, plus the unused `crates/decoder` and `crates/mailnoti`
  auxiliary crates and the stale `kubernetes/` manifests.

---

## Dependencies & tooling

- **All dependencies updated to their latest releases; crates synced to `0.7.0`.**
  No functional/on-the-wire change intended; validated against live cameras and the
  full unit-test suite. Notable major bumps and their migrations:
  - **nom 7 → 8** — associated-type `Parser` trait; `VerboseError` moved to the new
    `nom-language` crate. BC/UDP/media deserializers migrated to `.parse()` and a
    closure in place of the manual `Parser` impl.
  - **aes 0.8 → 0.9 / cfb-mode 0.8 → 0.9** (RustCrypto `cipher` 0.5) — CFB cipher
    types are no longer `Clone` and `AsyncStreamCipher` was removed; the AES-128 key
    is stored and the CFB cipher is built per call (same algorithm, identical output).
  - **rand 0.8 → 0.10** — `thread_rng()` → `rng()`, `Rng::gen()` → `random()`
    (now on `RngExt`).
  - **quick-xml 0.36 → 0.40** — serializer errors split into `SeError`; stricter
    XML-declaration validation.
  - **toml 0.8 → 1.1**, **thiserror 1 → 2**, **validator 0.18 → 0.20**,
    **rumqttc 0.24 → 0.25**, **gstreamer 0.24 → 0.25** (×4),
    **tikv-jemallocator 0.5 → 0.7**, **socket2 0.5 → 0.6**, **md5 0.7 → 0.8**,
    **delegate 0.12 → 0.13**, **env_logger → 0.11**, plus a full `cargo update`.
- **Earlier dependency maintenance** — bumped GStreamer and refreshed packages
  ahead of the larger dependency sweep.
- **Rust edition 2021** — both crates moved to edition 2021 (no behavioural change).
- **Docker / build** — improved the Dockerfile for development builds and added
  build caching; fixed the crate edition setting and cleared build warnings. Dropped
  `apt-get upgrade` from the build and runtime stages so images build reproducibly
  against the pinned base. Images are published to GitHub Container Registry
  (`ghcr.io/privatecoder/neolink`).

---

## Documentation

- **Full Camera Configuration Reference** added to the README (every option with
  its default).
- **Connection / Discovery Methods** section added to the README explaining each
  `discovery` option and its bandwidth characteristics.
- **[docs/connection-and-bandwidth.md](docs/connection-and-bandwidth.md)** — the
  discovery-method reference, the bitrate findings, and how to read the transport
  diagnostics.
- **[docs/reolink.md](docs/reolink.md)** — reverse-engineering notes on Reolink's
  SongP2P / relay behaviour and the RDT/p2p_udt flow control.
- **[docs/home-assistant.md](docs/home-assistant.md)** — using Neolink with Home
  Assistant: the go2rtc / WebRTC / MSE / HLS viewing path, the H264/H265 + AAC
  codec-vs-transport matrix, why an H265 stream can be slow to open, and the
  camera-side I-frame-interval (GOP) and CBR/VBR trade-offs.
