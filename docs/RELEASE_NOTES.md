# Release Notes

## 0.6.4-beta.14

### Highlight: choose always-connected or on-demand per camera

This branch had been hardcoded to on-demand (disconnect whenever idle). That's
ideal for battery cameras but unnecessary for mains-powered ones, where you'd
rather keep the camera connected and ready. A new per-camera `connect_mode`
setting lets one build do both:

- **`connect_mode = "always"` (default)** — connect at startup and stay
  connected, reconnecting automatically on drops. Add `idle_timeout_secs = N` to
  disconnect after N seconds idle (0 = never, the default).
- **`connect_mode = "on_demand"`** — connect only when something needs the camera
  and disconnect when idle; `relay_warm_seconds` controls the linger. Best for
  battery cameras.

The default is now always-connected (the "regular" behaviour). Existing configs
keep working unchanged; set `connect_mode = "on_demand"` on the cameras you want
to behave the old way. All the stream/bitrate/caching fixes from recent releases
apply to both modes. See the README "Connection Modes" section.

Also removed two dead/redundant things:

- **Push notifications (FCM)** — entirely removed (the `pushnoti` feature, the
  `fcm-push-listener`/`dirs`/`md5` deps, the standalone `crates/pushnoti` tool, and
  the `push_notifications` option). Google shut down the API it used, so it no
  longer worked. Wake a disconnected `on_demand` camera with an external trigger
  (a client, an MQTT command, or `/control/wakeup`).
- **`stream_tuning`** (`bitrate_kbps` / `interframe_speed`) — only resized the
  internal video buffer (not the camera's encode, and did not skip detection);
  `buffer_duration` already covers buffer sizing.

Configs that still contain `push_notifications` or `stream_tuning` are ignored.

## 0.6.4-beta.13

### Highlight: audio/video sync

Audio drifted ahead of video (about 0.4 s on the main stream, more on the sub
stream) and the offset was effectively permanent. The cause was timestamping every
frame by its *arrival* time while the audio path also accumulated frame durations —
so when the camera delivered media in bursts (which it does), the audio timeline ran
ahead of video.

Timestamps now come from a single media clock instead of arrival time:
- **Video** uses the camera's own capture timestamp carried on each frame.
- **Audio** (which has no timestamp) advances by its own frame durations, anchored
  once to the video clock.

Both then advance at true media rate from one origin, so they stay in sync and
playback is smooth even under bursty delivery. A/V drift now stays near zero.

## 0.6.4-beta.12

### Highlight: full-bitrate direct P2P streaming

High-bitrate streams (notably 4K main streams at ~4 MBit/s) previously would not
play reliably over remote/P2P connections — they started, stuttered, and stalled
within seconds. Investigation with a new per-second UDP transport heartbeat showed
the camera was delivering only ~340 kbps with **zero packet loss** on an otherwise
healthy **direct P2P** connection — i.e. the camera was deliberately throttling.

The trigger was the `maybe_latency` value Neolink reports to the camera in every
UDP ACK, which feeds the camera's adaptive bitrate. It was being computed from the
gap between the camera's ACKs rather than real latency, producing a steady ~21 ms
that the camera read as a poor link. Reporting `0` instead lets the camera serve
the full configured bitrate. Result: the 4K main stream now streams in real time
over direct P2P.

This is purely a transport-layer fix; the RTSP/decode side was never the cause.

### Also in this release

- **`MEDIA PATH` logging** — every connection now states whether media is flowing
  over direct P2P (LAN / device / dmap hole-punch) or routed through Reolink's
  relay servers. This finally disambiguates the `relay` discovery method, which
  prefers a direct P2P punch and only forwards through Reolink as a fallback.
- **Effective-config logging** — the active `discovery`, `relay_region`,
  `udp_gap_skip_ms`, encryption and stream settings are logged at connect time.
- **UDP transport heartbeat** (debug) — throughput, packet counters, the await
  watermark, and the legacy latency value, once per second, for diagnosing
  bandwidth/stall issues.
- **Documentation** — a new Connection / Discovery Methods section in the README
  and `docs/connection-and-bandwidth.md` explaining each `discovery` method and
  its bandwidth behaviour.

### Upgrading / verifying

No config changes are required. To confirm the fix on a high-bitrate stream, open
the camera's main stream and watch the heartbeat:

```bash
RUST_LOG='info,neolink_core::bc_protocol::connection::udpsource=debug' \
  neolink rtsp --config=neolink.toml
```

`in_kbps` should rise to roughly the camera's configured bitrate (e.g. ~4000 for a
4 MBit/s stream) and hold, with `resends=0` and `recieved_pending=0`.
