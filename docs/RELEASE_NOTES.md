# Release Notes

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
