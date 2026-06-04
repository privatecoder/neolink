# Connection / Discovery Methods & Bandwidth

This document explains how Neolink connects to a camera, what each `discovery`
method does, and the bandwidth characteristics of each path. It also documents
the diagnostics added in 0.6.4-beta.12.

## How a connection is established

Reolink cameras using the proprietary Baichuan (BC) protocol can be reached
several ways. When a stream is requested, Neolink registers with Reolink's
lookup servers (using `relay_server_region` to pick the region) to learn three
candidate addresses for the camera:

| Address | Meaning |
|---|---|
| `dev`   | The camera's **LAN** address (e.g. `192.168.1.50`). |
| `dmap`  | The camera's **public / NAT-mapped** address, registered with Reolink. Used for P2P hole-punching. |
| `relay` | A **Reolink relay server** that forwards traffic between you and the camera. |

The `discovery` setting selects which of these Neolink will try.

## `discovery` methods

Set per-camera in the config (`discovery = "..."`). Default is `relay`.

| Method   | Tries | Media path | Bandwidth |
|----------|-------|-----------|-----------|
| `local`  | LAN broadcast to the camera | **Direct P2P** on the LAN | Full (LAN speed). Only works on the same network. |
| `remote` | Connect to the camera's `dev` address | **Direct P2P** | Full, *if* the `dev` address is routable from the host (usually only on/near the LAN). |
| `map`    | Hole-punch to the camera's `dmap` (public) address | **Direct P2P** | Full. Works through NAT. No fallback — errors if the punch fails. |
| `relay`  | `dmap` hole-punch **first**, then fall back to the Reolink relay server | **Direct P2P** when the punch succeeds; **relayed** only if it fails | Full on the P2P path; the relay-server fallback is forwarded through Reolink and may be slower / rate-limited. |
| `cellular` | `map` + `relay` | As above | As above. |
| `debug`  | `local`, `remote`, `map`, `relay` (first to succeed wins) | Whatever wins | — |

**Recommended:** `relay` for remote cameras. Despite the name, it prefers a
**direct P2P** hole-punch and only forwards through Reolink's servers as a last
resort (when P2P is genuinely impossible). Use `map` to force pure P2P with no
relay fallback. Use `local` only on the same LAN.

### Which path am I actually on?

At connection time Neolink now logs a `MEDIA PATH:` line, e.g.:

```
MEDIA PATH: direct P2P to camera public/NAT-mapped address 37.x.x.x:52381 (dmap hole-punch, NOT via Reolink relay)
```

or, if it fell back to forwarding:

```
MEDIA PATH: routed THROUGH Reolink relay servers at <addr> (NOT direct P2P)
```

This disambiguates the relay-assisted path, whose P2P success and relay-server
fallback used to look identical in the logs.

## Bandwidth findings

- **Direct P2P (local / remote / map, and relay's P2P branch) carries the full
  camera bitrate.** A 4 MBit/s 4K main stream streams in real time over P2P.
- The throttle that previously capped *even direct P2P* at ~340 kbps was **not**
  inherent to any discovery method, nor a Reolink relay-server limit — it was a
  bug in the latency value Neolink reported to the camera (see below). It is
  fixed in 0.6.4-beta.12.
- The **relay-server fallback** (`relay()` — only used when the P2P punch fails)
  forwards your media through Reolink's infrastructure. Its throughput depends on
  those servers and is outside Neolink's control; prefer P2P where possible.

### The latency-feedback throttle (fixed in 0.6.4-beta.12)

Every UDP ACK Neolink sends the camera carries a `maybe_latency` field that feeds
the camera's adaptive-bitrate / congestion control. Neolink had been computing it
from the inter-arrival gap between the camera's own ACKs (essentially "how often
the camera acks our near-idle uplink"), not real link latency — yielding a steady
~21 ms that the camera interpreted as a poor link and responded to by downshifting
to a ~340 kbps floor. Neolink now reports `0` ("healthy link"), and the camera
serves full bitrate. Real packet loss is still recovered by the reliable-UDP layer
(retransmit + gap-skip), so this is safe on healthy links. On a genuinely lossy
link a future refinement could report a real congestion signal instead of a
constant `0`.

## Diagnostics

Enable the per-second UDP transport heartbeat:

```bash
RUST_LOG='info,neolink_core::bc_protocol::connection::udpsource=debug'
```

```
UDP HB: in_pkts=410 in_kbps=4086 delivered=410 resends=0 packets_want=1702 \
        sent_unacked=0 recieved_pending=0 ack_latency_us=22776 since_delivery=21ms win=1.00s
```

| Field | Meaning |
|---|---|
| `in_pkts` / `in_kbps` | UDP payload packets / throughput arriving from the camera this second. |
| `delivered` | Packets handed up to the demuxer this second. |
| `resends` | Retransmits Neolink issued (unacked sent packets). |
| `packets_want` | Next contiguous packet id awaited. Frozen = stalled awaiting a packet. |
| `sent_unacked` | Our outbound packets the camera hasn't acked. |
| `recieved_pending` | Packets buffered ahead of a gap (non-zero ⇒ a missing packet is blocking delivery). |
| `ack_latency_us` | The (legacy) computed latency value — **logged only**; Neolink now sends `0` to the camera. |

`udp_gap_skip_ms` controls how long the reassembler waits for a missing packet
before skipping it (default 120 ms; raise it on lossy links to favour completeness
over latency).
