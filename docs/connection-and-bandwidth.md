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

### `maybe_latency` is the bitrate lever (receiver-rate feedback)

Every UDP ACK Neolink sends carries a `maybe_latency` field. Despite the name it is
**not latency** — it is the **receiver's measured throughput in bytes/second**, which
the camera's CUBIC rate-controller consumes as its bandwidth estimate. Getting this
field right is the whole difference between a stalling stream and a full-rate one,
and its history is the P2P-bitrate investigation in three acts:

- **Pre-`beta.12`:** Neolink computed it from the inter-arrival gap between the
  camera's ACKs (a steady ~21 ms), which the camera read as a near-zero delivery
  rate and pinned the stream to a **~340 kbps floor**.
- **`beta.12`:** set it to a constant `0`. That cleared the 340 kbps floor, but `0`
  gives the camera *no* rate estimate, so it free-runs a **conservative default** —
  full rate on some camera models, but on others (and on higher-RTT/remote paths)
  it capped a 4 Mbit/s main stream at **~2 Mbit/s** (real-time ratio ~0.4 →
  constant buffering), even though the official app pulled the full ~4.2 Mbit/s from
  the *same* camera over a single connection.
- **`beta.15` (fix):** Neolink now reports its **actual received bytes over the
  trailing ~1 s**, latched ~once per second (`ack_recv_rate` in the heartbeat). The
  camera's estimate tracks the real link and CUBIC ramps to full rate — a positive
  feedback loop (more received → higher reported rate → camera sends faster). An
  affected camera went from 1.9 Mbit/s / 0.38× to ~4.7 Mbit/s / 0.93×; an
  already-fine camera is unchanged (no regression).

Confirmed against a **packet capture of the official client**: its `maybe_latency`
equals its measured receive rate at ratio ≈ 1.0 across the session, and freezes at
the last 1-second sample when the stream stops (proving the ~1 Hz latch). The **unit
matters** — an interim attempt reported bytes-per-**100 ms** (10× too small), which
the camera read as a slow path and throttled; it must be bytes-per-**second**.

How the camera's rate-controller reads the field:

| reported `maybe_latency` | camera behaviour |
|---|---|
| `0` | no estimate → conservative default (full on some models, ~2 Mbit/s on others) |
| a small/fixed value | believes the path is that slow → paces down to it (~340 kbps) |
| **real measured bytes/s** | estimate tracks reality → ramps to full bitrate ✅ |

Other facts established about this ACK (official-client disassembly + capture), none
of which were the lever — recorded so they aren't re-investigated:

- **`group_id` + `packet_id` are one 64-bit cumulative-ACK sequence**
  (`full_seq = group_id·2³⁰ + packet_id`, base `0x40000000`;
  `0xffffffff/0xffffffff` = "nothing acked yet"). Neolink hardcodes `group_id=0`,
  correct only below ~1.07e9 packets (~31 days at 400 pps) — a latent correctness
  bug to fix separately.
- The **selective-ACK payload** is the same truth-map Neolink already sends (empty
  in steady state — no loss). The `0x2a87cf3a` block once suspected as an ACK
  extension is actually a **standalone encrypted auth/handshake** packet at session
  start, not flow control. **ACK cadence** (~12 ms in the official client) is not the
  lever either — event-driven per-packet ACKs changed nothing.
- Also ruled out by measurement: camera uplink ceiling (app gets 4.2), path (same on
  direct P2P *and* relay), a receiver window (no such field on this wire),
  reordering/loss (`reorder_events=0`, in-order, lossless).

`UdpAck` wire format (`crates/core/src/bcudp/{ser,de}.rs`):

```
magic 0x2a87cf20 | connection_id i32 | unknown_a u32 (0) |
group_id u32 (hi 30 bits of seq) | packet_id u32 (lo 30 bits of seq) |
maybe_latency u32 (receiver bytes/sec) | payload_size u32 | payload (selective-ACK bitmap)
```

## Diagnostics

Enable the per-second UDP transport heartbeat:

```bash
RUST_LOG='info,neolink_core::bc_protocol::connection::udpsource=debug'
```

```
UDP HB: in_pkts=485 in_kbps=4817 delivered=485 resends=0 packets_want=10846 \
        sent_unacked=0 recieved_pending=0 reorder_events=0 max_reorder_depth=0 \
        max_pending=1 ack_recv_rate=606755 ack_latency_us=22776 since_delivery=21ms win=1.00s
```

| Field | Meaning |
|---|---|
| `in_pkts` / `in_kbps` | UDP payload packets / throughput arriving from the camera this second. |
| `delivered` | Packets handed up to the demuxer this second. |
| `resends` | Retransmits Neolink issued (unacked sent packets). |
| `packets_want` | Next contiguous packet id awaited. Frozen = stalled awaiting a packet. |
| `sent_unacked` | Our outbound packets the camera hasn't acked. |
| `recieved_pending` | Packets buffered ahead of a gap (non-zero ⇒ a missing packet is blocking delivery). |
| `reorder_events` | Packets that arrived ahead of the contiguous point this second (would force selective-ACK holes). `0` = perfectly in-order. |
| `max_reorder_depth` | Largest gap (in packets) between an out-of-order arrival and `packets_want`. |
| `max_pending` | Peak depth of the reassembly buffer this second. `1` = each packet delivered immediately (in-order). |
| `ack_recv_rate` | Bytes received in the trailing ~1 s — the value reported to the camera in `maybe_latency` (its bitrate ramps to track this). |
| `ack_latency_us` | The (legacy) computed latency value — **logged only**, NOT sent to the camera (`maybe_latency` carries `ack_recv_rate` instead). |

> **Reading throughput problems:** if `in_kbps` is stuck below the stream's
> configured bitrate while `resends=0`, `recieved_pending=0`, and `reorder_events=0`
> (clean, in-order, lossless), check that `ack_recv_rate` is tracking the real
> receive rate (and not stuck near 0) — a low/zero value tells the camera to throttle
> (see "`maybe_latency` is the bitrate lever" above).

## Tuning for jitter / loss

- **`udp_gap_skip_ms`** (default `120`) — how long the reassembler waits for a
  missing packet before skipping it. Raise on lossy links to favour completeness
  over latency; on a clean link it rarely triggers.
- **`buffer_duration`** (default `3000`, ms; aliases `buffer`, `duration`) — the
  size of Neolink's internal video buffer, expressed as ms of stream
  (`≈ bitrate/8 × buffer_duration`). Larger absorbs bursty/jittery delivery
  (smoother playback, more latency); smaller lowers latency at the cost of burst
  tolerance. This is the primary buffer-sizing knob (the removed `stream_tuning`
  settings only nudged the same buffer and have been dropped).

The full per-camera option list with defaults is in the README's
"Camera Configuration Reference".
