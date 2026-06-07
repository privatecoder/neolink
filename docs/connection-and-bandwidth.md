# Connection, Discovery & Bandwidth

Technical reference for how Neolink reaches a camera, the `discovery` methods, the
UDP transport flow control that governs stream bitrate, and the transport
diagnostics.

## Connection establishment

Reolink cameras speak the proprietary Baichuan (BC) protocol. To reach a camera,
Neolink registers with Reolink's lookup servers (region selected by
`relay_server_region`) and obtains three candidate addresses:

| Address | Meaning |
|---|---|
| `dev`   | The camera's **LAN** address (e.g. `192.168.1.50`). |
| `dmap`  | The camera's **public / NAT-mapped** address registered with Reolink, used for P2P hole-punching. |
| `relay` | A **Reolink relay server** that forwards traffic between client and camera. |

The `discovery` setting selects which addresses Neolink attempts.

## `discovery` methods

Set per-camera (`discovery = "..."`). Default is `relay`.

| Method   | Attempts | Media path | Bandwidth |
|----------|----------|-----------|-----------|
| `local`  | LAN broadcast to the camera | Direct P2P on the LAN | Full (LAN speed); same network only. |
| `remote` | Connect to the camera's `dev` address | Direct P2P | Full, if `dev` is routable from the host (usually only on/near the LAN). |
| `map`    | Hole-punch to the camera's `dmap` (public) address | Direct P2P | Full; works through NAT. No fallback — errors if the punch fails. |
| `relay`  | `dmap` hole-punch first, then the Reolink relay server | Direct P2P when the punch succeeds; relayed only on failure | Full on the P2P path; the relay fallback is forwarded through Reolink and may be rate-limited. |
| `cellular` | `map` + `relay` | As above | As above. |
| `debug`  | `local`, `remote`, `map`, `relay` (first to succeed) | Whichever wins | — |

`relay` prefers a direct P2P hole-punch and only forwards through Reolink's servers
when P2P is not possible. `map` forces pure P2P with no relay fallback. `local`
works only on the same LAN.

The wire-level message exchange behind each method (the UDP `UdpXml` catalog and the
per-method negotiation sequence) is documented in
[discovery-handshake.md](discovery-handshake.md).

### Media-path logging

At connection time Neolink logs the selected media path:

```
MEDIA PATH: direct P2P to camera public/NAT-mapped address <addr> (dmap hole-punch, NOT via Reolink relay)
```

```
MEDIA PATH: routed THROUGH Reolink relay servers at <addr> (NOT direct P2P)
```

This distinguishes the relay-assisted method's direct-P2P success from its
relay-server fallback, which are otherwise indistinguishable in the logs.

## Regions (`relay_server_region`)

UID resolution — turning a camera UID into its `dev` / `dmap` / `relay` addresses —
is performed by querying Reolink's P2P lookup servers (`p2p.reolink.com`,
`p2p1.reolink.com` … `p2p11.reolink.com`).

`relay_server_region` (alias `relay_region`) selects a single lookup server to query
instead of all of them:

| `relay_server_region` value | Lookup server |
|---|---|
| `North America (East US)` | `p2p1.reolink.com` |
| `Europe (Germany)` | `p2p2.reolink.com` |
| `Asia (Hong Kong)` | `p2p3.reolink.com` |
| `Middle East` | `p2p6.reolink.com` |
| `Europe (France)` | `p2p7.reolink.com` |
| `Europe (Unite Kingdom)` | `p2p8.reolink.com` |
| `North America (West US)` | `p2p9.reolink.com` |

Matching is case-insensitive and whitespace-trimmed, but the string must otherwise
match exactly — note the literal spelling `Unite Kingdom`.

Behaviour:

- **Set and recognised** — only that region's server is queried. Set this to the
  region the camera is registered in: lookup resolves faster, and the relay address
  returned (used only if the connection falls back to relaying) is region-local.
- **Unset, or an unrecognised value** — all lookup servers are queried in parallel
  (logged as `Unknown relay region, using all relay servers`). This still works but
  resolves more slowly and may yield a distant relay server for the fallback path.

Related fixed parameters: the negotiated transport MTU is `1350`; resolved UID
lookups are cached for 300 s.

## Bandwidth characteristics

- Direct P2P (`local` / `remote` / `map`, and the P2P branch of `relay`) carries the
  full camera bitrate; a 4 Mbit/s 4K main stream streams in real time.
- The relay-server fallback (used only when the P2P punch fails) forwards media
  through Reolink's infrastructure; its throughput depends on those servers and is
  outside Neolink's control. Prefer P2P where possible.

Stream bitrate is governed by the UDP transport flow control below, not by the
discovery method itself.

## UDP transport flow control

The camera is the reliable-UDP **sender** and runs **CUBIC** congestion control.
Its send rate is governed by receiver feedback carried in every UDP ACK
(`UdpAck`, magic `0x2a87cf20`).

### `maybe_latency` (receiver-rate feedback)

Despite its name, the `maybe_latency` field is **not latency**. It is the
**receiver's measured throughput in bytes per second**, which the camera's CUBIC
controller consumes as its bandwidth estimate.

- Neolink reports its **actual received bytes over the trailing ~1 second**, latched
  once per second (surfaced as `ack_recv_rate` in the heartbeat).
- The unit is bytes per **second**. A value scaled to a shorter window (e.g.
  bytes per 100 ms) under-reports by that factor and causes the camera to throttle.
- The reference (official) client populates this field with its own measured receive
  rate; the value tracks the receive rate at a ratio of ≈ 1.0 and holds the last
  1-second sample when the stream stops.

Camera response to the reported value:

| reported `maybe_latency` | camera behaviour |
|---|---|
| `0` | no estimate → conservative default (full rate on some models, ~2 Mbit/s on others) |
| a small/fixed value | treats the path as that slow → paces down to it (~340 kbps for a steady ACK-interval-derived value) |
| real measured bytes/s | estimate tracks the link → ramps to the full configured bitrate |

Genuine packet loss is handled separately by the reliable-UDP layer (resend +
gap-skip); it is not signalled through this field.

### `UdpAck` wire format

`crates/core/src/bcudp/{ser,de}.rs`:

```
magic 0x2a87cf20 | connection_id i32 | unknown_a u32 (0) |
group_id u32 (hi 30 bits of seq) | packet_id u32 (lo 30 bits of seq) |
maybe_latency u32 (receiver bytes/sec) | payload_size u32 | payload (selective-ACK bitmap)
```

- **Cumulative ACK sequence:** `group_id` and `packet_id` form one 64-bit sequence,
  `full_seq = group_id · 2³⁰ + packet_id`, base `0x40000000`.
  `group_id = 0xffffffff, packet_id = 0xffffffff` means "nothing acked yet".
  Neolink sets `group_id = 0`, which is valid until the sequence exceeds 2³⁰ packets
  (~1.07e9, ~31 days at 400 pps).
- **Selective ACK:** the `payload` is a truth-map bitmap of received packets since
  the cumulative point. It is empty in steady state (no loss).
- The ACK cadence does not affect the camera's rate; per-packet (event-driven) and
  periodic ACKs behave the same.

A separate `0x2a87cf3a` packet appears at session start; it is a standalone
encrypted auth/handshake packet and is not part of ACK / flow control.

## Diagnostics

Enable the per-second UDP transport heartbeat (note: `trace!` is compiled out of
release builds via `release_max_level_debug`, so these use `debug!`):

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
| `resends` | Retransmits issued for unacked sent packets. |
| `packets_want` | Next contiguous packet id awaited. A frozen value means delivery is stalled awaiting a packet. |
| `sent_unacked` | Outbound packets the camera has not acked. |
| `recieved_pending` | Packets buffered ahead of a gap; non-zero means a missing packet is blocking delivery. |
| `reorder_events` | Packets that arrived ahead of the contiguous point this second. `0` = perfectly in-order. |
| `max_reorder_depth` | Largest gap (in packets) between an out-of-order arrival and `packets_want`. |
| `max_pending` | Peak depth of the reassembly buffer this second. `1` = each packet delivered immediately. |
| `ack_recv_rate` | Bytes received in the trailing ~1 s; the value sent to the camera in `maybe_latency`. |
| `ack_latency_us` | Legacy computed latency value, logged only — not sent to the camera. |

Reading the heartbeat:

- `in_kbps` below the configured bitrate while `resends=0`, `recieved_pending=0`, and
  `reorder_events=0` (clean, in-order, lossless) → a flow-control issue; check that
  `ack_recv_rate` tracks the real receive rate and is not near `0`.
- `packets_want` frozen with `in_pkts=0` → the camera has **stopped sending**
  (disconnected or idle), which is distinct from packet loss.
- `packets_want` advancing slowly with `recieved_pending>0` → a gap is blocking
  delivery; `udp_gap_skip_ms` bounds how long the reassembler waits before skipping.

## Tuning for jitter / loss

- **`udp_gap_skip_ms`** (default `120`) — how long the reassembler waits for a
  missing packet before skipping it. Raise on lossy links to favour completeness over
  latency; on a clean link it rarely triggers.
- **`buffer_duration`** (default `3000` ms; aliases `buffer`, `duration`) — size of
  the internal video buffer, expressed as ms of stream
  (`≈ bitrate/8 × buffer_duration`). Larger absorbs bursty/jittery delivery (smoother
  playback, higher latency); smaller lowers latency at the cost of burst tolerance.

The full per-camera option list with defaults is in the README's "Camera
Configuration Reference".
