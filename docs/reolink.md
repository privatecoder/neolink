# Reolink SongP2P / RDT — empirical notes

Protocol-level observations about Reolink's P2P and reliable-UDP transport that
underpin the connection and streaming behaviour. The detailed, field-level specs live
in the topic docs; this file captures the SDK/protocol facts and the things that were
established empirically (disassembly + packet capture).

Cross-references:

- Connection paths, regions, bandwidth, the receiver-rate ACK field
  (`recv_bytes_per_sec`, reverse-engineered as `maybe_latency`), the `UdpAck` wire
  format, and diagnostics → [connection-and-bandwidth.md](connection-and-bandwidth.md)
- The UDP discovery message catalog and negotiation sequences →
  [discovery-handshake.md](discovery-handshake.md)
- BC control framing and encryption → [bc-protocol.md](bc-protocol.md)
- The BcMedia substream format and frame characteristics →
  [media-streams.md](media-streams.md)

## SongP2P behaviour

- The SDK uses **SongP2P** (`BC_P2P_TYPE_SONG`). Direct, relay, and LAN paths are
  attempted in parallel; the established connection reports a connection type and the
  set of available path types.
- A stream can run over **direct P2P even when a relay path is available.** Empirically:
  blocking the relay server does not interrupt a direct-P2P stream; blocking the direct
  peer does. The `relay` discovery method therefore prefers a direct hole-punch and
  uses the relay only as a fallback.
- Address discovery uses a DMAP / relay-map phase against region-specific Reolink
  lookup servers, which return the camera's `dev` / `dmap` / `relay` candidates plus a
  region relay list.
- The client issues no encoder-control commands during normal streaming (no
  `E_BC_CMD_SET_ENC_PROFILE` / `E_BC_CMD_IFRAME_PREVIEW`); the encode profile is
  whatever the camera is configured for.

## Stream types — the third ("Balanced") stream

Reolink cameras (RGM-203-class and similar) encode **three** parallel streams, not two.
The official client's live-quality menu maps 1:1 to them:

| Live-quality label    | Wire `<streamType>` | Encoder-config element | SDK enum (FYI)           |
|-----------------------|---------------------|------------------------|--------------------------|
| Clear                 | `mainStream`        | `<mainStream>`         | `E_BC_STREAM_MAIN` = 0   |
| Balanced              | `externStream`      | `<thirdStream>`        | `E_BC_STREAM_EXTERN` = 4 |
| Fluent                | `subStream`         | `<subStream>`          | `E_BC_STREAM_SUB` = 1    |

- The third stream is selected on the wire with `<streamType>externStream</streamType>`
  in the `Preview` body of the video-start message (`MSG_ID_VIDEO`, 3) — the same
  mechanism used for main/sub. **Use `externStream` as the selector, not `thirdStream`.**
  `thirdStream` is only the element *name* in the `GetEnc` / `<Compression>` encoder-config
  response (msg 56); it is **not** a valid `<streamType>`. The firmware's stream-type
  string table contains exactly `mainStream`, `subStream`, `externStream`, `mobileStream`.
- `mobileStream` (`E_BC_STREAM_MOBILE` = 2) is a separate, mobile-optimized type present
  in the firmware string table but not exposed by the cameras tested here — a
  known-but-untested fourth type.
- The third stream is **Baichuan-only**: it is not published over Reolink's own RTSP
  service (which offers only main/sub), and its encoder settings are read-only (not
  editable in the camera's Stream settings page).
- Its profile is **per-camera** — read it from the `GetEnc` `<Compression><thirdStream>`
  block (`<frame>`, `<width>`/`<height>`, `<videoEncType>`) or let the decoder learn it
  from the stream; do not hardcode. On a verified RGM-203 it encodes ~896×512 H.264
  ~20 fps ~1.2 Mbit/s, between main (4K H.265) and sub (360p H.264).
- **Multi-lens caveat:** on dual-lens / tracking / telephoto models the client reuses
  `E_BC_STREAM_EXTERN` for the *second lens* rather than a "balanced" stream. So
  `externStream` is really "the third stream slot", whose meaning is model-dependent —
  treat it as a generic third stream, not necessarily "balanced".

In Neolink this is `StreamKind::Extern`: configure it per camera with `stream = "Extern"`
(or `"All"` = main + extern + sub) and reach it on the RTSP path `/<name>/externStream`
(alias `/<name>/extern`). See [media-streams.md](media-streams.md) for the per-stream
wire values and [bc-protocol.md](bc-protocol.md) for `GetEnc` (`<Compression>`).

## RDT / p2p_udt flow control

The camera is the reliable-transport **sender** running **CUBIC**. Its send rate is
governed by receiver feedback in the p2p_udt ACK (the same packet as Neolink's
`UdpAck`, magic `0x2a87cf20`).

- **The receiver-rate field at offset `0x14` is the bandwidth lever** — reverse-engineered
  as `maybe_latency` (and named `recv_bytes_per_sec` in the code), it is the receiver's
  measured throughput in **bytes per second**, not latency. The receiver reports its
  actual received bytes over a ~1-second window; this lets CUBIC ramp a high-bitrate
  stream to full rate. (Field-level detail and the camera-response table:
  [connection-and-bandwidth.md](connection-and-bandwidth.md) → "UDP transport flow
  control".)
- **Cumulative ACK** = `group_id · 2³⁰ + packet_id` (64-bit sequence). **Selective
  ACK** = the ACK payload bitmap, empty when lossless.
- There is **no receiver-window field** on this ACK wire.
  `RDT_Set_Max_Pending_ACK_Number` exists in the SDK but lives in a different module
  and is not present on the p2p_udt ACK.
- The `0x2a87cf3a` packet at session start is a standalone encrypted auth/handshake
  packet, not part of flow control.

## Practical implications

- Size reassembly limits for the main stream's ~240 KB I-frame bursts (~200 packets);
  see [media-streams.md](media-streams.md).
- Report measured received bytes per second in the ACK receiver-rate field
  (`recv_bytes_per_sec`, reverse-engineered as `maybe_latency`, latched ~1 Hz) so the
  camera ramps to full bitrate — it is the receiver's job to
  feed the sender a truthful delivery-rate estimate.
- Do not assume the relay path is in use; direct P2P may carry the stream even when a
  relay is available.
