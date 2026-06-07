# Media substream (BcMedia)

The format the camera streams video and audio in, how a stream is requested, and the
frame characteristics that drive reassembly. Implemented in
`crates/core/src/bcmedia/` (`model.rs`, `de.rs`, `ser.rs`) and
`crates/core/src/bc_protocol/stream.rs`.

Media is delivered as `Binary` payloads inside BC control packets (see
[bc-protocol.md](bc-protocol.md)) over the negotiated transport (see
[discovery-handshake.md](discovery-handshake.md) and
[connection-and-bandwidth.md](connection-and-bandwidth.md)). RTSP-side timestamping of
the demuxed frames is covered in [av-timestamping.md](av-timestamping.md).

## Requesting a stream

`BcCamera::start_video(stream, buffer_size, strict)` opens a stream and returns a
handle; dropping it stops the stream. It sends `MSG_ID_VIDEO` (3) with a `Preview` XML
body and expects a `200` reply before media flows. Per stream kind it sets three
distinct values:

| `StreamKind` | wire name | `stream_type` (header) | `Preview.handle` |
|---|---|---|---|
| `Main` | `mainStream` | 0 | 0 |
| `Sub` | `subStream` | 1 | 256 |
| `Extern` | `externStream` | 0 | 1024 |

`Extern` is "between SD and HD" on cameras that support it, otherwise equivalent to
`Sub`. Stream selection is configured per camera via `stream` (`Main` | `Sub` |
`Both` | `All` | `Extern` | `None`); `All` = Main+Extern+Sub, `Both` = Main+Sub.

## Frame format

Each `BcMedia` frame begins with a little-endian `u32` magic that selects the type:

| Frame | Magic | Carries |
|---|---|---|
| `InfoV1` | `0x31303031` | resolution, fps, start/end timestamps |
| `InfoV2` | `0x32303031` | same fields as InfoV1 |
| `Iframe` | `0x63643030`–`0x63643039` | keyframe (magic low nibble encodes channel) |
| `Pframe` | `0x63643130`–`0x63643139` | predicted frame |
| `Aac` | `0x62773530` | AAC audio (ADTS) |
| `Adpcm` | `0x62773130` | ADPCM audio (DVI-4) |

**8-byte padding:** after a frame's payload, bytes are consumed to round the payload
up to an 8-byte boundary. Info frames have no payload/padding.

### Video frames (Iframe / Pframe)

After the magic: a 4-byte ASCII codec tag — `"H264"` or `"H265"` (→ `VideoType`) —
then `payload_size: u32`, `additional_header_size: u32`, `microseconds: u32` (capture
timestamp), an unknown `u32`, the additional header, the payload, and padding.

- **I-frames** additionally extract `time: Option<u32>` (POSIX seconds) from the
  additional header when `additional_header_size >= 4`; extra bytes beyond that are
  skipped. P-frames consume the additional header without extracting `time`.
- Only video frames carry a timestamp (`microseconds`). This is the single media clock
  the RTSP factory uses (see [av-timestamping.md](av-timestamping.md)).

### Audio frames (Aac / Adpcm)

Audio frames carry **no timestamp**.

- **AAC** is ADTS-framed; its struct holds only the data. Duration is computed by
  parsing the ADTS headers (sampling-frequency index → Hz, raw-block count × 1024
  samples). Header length is 7 or 9 bytes depending on CRC.
- **ADPCM** is DVI-4 / IMA: a 4-byte sub-header (a `0x0100` data magic + a half-block
  `u16`) precedes the data, which is "4 bytes of predictor state + one ADPCM block".
  Fixed 8000 Hz for duration purposes (`samples = block_size × 2`).

### Info frames

`InfoV1` / `InfoV2` (identical layout) carry `video_width`, `video_height`, `fps`
(an index into an FPS table on older cameras), and start/end timestamps split into
`u8` year/month/day/hour/min/seconds fields. The end timestamp is mainly relevant to
SD-card recordings.

## Frame-size characteristics & reassembly

Frame sizes drive reassembly-buffer requirements. The transport negotiates an MTU of
1350 (≈1.2–1.3 KB payload per packet):

**Sub stream** — mostly small frames (tens to a few hundred bytes) with periodic
~16–17 KB frames every ~15 frames. Large frames span ~14–15 packets; small frames are
a single packet. Easy to reassemble.

**Main stream** — typical frames ~17–20 KB (~14–18 packets), with periodic I-frame
bursts up to ~240 KB (~190–200 packets) roughly every ~30 frames; average ~27 KB. The
reassembler must accommodate ~240 KB / ~200-packet frames.

Representative 4K profile: Main `3840×2160 @ 20 fps, ~4096 kbps, GOP 2`; Sub
`640×360 @ 10 fps, ~256 kbps, GOP 4`.
