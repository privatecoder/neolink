# A/V Timestamping (RTSP output)

Technical reference for how the RTSP factory assigns presentation timestamps (PTS)
to video and audio, and how it keeps them in sync. Implemented in
`src/rtsp/factory.rs` (`send_to_sources`).

## Principle

PTS is derived from a **single media clock**, never from packet arrival time.

Media arrives from the camera in bursts. Audio in particular is constant-rate at
the source but is delivered bursty over the network, so timestamping by arrival time
makes the audio PTS race ahead of (or lag) video and destabilises the GStreamer
pipeline, producing periodic stalls. Both streams are therefore timed from one
content clock anchored to the camera's own video capture time.

## Video clock

- **Source:** the camera's per-frame capture timestamp (`microseconds`) carried on
  every `Iframe` / `Pframe` (see [media-streams.md](media-streams.md)).
- The value is a `u32` that **wraps roughly every 71 minutes**. The PTS is rebased by
  accumulating per-frame deltas with `wrapping_sub` (so a wrap is handled
  transparently) rather than subtracting an absolute base; the first frame is PTS 0.
- An implausibly large delta (> 10 s) is treated as a **camera clock reset**: the PTS
  is held and the reference re-synced, so a reset does not inject a huge jump.
- Helper: `video_ts_from_camera(micros) -> u64`.

## Audio clock

- Audio (`Aac` / `Adpcm`) carries **no camera timestamp**.
- It rides a **content clock**: each frame advances the audio PTS by its own decoded
  duration (AAC from `duration_info()`, ADPCM from `duration()`).
- The audio clock is **anchored once to the video clock**: on the first audio frame
  the audio PTS is set to the current video PTS (`aud_anchored`), then advances
  independently by per-frame duration.
- This keeps audio smooth (it is paced by content duration, not arrival) and aligned
  with video (it shares the video origin).

## Frame dropping and clock continuity

The pipeline drops frames under buffer pressure but never lets the clock skip:

- **P-frame priority drop:** when the video AppSrc buffer is filling, P-frames are
  dropped and I-frames preserved (an I-frame can restart decode; a dropped I-frame
  cannot). P-frames are dropped once the buffer reaches ~80 % of capacity in the send
  path; a separate backpressure guard drops at `0.85` of AppSrc capacity (`0.92` for
  high-bitrate streams).
- **The clock still advances on dropped frames.** The drop paths call
  `video_ts_from_camera` (via `video_microseconds`) so the video PTS stays continuous
  across gaps, which keeps the audio anchor valid and prevents PTS discontinuities.

## Diagnostics

The factory heartbeat (per second) reports sync health:

| Field | Meaning |
|---|---|
| `av_drift_ms` | `vid_ts − aud_ts` in ms; near `0` = in sync. Negative = audio ahead of video. |
| `vid_push` / `aud_push` | Frames pushed to the video / audio AppSrc this interval. |
| `dropP_pressure` / `dropP_backpressure` | P-frames dropped for buffer pressure / backpressure. |
| `vid_buf` | Current video AppSrc fill vs capacity. |
| `last_media` | Time since the last media frame (climbs when the camera goes silent). |

## Constraints

- Do **not** timestamp audio (or video) from arrival time; it reintroduces drift and
  pipeline stalls.
- Any frame-drop or catch-up path must keep advancing the video clock so it stays
  continuous.
