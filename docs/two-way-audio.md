# Two-way audio (talk)

Sending audio *to* a camera's speaker. Implemented in `src/talk/`
(`cmdline.rs`, `gst.rs`, `mod.rs`) and `crates/core/src/bc_protocol/talk.rs`.

The camera expects **ADPCM (DVI-4 / IMA), mono**. Neolink captures audio (mic or
file), runs it through a GStreamer pipeline that resamples and encodes to ADPCM, then
streams the ADPCM blocks to the camera as binary BC packets.

## CLI

```
neolink talk <camera> [--microphone | --file-path <path>] [options]
```

| Option | Default | Meaning |
|---|---|---|
| `<camera>` | — | camera name from the config |
| `-f, --file-path <path>` | — | audio file to send (conflicts with `--microphone`) |
| `-m, --microphone` | — | use a microphone source (conflicts with `--file-path`) |
| `-i, --input-src <str>` | `autoaudiosrc` | explicit source, e.g. `alsasrc device=hw:1` |
| `-v, --volume <f32>` | `1.0` | input volume |
| `--noise-suppression` | off | enable WebRTC noise suppression |
| `--noise-suppression-level <i32>` | `1` | noise-suppression aggressiveness |
| `--echo-cancel` | off | enable WebRTC echo cancellation (see caveat) |
| `--echo-suppression-level <i32>` | `2` | echo-suppression level |

You must pass either `--file-path` or `--microphone`.

## GStreamer pipeline

```
<source> ! decodebin [! webrtcdsp <flags>] ! audioconvert ! audioresample
  ! audio/x-raw,rate=<camera_rate>,channels=1
  ! volume volume=<volume>
  ! queue ! adpcmenc blockalign=<block_align> layout=dvi
  ! appsink (caps: audio/x-adpcm, layout=dvi, channels=1, rate=<camera_rate>)
```

- The raw intermediate format is mono PCM at the camera's sample rate; the final
  encode is **ADPCM, DVI layout** (`adpcmenc … layout=dvi`).
- `webrtcdsp` is inserted only when `--noise-suppression` or `--echo-cancel` is set.
  Each option maps directly to the element's properties
  (`noise-suppression` + `noise-suppression-level`, `echo-cancel` +
  `echo-suppression-level`); when disabled they are set to `false`.
- Encoded ADPCM is pulled from the appsink and handed to the BC send path.

## BC send path

1. **Query ability** — `MSG_ID_TALKABILITY` (10) returns `TalkAbility`
   (`duplex_list`, `audio_stream_mode_list`, `audio_config_list`). All three must be
   non-empty.
2. **Pick format** from `audio_config_list[0]`: `audio_type` must be `adpcm`
   (else error), `sample_rate` (e.g. 16000), and `length_per_encoder` →
   `block_size = length_per_encoder/2 + 4` (the GStreamer `blockalign`). The camera's
   reply is the source of truth for sample rate and block size.
3. **Audio stream mode** — the first mode of the first ability is used, defaulting to
   `followVideoStream` if the list is empty. (Cameras advertising several
   `audioStreamMode` entries are handled; only the first is sent.)
4. **Configure** — `MSG_ID_TALKCONFIG` (201) with the chosen `TalkConfig`. A `422`
   reply (another client already talking, or a stale prior session) triggers a
   `talk_stop()` and one retry, mirroring the official client; a final `200` is
   required.
5. **Stream** — audio is sent as `MSG_ID_TALK` (202) packets with `binaryData=1` and a
   `Binary` payload; each ADPCM block is serialised as a `BcMedia::Adpcm` frame. The
   sender paces to real time (`play_length = samples / sample_rate`), then after EOS
   waits for in-flight audio to finish (+100 ms) and issues
   `talk_stop()` → `MSG_ID_TALKRESET` (11).

## Caveats

- **ADPCM only.** Both the pipeline caps and two core-side checks hard-require ADPCM;
  there is no A-law path.
- **Echo cancellation is best-effort / effectively non-functional.** `--echo-cancel`
  sets the `webrtcdsp` property, but this is a capture-only pipeline (mic/file →
  camera) with **no `webrtcechoprobe`** providing a far-end reference signal, so there
  is nothing for it to cancel against. Noise suppression does not need a reference and
  works as expected.
- Only the first talk ability / audio stream mode is used; multi-ability cameras are
  defended against (empty-list checks) but not iterated.
