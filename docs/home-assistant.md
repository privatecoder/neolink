# Neolink with Home Assistant

How Neolink fits into a Home Assistant (HA) viewing setup, and — more importantly —
how the browser-side playback stack (go2rtc, WebRTC, MSE, HLS) interacts with the
codecs Reolink cameras produce. Most "the stream takes a long time to open" or
"no audio" reports are decided here, not inside Neolink: Neolink hands HA a
standards-compliant RTSP stream; what happens after that is the card/go2rtc/browser
negotiation.

This document is generic. Replace `front-door` with your own camera name and
`homeassistant.local` with your host throughout.

## What Neolink exposes

- **RTSP** at `rtsp://<host>:<port>/<camera>/<stream>`. Each camera publishes its
  enabled streams under case-insensitive aliases:
  - Main: `/<camera>/main` (also `/Main`, `/mainStream`, `/MainStream`, …)
  - Sub: `/<camera>/sub` (also `/Sub`, `/subStream`, …)
  - A bare `/<camera>` path maps to the camera's stream when only one is enabled.
- **MQTT** (optional) for status/commands and HA discovery.

**Port note:** the Home Assistant add-on defaults RTSP to **8558**, not the upstream
default 8554, because HA's built-in go2rtc already listens on 8554. If you point a
client at 8554 you'll reach HA's go2rtc instead of Neolink and typically see a
*"wrong response on DESCRIBE"* error. Use the port Neolink actually bound.

## The Home Assistant viewing path

A browser cannot play RTSP directly. Something has to repackage the RTSP stream into
a transport the browser understands. In a typical HA setup that "something" is
**go2rtc**, which most live-view cards use under the hood:

```
Neolink (RTSP)  ─▶  go2rtc  ─▶  WebRTC / MSE / HLS / MJPEG  ─▶  browser
```

Common front-ends, all of which ultimately pull RTSP and go through go2rtc:

| Front-end | Notes |
|---|---|
| **WebRTC Camera** (AlexxIT) integration + card | Bundles go2rtc; can take an RTSP `url:` directly. |
| **Advanced Camera Card** | A card UI; with `live_provider: webrtc-card` it embeds the AlexxIT WebRTC card (so, go2rtc again). |
| **HA `go2rtc`** (built-in) / standalone go2rtc add-on | Define a named stream once, reference it from many cards. |
| **Generic Camera** integration | Can use the RTSP/`stream` component → HLS; simplest, higher latency. |

## Codecs vs. browser transports — the crux

Reolink main streams are commonly **H265 (HEVC)** video with **AAC** audio; sub
streams are often **H264**. The transport the card ends up using determines whether
those codecs play:

| Transport | H264 video | H265 video | AAC audio |
|---|---|---|---|
| **WebRTC** | ✅ everywhere | ⚠️ Safari/Apple only (Chrome HEVC-over-WebRTC is very limited) | ❌ not a WebRTC codec — go2rtc transcodes to Opus or drops audio |
| **MSE** (Media Source Extensions) | ✅ | ✅ in Chromium/Edge | ✅ |
| **HLS** (HA `stream` / LL-HLS) | ✅ | ✅ (fMP4) | ✅ |
| **MJPEG** | n/a (re-encoded) | n/a | ❌ no audio; CPU-heavy fallback |

Two practical takeaways:

- **H265 over WebRTC usually fails** outside Safari. go2rtc does **not** transcode
  video by default (that would need an ffmpeg source), so an H265 main stream offered
  to a WebRTC-first card typically can't render via WebRTC and must fall back to MSE.
- **AAC isn't carried by WebRTC.** Even with H264 video, a WebRTC session will drop
  or transcode the audio. If you want AAC to "just work," prefer MSE/HLS.

## Why the first open can be slow (the fallback cascade)

The AlexxIT WebRTC card (used directly, or via Advanced Camera Card's
`live_provider: webrtc-card`) tries playback modes **in priority order** — by default
`webrtc → mse → mp4 → mjpeg` — opening a fresh go2rtc connection for each until one
renders. go2rtc in turn opens a new RTSP pull to Neolink per attempt.

With an **H265 + AAC** main stream the usual sequence is:

1. **WebRTC** is tried first → fails to render H265 in a non-Safari browser → torn
   down after a second or two.
2. The card falls back to **MSE** → H265 plays in Chromium → this session sticks.

In Neolink's log this looks like several `RTSP client connected` events, only some of
which reach `Factory received new client … setting up pipeline`, each lasting ~1–2 s
and dropping, until one survives. Each `RTSP PLAY reached after <microseconds>` line
shows Neolink linked the pipeline essentially instantly — the wall-clock delay is the
card cycling modes and go2rtc renegotiating, not Neolink stalling. The same applies if
the stream is on-demand (`connect_mode: on_demand`): every probe re-triggers a relay
setup/teardown, adding to the churn.

## Recommended configurations

### Pin the playback mode (biggest win for H265)

Stop the card from trying WebRTC first for an H265 source. In the WebRTC card config,
set the `mode` order so it goes straight to MSE:

```yaml
# Advanced Camera Card example
cameras:
  - live_provider: webrtc-card
    webrtc_card:
      url: rtsp://homeassistant.local:8558/front-door/main
      mode: mse            # skip the webrtc→fail→fallback cascade for H265
    id: front-door
    title: Front Door
```

For the AlexxIT card used on its own:

```yaml
type: custom:webrtc-camera
url: rtsp://homeassistant.local:8558/front-door/main
mode: mse                  # or "webrtc,mse" on Safari, where WebRTC H265 works
```

Browser guidance: **Chrome/Edge** → `mse` (or `mse,webrtc`). **Safari** →
`webrtc,mse` is fine (Safari can do H265 over WebRTC and falls back to MSE).

### Use the H264 sub stream for live view

If the camera has a sub stream (usually H264), point the live view at it:

```
rtsp://homeassistant.local:8558/front-door/sub
```

H264 plays over WebRTC everywhere, so the fast path succeeds with no fallback. A
common pattern is **sub for live tiles, main for recording / fullscreen**. The sub
stream is also far lighter to decode for dashboards with many cameras.

### Define the stream once in go2rtc

Declaring the source in go2rtc (e.g. `/config/go2rtc.yaml`, or the go2rtc add-on
config) lets go2rtc hold a single RTSP connection and fan it out to multiple
consumers, instead of each card load opening its own pull:

```yaml
streams:
  front-door: rtsp://homeassistant.local:8558/front-door/main
  front-door-sub: rtsp://homeassistant.local:8558/front-door/sub
```

Cards then reference `front-door` by name. This reduces connection churn against
Neolink and gives you one place to add ffmpeg transcoding later if you ever do want
H265→H264 for WebRTC.

## Neolink-side options

These don't change the browser-side codec story, but they smooth the experience:

- **`connect_mode: always`** keeps the camera connection warm so a card's probe/
  fallback cycle doesn't repeatedly re-trigger on-demand relay setup. Trade-off: a
  constant camera connection and its bandwidth even when nobody is watching. See
  [connection-modes.md](connection-modes.md).
- **Stream selection** (`stream: Main | Sub | Both | All | …`): enable `Both` so the
  H264 sub path exists for live view alongside the H265 main. See
  [media-streams.md](media-streams.md).
- **RTSP port**: pick a free port (the add-on uses 8558) and use that exact port in
  every URL, including go2rtc's.

## I-frame interval (GOP)

The **I-frame (keyframe) interval** — sometimes labelled "interframe" or GOP in the
Reolink app — is a **camera-side encoding setting**. Neolink streams whatever the
camera produces and does not change it, so configure it per stream in the Reolink
app/web UI. Typical defaults are a keyframe roughly every ~2 s on the main stream and
~4 s on the sub. **Lowering it means more frequent keyframes.**

Why more frequent keyframes help the live-view experience:

- **Faster first picture on connect.** A decoder (and MSE/WebRTC) can only begin
  rendering at a keyframe, so a freshly-connecting client shows nothing until the next
  one arrives — up to a full interval of black. With a 2 s interval that's up to ~2 s
  per attempt, and since opening a card often makes several connection attempts (see
  the fallback cascade above), the wait repeats each time. A shorter interval gives a
  quicker first frame.
- **Faster recovery after packet loss.** On a lossy remote/relay link a damaged group
  of frames shows artifacts or freezes until the next keyframe. More frequent
  keyframes shorten that visible glitch. It also helps Neolink's backpressure
  catch-up, which drops predicted frames *until the next keyframe* — more keyframes
  means a clean picture resumes sooner.

The cost:

- **More bandwidth.** Keyframes are far larger than predicted frames (a 4K main-stream
  keyframe can be ~240 KB vs ~17–27 KB for a normal frame), so sending them more often
  is a real increase. Over P2P-relay/remote links the bigger, more frequent bursts
  also mean more reassembly pressure and more opportunity for loss.
- **Lower quality under a bitrate cap.** If the stream is bitrate-capped, more of the
  budget is spent on keyframes, reducing per-frame quality; uncapped, the bitrate
  simply rises.

Practical guidance:

- **Main (4K / H265):** usually leave it. Its keyframes are already the heavy part, so
  making them more frequent noticeably raises bandwidth and backpressure on anything
  but a clean LAN — for a startup saving of ~1 s.
- **Sub (low-res / H264):** a good candidate to shorten (e.g. ~1–2 s). Keyframes are
  cheap here, so if you use the sub for live tiles you get snappier opens and faster
  loss recovery at negligible cost.

This only affects the per-attempt "wait for the first keyframe"; it does **not** fix
the WebRTC→MSE fallback delay described above — pin the playback mode for that. If
fast startup is the goal, a persistent go2rtc stream helps more, since go2rtc can hold
a recent keyframe and hand it to new consumers immediately.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Live view takes ~10–20 s to appear, then works | WebRTC tried first on an H265 stream → fallback cascade | Pin `mode: mse`, or use the H264 sub stream |
| Black/no video in some browsers, fine in Safari | H265 over WebRTC unsupported off Apple | `mode: mse` (Chromium) or use sub (H264) |
| Video but no audio over WebRTC | AAC isn't a WebRTC codec | Use MSE/HLS, or let go2rtc transcode audio to Opus |
| *"Wrong response on DESCRIBE"* | Client hit port **8554** = HA's built-in go2rtc, not Neolink | Use Neolink's actual port (8558 in the add-on) |
| Repeated connect/disconnect in Neolink's log when opening a card | Card's mode-fallback + per-probe on-demand relay setup | Pin the mode; consider `connect_mode: always` or a go2rtc stream |
| Choppy/laggy 4K on a multi-camera dashboard | Decoding several H265 main streams | Use sub streams for the tiles |

None of the above indicate a Neolink fault on their own — a `RTSP PLAY reached after
<microseconds>` line means Neolink served the stream promptly. Look at which transport
the card settled on and the source codec first.
