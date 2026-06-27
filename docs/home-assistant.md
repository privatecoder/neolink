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

Three practical takeaways:

- **H265 over WebRTC usually fails** outside Safari. go2rtc does **not** transcode
  video by default (that would need an ffmpeg source), so an H265 main stream offered
  to a WebRTC-first card typically can't render via WebRTC and must fall back to MSE.
- **AAC isn't carried by WebRTC.** Even with H264 video, a WebRTC session will drop
  or transcode the audio. If you want AAC to "just work," prefer MSE/HLS.
- **The third ("extern") stream is often the best WebRTC source.** On single-lens
  cameras it's the app's "Balanced" live quality — H264 like sub but at a higher
  resolution/bitrate (e.g. 896×512 vs sub's 640×360); you can pick it in live view but
  its encoder config never appears in the app's Stream settings (only main + sub do).
  Point go2rtc at `/<name>/externStream` (Neolink `stream = "Extern"` or `"All"`)
  for a broadly-playable H264 stream sharper than sub, without the H265 main's WebRTC
  limits. The slot is **model-dependent**, though: on dual-lens / tracking models it's
  the *second lens* rather than a balanced stream, and some cameras don't serve it (it
  falls back to sub) — so check what yours actually delivers. See the
  [Streams table in the README](../README.md#streams-main-extern-sub).

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

## Opening an offline camera / surviving a reboot

If a stream's codec has already been seen once (it's cached), Neolink can serve the
pipeline from cache and connect to the camera in the background — so a card opened
while the camera is offline, or a camera that reboots while a card is open, no longer
fails or times out. On a fresh Neolink process, before that stream has produced its
first real keyframe, Neolink waits briefly (`startup_keyframe_wait_secs`, default 5s)
before serving the cached RTSP pipeline; if the keyframe arrives in that window, the
session starts straight on live video instead of exposing a keepalive-only stream to
go2rtc. If the wait expires, or after a real keyframe has already been cached in
memory, Neolink serves the cached pipeline immediately and streams a low-rate
placeholder plus matching silent audio during any gap.

Once the camera has produced a real keyframe for that stream, Neolink caches that
camera keyframe and replays it as the video placeholder. That replayed frame carries
the camera's own H264/H265 parameter sets (SPS/PPS/VPS), so the SDP/caps the client
negotiated match the live stream exactly. A locally encoded black frame is only used
as the cold-start fallback before any real keyframe has been seen for that stream.
When the camera comes back the session **hands off to live video on its own, with no
reconnect**. In debug logs you'll see `keepalive: replaying cached camera … IDR` when
the cached camera frame is used, `…: keepalive: encoded …` only for the cold-start
fallback, and `…: first camera keyframe received; switching from keepalive to live` on
recovery.

The placeholder is held **indefinitely by design** — there is no internal time limit,
so an always-on wall-dashboard card stays connected through an arbitrarily long outage
(a multi-minute reboot, a camera powered off overnight) and recovers whenever the
camera returns. The session only ends when the client itself disconnects (the card is
closed, or the viewer/go2rtc gives up on its own).

One prerequisite: the codec/rate are only cached after a *successful* view, so the
**first** open of a camera after the add-on (re)starts must happen while the camera is
online. After that, offline opens and reboots recover automatically. The silent-audio
part needs an AAC stream; if it can't be built, the stream falls back to the previous
behaviour (video keepalive only, which some players still drop during a long outage).

## Live view connects then drops in a reconnect loop

**Symptom.** After Neolink (re)starts — or intermittently — a camera's live view never
holds: it connects and drops within a second, retries every few seconds, and the card
shows nothing. Neolink's log shows the session counter climbing fast, many
`Client disconnected during keepalive (3 vid / 3 aud pushed …)` lines, and sessions that
reach `switching from keepalive to live` then `Client disconnected` almost immediately.

**Why it happens.** Two things combine:

1. **A cold-start race.** A Reolink camera takes ~2–3 s to connect over P2P after Neolink
   starts. The cached RTSP fast path, by design, can answer a client *immediately* from
   the cached codec info and serve a low-rate keepalive placeholder while the camera
   connects in the background. But go2rtc (Home Assistant's RTSP/WebRTC bridge) waits only
   a few hundred milliseconds for a real keyframe — it receives ~3 keepalive frames, sees
   no real video, and **disconnects**.
2. **go2rtc reconnect hysteresis.** That first bail tips go2rtc into an aggressive ~5 s
   reconnect storm. Once storming, it reconnects *faster than a stable stream can
   establish* and drops even healthy sessions, so it doesn't self-heal — only a calm
   restart (or the fixes below) clears it. Neolink isn't closing those sessions; the
   teardown is client-driven (you'll see the gstreamer `push_buffer => FLUSHING` in a
   debug log). Each retry also makes Neolink spin up a fresh per-client camera video
   stream, adding churn.

When Neolink happens to win the race (the camera produces a keyframe before go2rtc gives
up), the same stream plays perfectly for minutes — which is why the failure looks
intermittent and "a restart sometimes fixes it."

**The fixes (use both):**

- **Neolink side — remove the trigger.** `startup_keyframe_wait_secs` (default **5 s**, on
  by default) makes the cached fast path, on the first open of a stream since Neolink
  started, **wait for the camera's first real keyframe before answering** — so go2rtc's
  first PLAY gets real video and never bails on keepalive-only media. The cost is that the
  first open per camera after a restart shows a brief "connecting" (up to ~5 s, usually
  the ~2–3 s P2P connect) instead of an instant placeholder; warm opens are unaffected.
  Set it to `0` to restore the old immediate-serve behaviour.
- **Home Assistant side — remove the amplifier.** Don't give each card a raw
  `rtsp://…:8558/…` URL (every card then makes its own impatient, no-backoff connection).
  Register the cameras **once** as named go2rtc streams so a single go2rtc owns one
  producer per camera and reconnects with sane backoff, then reference them by **name**:

  ```yaml
  # go2rtc config
  streams:
    front-door:  rtsp://homeassistant.local:8558/front-door/sub
    driveway:    rtsp://homeassistant.local:8558/driveway/sub
  ```

  then in the card use `url: front-door` (the stream name, not an `rtsp://` URL). See
  ["Define the stream once in go2rtc"](#define-the-stream-once-in-go2rtc) for the exact
  card wiring and the built-in-go2rtc caveat.

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

## Bitrate: CBR vs VBR

Like the GOP, the rate-control mode is a **camera-side encoding setting** — Neolink
streams whatever the camera produces and the stream-info reply it reads carries no
CBR/VBR flag, so Neolink neither sets nor logs which mode is in use. Configure it per
stream in the Reolink app/web UI. The choice mostly comes down to **where you watch**.

**VBR (variable)** allocates bits where they're needed — more during motion, fewer on
a static scene — with the configured bitrate acting as a ceiling/target max.

- *Pros:* better quality per average bitrate; much lower bandwidth/storage when the
  scene is still (most of the time for a fixed camera); detail holds up during motion.
- *Cons:* **bitrate spikes** during motion. On a constrained upload / P2P-relay /
  remote link those bursts can exceed the available bandwidth → packet loss → Neolink's
  catch-up/backpressure dropping engages (the `CATCHUP enter` path), causing stutter
  and freeze-until-keyframe.

**CBR (constant)** holds a steady bitrate regardless of scene.

- *Pros:* predictable bandwidth — easy to provision for remote/relay, smoother
  delivery, more consistent latency and fewer loss-induced glitches.
- *Cons:* wastes bits on static scenes (more storage), and during heavy motion it
  can't exceed the cap, so quality drops (more blockiness) exactly when you want detail.

Guidance:

| Situation | Pick |
|---|---|
| Main stream on **LAN** (or recording to an NVR) | **VBR** — best quality + storage efficiency; LAN absorbs the spikes |
| Viewed **remotely / over P2P-relay** on limited upload, with stutter/backpressure | **CBR**, or keep VBR but **lower the max-bitrate cap** |
| **Sub stream** for live tiles | Either; **CBR** gives a steady, predictable small stream across many tiles |

**Interaction with GOP.** VBR motion spikes and large I-frames are the same root
stressor on a relay link. Shortening the GOP (more keyframes) *and* running VBR makes
the bursts compound. So on a remote/relay camera, lean toward **CBR (or VBR with a firm
cap) + the default longer GOP**; on LAN, **VBR + whatever GOP you like** is fine. See
[connection-and-bandwidth.md](connection-and-bandwidth.md) for how Neolink's transport
and backpressure handle these bursts.

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
