# Neolink

[![dependency status](https://deps.rs/repo/github/privatecoder/neolink/status.svg)](https://deps.rs/repo/github/privatecoder/neolink)

Neolink is a small program that acts as a proxy between Reolink IP cameras and
normal RTSP clients.
Certain cameras, such as the Reolink B800, do not implement ONVIF or RTSP, but
instead use a proprietary "Baichuan" protocol only compatible with their apps
and NVRs (any camera that uses "port 9000" will likely be using this protocol).
Neolink allows you to use NVR software such as Shinobi or Blue Iris to receive
video from these cameras instead.
The Reolink NVR is not required, and the cameras are unmodified.
Your NVR software connects to Neolink, which forwards the video stream from the
camera.

The Neolink project is not affiliated with Reolink in any way; everything it
does has been reverse engineered.

## This Fork

This is a hardened fork of Neolink, with improvements focused on connection
stability and on-demand connection behavior. It builds on the work of the
original Neolink authors (credited in `LICENSE` and `Cargo.toml`).


## Installation

Download from the
[release page](https://github.com/privatecoder/neolink/releases)

## Config/Usage

### RTSP

To use `neolink` you need a config file.

There's a more complete example in this repo's
[`sample_config.toml`](sample_config.toml), and the full per-camera option list is
in the [Camera Configuration Reference](#camera-configuration-reference) below, but
the following should work as a minimal example.

```toml
bind = "0.0.0.0"

[[cameras]]
name = "Camera01"
username = "admin"
password = "password"
uid = "ABCDEF0123456789"

[[cameras]]
name = "Camera02"
username = "admin"
password = "password"
uid = "BCDEF0123456789A"
address = "192.168.1.10"
```

### Connection / Discovery Methods

Neolink reaches a camera over the proprietary Baichuan protocol, and there are
several ways to find and connect to it. The per-camera `discovery` option selects
which. Reolink's lookup servers (region picked by `relay_server_region`) return up
to three candidate addresses: the camera's **LAN** address (`dev`), its **public /
NAT-mapped** address (`dmap`, used for P2P hole-punching), and a **Reolink relay
server** (`relay`, which forwards traffic on your behalf).

```toml
[[cameras]]
  discovery = "relay"   # default
```

| Method   | What it does | Media path | Bandwidth |
|----------|--------------|------------|-----------|
| `local`  | LAN broadcast | **Direct P2P** on the LAN | Full. Same network only. |
| `remote` | Connect to the camera's `dev` address | **Direct P2P** | Full, if the `dev` address is routable from the host. |
| `map`    | Hole-punch to the camera's public (`dmap`) address | **Direct P2P** | Full, works through NAT. No relay fallback. |
| `relay`  | `dmap` hole-punch first, Reolink relay server as fallback | **Direct P2P** when the punch works; **relayed** only if it can't | Full on P2P; the relay-server fallback is forwarded through Reolink and may be slower / rate-limited. |
| `cellular` | `map` + `relay` | as above | as above |
| `debug`  | tries everything, first success wins | whichever wins | — |

Despite its name, **`relay` prefers a direct P2P connection** and only forwards
through Reolink's servers when peer-to-peer is genuinely impossible — so it is the
recommended default for remote cameras. Use `map` to force pure P2P (no fallback),
or `local` on the same LAN.

**Bandwidth:** direct P2P (`local` / `remote` / `map`, and `relay`'s P2P branch)
carries the **full camera bitrate** — a 4 MBit/s 4K main stream streams in real
time. Only the Reolink relay-server *fallback* is subject to Reolink's
infrastructure. (A bug in the latency value Neolink reported to the camera once
throttled *even direct P2P* to ~340 kbps; this is now fixed.)

Each connection logs which path it actually took:

```
MEDIA PATH: direct P2P to camera public/NAT-mapped address 37.x.x.x:52381 (dmap hole-punch, NOT via Reolink relay)
```

See [docs/connection-and-bandwidth.md](docs/connection-and-bandwidth.md) for the
full reference and the transport diagnostics.

### MQTT

To use mqtt you will need to adjust your config file as such:

```toml
bind = "0.0.0.0"

[mqtt]
broker_addr = "127.0.0.1" # Address of the mqtt server
port = 1883 # mqtt servers port
credentials = ["username", "password"] # mqtt server login details

[[cameras]]
name = "Camera01"
username = "admin"
password = "password"
uid = "ABCDEF0123456789"
```

Then to start the mqtt+rtsp connection run the following:

```bash
./neolink mqtt-rtsp --config=neolink.toml
```

OR for only mqtt

```bash
./neolink mqtt --config=neolink.toml
```

Neolink will publish these messages:

Messages that are prefixed with `neolink/`

- `/status` Tracks the connection of neolink, `connected` for ready `offline`
  for not ready this is a LastWill message
- `/config` The configuration file used to start neolink, you can publish to
  this to **temporarily** alter the live configuration
- `/config/status` If you publish to `/config` then any errors from your
  publish config will show here, or `Ok(())` if no errors and finished loading

Messages that are prefixed with `neolink/{CAMERANAME}`

Control messages:

- `/control/led [on|off]` Turns status LED on/off
- `/control/ir [on|off|auto]` Turn IR lights on/off or automatically via light
  detection
- `/control/reboot` Reboot the camera
- `/control/ptz [up|down|left|right|in|out] (amount)` Control the PTZ
  movements, amount defaults to 32.0
- `/control/ptz/preset [id]` Move the camera to a PTZ preset
- `/control/ptz/assign [id] [name]` Set the current PTZ position to a preset ID
  and name
- `/control/zoom (amount)` Zoom the camera to the specified amount. Example: 1.0
  for normal and 3.5 for 3.5x zoom factor. This only works on cameras that support
  zoom
- `/control/pir [on|off]`
- `/control/floodlight [on|off]` Turns floodlight (if equipped) on/off
- `/control/floodlight_tasks [on|off]` Turns floodlight (if equipped) tasks on/off
  This is the automatic tasks such as on motion and night triggers
- `/control/wakeup (mins)` Force a camera connection for at least the given minutes
- `/control/siren on` Signal the siren, the message is always "on" as there is no
  "off" signal for the siren

Status Messages:

- `/status disconnected` Sent when the camera goes offline
- `/status/battery` Sent in reply to a `/query/battery` an XML encoded version
  of the battery status
- `/status/battery_level` A simple % value of current battery level, only
  published when `enable_battery` is true in the config
- `/status/pir` Sent in reply to a `/query/pir` an XML encoded version of the
  pir status
- `/status/motion` Contains the motion detection alarm status. `on` for motion
  and `off` for still, only published when `enable_moton` is true in the config
- `/status/ptz/preset` Sent in reply to a `/query/ptz/preset` an XML encoded
  version of the PTZ presets
- `/status/preview` a base64 encoded camera image updated every 2s. Not
  every camera supports the snapshot command needed for this. In such cases
  there will be no `/status/preview` message. Only published when
  `enable_preview` is true in the config
- `/status/floodlight_tasks` The current status of the floodlight tasks
  used updated every 2s by default

Query Messages:

- `/query/battery` Request that the camera reports its battery level
- `/query/pir` Request that the camera reports its pir status
- `/query/ptz/preset` Request that the camera reports its PTZ presets
- `/query/preview` Request that the camera post a base64 encoded jpeg
  of the stream to `/status/preview` now, ignoring the timer

### Controlling RTSP from MQTT

If neolink is started with `mqtt-rtsp` then the `/neolink/config` can be used
to control the RTSP

Changes made to the config by publishing to `/neolink/config` should be
reflected in the rtsp

These include changing the:

- Available users

```toml
[[users]]
  name = "me"
  pass = "mepass"
```

- Permitted users on a camera

```toml
[[cameras]]
  permitted_users = [ "me" ]
```

- Available streams

```toml
[[cameras]]
  stream = "Main"
```

Setting a value of `None` will disable the stream

- Disable the entire camera (mqtt updates and all)

```toml
[[cameras]]
  enabled = false
```

### MQTT Disable Features

Certain features like preview and motion detection may not be desired
you can disable them with the following config options.
Disabling these may help to conserve battery

```toml
bind = "0.0.0.0"

[mqtt]
broker_addr = "127.0.0.1" # Address of the mqtt server
port = 1883 # mqtt servers port
credentials = ["username", "password"] # mqtt server login details

[[cameras]]
name = "Camera01"
username = "admin"
password = "password"
uid = "ABCDEF0123456789"
[cameras.mqtt]
enable_motion = false        # motion detection
                             # (limited battery drain since it
                             # is a passive listening connection)
                             #
enable_light = false         # flood lights only available on some camera
                             # (limited battery drain since it
                             # is a passive listening connection)
                             #
enable_battery = false       # battery updates in `/status/battery_level`
                             #
enable_preview = false       # preview image in `/status/preview`
                             #
enable_floodlight = false    # preview image in `/status/floodlight_tasks`
                             #
battery_update = 2000        # Number of ms between `/status/battery_level` updates
                             #
preview_update = 2000        # Number of ms between `/status/preview` updates
                             #
floodlight_update = 2000     # Number of ms between `/status/floodlight_tasks` updates
```

#### MQTT Discovery

[MQTT Discovery](https://www.home-assistant.io/integrations/mqtt/#mqtt-discovery)
is partially supported. Currently, discovery is opt-in and camera features
must be manually specified.

```toml
[cameras.mqtt]
  # <see above>
  [cameras.mqtt.discovery]
  topic = "homeassistant"
  features = ["floodlight"]
```

Available features are:

- `floodlight`: This adds a light control to home assistant
- `camera`: This adds a camera preview to home assistant. It is only updated
  every 0.5s and cannot be much more than that since it is updated over mqtt
  not over RTSP. Not every camera supports the snapshot command needed for
  this. In such cases there will be no `/status/preview` message.
- `led`: This adds a switch to chage the LED status light on/off to home
  assistant
- `ir`: This adds a selection switch to chage the IR light on/off/auto to home
  assistant
- `motion`: This adds a motion detection binary sensor to home assistant
- `reboot`: This adds a reboot button to home assistant
- `pt`: This adds a selection of buttons to control the pan and tilt of the
  camera
- `battery`: This adds a battery level sensor to home assistant
- `siren`: Adds a siren button to home assistant

### Extra Camera Settings

Listed below are extra camera settings:

```toml
[[cameras]]
name = "Camera01"
username = "admin"
password = "password"
uid = "ABCDEF0123456789"
debug = false # Displays Debug XML messages from camera
enabled = true # Enable or Disable the camera
update_time = false # When camera connects, force the setting of the camera date/time to now. The default is false
```

- **Debug:** Will dump the various XMLs from the camera as they are recieved
and decrypted. Leave this off unless asked for it to fix an issue.

- **Enabled:** Useful if you want to remove a camera from rtsp without deleting
it from the config

- **update_time:** Used to FORCE an update on the camera time. Usually it checks
if it is needed but this
will force it regardless. (Mostly this was introduced to address a specific
ssue a user had)

### Camera Configuration Reference

All per-camera options (under `[[cameras]]`), with defaults. Sub-tables
(`[cameras.mqtt]`, `[cameras.pause]`) are documented in their own sections.

| Option | Default | Description |
|---|---|---|
| `name` | *(required)* | Camera name; used in the RTSP path and logs. |
| `uid` | – | Camera UID (for relay/P2P discovery). One of `uid`/`address` is required. |
| `address` | – | Camera `ip[:port]` for direct/LAN connections. |
| `username` | *(required)* | Camera login user. |
| `password` (alias `pass`) | – | Camera login password. |
| `stream` | `All` | Streams to serve: `Main`, `Sub`, `Both`, `All`, `Extern`, `None`. |
| `channel_id` (alias `channel`) | `0` | Channel on an NVR; `0` for a standalone camera. |
| `permitted_users` | *(all)* | List of `[[users]]` names allowed to view this camera. |
| `discovery` | `relay` | How to find/connect the camera — see [Connection / Discovery Methods](#connection--discovery-methods). |
| `relay_server_region` (alias `relay_region`) | – | Reolink lookup region, e.g. `"Europe (France)"`. |
| `connect_mode` (alias `connect`) | `always` | `always` or `on_demand` — see [Connection Modes](#connection-modes). |
| `idle_timeout_secs` (alias `idle_timeout`) | `0` | In `always` mode: disconnect after N s idle (`0` = never). Ignored in `on_demand`. |
| `relay_warm_seconds` (alias `relay_warm`) | `60` | In `on_demand` mode: keep the connection warm N s after the last client (`0` = disconnect immediately). Ignored in `always`. |
| `udp_gap_skip_ms` | `120` | Reliable-UDP: how long to wait for a missing packet before skipping it. Raise on lossy links (favours completeness over latency); on a clean link it rarely triggers. |
| `buffer_duration` (aliases `buffer`, `duration`) | `3000` | Size of Neolink's internal video buffer, expressed as ms of stream. Larger absorbs network jitter/bursts (smoother, more latency); smaller = lower latency, less burst tolerance. |
| `max_encryption` | `Aes` | `none`, `bcencrypt`, or `aes`. |
| `strict` | `false` | Error the media stream on unexpected packets instead of tolerating them. |
| `max_discovery_retries` (alias `retries`) | `10` | Discovery attempts before giving up. |
| `update_time` (alias `time`) | `false` | Force-set the camera clock to "now" on connect. |
| `debug` (alias `verbose`) | `false` | Dump decrypted XML from the camera (noisy; troubleshooting only). |
| `enabled` (alias `enable`) | `true` | Set `false` to disable a camera without deleting it. |
| `use_splash` (alias `splash`) | `true` | Show the `splash_pattern` ("Stream not Ready") instead of a **404** in the brief window before the real stream factory is mounted (helps clients like Blue Iris that give up forever on a 404), and as a fallback if the video codec can't be determined. **Not** a live "connecting" placeholder — it does not play during a connect and transition to the real stream. |
| `splash_pattern` (alias `pattern`) | `Snow` | Splash look: `Snow`, `Smpte`, `Black`, `White`, `Red`, `Green`, … |

Global (top-level) options: `bind` (default `0.0.0.0`), the RTSP port (default
`8554`, or via `NEO_LINK_PORT` / the `--config` command), `[mqtt]` (broker), and
`[[users]]` (RTSP auth) — see the relevant sections above.

### Pause

To use the pause feature you will need to adjust your config file as such:

```toml
bind = "0.0.0.0"

[[cameras]]
name = "Camera01"
username = "admin"
password = "password"
uid = "ABCDEF0123456789"
  [cameras.pause]
  on_motion = true # Should pause when no motion
  on_client = true # Should pause when no rtsp client
  timeout = 2.1 # How long to wait after motion stops before pausing
```

Then start the rtsp server as usual:

```bash
./neolink rtsp --config=neolink.toml
```

### Connection Modes

Each camera chooses how it maintains its connection via `connect_mode`:

```toml
[[cameras]]
  connect_mode = "always"      # default: connect at startup and stay connected
  # connect_mode = "on_demand" # connect only when needed (best for battery cams)
```

**`always` (default)** — the camera connects at startup and stays connected,
reconnecting automatically if the link drops. Optionally set `idle_timeout_secs`
to drop the connection after a period with no active use (and reconnect on
demand). `0` (the default) means never disconnect:

```toml
[[cameras]]
  connect_mode = "always"
  idle_timeout_secs = 0     # 0 = stay connected forever (default)
  # idle_timeout_secs = 30  # disconnect after 30s idle, reconnect on demand
```

**`on_demand`** — the camera stays disconnected until something needs it, and
disconnects again when idle. Good for battery-powered cameras. Use
`relay_warm_seconds` to keep the connection warm briefly after the last client
leaves so a quick re-view reconnects instantly:

```toml
[[cameras]]
  connect_mode = "on_demand"
  relay_warm_seconds = 60   # linger 60s after last client (0 = disconnect immediately)
```

A camera connects on demand when:
- an RTSP client connects to view the stream
- an MQTT command is executed
- motion detection is active

You can monitor the lifecycle in the logs:
```
[INFO] Permit acquired, connecting to camera relay
[INFO] Camera relay established
[INFO] All permits dropped, disconnecting from camera relay
[INFO] Camera relay disconnected
```

> Note: a fully-disconnected `on_demand` camera cannot observe motion itself, and
> push-notification wake-ups are no longer available (Google removed the API the
> camera used). Wake such cameras with an external trigger (an RTSP client, an
> MQTT command, or `/control/wakeup`).

You can make neolink stop active streams when there are no rtsp clients using

```toml
[cameras.pause]
  on_client = true # Should pause when no rtsp client
```

Once in the disconnected state. Neolink will stay disconnected until there is a
new requested activation such as a client connecting or an mqtt command.

> Push-notification wake-ups (FCM) were removed — Google shut down the API they
> relied on — so a fully-disconnected camera can't be woken by motion on its own;
> use an external trigger (a client, an MQTT command, or `/control/wakeup`).

### Docker

[Docker](https://github.com/privatecoder/neolink/pkgs/container/neolink) builds
are published to GitHub Container Registry. The `latest` tag tracks the most
recent release; each pushed tag also gets its own image tag.

```bash
docker pull ghcr.io/privatecoder/neolink

# Add `-e "RUST_LOG=debug"` to run with debug logs
#
# --network host is only needed if you require to connect
# via local broadcasts. If you can connect via any other
# method then normal bridge mode should work fine
# and you can ommit this option. Not all OSes support
# network=host, notably macos lacks this option.
docker run --network host --volume=$PWD/config.toml:/etc/neolink.toml ghcr.io/privatecoder/neolink
```

#### Environmental Variables

There are currently 2 environmental variables available as part of the container:

- `NEO_LINK_MODE`: defaults to `"rtsp"` if not set, other options are "mqtt" or "mqtt-rtsp".
- `NEO_LINK_PORT`: defaults to `8554`, set this to your required port value.

### Image

You can write an image from the stream to disk using:

```bash
neolink image --config=config.toml --file-path=filepath CameraName
```

Where filepath is the path to save the image to and CameraName is the name of
the camera from the config to save the image from.

File is always jpeg and the extension given in filepath will be added or changed
to reflect this.

Some cameras do not support the SNAP command that is used to generate the image
on the camera. If this is the case with your camera you can try the
`--use-stream` option which will instead create a jpeg by transcoding the video
stream.

### Battery Levels

You can get the battery level and status using

```bash
neolink battery --config=config.toml CameraName
```

This will produce an xml formatted battery status on stdout for processing

### PIR

You can control pir using

```bash
neolink pir --config=config.toml CameraName [on|off]
```

This will turn the PIR on or off

### Reboot

You can reboot a camera using

```bash
neolink reboot --config=config.toml CameraName
```

### Status LED

You can control the status LED using

```bash
neolink status-light --config=config.toml CameraName [on|off]
```

### Talk

You can talk over the camera using

```bash
neolink talk --config=config.toml --adpcm-file=data.adpc\
               --sample-rate=16000 --block-size=512 CameraName
```

Where the sounds is ADPCM encoded

or

```bash
neolink talk --config=config.toml --microphone  CameraName
```

Which uses the default microphone which depends on
[gstreamer](https://gstreamer.freedesktop.org/documentation/autodetect/autoaudiosrc.html?gi-language=c#autoaudiosrc-page)

### PTZ

You can control the PTZ using

```bash
neolink ptz --config=config.toml CameraName control 32 [left|right|up|down|in|out]
```

Where 32 is the speed. Not all cameras support speed

Some cameras also support preset positions

```bash
# Print the list of preset positions
neolink ptz --config=config.toml CameraName preset
# Move the camera to preset ID 0
neolink ptz --config=config.toml CameraName preset 0
# Save the current position as preset ID 0 with name PresetName
neolink ptz --config=config.toml CameraName assign 0 PresetName
```

To change the zoom level use the following:

```bash
# Zoom the camera to 2.5x
neolink ptz --config=config.toml CameraName zoom 2.5
```

With 1.0 being normal and 2.5 being 2.5x zoom

## License

Neolink is free software, released under the GNU Affero General Public License
v3.

This means that if you incorporate it into a piece of software available over
the network, you must offer that software's source code to your users.

## Donations

If you find this code helpful please consider supporting development.

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/G2G5HOYIZ)
