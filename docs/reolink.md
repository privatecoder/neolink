# Reolink P2P / Relay Findings

## Topline
- The SDK uses **SongP2P (BC_P2P_TYPE_SONG)** and attempts **direct + relay + LAN** in parallel.
- In practice, the stream can be **direct P2P** even when relay is available (blocking relay does not stop stream; blocking direct peer does).
- The relay-assisted map step appears to use a **DMAP / relay map** phase ("Q ok" and "2rly" entries), which lines up with the relay server IPs listed below.

## P2P / Relay Evidence
- Connection type: `connType=3`, `connRet=0` in most runs.
- P2P type: `BC_P2P_TYPE_SONG`, `p2pRet=0`.
- Summary shows all paths available:
  - `direct=yes relay=yes lan=yes connFdTypes=3,2`
- **Direct P2P** peer (external): `37.15.103.160:54392` (from `2p:` line).
- **LAN** peer: `192.168.1.10:54392` (from `2sub:` line).
- **Relay** peer: `13.38.26.165:<port>` (from `2rly:` line), with relay map / Q ok on `13.38.26.165:58200`.
- **Mapping / server list** includes multiple relay / map servers, example subset:
  - `119.23.230.154` (server from SongP2P debug)
  - `13.38.26.165` (relay/map)
  - `43.204.92.13`, `35.156.78.129`, `35.152.156.39`, `52.56.65.139`, `51.16.221.160`, `54.252.82.145`, `129.158.224.26`, `129.213.189.52`, `132.226.204.68`, `144.24.42.1`, `16.163.195.33`, `15.188.197.53`
- Debug shows `server=119.23.230.154` and `ver=2022.06.10`.

## Blocking Test Result (Direct vs Relay)
- Blocking relay server `13.38.26.165` **did not** stop the stream.
- Blocking direct peer `37.15.103.160` **did** stop the stream.
- Conclusion: stream was using **direct P2P**, with relay available as fallback.

## LiveOpen Path (Render vs Data)
- Probe mode (`--bcsdk-data-probe`) uses `BCSDK_LiveOpen2` (compressed/data frames) and then switches to `BCSDK_LiveOpen` (render) after the data probe finishes.
- Both main and sub follow the same path in probe mode.

## Stream Configuration (Encode Table)
- Main: `3840x2160`, `fps=20`, `bitrate=4096`, `gop=2`
- Sub: `640x360`, `fps=10`, `bitrate=256`, `gop=4`

## Compressed Frame Sizes (DATA_FRAME_DESC)

### Sub stream (100 frames)
- `avg=1236.2`, `min=20`, `max=16940`, `maxOverAvg=13.70`
- Pattern: periodic large frames (~16-17 KB) about every ~15 frames, many very small frames (20-229 B).
- Estimated packets (MTU=1200):
  - Large frames: ~14-15 packets
  - Small frames: 1 packet

### Main stream (100 frames)
- `avg=27221.6`, `min=16698`, `max=238005`, `maxOverAvg=8.74`
- Pattern: periodic very large frames (~226-238 KB) about every ~30 frames, most frames ~17-20 KB.
- Estimated packets (MTU=1200):
  - Large frames: ~189-199 packets
  - Typical frames: ~14-18 packets

## First Frame Properties (Observed)
- Sub first data frame: `640x360`, `frameRate=15`, `length~16-17 KB`, `extlen=0`.
- Main first data frame: `3840x2160`, `frameRate=15`, `length~238 KB`, `extlen=120`.
- Render frames show `format=0` for both streams.

## Encoder / IDR Requests
- No evidence of the app sending `E_BC_CMD_SET_ENC_PROFILE` or `E_BC_CMD_IFRAME_PREVIEW`.
- No native calls matching `SetEnc*` / `IFramePreview*` observed in logs.

## Transport / Buffer Hints
- No recv window / buffer / queue hints were surfaced by SDK logs during runs.
- Fragmentation fields in DATA_FRAME_DESC were not exposed (`fragFields=NA`).

## Practical Implications for Rust P2P Implementation
- **Main stream is bursty** with very large I-frame bursts (200-ish packets). Reassembly limits must accommodate ~240 KB frames.
- **Sub stream is mostly tiny** with periodic larger frames (~17 KB). Easier to reassemble.
- Ensure relay map / DMAP phase succeeds ("Q ok" to relay server), but do not assume relay path is used; direct P2P may carry the stream.
- Use SongP2P detail/debug to match relay servers and direct peer addresses.
