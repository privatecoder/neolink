# Neolink documentation

Technical reference for how Neolink communicates with Reolink "Baichuan" (BC) P2P
cameras and serves them over RTSP. These documents describe how the system actually
works (grounded in the source), organised one topic per file.

## End-to-end picture

```
config ─▶ NeoReactor ─▶ NeoCam ──(on demand)──▶ discovery/lookup ─▶ P2P/relay connect
                                                      │
                                                      ▼
                                          BC control channel (login, encryption,
                                          stream requests, PTZ, talk, …)
                                                      │
                                       ┌──────────────┴───────────────┐
                                       ▼                              ▼
                              media substream (BcMedia)        two-way audio (talk)
                                       │
                                       ▼
                              RTSP factory (timestamping) ─▶ RTSP clients
```

1. **Ownership & lifecycle** — the reactor lazily creates a per-camera actor; a
   permit system connects on demand and disconnects when idle.
2. **Discovery** — a UID is resolved via Reolink lookup servers (region-selectable)
   and a connection is negotiated over LAN, P2P hole-punch, or relay.
3. **BC control protocol** — an encrypted, framed request/reply channel.
4. **Media** — video/audio arrive as the BcMedia substream and are re-timestamped for
   RTSP; audio can also be sent back to the camera (talk).
5. **Transport & bandwidth** — a reliable-UDP layer with CUBIC flow control governs
   stream bitrate.

## Documents

| Topic | File |
|---|---|
| Camera-ownership model (reactor / actors / permits) | [architecture.md](architecture.md) |
| Connection modes & lifecycle (`connect_mode`, reconnect, backoff) | [connection-modes.md](connection-modes.md) |
| Connection, discovery methods, regions, bandwidth, UDP flow control, diagnostics | [connection-and-bandwidth.md](connection-and-bandwidth.md) |
| UDP discovery / P2P negotiation handshake (message catalog & sequences) | [discovery-handshake.md](discovery-handshake.md) |
| BC control protocol (framing, modern/legacy, binary mode, encryption, message IDs) | [bc-protocol.md](bc-protocol.md) |
| Media substream format (BcMedia), codecs, stream kinds | [media-streams.md](media-streams.md) |
| A/V timestamping on the RTSP output | [av-timestamping.md](av-timestamping.md) |
| Two-way audio (talk) | [two-way-audio.md](two-way-audio.md) |
| Reolink SongP2P / RDT — empirical notes | [reolink.md](reolink.md) |
| Home Assistant integration (go2rtc, WebRTC/MSE/HLS, H264/H265, viewing cards) | [home-assistant.md](home-assistant.md) |

Per-camera configuration options and their defaults are documented in the project
`README.md` ("Camera Configuration Reference").
