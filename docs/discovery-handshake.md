# UDP discovery & P2P negotiation handshake

The UDP message exchange that turns a camera UID into a live media connection.
Implemented in `crates/core/src/bc_protocol/connection/discovery.rs`, with wire types
in `crates/core/src/bcudp/{model.rs,xml.rs}`.

For the higher-level `discovery` method/region selection and bandwidth, see
[connection-and-bandwidth.md](connection-and-bandwidth.md). For the reliable-UDP
data/ACK layer that runs *after* a connection is established, see the "UDP transport
flow control" section there.

## BcUdp packet types

All UDP packets start with a `u32` magic:

| Packet | Magic | Purpose |
|---|---|---|
| `Discovery` (`UdpDiscovery`) | `0x2a87cf3a` | Carries a `UdpXml` negotiation message. Header: magic, payload size, constant `0x00000001`, checksum; plus a `tid` (transmission id, also used as the XML encryption key). |
| `Ack` (`UdpAck`) | `0x2a87cf20` | Acknowledges received `Data`; see [connection-and-bandwidth.md](connection-and-bandwidth.md). |
| `Data` (`UdpData`) | `0x2a87cf10` | Carries (a fragment of) a BC control/media packet. |

For `Data`/`Ack`, `connection_id` is the peer id from negotiation: the **client id
(`cid`)** when received from the camera, the **device id (`did`)** when sent to it.

## Identifiers

| Id | Meaning | Origin |
|---|---|---|
| `cid` | Client id | Random `i32` generated once per discovery session. |
| `did` | Device (camera) id | Chosen by the camera; learned from `D2C_C_R`/`D2C_T`/`D2C_CFM`. |
| `sid` | Session id | Assigned by the Reolink register server (`R2C_C_R`); `0` for the LAN path. |
| `tid` | Per-message transmission id | Random `u8`; routes discovery replies and keys the XML. `tid = 0` subscribes to all replies. |

## UdpXml message catalog

All `UdpXml` messages are wrapped in a `<P2P>` element. Grouped by phase. (Some
variants exist in the type but are unused by the connection logic: `C2D_S`, `R2C_T`,
`C2R_HB`.)

**Lookup / registration (client ↔ Reolink middle-man servers)**

| Message | Direction | Key fields | Purpose |
|---|---|---|---|
| `C2M_Q` | client → middle-man | `uid`, `p` (os) | Query a UID (sent to a Reolink p2p server, port 9999). |
| `M2C_Q_R` | middle-man → client | `reg`, `relay`, `log`, `t` (each `ip:port`) | Reply with server locations (only `reg`/`relay` are used). |
| `C2R_C` | client → register | `uid`, `cli` (local ip:port), `relay`, `cid`, `family` (4=IPv4), `p`, `r` (rev) | Connect/registration request. |
| `R2C_C_R` | register → client | `dev`, `dmap`, `relay`, `relayt`, `nat`, `sid`, `rsp` (0 ok / -3 fail), `ac` | Reply with the camera's addresses and the session id. |
| `C2R_CFM` | client → register | `sid`, `conn`, `rsp`, `cid`, `did` | Confirm the established connection. |
| `R2C_DISC` | relay → client | `sid` | Relay terminated the session. |

**Client ↔ device negotiation**

| Message | Direction | Key fields | Purpose |
|---|---|---|---|
| `C2D_C` | client → device | `uid`, `cli` (port), `cid`, `mtu`, `p` | Direct connect for a known UID (LAN). |
| `D2C_C_R` | device → client | `timer`, `rsp`, `cid`, `did` | Reply to `C2D_C`. |
| `C2D_T` | client → device | `sid`, `conn` (`local`/`map`/`relay`), `cid`, `mtu` | Connect attempt over a path. |
| `D2C_T` | device → client | `sid`, `conn`, `cid`, `did` | Device offer. |
| `C2D_A` | client → device | `sid`, `conn`, `cid`, `did`, `mtu` | Accept a `D2C_T`. |
| `D2C_CFM` | device → client | `sid`, `conn`, `rsp`, `cid`, `did`, `time_r` | Connection confirmed. |

**Heartbeat / disconnect**

| Message | Direction | Purpose |
|---|---|---|
| `C2D_HB` | client → device | Keep-alive, sent every **1 s** once connected (`cid`, `did`). |
| `D2C_HB` | device → client | Camera-side keep-alive. |
| `C2D_DISC` / `D2C_DISC` | either | Disconnect (`D2C_DISC` → `CameraTerminate`). |

## Negotiation sequences

The `discovery` method (cumulative: `local` ⊂ `remote` ⊂ `map` ⊂ `relay`) decides
which paths are attempted; they race concurrently and the first success wins. Methods
other than `local` first perform the lookup/registration phase.

### Lookup / registration (remote, map, relay)

1. Cache check (5 min TTL); on hit, skip to step 3.
2. `C2M_Q → M2C_Q_R` against each relay host on port 9999 — yields the `reg` and
   `relay` server addresses.
3. `C2R_C → R2C_C_R` against `reg` — yields the camera's `dev` / `dmap` / `relay`
   addresses and the `sid`. `rsp == -1/-3` → registration error. The result is cached.

### Local (LAN)

`C2D_C → D2C_C_R` broadcast to ports 2015/2018; the reply with a matching `cid`
provides `did`. `sid = 0`. Start `C2D_HB`. Media flows direct over the LAN.

### Remote / dev

After registration, race two sub-paths against `reg_result.dev`:

- *client-initiated:* `C2D_T(conn=local) → D2C_CFM`, then `C2R_CFM(local)` to `reg`.
- *device-initiated:* wait for `D2C_T(conn=local)` → reply `C2D_A(local)` → `D2C_CFM`
  → `C2R_CFM(local)`.

### Map / dmap (P2P hole-punch)

Up to 5 punch attempts ~500 ms apart: `C2D_T(conn=map)` to `reg_result.dmap`; await
`D2C_T(conn=map)` from the dmap ip (port may differ); reply `C2D_A(map)`; await
`D2C_CFM(map)`; send `C2R_CFM(map)` to `reg`. Start `C2D_HB`. Media flows direct P2P.

### Relay (and relay-assisted P2P)

The `relay` method first tries the map hole-punch with a 3 s timeout; on success it
uses that direct path, otherwise it falls back to relaying:

`C2D_T(conn=relay) → D2C_CFM(conn=relay)` against `reg_result.relay`, then
`C2R_CFM(relay)` to `reg`. All traffic then routes through the Reolink relay server.

On success every method returns a `DiscoveryResult { socket, addr, camera_id (did),
client_id (cid) }`; from here the reliable-UDP data/ACK layer takes over.

## Timeouts, retries & constants

| Constant | Value |
|---|---|
| MTU (sent in `C2D_*`) | `1350` |
| UID-lookup cache TTL | `300 s` |
| Overall reply timeout (`MAXIMUM_WAIT`) | `15 s` → `DiscoveryTimeout` |
| TCP probe wait | `4 s` |
| Resend interval (`RESEND_WAIT`) | `500 ms` |
| Map hole-punch attempts | `5` |
| Relay-assisted map timeout before relay fallback | `3 s` |
| `C2R_CFM` send-and-forget repeats | `5` |
| Local UDP source port range | `53500–54000` (4 MiB send/recv buffers) |
| Registration retry backoff | `1,2,4,8,16,32 s` capped at 60 s, bounded by `max_discovery_retries` (`0` = infinite) |
| Relay hosts | `p2p.reolink.com`, `p2p1…p2p11.reolink.com`, port 9999 (see [connection-and-bandwidth.md](connection-and-bandwidth.md) → Regions) |

> The camera's `D2C_C_R.timer` (`def`/`hb`/`hbt`) values are received but not acted
> upon; the heartbeat cadence is the hard-coded 1 s.
