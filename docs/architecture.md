# Architecture: the camera-ownership model

How Neolink owns camera connections internally and lets many subsystems (RTSP,
MQTT, motion, one-shot CLI commands) share a single physical connection without
locking. Implemented in `src/common/` (`reactor.rs`, `neocam.rs`, `instance.rs`,
`usecounter.rs`, `camthread.rs`, `mdthread.rs`).

The design is a **channel-actor pattern**: each layer owns its state in a task and
is reached only over channels, handing out cheap `Clone`able handles.

## Layers

```
NeoReactor ──owns──> NeoCam (one per camera) ──hands out──> NeoInstance (cheap handle)
    │                    │                                        │
 watch<Config>      JoinSet of tasks                         run_task / run_passive_task
 HashMap<name,…>    (connect loop, camthread, md, …)         (talk to NeoCam over channels)
```

### NeoReactor — top-level owner

`NeoReactor` (`reactor.rs`) is `Clone` and holds only a `CancellationToken`, an mpsc
command sender, and the `JoinSet`. The camera map lives **inside** a single spawned
task; all access goes through an mpsc command channel (`HangUp`, `Config`,
`UpdateConfig`, `Get`), so the `HashMap<name, NeoCam>` is never touched directly.

- The live `Config` is held in a `watch` channel; subscribers get a
  `watch::Receiver<Config>`.
- **Cameras are created lazily.** `get(name)` uses `HashMap::entry`: an existing
  camera returns a new `NeoInstance` (subscribe); a missing one is constructed from
  the matching `CameraConfig` only if that entry exists and is `enabled`. An unknown
  name returns "Camera `<name>` not found in config".
- **`update_config` diffs** the new config: it drops cameras that are gone or
  disabled (`retain`), pushes the new `CameraConfig` to survivors, and replaces the
  config watch. New cameras are not eagerly created — they appear on the next `get`.

### NeoCam — owns one physical camera

`NeoCam` (`neocam.rs`) spawns a `JoinSet` of long-lived tasks and communicates
outward only by handing out `NeoInstance`s. It starts **disconnected** — the camera
connects when something first needs it (via the permit system). Tasks:

1. **Command dispatcher** — handles `Instance`, `Motion`, `Config`, `Connect`,
   `Disconnect`, `GetPermit`, `GetUid`. `Connect`/`Disconnect` flip a
   `watch<NeoCamThreadState>` (`Connected`/`Disconnected`) idempotently.
2. **`NeoCamThread`** (`camthread.rs`) — the actual login/connection loop (see
   [connection-modes.md](connection-modes.md)).
3. **`NeoCamMdThread`** (`mdthread.rs`) — the motion-detection event loop.
4. **Motion-permit watcher** — while motion is active it holds a permit (with a 30 s
   grace after motion stops), keeping the camera connected during events.
5. **On-demand connect loop** — branches on `connect_mode` to decide when to
   `connect()` / `disconnect()` (see [connection-modes.md](connection-modes.md)).

Startup info-queries (model/firmware/UID) were intentionally removed so an
`on_demand` camera truly never connects at boot.

### NeoInstance — the shared handle

`NeoInstance` (`instance.rs`) is a cheap `Clone` (a couple of channel endpoints + a
cancel token). It owns nothing; it talks to `NeoCam` over channels. Key helpers:

- **`run_task(f)`** runs `f(&BcCamera)` while **holding an active permit**, keeping
  the camera connected for the duration. Used by MQTT commands and one-shot CLI
  subcommands.
- **`run_passive_task(f)`** is the same but takes **no permit**, so the camera may
  disconnect for inactivity during the call. Streams and motion detection use this.
- Both auto-retry across reconnects: they watch a `watch<Weak<BcCamera>>` and, when
  the camera object is swapped (a reconnect publishes a new `Weak`), re-run `f` on the
  new object. Retryable errors (dropped/reset/timeout connections, and a transient
  `CameraServiceUnavailable{400}` retried up to 5×) trigger the re-run; fatal errors
  propagate.

### UseCounter / Permit — connection lifecycle

A `Permit` (`usecounter.rs`) is an **RAII token** over a `watch<u32>` use-count.
Taking a permit increments the count; dropping it decrements (a dropped permit
auto-releases). The on-demand connect loop waits on:

- `aquired_users()` — resolves when the count is `> 0` (something needs the camera).
- `dropped_users()` — resolves when the count is `0` (nothing needs it).

Permit holders: RTSP stream clients, MQTT commands and all one-shot CLI subcommands
(via `run_task`), and the motion-event watcher. The connect loop itself holds a
deactivated (watch-only) permit.

## Putting it together

A client connecting to an RTSP path → the factory takes a permit → the on-demand
loop sees `aquired_users()` → it flips the state watch to `Connected` →
`NeoCamThread` performs discovery + login and publishes the live `BcCamera` over the
`Weak` watch → in-flight `run_passive_task` stream loops pick up the new camera and
start pulling media. When the last client leaves, the permit drops →
`dropped_users()` → (after the warm/idle grace) the loop disconnects.

See also:

- [connection-modes.md](connection-modes.md) — the connect/disconnect rules and backoff.
- [connection-and-bandwidth.md](connection-and-bandwidth.md) — how the connection is
  actually established and how stream bitrate is governed.
