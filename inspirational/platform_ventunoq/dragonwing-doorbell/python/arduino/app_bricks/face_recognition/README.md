# `arduino:face_recognition` — app-local custom brick (Ventuno Q / HTP v75)

A **face detection + 512-d ArcFace embedding** brick for Arduino App Lab on the
**Ventuno Q** (`arduino,monza` / QCS8300, HTP v75). It runs InsightFace
`buffalo_sc` (det_500m + w600k_mbf) on the NPU inside a companion container and
streams raw faces back to the app. It is the prototype-side analogue of the
production `NpuFaceEngine` in `doorbell.py`.

## Why it lives inside the app

This brick is **fully isolated to the app folder** — no platform files are
written, no base image is rebuilt — because of how App Lab's base image starts
an app:

```
run.sh:  cd /app && exec python /app/python/main.py
```

so `/app/python` is on `sys.path`. Placing the brick at
`python/arduino/app_bricks/face_recognition/` namespace-merges it with the stock
`arduino.app_bricks` package. That makes it both:

1. **importable** as `from arduino.app_bricks.face_recognition import FaceRecognition`, and
2. **discovered** by `arduino-bricks-list-modules` (which scans `/app/python` via
   `list_installed_packages_pkg_resources`), so `--provision-compose` copies its
   compose into `.cache/` and the daemon merges the runner service into the app's
   generated `app-compose.yaml`.

This satisfies the project's isolation constraint: the working IQ-8275
Ubuntu/QLI setup is untouched.

## The files, and the one that must not be missing

| File | Purpose |
|------|---------|
| `__init__.py` | the client API (`FaceRecognition`) — same surface as `NpuFaceEngine` |
| `brick_config.yaml` | manifest: `id: arduino:face_recognition`, ports, `supported_boards: [ventunoq]`, `require_container: true`, `mount_devices_into_container: true` |
| **`brick_compose.yaml`** | **base** runner service — its existence by this exact name is what flips `require_container` on |
| `brick_compose.ventunoq.yaml` | ventunoq override: forwards the HTP v75 device nodes + DSP skels |

> **CRITICAL gotcha (verified on the board):** `ArduinoBrick.require_container`
> is derived purely from the presence of a file named exactly
> `brick_compose.yaml`. A board-specific `brick_compose.ventunoq.yaml` **alone**
> does **not** flip it — `get_compose_file()` only checks the base name. If you
> ship only the ventunoq file, the runner service is silently never merged and
> the brick runs with no NPU container. **Ship both.** Keep the service name
> (`face-recognition-runner`) identical across both files — the client resolves
> the runner host from the first service name.

## Client ↔ runner contract

Mirrors the stock `video_object_detection` brick exactly — two channels:

| Channel | Transport | Direction | Payload |
|---------|-----------|-----------|---------|
| results | `ws://<runner>:4912` | runner → client | JSON `{"type":"hello"}` then `{"type":"faces","faces":[{bbox,kps,embedding,score}]}` |
| frames | `tcp://<runner>:5050` | client → runner | length-prefixed JPEG (`>I` u32 length + bytes) |

Face coordinates are in the app's **320×240 inference space** (`INF_W×INF_H`),
so every `draw_*` helper in `doorbell_core` maps them onto the 1920×1080 canvas
unchanged.

## Public API (drop-in for `NpuFaceEngine`)

```python
from arduino.app_bricks.face_recognition import FaceRecognition

eng = FaceRecognition(on_ready=lambda: print("ready"))
eng.start()
eng.submit(inf_frame_320x240_bgr)      # latest-wins
faces = eng.get_faces()                # [Face: .bbox .kps .embedding .det_score]
eng.fps_counter.fps()
eng.stop()
```

`recognize_all(faces, face_db)` from `doorbell_core` then annotates each face
with `.rec_name` / `.rec_sim` — identity, the 2-minute TTL store and QR-triggered
enrollment all stay in the app, exactly as in `doorbell.py`.

## Deliberate design split

The brick is **stateless**: detection + embeddings only. The identity database,
TTL expiry, QR authorization and the whole state machine live in the **app**
(reused verbatim from `doorbell_core`). This keeps the NPU runner a pure
inference service and leaves policy where it already is.

## Extra Python deps

The client needs `websocket-client`. Add it to the app's `python/requirements.txt`
(installed into the uv venv by `run.sh`):

```
websocket-client
pyyaml
```

## Status / to verify on board

- [ ] `arduino-bricks-list-modules --provision-compose` discovers the brick and
      emits `brick_compose.ventunoq.yaml` into `.cache/`.
- [ ] The runner image `face-recognition-runner:0.1.0` is built and reachable
      (task #43).
- [ ] `resolve_address()` on the service name is an identity on ventunoq; if the
      runner is only reachable via `HOST_IP`, set `FACE_RECOGNITION_HOST`.
- [ ] HTP v75 session actually comes up in the runner (not a CPU fallback).
