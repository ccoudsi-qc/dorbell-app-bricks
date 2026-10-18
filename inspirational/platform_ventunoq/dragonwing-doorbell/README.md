# Dragonwing Doorbell — Arduino App Lab app (Ventuno Q)

An autonomous **face-recognition doorbell** for the **Ventuno Q**
(`arduino,monza` / QCS8300, HTP v75), packaged as an **Arduino App Lab** example.
It runs InsightFace `buffalo_sc` on the NPU and shows the full Dragonwing UI in a
webview.

This is the **prototype-side** of a larger "prototype → production" story: the
same demo also runs as a bare-metal / containerized app on the IQ-8275. See the
[production repo](https://github.com/munoz0raul/dorbell) for that path and the
narrative that connects the two.

> **New Ventuno Q?** Start with [`../../../install-bricks.md`](../../../install-bricks.md)
> — it takes a fresh Ubuntu + App Lab board to a running doorbell step by step.

## What makes it an App Lab app

| Piece | Role |
|---|---|
| `app.yaml` | declares two bricks: `arduino:face_recognition` + `arduino:web_ui` |
| `python/arduino/app_bricks/face_recognition/` | an **app-local custom brick** — detection + 512-d embeddings on the NPU runner |
| `python/main.py` | the app loop: camera → brick → recognition/state machine → canvas → web_ui |
| `python/doorbell_core.py` | the UI + state machine, imported verbatim (hardware-agnostic) |
| `assets/index.html` | the webview front-end (renders the JPEG stream, forwards keys) |
| `run_applab.sh` | board launcher (see **Build & run**) |

`doorbell_core.py` is a **byte-for-byte snapshot** of the production
`doorbell.py`. It is imported, not forked — so UI/behavior stay in lockstep with
the production app. Only the two I/O ends change (below).

## Architecture — two I/O swaps, nothing else

```
 ┌─────────────────────────── app container (python-apps-base) ───────────────────────────┐
 │  main.py                                                                                 │
 │    camera ─► FaceRecognition.submit(frame) ──TCP :5050 JPEG──►┐                          │
 │             FaceRecognition.get_faces() ◄──WS :4912 JSON──────┤                          │
 │    recognize_all + state machine + draw_* (doorbell_core)     │                          │
 │    canvas ──JPEG/base64──► web_ui.send_message("frame") ──────┼──► webview :7000          │
 │    web_ui.on_message("key") ◄── keydown ──────────────────────┼──◄ webview               │
 └───────────────────────────────────────────────────────────────┼──────────────────────────┘
                                                                  │
 ┌──────────── face-recognition-runner container (HTP v75) ───────▼──────────┐
 │  det_500m + w600k_mbf on the NPU → faces {bbox, kps, embedding, score}     │
 └────────────────────────────────────────────────────────────────────────────┘
```

1. **Inference:** the in-process `NpuFaceEngine` becomes the
   `arduino:face_recognition` brick client, which streams frames to the NPU
   **[runner](../../../runner/README.md)** and returns faces. Its surface is
   identical (`submit`/`get_faces`/`fps_counter`/`start`/`stop`/`on_ready`), so
   the main loop is unchanged.
2. **Display:** `cv2.imshow()` / `cv2.waitKey()` become the composited 1920×1080
   canvas, JPEG-encoded and streamed into the **web_ui** webview; key events come
   back over the same Socket.IO channel.

## Layout

```
dragonwing-doorbell/
├── app.yaml                       # bricks: face_recognition + web_ui (hand-authored)
├── run_applab.sh                  # board launcher: private data dir + app start
├── assets/
│   ├── index.html                 # webview: renders the JPEG stream, forwards keys
│   └── libs/
│       └── socket.io.min.js       # Socket.IO 4.x client (web_ui serves no client lib)
└── python/
    ├── main.py                    # app loop (doorbell_core + web_ui I/O)
    ├── doorbell_core.py           # verbatim snapshot of production doorbell.py
    ├── requirements.txt           # psutil, websocket-client, pyyaml (cv2+numpy from base)
    └── arduino/app_bricks/face_recognition/
        ├── __init__.py            # the brick client (see its README)
        ├── brick_config.yaml
        ├── brick_compose.yaml           # base — flips require_container on
        ├── brick_compose.ventunoq.yaml  # HTP v75 device forwarding
        └── README.md
```

The NPU **runner** (the brick's companion container) lives at the repo root in
[`runner/`](../../../runner/README.md).

## Controls (in the webview)

Enrollment is normally **automatic**: show a LinkedIn QR to the camera → 10 s
authorize animation → ~30-frame capture inside the enrollment ring. The keys
below drive the same flows from the browser:

| Key | Action |
|-----|--------|
| `G` | authorize a guest (same flow as a LinkedIn QR) |
| `C` | clear the face database |
| `[` / `]` | decrease / increase the idle timeout (5–60 s) |
| `-` / `=` | decrease / increase the enrolled-face TTL (60–600 s) |
| `Esc` | back to Standby |
| `Q` | quit the app loop |

## Build & run on the board

> `app start` is **state-changing** — it stops any running app and rebuilds the
> compose. Coordinate before running it on a shared board.

### Why not a plain `arduino-app-cli app start`?

`app start` validates every brick in `app.yaml` against the **static** registry
`<data-dir>/assets/<ver>/bricks-list.yaml`. The `arduino:face_recognition` brick
lives **inside the app**, so it is absent from the stock registry and a plain
start fails with:

```
[ERROR] Brick "arduino:face_recognition" Not Found
```

The isolation-preserving fix (verified on-board 2026-10-18) is to launch the CLI
against a **private data dir** (`ARDUINO_APP_CLI__DATA_DIR`) whose registry =
stock + our brick, built **without editing any stock file**. `run_applab.sh`
automates exactly that.

### Launch

```bash
# On the Ventuno Q, with this app under ~/ArduinoApps/dragonwing-doorbell
# (or wherever App Lab keeps its apps):

# 1. Build the NPU runner image once (see ../../../runner/README.md), tagged:
#    ghcr.io/arduino/app-bricks/face-recognition-runner:0.1.0  (or set DOCKER_REGISTRY_BASE)

# 2. Build the private data dir + start the app (one command):
bash run_applab.sh

# 3. Open the webview (App Lab opens it automatically; or browse to):
#    http://<board>:7000
```

`run_applab.sh` builds the private data dir once (symlinks every stock asset,
then regenerates `bricks-list.yaml` + the `compose/` tree with the **same
provisioner the daemon uses**, so the app-local brick is included), then runs
`app start` under `ARDUINO_APP_CLI__DATA_DIR`. Flags:

| Command | Effect |
|---------|--------|
| `bash run_applab.sh` | build data dir (if needed) + `app start` |
| `bash run_applab.sh --rebuild` | force-refresh the data dir, then start |
| `bash run_applab.sh --setup-only` | build the data dir, don't start |
| `bash run_applab.sh --stop` | stop the running app |

To verify brick discovery on its own (what the launcher does internally):

```bash
docker run --rm --entrypoint arduino-bricks-list-modules \
  -v "$PWD":/app -v /tmp/prov:/out \
  ghcr.io/arduino/app-bricks/python-apps-base:0.12.0 --provision-compose -o /out
#  → expect arduino:face_recognition listed, require_container=True,
#    and /tmp/prov/compose/arduino/face_recognition/brick_compose.ventunoq.yaml emitted
```

## Status

**Running on the board (verified 2026-10-18, Ventuno Q "Monaco Monza addons").**
Both containers start under `run_applab.sh`; the NPU runner comes up on
`QNNExecutionProvider` (HTP v75, 0 `QNN_DEVICE_ERROR_INVALID_CONFIG`), web_ui
serves at `:7000`, and the canvas stream + key channel are live end-to-end. See
the brick README's checklist and [`../../../runner/README.md`](../../../runner/README.md).
