# Install the Dragonwing Doorbell on a fresh Ventuno Q

This takes a **fresh Ventuno Q** (`arduino,monza` / QCS8300, HTP v75) running
**Ubuntu** with **Arduino App Lab** to a running face-recognition doorbell. It
assumes nothing beyond a flashed board and network access.

> Verified on-board 2026-10-18 (Ventuno Q "Monaco Monza addons", Ubuntu 24.04,
> `arduino-app-cli` 0.13.0 / App Lab 0.10.0, Docker 29.1.3, assets version 0.12.0).

The end state: two containers (the `web_ui` webview + the NPU
**face-recognition-runner**) running under App Lab, the doorbell UI live at
`http://<board>:7000`, recognition on the **HTP v75** NPU.

---

## 0. Prerequisites

On the board you need, already installed by the Ubuntu + App Lab image:

- **Arduino App Lab** and `arduino-app-cli` on `PATH` (`arduino-app-cli version`).
- **Docker** running and usable by your user (`docker ps` works without sudo —
  if not, `sudo usermod -aG docker $USER` then re-login).
- The stock App Lab data dir at `/var/lib/arduino-app-cli/assets/<ver>/` (this is
  what the private-data-dir launcher symlinks against).
- A **USB webcam** plugged into the board — the app reads it directly (see
  "Camera" below). Without one the app runs on synthetic frames (useful for a
  smoke test, but no real recognition).

Confirm:

```bash
arduino-app-cli version
docker ps
ls /var/lib/arduino-app-cli/assets        # e.g. 0.12.0
ls -l /dev/video*                          # expect a USB-cam node, e.g. /dev/video0
```

### Base images

The runner image builds `FROM ghcr.io/arduino/app-bricks/ei-qnn-models-runner:0.12.1`,
and `run_applab.sh` runs `ghcr.io/arduino/app-bricks/python-apps-base:0.12.0` as its
provisioner. On a board set up with Arduino App Lab these are normally already
present:

```bash
docker images | grep -E "ei-qnn-models-runner|python-apps-base"
```

If either is missing, the board needs network egress to `ghcr.io` to pull them
(`docker pull ghcr.io/arduino/app-bricks/ei-qnn-models-runner:0.12.1` and
`…/python-apps-base:0.12.0`). Do this once before step 3/4.

### Camera

The app container reads the camera **directly** via V4L2: `CameraThread` probes
`/dev/video*` and opens the first node that yields a real frame (falling back to
synthetic frames if none do). App Lab bind-mounts the whole host `/dev` into the
app container and adds the matching device-cgroup rule, so a USB webcam that
enumerates on the host (e.g. `/dev/video0`) is visible to the app with **no
per-app configuration** — nothing in `app.yaml` or the brick composes forwards the
camera. Just make sure the webcam is plugged in and shows up in `ls /dev/video*`
on the host before launching. (Verified on-board 2026-10-18: app logs
`[Camera] Opened /dev/video0`.)

An HDMI display is **not** required — the UI is a webview you open from any browser
on the network (step 5).

---

## 1. Get the app onto the board

Clone this repo (or copy just the app folder) into App Lab's apps directory —
typically `~/ArduinoApps/`:

```bash
mkdir -p ~/ArduinoApps
cd ~/ArduinoApps
git clone https://github.com/munoz0raul/dorbell-app-bricks.git
# App Lab looks for an app folder with an app.yaml:
ln -s dorbell-app-bricks/inspirational/platform_ventunoq/dragonwing-doorbell \
      ~/ArduinoApps/dragonwing-doorbell
```

(You can also copy the `dragonwing-doorbell/` folder directly to
`~/ArduinoApps/dragonwing-doorbell` — the symlink just avoids duplicating files.)

---

## 2. Stage the NPU models

The `buffalo_sc` ONNX models are **not committed** (large, board-pinned). Stage
them under `runner/models/buffalo_sc/` before building the runner image. If the
board has internet egress, let InsightFace fetch the pack once:

```bash
pip3 install --break-system-packages insightface onnxruntime
python3 - <<'PY'
from insightface.app import FaceAnalysis
FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"]).prepare(ctx_id=-1, det_size=(320,320))
PY
# -> ~/.insightface/models/buffalo_sc/{det_500m,w600k_mbf}.onnx

mkdir -p ~/ArduinoApps/dorbell-app-bricks/runner/models/buffalo_sc
cp ~/.insightface/models/buffalo_sc/det_500m.onnx \
   ~/.insightface/models/buffalo_sc/w600k_mbf.onnx \
   ~/ArduinoApps/dorbell-app-bricks/runner/models/buffalo_sc/
```

(Alternatively, mount the models at runtime to `/opt/models/buffalo_sc` and skip
the COPY — see [`runner/README.md`](runner/README.md).)

---

## 3. Build the NPU runner image

The brick's companion container runs InsightFace on the HTP v75 via
`onnxruntime-qnn`. Build it **from the repo root** so the build context includes
`runner/`:

```bash
cd ~/ArduinoApps/dorbell-app-bricks
docker build -t ghcr.io/arduino/app-bricks/face-recognition-runner:0.1.0 \
  -f runner/Dockerfile .
```

The brick compose references
`${DOCKER_REGISTRY_BASE:-ghcr.io/arduino/}app-bricks/face-recognition-runner:0.1.0`
— so tagging it under `ghcr.io/arduino/` locally (as above) lets App Lab find it
with no registry push. If you push it elsewhere, set `DOCKER_REGISTRY_BASE`.

> **Why no host QNN libs?** `onnxruntime-qnn` bundles its own version-matched QNN
> 2.51 runtime (including the v75 skels). The compose forwards only the
> kernel-coupled plumbing (`/dev/fastrpc-cdsp`, `/dev/dma_heap/system`). Full
> rationale + the `ADSP_LIBRARY_PATH` skew gotcha are in
> [`runner/README.md`](runner/README.md).

---

## 4. Start the app

A plain `arduino-app-cli app start` (and the App Lab UI's **Run** button) validates
every brick in `app.yaml` against the **stock** registry at
`/var/lib/arduino-app-cli/assets/<ver>/bricks-list.yaml`. The
`arduino:face_recognition` brick lives *inside* the app, so out of the box it is
absent from that registry and you get:

```
[ERROR] Brick "arduino:face_recognition" Not Found
```

There are **two ways** to launch, depending on whether you want full isolation or
UI integration. Pick one.

### 4A. Install the brick into App Lab — run it like a stock app *(recommended)*

This registers the brick into App Lab's **stock** data dir so the app behaves
exactly like the built-in examples (e.g. *Face Detector on Camera*): the App Lab
**Stop/Run** buttons work, and the kiosk **auto-opens the webview on the board
screen**.

```bash
cd ~/ArduinoApps/dragonwing-doorbell
bash install-brick.sh
```

`install-brick.sh` is **additive and reversible**. It:

1. runs the **same provisioner the daemon uses** (`arduino-bricks-list-modules` in
   the `python-apps-base` image) to generate the brick's registry entry + compose
   — no hand-authored YAML;
2. **backs up** the stock `bricks-list.yaml` (to `~/.doorbell-stock-backup/<ts>/`);
3. **appends only** the `arduino:face_recognition` entry to the stock registry
   (no stock brick is modified);
4. copies **only** `compose/arduino/face_recognition/` into the stock compose tree;
5. **restarts the daemon** so it reloads the registry.

> **Step 5 is not optional.** The `arduino-app-cli` daemon caches the brick list in
> memory at startup, so editing the registry file alone leaves the UI still
> reporting `Brick not found`. The script restarts the daemon for you (it will use
> `sudo`, which prompts in a terminal); if it can't restart automatically it prints
> the one command to run:
> ```bash
> sudo systemctl restart arduino-app-cli.service
> ```

After that, open App Lab, select **Dragonwing Doorbell**, and hit **Run** — the
containers come up and the webview opens on the board. To go back to pristine
stock at any time:

```bash
bash install-brick.sh --uninstall      # restores the registry backup + removes the compose dir + restarts the daemon
```

### 4B. Launch isolated, without touching the stock data dir

If you'd rather not modify the stock data dir at all (strict isolation), use
`run_applab.sh`. It builds a **private data dir** (`ARDUINO_APP_CLI__DATA_DIR`)
whose registry = stock + our brick, regenerated with the same provisioner, then
starts the app against it — **without editing any stock file**. The trade-off: the
App Lab UI and the on-screen kiosk don't see the app, so you launch (and open the
webview) yourself.

```bash
cd ~/ArduinoApps/dragonwing-doorbell
bash run_applab.sh
```

| Command | Effect |
|---------|--------|
| `bash run_applab.sh` | build the private data dir (if needed) + `app start` |
| `bash run_applab.sh --rebuild` | force-refresh the data dir, then start |
| `bash run_applab.sh --setup-only` | build the data dir, don't start |
| `bash run_applab.sh --stop` | stop the running app |

> `app start` is **state-changing** — it stops any running app and rebuilds the
> compose. Coordinate before running it on a shared board.

---

## 5. Open the webview

If you used **4A**, the App Lab kiosk opens the webview automatically on the board
screen when the app starts (like the stock examples). Otherwise — or from another
machine — browse to:

```
http://<board>:7000
```

You should see the live camera canvas. Enrollment is normally automatic (show a
LinkedIn QR to the camera). From the browser you can also drive the flows by key:

| Key | Action |
|-----|--------|
| `G` | authorize a guest (same flow as a LinkedIn QR) |
| `C` | clear the face database |
| `[` / `]` | decrease / increase the idle timeout (5–60 s) |
| `-` / `=` | decrease / increase the enrolled-face TTL (60–600 s) |
| `Esc` | back to Standby |
| `Q` | quit the app loop |

---

## 6. Verify the NPU is actually in use

CPU fallback is **silent** — always confirm. Check the runner's logs:

```bash
docker logs <face-recognition-runner container> 2>&1 | grep -i qnn
```

Expect:

```
[runner] QNN registered — qnn_version=2.51.0 adsp=…/onnxruntime_qnn
[runner] engine ready — det/rec providers=['QNNExecutionProvider', 'CPUExecutionProvider']
```

A providers list that **starts** with `CPUExecutionProvider` means QNN didn't load
— see the `ADSP_LIBRARY_PATH` skew gotcha in [`runner/README.md`](runner/README.md).

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `Brick "arduino:face_recognition" Not Found` (UI Run, or plain `app start`) | The custom brick isn't in the stock registry. Install it with `install-brick.sh` (step 4A), or launch via `run_applab.sh` (step 4B). |
| UI **still** says `Brick not found` right after `install-brick.sh` | The daemon caches the registry in memory. Restart it: `sudo systemctl restart arduino-app-cli.service` (the script does this; it's needed if the auto-restart was skipped). |
| Webview stuck at "Connecting… / offline" | The bundled Socket.IO client (`assets/libs/socket.io.min.js`) must match the server's Socket.IO v4. It's committed in this repo; don't point the script tag at `/socket.io/socket.io.js` (that's the Engine.IO server, which returns HTTP 400). |
| Runner can't find the models | Stage them in `runner/models/buffalo_sc/` before building (step 2), or mount them at `/opt/models/buffalo_sc`. |
| `providers` starts with `CPUExecutionProvider` | HTP didn't initialize. Confirm `/dev/fastrpc-cdsp` + `/dev/dma_heap/system` are forwarded (they're in `brick_compose.ventunoq.yaml`) and that `ADSP_LIBRARY_PATH` points at the wheel's skels, not the board's. |
| Compose can't pull the runner image | Build it locally tagged under `ghcr.io/arduino/` (step 3), or set `DOCKER_REGISTRY_BASE`. |
| App shows a synthetic/animated face, not the camera | No USB webcam detected. Confirm `ls /dev/video*` on the host shows a cam node and that the app logs `[Camera] Opened /dev/videoN` (not `No camera — using synthetic frames`). |
| `docker pull` of a base image fails | The board needs egress to `ghcr.io` for `ei-qnn-models-runner:0.12.1` + `python-apps-base:0.12.0` (see "Base images" in §0). |

---

See the app's [README](inspirational/platform_ventunoq/dragonwing-doorbell/README.md)
for the architecture, and [munoz0raul/dorbell](https://github.com/munoz0raul/dorbell)
for the production (IQ-8275) side of the same demo.
