# face-recognition-runner

The NPU inference server behind the `arduino:face_recognition` brick. Runs
InsightFace `buffalo_sc` (det_500m + w600k_mbf) on the **HTP v75** via
**onnxruntime-qnn** (the QNN execution provider, delivered as a pip wheel that
bundles its own board-matched QNN runtime), and speaks the two-channel contract the
brick client expects.

> **Validated on-board 2026-10-18** (Ventuno Q, "Monaco Monza addons"): the real
> det+rec engine loads on `QNNExecutionProvider` and runs a 320×240 frame through
> det_500m + w600k_mbf in **~17–70 ms** (vs ~267 ms on CPU for the same 6-face
> image), producing embeddings that match the CPU path to ~3 decimals (the expected
> FP16-vs-FP32 delta).

## Contract

| Channel | Transport | Direction | Payload |
|---------|-----------|-----------|---------|
| frames | TCP `:5050` | client → runner | length-prefixed JPEG (`>I` u32 + bytes) |
| results | WS `:4912` | runner → client | `{"type":"hello"}`, then `{"type":"faces","faces":[…]}` |

Each face: `{"bbox":[x1,y1,x2,y2], "kps":[[x,y]×5], "embedding":[512 floats], "score":f}`.

The det/rec pre- and post-processing is **copied verbatim from `doorbell_core`**
(the production NPU path), so embeddings match what the production IQ-8275 app
produces. Faces are returned in the **same pixel space as the incoming frame** —
the app sends 320×240 inference frames, so faces come back in 320×240
(`INF_W×INF_H`) and every `draw_*` maps them to the 1920×1080 canvas unchanged.

## How the NPU is reached (the important part)

The planned `aihub-onnx-models-runner` base **does not exist** on the board. What
*does* exist and works:

- **Base image:** `ghcr.io/arduino/app-bricks/ei-qnn-models-runner:0.12.1` — already
  on the board (no pull), Debian 13 aarch64, Python 3.13 + pip. We ignore its Edge
  Impulse Node.js runner and override the entrypoint. It runs as **root**, which is
  how it opens the group-restricted `/dev/fastrpc-cdsp` + `/dev/dma_heap/system`
  nodes — the stock Arduino NPU runners use neither `group_add` nor `privileged`,
  and neither do we.
- **The EP:** `pip install onnxruntime-qnn==2.7.0` — a real **linux-aarch64/cp313**
  wheel (86 MB) that is plain `onnxruntime` 1.31 **plus** an `onnxruntime_qnn` helper
  bundling its **own version-matched QNN 2.51** runtime, including the host
  `libQnnHtpV75Stub.so` and DSP `libQnnHtpV75Skel.so`. This is the same "pin the host
  QNN to the board QAIRT" discipline the production IQ-8275 container uses — just
  delivered as a wheel.
- **Registration** is the ORT 1.31 **plugin-EP** model (not the old provider tuple):
  ```python
  import onnxruntime as ort, onnxruntime_qnn as oq
  ort.register_execution_provider_library(oq.EP_NAME, oq.get_library_path())
  qnn = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
  so = ort.SessionOptions()
  so.add_provider_for_devices(qnn, {"backend_path": oq.get_qnn_htp_path()})
  sess = ort.InferenceSession(onnx_path, so)
  ```
  `runner.py` does exactly this in `Engine._register_qnn` / `_make_session`.

### ⚠️ The `ADSP_LIBRARY_PATH` skew gotcha

`ADSP_LIBRARY_PATH` **must** point at the **wheel's own bundled skels**
(`onnxruntime_qnn.LIB_DIR_FULL_PATH`), **not** the board's
`/usr/share/qcom/.../dsp/cdsp`. The board skels are a *different QNN generation*
than the wheel's host `libQnnHtp.so`, so pointing there makes HTP device creation
fail with `QNN_DEVICE_ERROR_INVALID_CONFIG` and the session **silently falls back to
CPU**. `runner.py` sets `ADSP_LIBRARY_PATH` to the wheel dir at startup; the
ventunoq compose therefore does **not** forward `/usr/share/qcom` (unlike the stock
EI runner, whose QNN lives in `/usr/lib`). Override only with
`FACE_RECOGNITION_KEEP_ADSP_PATH=1` if you really mean to.

**HTP-liveness check** (CPU fallback is silent — always verify): a float `Relu` on
`[-1,2,-3,4]` returns exact `[0,2,0,4]` on CPU but `[0, 2.00000024, 0, 4.0000005]`
(FP16 rounding) on the HTP, and the HTP path logs
`hexagon-sdk-…/ipc/fastrpc/rpcmem`. The runner also prints its providers on start:
```
[runner] QNN registered — qnn_version=2.51.0 adsp=…/onnxruntime_qnn
[runner] engine ready — det/rec providers=['QNNExecutionProvider', 'CPUExecutionProvider']
```
A providers list that starts with `CPUExecutionProvider` means QNN didn't load.

## Models (not committed)

Stage the buffalo_sc ONNX before building (gitignored — same models the production
app downloads to `~/.insightface/models/buffalo_sc`):

```
runner/models/buffalo_sc/det_500m.onnx
runner/models/buffalo_sc/w600k_mbf.onnx
```

Either `COPY` them into the image (default) or mount them at runtime to
`/opt/models/buffalo_sc` and skip the copy. If the board has egress, the quickest
way to obtain them is to let insightface fetch the pack once:
```python
from insightface.app import FaceAnalysis
FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"]).prepare(ctx_id=-1, det_size=(320,320))
# -> ~/.insightface/models/buffalo_sc/{det_500m,w600k_mbf}.onnx
```

## Build

```bash
# From the repo root (build context must include runner/):
docker build -t ghcr.io/arduino/app-bricks/face-recognition-runner:0.1.0 \
  -f runner/Dockerfile .

# To override the base (default is the on-board ei-qnn-models-runner:0.12.1):
docker build --build-arg BASE=<other-base> \
  -t ghcr.io/arduino/app-bricks/face-recognition-runner:0.1.0 \
  -f runner/Dockerfile .
```

The brick compose references
`${DOCKER_REGISTRY_BASE:-ghcr.io/arduino/}app-bricks/face-recognition-runner:0.1.0`
— so either tag it under `ghcr.io/arduino/` locally (as above) or set
`DOCKER_REGISTRY_BASE` to wherever you push it.

## Backend selection

`FACE_RECOGNITION_BACKEND=qnn|cpu` (default `qnn`). The QNN path registers the
plugin-EP and loads det/rec on the HTP; it falls back to `CPUExecutionProvider` if
the wheel is absent or the `/dev/fastrpc-cdsp` + `/dev/dma_heap/system` nodes aren't
forwarded — so the runner also works off-board for development.

## HTP enablement — what the compose forwards

`brick_compose.ventunoq.yaml` forwards only the kernel-coupled plumbing the QNN EP
needs (the wheel supplies the skels itself):

- `devices`: `/dev/dma_heap/system`, `/dev/fastrpc-cdsp`
- `volumes`: `/sys/firmware/devicetree/base/model` (board id the FastRPC lib reads)
- **no** `/usr/share/qcom` mount, **no** `ADSP_LIBRARY_PATH` override (see the skew
  gotcha above)

These are the Ventuno Q analog of the IQ-8275 CDI mounts — same FastRPC/DSP
forwarding, expressed the Arduino compose way, minus the host skels the wheel makes
redundant.
