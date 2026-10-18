#!/usr/bin/env python3
"""
face-recognition-runner — the NPU server behind the arduino:face_recognition brick.

Server end of the two-channel contract the brick client speaks (mirrors Arduino's
ei-qnn-models-runner):

    * frames  — TCP :5050   length-prefixed JPEG (client -> runner)
    * results — WS  :4912   JSON faces            (runner -> client)

It runs InsightFace buffalo_sc (det_500m + w600k_mbf) on the HTP v75 via
onnxruntime-qnn (the QNN plugin-EP, ORT 1.31 + bundled QNN 2.51), and returns,
per face:
    { "bbox":[x1,y1,x2,y2], "kps":[[x,y]*5], "embedding":[512 floats], "score":f }

The det/rec pre- and post-processing is IDENTICAL to doorbell_core's NPU path
(the same _det_preprocess/_det_postprocess/_rec_preprocess/_align_crop), so the
embeddings the app matches against are the same ones the production app produces.
Coordinates are returned in the SAME space as the incoming frame (the app sends
320x240 inference frames, so faces come back in 320x240 — doorbell_core's INF_W/
INF_H space — and every draw_* maps them to the 1920x1080 canvas unchanged).

Backend selection:
    FACE_RECOGNITION_BACKEND = qnn | cpu   (default: qnn; falls back to cpu)
The QNN plugin-EP (onnxruntime-qnn) drives the HTP v75. If the EP/DSP is
unavailable (wheel absent, no fastrpc/DMA device nodes) it falls back to
CPUExecutionProvider so the brick still works off-board for development.
ADSP_LIBRARY_PATH is pointed at the wheel's OWN bundled v75 skels at startup —
NOT the board's /usr/share/qcom skels, which are a different QNN generation and
make HTP device creation fail (QNN_DEVICE_ERROR_INVALID_CONFIG → silent CPU
fallback). Override with FACE_RECOGNITION_KEEP_ADSP_PATH=1 if you know better.
"""

import os
import io
import json
import time
import socket
import struct
import threading
import asyncio
from typing import Optional, List

import numpy as np
import cv2

# ── ports / config ────────────────────────────────────────────────────────────
WS_PORT = int(os.environ.get("FACE_RECOGNITION_WS_PORT", "4912"))
TCP_PORT = int(os.environ.get("FACE_RECOGNITION_TCP_PORT", "5050"))
DET_THRESHOLD = float(os.environ.get("FACE_RECOGNITION_DET_THRESHOLD", "0.5"))
NMS_THRESHOLD = float(os.environ.get("FACE_RECOGNITION_NMS_THRESHOLD", "0.4"))
BACKEND = os.environ.get("FACE_RECOGNITION_BACKEND", "qnn").lower()

# ONNX model locations inside the runner image. buffalo_sc det + rec.
MODEL_DIR = os.environ.get("FACE_RECOGNITION_MODEL_DIR",
                           "/opt/models/buffalo_sc")
DET_ONNX = os.path.join(MODEL_DIR, "det_500m.onnx")
REC_ONNX = os.path.join(MODEL_DIR, "w600k_mbf.onnx")

# ── det/rec pre-post, copied verbatim from doorbell_core (the NPU path) ─────────
_DET_STRIDES = [8, 16, 32]
_DET_FEAT_MAPS = [40, 20, 10]      # 320/8, 320/16, 320/32
_DET_NUM_ANCHORS = 2


def _det_preprocess(bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(bgr, (320, 320))
    img = img[:, :, ::-1].astype(np.float32)
    img = (img - 127.5) / 128.0
    return img.transpose(2, 0, 1)[np.newaxis]


def _det_postprocess(outputs, orig_h, orig_w, det_thresh=0.5, nms_thresh=0.4):
    scores_list = outputs[0:3]
    boxes_list = outputs[3:6]
    kps_list = outputs[6:9]

    sx = orig_w / 320.0
    sy = orig_h / 320.0
    boxes_out, scores_out, kps_out = [], [], []

    for i, stride in enumerate(_DET_STRIDES):
        fm = _DET_FEAT_MAPS[i]
        scores = scores_list[i].reshape(-1)
        boxes = boxes_list[i].reshape(-1, 4)
        kps = kps_list[i].reshape(-1, 5, 2)

        ys, xs = np.meshgrid(np.arange(fm), np.arange(fm), indexing='ij')
        cx = np.repeat((xs.flatten() + 0.5) * stride, _DET_NUM_ANCHORS)
        cy = np.repeat((ys.flatten() + 0.5) * stride, _DET_NUM_ANCHORS)

        mask = scores >= det_thresh
        if not np.any(mask):
            continue

        sc = scores[mask]
        bx = boxes[mask]
        kp = kps[mask]
        acx = cx[mask]
        acy = cy[mask]

        x1 = (acx - bx[:, 0] * stride) * sx
        y1 = (acy - bx[:, 1] * stride) * sy
        x2 = (acx + bx[:, 2] * stride) * sx
        y2 = (acy + bx[:, 3] * stride) * sy

        kp_dec = np.zeros_like(kp)
        kp_dec[:, :, 0] = (acx[:, None] + kp[:, :, 0] * stride) * sx
        kp_dec[:, :, 1] = (acy[:, None] + kp[:, :, 1] * stride) * sy

        boxes_out.append(np.stack([x1, y1, x2, y2], axis=1))
        scores_out.append(sc)
        kps_out.append(kp_dec)

    if not boxes_out:
        return []

    boxes_all = np.concatenate(boxes_out)
    scores_all = np.concatenate(scores_out)
    kps_all = np.concatenate(kps_out)

    keep = cv2.dnn.NMSBoxes(
        boxes_all.tolist(), scores_all.tolist(), det_thresh, nms_thresh)
    if len(keep) == 0:
        return []

    faces = []
    for k in np.array(keep).flatten():
        faces.append({'bbox': boxes_all[k], 'kps': kps_all[k],
                      'score': float(scores_all[k])})
    return faces


def _rec_preprocess(bgr: np.ndarray) -> np.ndarray:
    if bgr.shape[0] != 112 or bgr.shape[1] != 112:
        bgr = cv2.resize(bgr, (112, 112))
    img = bgr[:, :, ::-1].astype(np.float32)
    img = (img - 127.5) / 128.0
    return img.transpose(2, 0, 1)[np.newaxis]


_norm_crop = None


def _align_crop(frame: np.ndarray, kps: np.ndarray) -> np.ndarray:
    global _norm_crop
    if _norm_crop is None:
        from insightface.utils.face_align import norm_crop as _nc
        _norm_crop = _nc
    return _norm_crop(frame, landmark=np.asarray(kps, dtype=np.float32),
                      image_size=112)


# ── the HTP engine ──────────────────────────────────────────────────────────
class Engine:
    """det + rec on the NPU via onnxruntime-qnn (CPU fallback for dev)."""

    def __init__(self):
        self.det = None
        self.rec = None
        self._det_in = None
        self._rec_in = None
        self.ready = False
        self.backend = None

    def load(self):
        import onnxruntime as ort

        qnn_ok = False
        if BACKEND == "qnn":
            qnn_ok = self._register_qnn(ort)
            if not qnn_ok:
                print("[runner] QNN EP unavailable — falling back to CPU", flush=True)

        self.det = self._make_session(ort, DET_ONNX, qnn_ok)
        self.rec = self._make_session(ort, REC_ONNX, qnn_ok)
        self._det_in = self.det.get_inputs()[0].name
        self._rec_in = self.rec.get_inputs()[0].name
        self.backend = self.det.get_providers()[0]
        self.ready = True
        print(f"[runner] engine ready — det/rec providers={self.det.get_providers()}",
              flush=True)

    # onnxruntime-qnn 2.7.0 exposes the HTP via the plugin-EP model (ORT 1.31), not
    # the old provider-tuple API. The wheel bundles its OWN version-matched QNN 2.51
    # skels; ADSP_LIBRARY_PATH MUST point at them (NOT the board's /usr/share/qcom
    # skels — mixing generations fails HTP device creation with
    # QNN_DEVICE_ERROR_INVALID_CONFIG and silently falls back to CPU).
    _qnn_devices = None
    _qnn_backend_path = None

    def _register_qnn(self, ort) -> bool:
        try:
            import onnxruntime_qnn as oq
        except Exception as e:  # pragma: no cover - env without the wheel
            print(f"[runner] onnxruntime_qnn import failed: {e!r}", flush=True)
            return False
        # Point the DSP loader at the wheel's bundled v75 skels before any session is
        # created. An override env is honoured only if the caller really means it.
        if os.environ.get("FACE_RECOGNITION_KEEP_ADSP_PATH") != "1":
            os.environ["ADSP_LIBRARY_PATH"] = oq.LIB_DIR_FULL_PATH
        try:
            ort.register_execution_provider_library(oq.EP_NAME, oq.get_library_path())
        except Exception as e:
            print(f"[runner] register QNN EP failed: {e!r}", flush=True)
            return False
        devs = [d for d in ort.get_ep_devices() if d.ep_name == oq.EP_NAME]
        if not devs:
            print("[runner] no QNN device visible (fastrpc/DMA nodes forwarded?)",
                  flush=True)
            return False
        self._qnn_devices = devs
        self._qnn_backend_path = oq.get_qnn_htp_path()
        print(f"[runner] QNN registered — qnn_version="
              f"{oq.build_and_package_info.qnn_version} "
              f"adsp={os.environ.get('ADSP_LIBRARY_PATH')}", flush=True)
        return True

    def _make_session(self, ort, onnx_path, qnn_ok):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if qnn_ok:
            # htp_performance_mode would be nice but the plugin-EP device config is
            # strict (unknown keys -> INVALID_CONFIG); keep options minimal.
            so.add_provider_for_devices(
                self._qnn_devices, {"backend_path": self._qnn_backend_path})
            return ort.InferenceSession(onnx_path, sess_options=so)
        return ort.InferenceSession(onnx_path, sess_options=so,
                                    providers=["CPUExecutionProvider"])

    def infer(self, frame: np.ndarray) -> list:
        """det → align → rec for one frame. Returns face dicts (JSON-ready)."""
        h, w = frame.shape[:2]
        pre = _det_preprocess(frame)
        det_outs = self.det.run(None, {self._det_in: pre})
        # det_outs is a list of 9 arrays (scores×3, boxes×3, kps×3), same order
        # doorbell_core assumes.
        raw = _det_postprocess(det_outs, h, w, DET_THRESHOLD, NMS_THRESHOLD)

        faces = []
        for fd in raw:
            x1, y1, x2, y2 = fd['bbox']
            xi1, yi1 = max(0, int(x1)), max(0, int(y1))
            xi2, yi2 = min(w, int(x2)), min(h, int(y2))
            if xi2 <= xi1 or yi2 <= yi1:
                continue
            try:
                crop = _align_crop(frame, fd['kps'])
            except Exception:
                crop = frame[yi1:yi2, xi1:xi2]
            rpre = _rec_preprocess(crop)
            rout = self.rec.run(None, {self._rec_in: rpre})
            emb = np.asarray(rout[0], dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(emb)
            if norm > 1e-6:
                emb = emb / norm
            faces.append({
                "bbox": [float(v) for v in fd['bbox']],
                "kps": [[float(x), float(y)] for x, y in fd['kps']],
                "embedding": [float(v) for v in emb],
                "score": float(fd['score']),
            })
        return faces


# ── shared latest-faces state (frame intake → ws broadcast) ────────────────────
class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_faces: list = []
        self.seq = 0

    def set(self, faces):
        with self.lock:
            self.latest_faces = faces
            self.seq += 1

    def get(self):
        with self.lock:
            return self.latest_faces, self.seq


SHARED = Shared()
ENGINE = Engine()


# ── TCP frame intake (client → runner) ─────────────────────────────────────────
def _recv_exact(conn: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _handle_frame_conn(conn: socket.socket, addr):
    print(f"[runner] frame client connected: {addr}", flush=True)
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        while True:
            hdr = _recv_exact(conn, 4)
            if hdr is None:
                break
            (length,) = struct.unpack(">I", hdr)
            if length <= 0 or length > 32 * 1024 * 1024:
                print(f"[runner] bad frame length {length}, dropping conn", flush=True)
                break
            payload = _recv_exact(conn, length)
            if payload is None:
                break
            arr = np.frombuffer(payload, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if not ENGINE.ready:
                continue
            try:
                faces = ENGINE.infer(frame)
                SHARED.set(faces)
            except Exception as e:
                print(f"[runner] infer error: {e}", flush=True)
    except Exception as e:
        print(f"[runner] frame conn error: {e}", flush=True)
    finally:
        try:
            conn.close()
        except Exception:
            pass
        print(f"[runner] frame client disconnected: {addr}", flush=True)


def _frame_server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", TCP_PORT))
    srv.listen(4)
    print(f"[runner] frame intake on tcp :{TCP_PORT}", flush=True)
    while True:
        conn, addr = srv.accept()
        threading.Thread(target=_handle_frame_conn, args=(conn, addr),
                         daemon=True).start()


# ── websocket results (runner → client) ─────────────────────────────────────────
async def _ws_handler(websocket):
    peer = getattr(websocket, "remote_address", "?")
    print(f"[runner] ws client connected: {peer}", flush=True)
    try:
        await websocket.send(json.dumps({"type": "hello",
                                         "backend": ENGINE.backend or BACKEND}))
    except Exception:
        return
    last_seq = -1
    try:
        while True:
            faces, seq = SHARED.get()
            if seq != last_seq:
                last_seq = seq
                await websocket.send(json.dumps({"type": "faces", "faces": faces}))
            await asyncio.sleep(0.02)   # ~50 Hz push ceiling; seq-gated
    except Exception as e:
        print(f"[runner] ws conn closed: {e}", flush=True)


async def _ws_main():
    import websockets
    async with websockets.serve(_ws_handler, "0.0.0.0", WS_PORT,
                                ping_interval=20, ping_timeout=20):
        print(f"[runner] results websocket on ws :{WS_PORT}", flush=True)
        await asyncio.Future()  # run forever


def main():
    print(f"[runner] starting — backend={BACKEND} det_thresh={DET_THRESHOLD}",
          flush=True)
    print(f"[runner] models: det={DET_ONNX} rec={REC_ONNX}", flush=True)

    # Load the HTP engine in a thread so the sockets come up immediately (the
    # client's LOADING gate is released by the ws 'hello', and faces start once
    # the engine is ready).
    def _load():
        try:
            ENGINE.load()
        except Exception as e:
            print(f"[runner] FATAL: engine load failed: {e}", flush=True)
    threading.Thread(target=_load, daemon=True).start()

    threading.Thread(target=_frame_server, daemon=True).start()
    asyncio.run(_ws_main())


if __name__ == "__main__":
    main()
