"""
Dragonwing Doorbell — doorbell.py
Autonomous face-recognition doorbell demo for the Qualcomm IQ-8275 (QCS8275 / HTP v75).

Stage 2 target: IQ8 + Ubuntu 24.04 + USB webcam + HDMI display.
  - InsightFace buffalo_sc, det_size=(320,320) for low latency on aarch64 CPU
  - V4L2 camera capture, synthetic-frame fallback when no camera is present
  - Dual inference backends, switchable live:
      CPU  — InsightFace / onnxruntime (CPUExecutionProvider)
      NPU  — HTP v75 via qai_appbuilder in-process QNN contexts (det + rec on HTP)
  - Auto-discovers Xwayland auth from the running GNOME session
  - Dragonwing-branded full-screen UI drawn on an OpenCV canvas

Usage:
  python3 doorbell.py                  # CPU backend
  python3 doorbell.py --backend npu    # NPU (HTP v75) via qai_appbuilder in-process QNN

Controls (click the window first):
  B      — toggle CPU / NPU backend live
  G      — authorize a guest (random name, same flow as a LinkedIn QR)
  C      — clear the face database
  [ / ]  — decrease / increase the idle timeout (no-face -> Standby), 5-60s
  - / =  — decrease / increase how long enrolled faces are stored, 60-600s
  ESC    — back to Standby
  Q      — quit

Auto behaviour:
  Face detected        → Standby becomes Detection, recognition runs each frame
  No face for idle_to  → returns to Standby
  LinkedIn QR detected → triggers Registration (~30 aligned frames over ~15 s, data_ttl TTL)
  Any QR is read, but only a LinkedIn QR authorizes/enrolls.
"""

import cv2
import numpy as np
import time
import threading
import random
import re
import os
import glob
import argparse
import collections
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List


def _setup_display():
    """Auto-discover Xwayland auth from the running GNOME session."""
    if not os.environ.get('DISPLAY'):
        os.environ['DISPLAY'] = ':0'
    if not os.environ.get('XAUTHORITY'):
        matches = glob.glob('/run/user/1000/.mutter-Xwaylandauth.*')
        if matches:
            os.environ['XAUTHORITY'] = matches[0]
            print(f"[Display] XAUTHORITY={matches[0]}")
        else:
            print("[Display] WARNING: no Xwayland auth found")

_setup_display()


# ── FPS counter ───────────────────────────────────────────────────────────────

class FpsCounter:
    """Rolling average FPS over the last N timestamps."""
    def __init__(self, window: int = 30):
        self._times: collections.deque = collections.deque(maxlen=window)

    def tick(self):
        self._times.append(time.perf_counter())

    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0


# ── NPU inference engine (qnn-net-run batch) ──────────────────────────────────
#
# Pre/post-processing helpers shared by CPU and NPU paths.

ONNX_DIR = os.path.expanduser("~/.insightface/models/buffalo_sc")
CTX_DIR  = os.path.expanduser("~/qnn_context")

_DET_STRIDES    = [8, 16, 32]
_DET_FEAT_MAPS  = [40, 20, 10]   # 320/8, 320/16, 320/32
_DET_NUM_ANCHORS = 2


def _det_preprocess(bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(bgr, (320, 320))
    img = img[:, :, ::-1].astype(np.float32)
    img = (img - 127.5) / 128.0
    return img.transpose(2, 0, 1)[np.newaxis]


def _det_postprocess(outputs, orig_h, orig_w, det_thresh=0.5, nms_thresh=0.4):
    scores_list = outputs[0:3]
    boxes_list  = outputs[3:6]
    kps_list    = outputs[6:9]

    sx = orig_w / 320.0
    sy = orig_h / 320.0
    boxes_out, scores_out, kps_out = [], [], []

    for i, stride in enumerate(_DET_STRIDES):
        fm = _DET_FEAT_MAPS[i]
        scores = scores_list[i].reshape(-1)
        boxes  = boxes_list[i].reshape(-1, 4)
        kps    = kps_list[i].reshape(-1, 5, 2)

        ys, xs = np.meshgrid(np.arange(fm), np.arange(fm), indexing='ij')
        cx = np.repeat((xs.flatten() + 0.5) * stride, _DET_NUM_ANCHORS)
        cy = np.repeat((ys.flatten() + 0.5) * stride, _DET_NUM_ANCHORS)

        mask = scores >= det_thresh
        if not np.any(mask):
            continue

        sc  = scores[mask]
        bx  = boxes[mask]
        kp  = kps[mask]
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

    boxes_all  = np.concatenate(boxes_out)
    scores_all = np.concatenate(scores_out)
    kps_all    = np.concatenate(kps_out)

    keep = cv2.dnn.NMSBoxes(
        boxes_all.tolist(), scores_all.tolist(), det_thresh, nms_thresh)
    if len(keep) == 0:
        return []

    faces = []
    for k in keep.flatten():
        faces.append({'bbox': boxes_all[k], 'kps': kps_all[k], 'score': scores_all[k]})
    return faces


def _rec_preprocess(bgr: np.ndarray) -> np.ndarray:
    # Expects a 112x112 face crop (already landmark-aligned by norm_crop). The
    # resize is a cheap safety net for off-size input; aligned crops pass through.
    if bgr.shape[0] != 112 or bgr.shape[1] != 112:
        bgr = cv2.resize(bgr, (112, 112))
    img = bgr[:, :, ::-1].astype(np.float32)
    img = (img - 127.5) / 128.0
    return img.transpose(2, 0, 1)[np.newaxis]


# InsightFace's 5-point similarity alignment (the same transform the CPU
# FaceAnalysis path applies before recognition). ArcFace embeddings are only
# discriminative on aligned crops — feeding a raw, stretched bbox makes every
# face's embedding collapse together, so two different people can score a false
# 0.5 match. Imported lazily so a missing insightface never breaks det-only use.
_norm_crop = None


def _align_crop(frame: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """112x112 face aligned from the 5 landmarks; falls back to None on failure."""
    global _norm_crop
    if _norm_crop is None:
        from insightface.utils.face_align import norm_crop as _nc
        _norm_crop = _nc
    return _norm_crop(frame, landmark=np.asarray(kps, dtype=np.float32), image_size=112)



DET_CTX_PATH = os.path.join(CTX_DIR, "det_500m_qcs8275.bin")
REC_CTX_PATH = os.path.join(CTX_DIR, "w600k_mbf_qcs8275.bin")

# QNN (Qualcomm Neural Network) runtime libraries. qai_appbuilder bundles a
# matching set under its package libs/ and resolves them automatically, so we
# let QNNConfig.Config() pick those (proven to drive this board's HTP v75). The
# board's own /usr/lib set is left as documentation of the on-device SDK.
QNN_BACKEND_LIB = "/usr/lib/libQnnHtp.so"

# det (det_500m) emits 9 output tensors in this order; _det_postprocess reshapes
# each one itself, so we only rely on the ORDER here (scores×3, boxes×3, kps×3).
# rec (w600k_mbf) emits a single 512-d embedding.
_DET_NUM_OUTPUTS = 9
_REC_EMB_SIZE    = 512


class NpuFaceEngine(threading.Thread):
    """
    Real HTP (Hexagon Tensor Processor) NPU engine — in-process via qai_appbuilder.

    The det/rec QNN context binaries (det_500m_qcs8275.bin, w600k_mbf_qcs8275.bin)
    are loaded ONCE into the HTP through qai_appbuilder.QNNContext and then driven
    with direct .Inference() calls — no per-frame subprocess, no file round-trip.

    This replaces the earlier qnn-net-run CLI engine, which paid a fixed ~280ms
    FastRPC+model-load cost on EVERY call and so had to batch 16 frames behind a
    0.10s window just to stay usable — the batch window being the dominant source
    of the overlay lag (box/dots trailing a moving face).

    Timing (measured on QCS8275, in-process):
      One-time load:  det ~185ms + rec ~15ms  (at startup, off the hot path)
      Per-frame HTP:  det ~2ms + rec ~1.3ms   (vs ~280ms spawn each before)

    With the spawn gone, batching has no purpose: we run latest-frame-wins exactly
    like the CPU FaceEngine, so the overlay tracks the live face with ~3ms of
    inference latency instead of a ~560ms batch stall.

    CPU usage when active stays low (~5-10%): the process blocks waiting on the HTP
    rather than spreading work across all 8 cores like the ORT CPU path (~95%).
    """

    def __init__(self, on_ready):
        super().__init__(daemon=True)
        self._pending: Optional[np.ndarray] = None
        self._pending_lock = threading.Lock()
        self._faces: list = []
        self._faces_lock = threading.Lock()
        self.running = True
        self.on_ready = on_ready
        self.fps_counter = FpsCounter(window=30)
        self._det = None
        self._rec = None

    def submit(self, frame: np.ndarray):
        with self._pending_lock:
            self._pending = frame  # always keep latest, drop older

    def get_faces(self) -> list:
        with self._faces_lock:
            return list(self._faces)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _infer_det(self, frame: np.ndarray) -> list:
        """Run det on one frame; return raw_face dicts (bbox/kps/score)."""
        h, w = frame.shape[:2]
        pre = _det_preprocess(frame)
        try:
            outs = self._det.Inference([pre])   # list of 9 flat float32 arrays
        except Exception as e:
            print(f"[NPU] det inference error: {e}")
            return []
        if not isinstance(outs, (list, tuple)) or len(outs) < _DET_NUM_OUTPUTS:
            return []
        try:
            return _det_postprocess(outs, h, w)
        except Exception as e:
            print(f"[NPU] det postprocess error: {e}")
            return []

    def _infer_rec(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Run rec on one 112-aligned crop; return L2-normalized embedding."""
        pre = _rec_preprocess(crop)
        try:
            out = self._rec.Inference([pre])
        except Exception as e:
            print(f"[NPU] rec inference error: {e}")
            return None
        emb = out[0] if isinstance(out, (list, tuple)) else out
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 1e-6 else emb

    def _process_frame(self, frame: np.ndarray):
        """Full det→crop→rec pipeline for one frame; updates self._faces."""
        h, w = frame.shape[:2]
        raw_faces = self._infer_det(frame)

        result_faces = []
        for fd in raw_faces:
            x1, y1, x2, y2 = fd['bbox']
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            # Align from the 5 landmarks before recognition (not a raw bbox crop),
            # so the embedding matches what the CPU InsightFace path produces.
            try:
                crop = _align_crop(frame, fd['kps'])
            except Exception as e:
                print(f"[NPU] align failed, falling back to bbox crop: {e}")
                crop = frame[y1:y2, x1:x2]
            emb = self._infer_rec(crop)
            if emb is None:
                continue
            face = type('Face', (), {
                'bbox': fd['bbox'], 'kps': fd['kps'],
                'landmark_2d_106': None,
                'det_score': fd['score'], 'embedding': emb,
            })()
            result_faces.append(face)

        with self._faces_lock:
            self._faces = result_faces
        self.fps_counter.tick()

    # ── thread entry ─────────────────────────────────────────────────────────

    def run(self):
        det_ok = os.path.exists(DET_CTX_PATH)
        rec_ok = os.path.exists(REC_CTX_PATH)
        if not det_ok or not rec_ok:
            print(f"[NPU] ERROR: context binaries missing")
            print(f"  det: {DET_CTX_PATH}  exists={det_ok}")
            print(f"  rec: {REC_CTX_PATH}  exists={rec_ok}")
            self.on_ready()
            return

        try:
            from qai_appbuilder import (QNNContext, QNNConfig, Runtime,
                                        PerfProfile, ProfilingLevel, LogLevel)
        except Exception as e:
            print(f"[NPU] ERROR: qai_appbuilder import failed: {e}")
            print(f"  pip3 install --break-system-packages qai-appbuilder")
            self.on_ready()
            return

        try:
            # Config() resolves the bundled QNN libs and points ADSP_LIBRARY_PATH
            # at the matching HTP skels — proven to drive this board's HTP v75.
            QNNConfig.Config(Runtime.HTP, LogLevel.ERROR, ProfilingLevel.OFF)
            t0 = time.perf_counter()
            self._det = QNNContext("det", DET_CTX_PATH)
            self._rec = QNNContext("rec", REC_CTX_PATH)
            PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)
            load_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            print(f"[NPU] ERROR: failed to load HTP contexts: {e}")
            self.on_ready()
            return

        print(f"[NPU] in-process HTP engine ready  (load {load_ms:.0f}ms)")
        print(f"[NPU] det_ctx={DET_CTX_PATH}")
        print(f"[NPU] rec_ctx={REC_CTX_PATH}")
        self.on_ready()

        while self.running:
            with self._pending_lock:
                frame = self._pending
                self._pending = None
            if frame is not None:
                self._process_frame(frame)
            else:
                time.sleep(0.005)

        # Release HTP contexts deterministically on shutdown.
        try:
            if self._det is not None:
                self._det.release()
            if self._rec is not None:
                self._rec.release()
        except Exception:
            pass

    def stop(self):
        self.running = False


# QR codes whose URLs match this pattern trigger registration.
# Name is extracted from the LinkedIn username segment.
QR_URL_PATTERN = re.compile(r'linkedin\.com/in/([^/?]+)', re.IGNORECASE)


# ── State machine ─────────────────────────────────────────────────────────────

class State(Enum):
    LOADING       = auto()
    STANDBY       = auto()
    DETECTION     = auto()
    RECOGNIZED    = auto()
    UNKNOWN       = auto()
    QR_AUTHORIZED = auto()  # 5s "AUTHORIZED!" anim + 5s countdown
    REGISTRATION  = auto()


@dataclass
class FaceEntry:
    name: str
    added_at: float
    ttl: float = 120.0
    embeddings: List[np.ndarray] = field(default_factory=list)

    def is_expired(self) -> bool:
        return time.time() - self.added_at > self.ttl

    def remaining(self) -> float:
        return max(0.0, self.ttl - (time.time() - self.added_at))

    def mean_embedding(self) -> Optional[np.ndarray]:
        if not self.embeddings:
            return None
        return np.mean(self.embeddings, axis=0)


@dataclass
class AppState:
    state: State = State.LOADING
    face_db: dict = field(default_factory=dict)
    reg_frames: int = 0
    reg_target: int = 30
    reg_name: str = ""
    last_match: Optional[str] = None
    status_msg: str = "Loading models..."
    last_face_time: float = 0.0
    qr_authorized_at: float = 0.0   # when QR_AUTHORIZED state began
    reg_start_at: float = 0.0       # when REGISTRATION capture began
    idle_timeout: float = 5.0       # no-face seconds before returning to Standby (live-adjustable)
    data_ttl: float = 120.0         # seconds an enrolled face is kept (live-adjustable)
    reg_blocked: bool = False       # two faces in the enroll O → capture paused ("one at a time")
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add_face(self, name: str, embeddings: list):
        entry = FaceEntry(name=name, added_at=time.time(), ttl=self.data_ttl)
        entry.embeddings = [e.copy() for e in embeddings]
        self.face_db[name] = entry

    def prune_expired(self):
        expired = [k for k, v in self.face_db.items() if v.is_expired()]
        for k in expired:
            del self.face_db[k]
            print(f"[DB] Expired: {k}")


# ── QR Detector ───────────────────────────────────────────────────────────────

class QRDetector:
    def __init__(self):
        self._detector = cv2.QRCodeDetector()
        self._cooldown_until = 0.0  # avoid re-triggering immediately

    def detect(self, frame: np.ndarray) -> Optional[str]:
        """Read *any* QR in the frame, but authorize only a LinkedIn profile.

        Returns the LinkedIn username slug to enroll under, or None. Non-LinkedIn
        QRs are decoded but ignored — the demo reads everything, acts on LinkedIn.
        """
        if time.time() < self._cooldown_until:
            return None
        try:
            # Multi-read: catches any/all codes in frame (more robust than the
            # single-code detectAndDecode, which misses off-center or paired QRs).
            ok, datas, _, _ = self._detector.detectAndDecodeMulti(frame)
        except cv2.error:
            # OpenCV 5.0's QR decoder can throw on degenerate candidate
            # contours (zero-area src points) for certain camera frames.
            # It's a transient per-frame failure — never let it kill the
            # unattended demo; just skip QR detection for this frame.
            return None
        if not ok or not datas:
            return None
        for data in datas:
            if not data:
                continue
            m = QR_URL_PATTERN.search(data)
            if m:
                self._cooldown_until = time.time() + 5.0  # 5s cooldown
                return m.group(1).split('?')[0]
        return None


# ── Recognition ───────────────────────────────────────────────────────────────

RECOGNITION_THRESHOLD = 0.35
NO_FACE_TIMEOUT = 5.0


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def recognize_face(embedding: np.ndarray, face_db: dict) -> Optional[str]:
    name, _ = recognize_face_scored(embedding, face_db)
    return name


def recognize_face_scored(embedding: np.ndarray, face_db: dict):
    """Best match as (name, sim). name is None when below threshold; sim is the
    top cosine similarity regardless (so the UI can show the real score)."""
    best_name, best_sim = None, -1.0
    for name, entry in face_db.items():
        mean_emb = entry.mean_embedding()
        if mean_emb is None:
            continue
        sim = cosine_sim(embedding, mean_emb)
        if sim > best_sim:
            best_sim = sim
            best_name = name
    if best_sim >= RECOGNITION_THRESHOLD:
        return best_name, best_sim
    return None, max(0.0, best_sim)


def recognize_all(faces: list, face_db: dict) -> None:
    """Annotate every face in-place with .rec_name (str|None) and .rec_sim (float).

    Each face is matched independently so a crowd shows a correct per-face verdict
    — the known person's box goes green, everyone else red — instead of one global
    state painting all boxes the same color.
    """
    for f in faces:
        emb = getattr(f, 'embedding', None)
        if emb is None:
            f.rec_name, f.rec_sim = None, 0.0
        else:
            f.rec_name, f.rec_sim = recognize_face_scored(emb, face_db)


# ── Multi-face selection + enrollment gate ──────────────────────────────────────
# InsightFace returns faces in detection/NMS order, not by intent — so with more
# than one person in frame we must choose. Unlock is driven by the *largest*
# (closest) face; enrollment is gated to the one face a visitor places inside the
# centered "O" ring, so a bystander can never be captured by accident.

def face_area(face) -> float:
    x1, y1, x2, y2 = face.bbox
    return float((x2 - x1) * (y2 - y1))


def largest_face(faces):
    """The closest face (biggest bbox). None if the list is empty."""
    return max(faces, key=face_area) if faces else None


# Enrollment gate ellipse, in normalized inference-frame coords. A tall oval,
# centered to match the standby focus; radii chosen so a face at a natural
# standing distance fills it comfortably.
GATE_CX, GATE_CY = 0.5, 0.46
GATE_RX, GATE_RY = 0.26, 0.36


def gate_ellipse_px(inf_w: int, inf_h: int):
    """Gate center + radii in inference-frame pixels (same space as face.bbox/kps)."""
    return (GATE_CX * inf_w, GATE_CY * inf_h, GATE_RX * inf_w, GATE_RY * inf_h)


def _in_ellipse(x, y, cx, cy, rx, ry) -> bool:
    if rx <= 0 or ry <= 0:
        return False
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0


def face_in_gate(face, inf_w: int, inf_h: int) -> bool:
    """True when the face is positioned inside the enrollment O.

    Requires the bbox center AND all 5 landmarks inside the ellipse — tighter
    than a center-only test, so a face merely clipping the ring doesn't qualify.
    """
    cx, cy, rx, ry = gate_ellipse_px(inf_w, inf_h)
    x1, y1, x2, y2 = face.bbox
    if not _in_ellipse((x1 + x2) / 2, (y1 + y2) / 2, cx, cy, rx, ry):
        return False
    if face.kps is None:
        return False
    for x, y in face.kps:
        if not _in_ellipse(x, y, cx, cy, rx, ry):
            return False
    return True


# ── Face engine (background thread) ───────────────────────────────────────────

class FaceEngine(threading.Thread):
    def __init__(self, on_ready):
        super().__init__(daemon=True)
        self._pending: Optional[np.ndarray] = None
        self._pending_lock = threading.Lock()
        self._faces: list = []
        self._faces_lock = threading.Lock()
        self.running = True
        self.on_ready = on_ready
        self.fa = None
        self.fps_counter = FpsCounter(window=30)

    def submit(self, frame: np.ndarray):
        with self._pending_lock:
            self._pending = frame  # always keep latest, drop older

    def get_faces(self) -> list:
        with self._faces_lock:
            return list(self._faces)

    def run(self):
        from insightface.app import FaceAnalysis
        try:
            self.fa = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
            self.fa.prepare(ctx_id=0, det_size=(320, 320))
        except Exception as e:
            # On a freshly flashed board the buffalo_sc ONNX models are not staged
            # and ~/.insightface is a read-only mount, so InsightFace cannot
            # auto-download them -> this CPU engine stays unavailable. The NPU
            # backend is unaffected; the B-toggle to CPU simply shows no faces
            # until the models are staged. Degrade gracefully instead of killing
            # the thread with an unhandled traceback.
            print(f"[CPU] InsightFace unavailable: {e}", flush=True)
            print("[CPU] stage ~/.insightface/models/buffalo_sc to enable the CPU backend", flush=True)
            self.fa = None
            self.on_ready()
            return
        self.on_ready()
        while self.running:
            with self._pending_lock:
                frame = self._pending
                self._pending = None
            if frame is not None:
                faces = self.fa.get(frame)
                with self._faces_lock:
                    self._faces = faces
                self.fps_counter.tick()
            else:
                time.sleep(0.01)

    def stop(self):
        self.running = False


# ── Camera thread ─────────────────────────────────────────────────────────────

class CameraThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self.running = True
        self.cap = self._open_camera()
        if self.cap is None or not self.cap.isOpened():
            print("[Camera] No camera — using synthetic frames")

    @staticmethod
    def _open_camera():
        """
        Open the first V4L2 index that both opens and yields a real frame.
        A USB webcam can re-enumerate across reboots/replugs (e.g. video0 ->
        video1), so we probe indices instead of hardcoding 0 — otherwise a
        moved camera leaves the app on synthetic frames or a stale handle.

        We probe the actual /dev/video* nodes present (sorted by index), not a
        fixed range: on Ubuntu the USB cam is usually video0, but on Qualcomm
        Linux (QLI) the on-SoC Iris codec + ISP pipelines occupy video0..27 and
        the USB webcam enumerates high (e.g. /dev/video28). The read() check
        skips the non-capture ISP/codec nodes, which open but yield no frame.
        """
        nodes = sorted(glob.glob('/dev/video*'),
                       key=lambda p: int(re.sub(r'\D', '', p) or -1))
        indices = [int(re.sub(r'\D', '', p)) for p in nodes] or list(range(6))
        for idx in indices:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    print(f"[Camera] Opened /dev/video{idx}")
                    return cap
            cap.release()
        return None

    def run(self):
        t = 0.0
        while self.running:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self._lock:
                        self._frame = frame
            else:
                h, w = 480, 640
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                t += 0.02
                for y in range(h):
                    b = int(20 + 60 * abs(np.sin(t + y * 0.005)))
                    frame[y, :, 0] = b
                noise = np.random.randint(0, 12, (h, w, 3), dtype=np.uint8)
                frame = cv2.add(frame, noise)
                with self._lock:
                    self._frame = frame
            time.sleep(1 / 30)

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self.running = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()


# ── Registration driver ───────────────────────────────────────────────────────

class RegistrationDriver(threading.Thread):
    def __init__(self, app_state: AppState, face_engine: FaceEngine, on_done):
        super().__init__(daemon=True)
        self.app_state = app_state
        self.face_engine = face_engine
        self.on_done = on_done
        self._embeddings: List[np.ndarray] = []

    def run(self):
        while True:
            with self.app_state.lock:
                if self.app_state.state != State.REGISTRATION:
                    return
                n = self.app_state.reg_frames
                target = self.app_state.reg_target
                name = self.app_state.reg_name

            faces = self.face_engine.get_faces()
            # Only a single face placed inside the enrollment O is captured. If two
            # people are in the ring we pause and ask for one at a time, so we never
            # enroll (or average in) a bystander. An empty ring simply waits.
            in_gate = [f for f in faces
                       if f.embedding is not None and face_in_gate(f, INF_W, INF_H)]
            captured = False
            if len(in_gate) == 1:
                self._embeddings.append(in_gate[0].embedding.copy())
                captured = True
                with self.app_state.lock:
                    self.app_state.reg_blocked = False
            elif len(in_gate) >= 2:
                with self.app_state.lock:
                    self.app_state.reg_blocked = True
            else:
                with self.app_state.lock:
                    self.app_state.reg_blocked = False

            if captured:
                with self.app_state.lock:
                    self.app_state.reg_frames += 1
                    done = self.app_state.reg_frames >= self.app_state.reg_target
                    name = self.app_state.reg_name
                if done:
                    self.on_done(name, self._embeddings)
                    return

            # ~0.5 s between captures: 30 frames spans ~15 s so the visitor has
            # time to turn their head and vary pose for a more robust embedding.
            # Capture only advances while a lone face is in the O, so pausing to
            # reposition simply lengthens enrollment rather than wasting a frame.
            time.sleep(0.5)


# ── Drawing ───────────────────────────────────────────────────────────────────

# Native panel resolution. The demo HDMI panel is 1920x1080, so we author and
# render the canvas at that size — drawing every Hershey glyph and vector stroke
# directly into the panel's pixel grid instead of drawing at 720p and letting the
# fullscreen window interpolate it up 1.5x (which is what made text/logo soft).
W, H = 1920, 1080
S = H / 720.0            # UI scale vs. the original 720p authoring (= 1.5)


def _s(v) -> int:
    """Scale a 720p-authored pixel size/offset to the native canvas."""
    return int(round(v * S))


def _th(v) -> int:
    """Scale a stroke thickness, never below 1px."""
    return max(1, int(round(v * S)))

# ── Dragonwing theme ────────────────────────────────────────────────────────
# Palette sampled from the Dragonwing brand references (assets/1.png–3.png).
# Ground is deep indigo #31017D; the accent is electric violet #5B0CD9.
# OpenCV is BGR, so every tuple below is (B, G, R).

INDIGO      = (125,   1,  49)   # #31017D  brand ground
INDIGO_DEEP = ( 90,   1,  35)   # darker indigo for gradient base / panels
INDIGO_LIFT = (150,  10,  66)   # lifted indigo for gradient top / fills
VIOLET      = (217,  12,  91)   # #5B0CD9  primary accent / brand chrome
VIOLET_LO   = (150,  30,  70)   # dim violet — inactive lines, secondary rules
VIOLET_HI   = (245, 120, 170)   # bright violet highlight / glow
INK         = (245, 240, 255)   # near-white primary text
MIST        = (190, 155, 205)   # secondary text (tinted from the violet hue, never gray)

# Access-control semantics stay universal, retuned to read on indigo.
UNLOCK      = (120, 220,  90)   # known / unlocked — green
LOCK        = ( 70,  70, 235)   # unknown / locked — red
DETECT      = (235, 190,  60)   # recognizing — cyan-amber that sits on indigo

# Legacy aliases kept so any out-of-scope reference still resolves.
GREEN, RED, AMBER, PURPLE, WHITE, DIM, DARK = (
    UNLOCK, LOCK, DETECT, VIOLET, INK, VIOLET_LO, INDIGO_DEEP)

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_PLAIN


def _detect_platform() -> str:
    """Build the demo title — the live Board + OS this binary is running on —
    read from the filesystem so the SAME file self-labels across configs:
    IQ8 + Ubuntu, IQ8 + QLI, Ventuno Q + Ubuntu. ASCII-only (Hershey), joined
    with ' + ' (never a Unicode dot).

    Board: /proc/device-tree/model (falls back to /sys/devices/soc0/machine).
    OS:    /etc/os-release (NAME + VERSION_ID; QLI/Qualcomm Linux recognized).
    """
    board = ""
    try:
        with open("/proc/device-tree/model", "rb") as f:
            board = f.read().decode("ascii", "ignore").replace("\x00", "").strip()
    except Exception:
        board = ""
    if not board:
        try:
            board = open("/sys/devices/soc0/machine").read().strip()
        except Exception:
            board = ""
    # Trim the vendor prefix and noise tokens so the title stays short.
    board = re.sub(r'^\s*Qualcomm Technologies,?\s*Inc\.?\s*', '', board, flags=re.I)
    board = re.sub(r'\bSKU\b', '', board, flags=re.I)
    board = re.sub(r'\s+', ' ', board).strip() or "Unknown Board"

    os_name = ""
    try:
        kv = {}
        for line in open("/etc/os-release"):
            if "=" in line:
                k, v = line.rstrip().split("=", 1)
                kv[k] = v.strip().strip('"')
        blob = (kv.get("ID", "") + kv.get("NAME", "") + kv.get("PRETTY_NAME", "")).lower()
        if re.search(r'qli|qcom|qualcomm', blob):
            ver = kv.get("VERSION_ID", "") or kv.get("BUILD_ID", "")
            os_name = ("QLI " + ver).strip()
        else:
            os_name = (kv.get("NAME", "") + " " + kv.get("VERSION_ID", "")).strip()
    except Exception:
        os_name = ""
    os_name = os_name or "Unknown OS"

    return f"{board}  +  {os_name}"


PLATFORM_LABEL = _detect_platform()


# ── Official Dragonwing mark (embedded) ──────────────────────────────────────
# The real circular dragon-head mark, cropped from assets/Qualcomm-Dragonwing-
# logo.jpeg and recolored light-on-transparent so it reads on the indigo bar.
# Embedded as base64 so the single-file scp-to-board deploy needs no assets dir.
_DRAGON_MARK_PNG_B64 = (
    "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"
    "RK5CYII="
)
_DRAGON_MARK: Optional[np.ndarray] = None


def _load_dragon_mark() -> Optional[np.ndarray]:
    """Decode the embedded mark to a BGRA array once; None on any failure."""
    global _DRAGON_MARK
    if _DRAGON_MARK is None:
        try:
            import base64
            raw = base64.b64decode(_DRAGON_MARK_PNG_B64)
            arr = np.frombuffer(raw, np.uint8)
            _DRAGON_MARK = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        except Exception:
            _DRAGON_MARK = None
    return _DRAGON_MARK


def draw_logo_mark(canvas: np.ndarray, cx: int, cy: int, size: int,
                   fallback_color: tuple = VIOLET):
    """
    Blit the official Dragonwing mark centered at (cx, cy), scaled to `size`.
    Alpha-composited over whatever is already on the canvas. Falls back to the
    drawn vector glyph if the embedded asset can't be decoded.
    """
    mark = _load_dragon_mark()
    if mark is None or mark.shape[2] != 4:
        draw_dragon_mark(canvas, cx, cy, size // 2, fallback_color, bg=INDIGO_DEEP)
        return
    m = cv2.resize(mark, (size, size), interpolation=cv2.INTER_AREA)
    x0, y0 = cx - size // 2, cy - size // 2
    x1, y1 = x0 + size, y0 + size
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x1), min(H, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sub = m[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    a = sub[:, :, 3:4].astype(np.float32) / 255.0
    roi = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
    roi = roi * (1.0 - a) + sub[:, :, :3].astype(np.float32) * a
    canvas[cy0:cy1, cx0:cx1] = roi.astype(np.uint8)


# Precomputed vertical indigo gradient (INDIGO_DEEP at top → INDIGO_LIFT at bottom).
_GRADIENT = None


def _build_gradient() -> np.ndarray:
    col = np.zeros((H, 1, 3), dtype=np.uint8)
    for y in range(H):
        f = y / (H - 1)
        col[y, 0] = [int(INDIGO_DEEP[c] + (INDIGO_LIFT[c] - INDIGO_DEEP[c]) * f)
                     for c in range(3)]
    return np.repeat(col, W, axis=1)


def brand_gradient(canvas: np.ndarray):
    """Fill the canvas with the vertical indigo brand gradient."""
    global _GRADIENT
    if _GRADIENT is None or _GRADIENT.shape[:2] != canvas.shape[:2]:
        _GRADIENT = _build_gradient()
    canvas[:] = _GRADIENT


def draw_wave_field(canvas: np.ndarray, t: float, intensity: float = 1.0):
    """
    The swept contour-line motif from assets/1.png: a fan of fine curves
    sweeping across the field. Rendered in dim violet over the indigo ground;
    animates by phase-shifting the sweep. Cheap — polylines over a sparse grid.
    """
    cx, cy = int(W * 0.80), int(H * 0.42)   # off-centre focus, like the reference
    n_lines = 22
    step = _s(40)
    for i in range(n_lines):
        spread = 0.6 + i * 0.12                     # each line fans out further
        phase = t * 0.25 + i * 0.14
        pts = []
        for x in range(0, W + step, step):
            # curve that bows toward the focus and drifts with phase
            dx = (x - cx) / W
            y = cy + int(np.sin(dx * 3.1 + phase) * (120 + 60 * spread) * spread * 0.5 * S)
            pts.append((x, y))
        shade = 0.20 + 0.5 * (i / n_lines)
        col = tuple(int(INDIGO[c] + (VIOLET_LO[c] - INDIGO[c]) * shade * intensity)
                    for c in range(3))
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, col, _th(1), cv2.LINE_AA)


def _rot(px, py, cx, cy, ang):
    c, s = np.cos(ang), np.sin(ang)
    return (int(cx + (px - cx) * c - (py - cy) * s),
            int(cy + (px - cx) * s + (py - cy) * c))


def draw_dragon_mark(canvas: np.ndarray, cx: int, cy: int, r: int,
                     color: tuple, bg: Optional[tuple] = None, glow: bool = False):
    """
    The Dragonwing dragon-head glyph from assets/2.png: a crescent sweep
    enclosing a hooked head with a negative-space eye. Built from filled
    ellipses/polygons so it scales cleanly on the canvas.
    """
    if glow:
        halo = canvas.copy()
        cv2.circle(halo, (cx, cy), int(r * 1.15), VIOLET, -1, cv2.LINE_AA)
        cv2.addWeighted(halo, 0.18, canvas, 0.82, 0, canvas)

    bg = bg if bg is not None else INDIGO_DEEP

    # Outer body: a thick crescent (large disc with an offset disc carved out).
    cv2.circle(canvas, (cx, cy), r, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, (int(cx - r * 0.42), int(cy - r * 0.30)),
               int(r * 0.86), bg, -1, cv2.LINE_AA)

    # Head/beak hook sweeping up and to the right.
    beak = np.array([
        _rot(cx + r * 0.10, cy - r * 0.86, cx, cy, 0),
        _rot(cx + r * 0.95, cy - r * 0.55, cx, cy, 0),
        _rot(cx + r * 0.55, cy - r * 0.18, cx, cy, 0),
        _rot(cx + r * 0.18, cy - r * 0.40, cx, cy, 0),
    ], dtype=np.int32)
    cv2.fillConvexPoly(canvas, beak, color, cv2.LINE_AA)

    # Flared tail curl sweeping down and to the right — the official-mark
    # geometry that closes the head + body into a near-circular emblem.
    tail_th = max(2, int(r * 0.17))
    cv2.ellipse(canvas, (int(cx + r * 0.12), int(cy + r * 0.44)),
                (int(r * 0.74), int(r * 0.54)), 0, -34, 122,
                color, tail_th, cv2.LINE_AA)

    # Inner curl (the eye socket), then the negative-space eye.
    cv2.circle(canvas, (int(cx + r * 0.30), int(cy + r * 0.12)),
               int(r * 0.34), bg, -1, cv2.LINE_AA)
    cv2.circle(canvas, (int(cx + r * 0.34), int(cy + r * 0.06)),
               int(r * 0.14), color, -1, cv2.LINE_AA)


def put_centered(img, text, cy, scale, color, thickness=1):
    """Centered text. `scale`/`thickness` are in logical 720p units (×S here);
    `cy` is a native-canvas coordinate."""
    scale *= S
    thickness = _th(thickness)
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    x = (img.shape[1] - tw) // 2
    cv2.putText(img, text, (x, cy), FONT, scale, color, thickness, cv2.LINE_AA)


def put_tracked(img, text, cx, cy, scale, color, thickness, spacing):
    """Centered text with manual letter-spacing — used for the DRAGONWING wordmark.
    `scale`/`thickness` are logical 720p units (×S here); `cx`/`cy`/`spacing`
    are native-canvas pixels."""
    scale *= S
    thickness = _th(thickness)
    sizes = [cv2.getTextSize(ch, FONT, scale, thickness)[0][0] for ch in text]
    total = sum(sizes) + spacing * (len(text) - 1)
    x = cx - total // 2
    for ch, w in zip(text, sizes):
        cv2.putText(img, ch, (x, cy), FONT, scale, color, thickness, cv2.LINE_AA)
        x += w + spacing


def draw_wordmark(canvas: np.ndarray, cx: int, cy: int, scale: float = 1.0):
    """DRAGONWING tracked wordmark + DOORBELL subtitle lockup."""
    put_tracked(canvas, "DRAGONWING", cx, cy, 0.95 * scale, INK, 2, _s(10 * scale))
    put_tracked(canvas, "D O O R B E L L", cx, cy + _s(34 * scale),
                0.5 * scale, VIOLET_HI, 1, _s(2 * scale))


def legibility_halo(canvas: np.ndarray, cx: int, cy: int, w: int, h: int,
                    strength: float = 0.72):
    """
    The one permitted 'shadow' (DESIGN.md Legibility-Halo Rule): a soft
    Deep-Indigo pad behind critical text or the mark when it sits over the
    camera feed or the wave field. Elliptical radial falloff, never a hard
    rectangle — for contrast, never decoration.
    """
    x0, y0 = max(0, cx - w // 2), max(0, cy - h // 2)
    x1, y1 = min(W, cx + w // 2), min(H, cy + h // 2)
    if x1 <= x0 or y1 <= y0:
        return
    region = canvas[y0:y1, x0:x1].astype(np.float32)
    rh, rw = region.shape[:2]

    # Elliptical alpha: 1 at centre, feathering to 0 at the edge.
    yy, xx = np.ogrid[0:rh, 0:rw]
    nx = (xx - rw / 2) / (rw / 2)
    ny = (yy - rh / 2) / (rh / 2)
    d = np.sqrt(nx * nx + ny * ny)
    alpha = np.clip(1.0 - d, 0.0, 1.0) ** 1.6 * strength
    alpha = alpha[..., None]

    pad = np.array(INDIGO_DEEP, dtype=np.float32)
    region = region * (1.0 - alpha) + pad * alpha
    canvas[y0:y1, x0:x1] = region.astype(np.uint8)


def draw_access_banner(canvas: np.ndarray, text: str, color: tuple, t: float):
    """
    The signature verdict component (DESIGN.md Access Banner + Flat-Verdict
    Rule): a full-width, flat, square-cornered Deep-Indigo block carrying the
    verdict in display-scale type — the loudest thing on screen. No gradient,
    glow, or halo on the block; its certainty is pure solid color and size.
    The type pulses subtly on appearance.
    """
    # Low band, clear of the face: sits in the lower third, above the status bar.
    band_h = _s(104)
    y1 = H - _s(96)
    y0 = y1 - band_h

    # Flat Deep-Indigo block, full width, square corners.
    canvas[y0:y1, 0:W] = INDIGO_DEEP
    # Thin signal-color rules top and bottom seat the block without softening it.
    cv2.line(canvas, (0, y0), (W, y0), color, _th(3), cv2.LINE_AA)
    cv2.line(canvas, (0, y1), (W, y1), color, _th(3), cv2.LINE_AA)

    # Display-scale verdict, subtly pulsing on appearance.
    pulse = 1.0 - 0.5 * np.exp(-t * 5.0) * abs(np.sin(t * 9.0))
    scale = 1.7 * pulse * S
    thickness = _th(4)
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    tx = (W - tw) // 2
    ty = y0 + (band_h + th) // 2
    cv2.putText(canvas, text, (tx, ty), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_face_overlays(canvas: np.ndarray, faces: list, color: tuple,
                        sx: float, sy: float, ox: int, oy: int,
                        per_face: bool = False):
    """
    Draw bounding box + 5 key-points. faces are in inference-frame coords;
    sx/sy/ox/oy map them onto the letterboxed display rectangle. The dense
    106-landmark cloud is intentionally skipped — it costs 100+ draws per
    face per frame and adds visual noise without aiding the demo read.

    per_face=True colors each box by its own recognition verdict (green when
    face.rec_name is set, red otherwise) and labels it with the name + score,
    so a mixed crowd reads correctly. per_face=False uses the single `color`
    for every box (used for the neutral/enroll states).
    """
    for face in faces:
        x1, y1, x2, y2 = face.bbox
        rec_name = getattr(face, 'rec_name', None) if per_face else None
        if per_face:
            box_c = UNLOCK if rec_name else LOCK
        else:
            box_c = color
        px1, py1 = int(x1 * sx + ox), int(y1 * sy + oy)
        px2, py2 = int(x2 * sx + ox), int(y2 * sy + oy)
        cv2.rectangle(canvas, (px1, py1), (px2, py2), box_c, _th(2))
        if face.kps is not None:
            for x, y in face.kps:
                cv2.circle(canvas, (int(x * sx + ox), int(y * sy + oy)), _s(4), box_c, -1)

        if per_face:
            # Name + similarity above the box: 'raul 0.62' when matched, else 'unknown'.
            sim = getattr(face, 'rec_sim', 0.0)
            tag = f"{rec_name} {sim:.2f}" if rec_name else "unknown"
            (tw, th), _ = cv2.getTextSize(tag, FONT, 0.6 * S, _th(2))
            ly = max(py1 - _s(8), th + _s(4))
            cv2.rectangle(canvas, (px1, ly - th - _s(6)), (px1 + tw + _s(10), ly + _s(4)),
                          INDIGO_DEEP, -1)
            cv2.putText(canvas, tag, (px1 + _s(5), ly),
                        FONT, 0.6 * S, box_c, _th(2), cv2.LINE_AA)
        else:
            # confidence score
            score = getattr(face, 'det_score', None)
            if score is not None:
                cv2.putText(canvas, f"{score:.2f}",
                            (px1, py1 - _s(6)),
                            FONT, 0.45 * S, box_c, _th(1), cv2.LINE_AA)


def draw_loading(canvas: np.ndarray, t: float):
    brand_gradient(canvas)
    draw_wave_field(canvas, t, intensity=0.7)
    cx, cy = W // 2, H // 2 - _s(40)
    pulse = 0.5 + 0.5 * np.sin(t * 2.2)
    # Real Dragonwing mark with a pulsing violet legibility glow behind it.
    halo = canvas.copy()
    cv2.circle(halo, (cx, cy), _s(int(100 + 10 * pulse)), VIOLET, -1, cv2.LINE_AA)
    cv2.addWeighted(halo, 0.18, canvas, 0.82, 0, canvas)
    draw_logo_mark(canvas, cx, cy, _s(188))

    # Orbiting progress arc around the mark
    ang = (t * 2.0) % (2 * np.pi)
    deg = int(np.degrees(ang))
    cv2.ellipse(canvas, (cx, cy), (_s(120), _s(120)), 0, deg, deg + 90,
                VIOLET_HI, _th(3), cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), _s(120), VIOLET_LO, _th(1), cv2.LINE_AA)

    dots = "." * (int(t * 2) % 4)
    glow_c = tuple(int(MIST[c] + (VIOLET_HI[c] - MIST[c]) * pulse) for c in range(3))
    legibility_halo(canvas, cx, cy + _s(215), _s(440), _s(170), strength=0.6)
    draw_wordmark(canvas, cx, cy + _s(180), 1.1)
    put_centered(canvas, f"Initializing neural engines{dots}", cy + _s(250), 0.6, glow_c, 1)
    put_centered(canvas, "InsightFace buffalo_sc    HTP v75", cy + _s(282), 0.46, VIOLET_LO, 1)


def draw_standby(canvas: np.ndarray, t: float, db_count: int):
    brand_gradient(canvas)
    draw_wave_field(canvas, t, intensity=1.0)

    # Large dragon-mark watermark anchored off-centre, echoing the reference art.
    # The real mark, blitted faint so it reads as a background watermark, not chrome.
    wm = canvas.copy()
    draw_logo_mark(wm, int(W * 0.5), int(H * 0.44), _s(300))
    cv2.addWeighted(wm, 0.12, canvas, 0.88, 0, canvas)

    cx, cy = W // 2, H // 2
    pulse = 0.5 + 0.5 * np.sin(t * 2.4)

    # Pulsing violet ring enclosing a lock glyph
    r_outer = _s(104 + 14 * pulse)
    ring_c = tuple(int(VIOLET_LO[c] + (VIOLET[c] - VIOLET_LO[c]) * pulse) for c in range(3))
    cv2.circle(canvas, (cx, cy), r_outer, ring_c, _th(2), cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), r_outer - _s(16), VIOLET_LO, _th(1), cv2.LINE_AA)

    # Padlock glyph, violet body on indigo
    lw, lh = _s(46), _s(36)
    lx, ly = cx - lw // 2, cy - lh // 2 + _s(8)
    cv2.rectangle(canvas, (lx, ly), (lx + lw, ly + lh), VIOLET, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (lx, ly), (lx + lw, ly + lh), VIOLET_HI, _th(1), cv2.LINE_AA)
    shackle_c = tuple(int(VIOLET[c] + (VIOLET_HI[c] - VIOLET[c]) * pulse) for c in range(3))
    cv2.ellipse(canvas, (cx, ly), (lw // 3, lh // 2), 0, 180, 360, shackle_c, _th(4), cv2.LINE_AA)
    cv2.circle(canvas, (cx, ly + lh // 2 - _s(2)), _s(5), INDIGO_DEEP, -1, cv2.LINE_AA)

    legibility_halo(canvas, cx, cy + _s(142), _s(430), _s(110), strength=0.62)
    put_centered(canvas, "DOORBELL READY", cy + _s(128), 0.95, INK, 2)
    put_centered(canvas, "Awaiting visitor", cy + _s(162), 0.56, MIST, 1)

    # DB badge, top-right — sits below the HUD bar to avoid the FPS badge
    badge = f"{db_count} enrolled" if db_count else "no faces enrolled"
    (tw, _), _ = cv2.getTextSize(badge, FONT, 0.5 * S, 1)
    cv2.circle(canvas, (W - tw - _s(30), _s(50)), _s(4), UNLOCK if db_count else VIOLET_LO, -1, cv2.LINE_AA)
    cv2.putText(canvas, badge, (W - tw - _s(18), _s(55)), FONT, 0.5 * S, MIST, _th(1), cv2.LINE_AA)


def composite_frame(canvas: np.ndarray, frame: np.ndarray):
    """Letterbox-fit frame into canvas over the indigo ground, return (sx, sy, ox, oy)."""
    fh, fw = frame.shape[:2]
    scale = min(W / fw, H / fh)
    dw, dh = int(fw * scale), int(fh * scale)
    ox, oy = (W - dw) // 2, (H - dh) // 2
    brand_gradient(canvas)
    canvas[oy:oy + dh, ox:ox + dw] = cv2.resize(frame, (dw, dh))
    return scale, scale, ox, oy


def draw_camera_state(canvas: np.ndarray, frame: np.ndarray, faces: list,
                       inf_w: int, inf_h: int,
                       border: tuple, label: str, label_color: tuple,
                       reg_progress: Optional[tuple] = None,
                       per_face: bool = False):
    scale, _, ox, oy = composite_frame(canvas, frame)

    fh, fw = frame.shape[:2]
    # Map inference-frame coords (inf_w×inf_h) onto the *letterboxed* display
    # rectangle, not the full canvas: through the camera frame, then the same
    # scale+offset the frame was composited with. Keeps the box on the face.
    dw, dh = fw * scale, fh * scale
    lm_sx = dw / inf_w
    lm_sy = dh / inf_h
    draw_face_overlays(canvas, faces, border, lm_sx, lm_sy, ox, oy, per_face=per_face)

    # Corner brackets in the state color (violet when neutral, green/red/amber per state)
    fx2, fy2 = ox + int(fw * scale), oy + int(fh * scale)
    bl = _s(54)
    bt = _th(3)
    for bx, by, sx2, sy2 in [(ox, oy, 1, 1), (fx2, oy, -1, 1),
                             (ox, fy2, 1, -1), (fx2, fy2, -1, -1)]:
        cv2.line(canvas, (bx, by), (bx + sx2 * bl, by), border, bt, cv2.LINE_AA)
        cv2.line(canvas, (bx, by), (bx, by + sy2 * bl), border, bt, cv2.LINE_AA)

    # Bottom status bar — indigo wash + logo lockup + state label. Seated above
    # the bottom hint strip so the logo and verdict never sit under the controls.
    bar_h = _s(84)
    panel_y = H - _s(22) - bar_h          # clear of the hint bar (H - 22 tall)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, panel_y), (W, H - _s(22)), INDIGO_DEEP, -1)
    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)
    cv2.line(canvas, (0, panel_y), (W, panel_y), VIOLET_LO, _th(1), cv2.LINE_AA)
    draw_logo_mark(canvas, _s(54), panel_y + bar_h // 2, _s(56))

    # Status dot + label, left-aligned after the mark — the sole verdict now.
    cv2.circle(canvas, (_s(108), panel_y + bar_h // 2), _s(7), label_color, -1, cv2.LINE_AA)
    cv2.putText(canvas, label.strip(), (_s(126), panel_y + bar_h // 2 + _s(9)),
                FONT, 1.0 * S, label_color, _th(2), cv2.LINE_AA)

    if reg_progress:
        n, total = reg_progress
        count_txt = f"{n}/{total}"
        (ctw, _), _ = cv2.getTextSize(count_txt, FONT, 0.5 * S, _th(1))
        right_pad = _s(30) + ctw          # room for the count after the bar
        bar_x = _s(620)
        bar_w = W - bar_x - right_pad
        bar_y = panel_y + bar_h // 2 - _s(6)
        bh = _s(12)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bh),
                      INDIGO_LIFT, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bh),
                      VIOLET_LO, _th(1), cv2.LINE_AA)
        fill = int(bar_w * n / total) if total else 0
        if fill > 0:
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill, bar_y + bh),
                          VIOLET, -1, cv2.LINE_AA)
        cv2.putText(canvas, count_txt, (bar_x + bar_w + _s(10), bar_y + bh),
                    FONT, 0.5 * S, MIST, _th(1), cv2.LINE_AA)


def draw_qr_authorized(canvas: np.ndarray, elapsed: float, name: str):
    """Phase 1 (0-5s): QR accepted — violet burst + dragon mark + AUTHORIZED."""
    brand_gradient(canvas)
    draw_wave_field(canvas, elapsed, intensity=0.6)
    cx, cy = W // 2, H // 2 - _s(20)

    # Expanding violet ring burst
    for i in range(4):
        phase = elapsed * 2.2 - i * 0.45
        frac = phase % 1.0
        r = int((_s(70) + _s(150) * frac) * (1.0 - frac * 0.4))
        alpha = max(0.0, 1.0 - frac)
        color = tuple(int(INDIGO[c] + (VIOLET_HI[c] - INDIGO[c]) * alpha) for c in range(3))
        if r > 0:
            cv2.circle(canvas, (cx, cy), r, color, max(1, _th(4 * alpha)), cv2.LINE_AA)

    # Dragon mark bursts to full presence — the real mark with a pulsing glow.
    pulse = 0.6 + 0.4 * abs(np.sin(elapsed * 5))
    halo = canvas.copy()
    cv2.circle(halo, (cx, cy), int(_s(96) * pulse), VIOLET, -1, cv2.LINE_AA)
    cv2.addWeighted(halo, 0.20, canvas, 0.80, 0, canvas)
    mark_r = int(_s(158) * pulse)
    draw_logo_mark(canvas, cx, cy, mark_r)

    # Green confirmation check, ring'd — the one semantic (unlock) accent
    scale_in = min(1.0, elapsed * 2.5)
    cv2.circle(canvas, (cx, cy), mark_r // 2 + _s(14), UNLOCK, _th(2), cv2.LINE_AA)

    legibility_halo(canvas, cx, cy + _s(180), _s(460), _s(120), strength=0.55)
    put_centered(canvas, "AUTHORIZED", cy + _s(150), 1.6 * scale_in, INK, 3)
    put_centered(canvas, name, cy + _s(196), 0.72 * scale_in, VIOLET_HI, 1)


def draw_qr_countdown(canvas: np.ndarray, elapsed: float, name: str, countdown: int):
    """Phase 2 (5-10s): get-ready countdown before capture begins."""
    brand_gradient(canvas)
    draw_wave_field(canvas, elapsed + 3.0, intensity=0.6)
    cx, cy = W // 2, H // 2 - _s(20)

    # Depleting violet sweep ring
    frac = max(0.0, min(1.0, 1.0 - elapsed / 5.0))
    cv2.ellipse(canvas, (cx, cy), (_s(118), _s(118)), -90, 0, 360, INDIGO_LIFT, _th(8), cv2.LINE_AA)
    cv2.ellipse(canvas, (cx, cy), (_s(118), _s(118)), -90, 0, int(360 * frac), VIOLET, _th(8), cv2.LINE_AA)

    # Big countdown digit
    digit = str(max(0, countdown))
    txt_scale = 4.6 + 0.6 * abs(np.sin(elapsed * 3.14))
    legibility_halo(canvas, cx, cy + _s(20), _s(240), _s(240), strength=0.5)
    put_centered(canvas, digit, cy + _s(42), txt_scale, INK, 6)

    legibility_halo(canvas, cx, cy + _s(185), _s(420), _s(110), strength=0.55)
    put_centered(canvas, "GET READY", cy + _s(168), 0.9, VIOLET_HI, 2)
    put_centered(canvas, f"Enrolling  {name}", cy + _s(206), 0.56, MIST, 1)


def draw_enroll_gate(canvas: np.ndarray, frame: np.ndarray, faces: list,
                     inf_w: int, inf_h: int, name: str, phase: str,
                     countdown: Optional[int] = None,
                     reg_progress: Optional[tuple] = None,
                     blocked: bool = False, t: float = 0.0):
    """Live feed with a centered 'O' the visitor places their face in.

    Everything outside the O is dimmed to brand indigo so the ring is the clear
    target; only a face inside the O is captured (see RegistrationDriver). Used
    for both the get-ready countdown (phase='countdown') and the capture itself
    (phase='capture'). Falls back to no overlay-only drawing — the caller guards
    the frame-less case.
    """
    scale, _, ox, oy = composite_frame(canvas, frame)
    fh, fw = frame.shape[:2]
    lm_sx = fw * scale / inf_w
    lm_sy = fh * scale / inf_h

    # Gate ellipse in display coords (same transform draw_face_overlays uses).
    gcx, gcy, grx, gry = gate_ellipse_px(inf_w, inf_h)
    dcx, dcy = int(gcx * lm_sx + ox), int(gcy * lm_sy + oy)
    drx, dry = int(grx * lm_sx), int(gry * lm_sy)

    # Dim everything outside the O down to brand indigo. One addWeighted through
    # an inverted elliptical mask keeps the live face crisp inside the ring.
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(mask, (dcx, dcy), (drx, dry), 0, 0, 360, 255, -1)
    fill = np.empty_like(canvas)
    fill[:] = INDIGO_DEEP
    dimmed = cv2.addWeighted(canvas, 0.28, fill, 0.72, 0)
    outside = mask == 0
    canvas[outside] = dimmed[outside]

    # Faces correctly inside the O (echo exactly what will be captured).
    inside = [f for f in faces if face_in_gate(f, inf_w, inf_h)]
    one_ready = len(inside) == 1 and not blocked

    # The ring itself — pulsing violet, green when exactly one face is ready.
    pulse = 0.5 + 0.5 * np.sin(t * 3.0)
    ring_c = UNLOCK if one_ready else tuple(
        int(VIOLET_LO[c] + (VIOLET_HI[c] - VIOLET_LO[c]) * pulse) for c in range(3))
    cv2.ellipse(canvas, (dcx, dcy), (drx, dry), 0, 0, 360, ring_c, _th(4), cv2.LINE_AA)
    cv2.ellipse(canvas, (dcx, dcy), (drx + _s(10), dry + _s(10)), 0, 0, 360,
                VIOLET_LO, _th(1), cv2.LINE_AA)

    draw_face_overlays(canvas, inside, ring_c, lm_sx, lm_sy, ox, oy)

    # Guidance under the O.
    ty = min(dcy + dry + _s(52), H - _s(78))
    legibility_halo(canvas, dcx, ty - _s(14), _s(520), _s(150), strength=0.5)
    if blocked:
        put_centered(canvas, "ONE PERSON AT A TIME", ty, 0.9, DETECT, 2)
        put_centered(canvas, "Only one face in the ring, please", ty + _s(34), 0.5, MIST, 1)
    elif phase == "countdown":
        if countdown is not None:
            # Digit seated just inside the top of the ring so the title bar never clips it.
            put_centered(canvas, str(max(0, countdown)), dcy - dry + _s(70),
                         1.8 + 0.4 * pulse, INK, 3)
        put_centered(canvas, "PLACE YOUR FACE IN THE RING", ty, 0.86, INK, 2)
        put_centered(canvas, f"Enrolling  {name}", ty + _s(34), 0.5, VIOLET_HI, 1)
    else:  # capture
        head = "SLOWLY TURN YOUR HEAD" if one_ready else "PLACE YOUR FACE IN THE RING"
        put_centered(canvas, head, ty, 0.86, INK, 2)
        if reg_progress:
            n, total = reg_progress
            bw, bh = _s(360), _s(12)
            bx, by = dcx - bw // 2, ty + _s(22)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), INDIGO_LIFT, -1, cv2.LINE_AA)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), VIOLET_LO, _th(1), cv2.LINE_AA)
            fillw = int(bw * n / total) if total else 0
            if fillw > 0:
                cv2.rectangle(canvas, (bx, by), (bx + fillw, by + bh), VIOLET, -1, cv2.LINE_AA)
            put_centered(canvas, f"{n}/{total}  {name}", by + bh + _s(22), 0.46, MIST, 1)


def draw_hud(canvas: np.ndarray, state: State, status: str, db_count: int,
             fps: float = 0.0, backend: str = "cpu",
             idle_timeout: float = 5.0, data_ttl: float = 120.0):
    hint = (f"B CPU/NPU   G guest   C clear   [ ] idle {idle_timeout:.0f}s   "
            f"- = store {data_ttl:.0f}s   ESC standby   Q quit")

    # Top bar — demo title: the live Board + OS this binary runs on (auto-detected),
    # so the same file self-labels as IQ8+Ubuntu / IQ8+QLI / Ventuno Q+Ubuntu.
    # A touch bigger than mono and in the Simplex face so it reads as a title, not
    # a state name. Backend/FPS live in the telemetry panel; the per-state verdict
    # lives in the status strip, so the top stays clean.
    cv2.rectangle(canvas, (0, 0), (W, _s(28)), INDIGO_DEEP, -1)
    cv2.line(canvas, (0, _s(28)), (W, _s(28)), VIOLET_LO, _th(1), cv2.LINE_AA)
    cv2.putText(canvas, PLATFORM_LABEL, (_s(12), _s(20)),
                FONT, 0.56 * S, INK, _th(1), cv2.LINE_AA)

    # Bottom hint bar
    cv2.rectangle(canvas, (0, H - _s(22)), (W, H), INDIGO_DEEP, -1)
    cv2.putText(canvas, hint, (_s(10), H - _s(7)), FONT_MONO, 0.95 * S, VIOLET_LO, _th(1), cv2.LINE_AA)


# ── System stats monitor ───────────────────────────────────────────────────────

class StatsMonitor(threading.Thread):
    """Polls CPU % and RAM every 0.5 s in a background thread."""
    def __init__(self):
        super().__init__(daemon=True)
        self._lock = threading.Lock()
        self._cpu_per_core: list = []
        self._cpu_total: float = 0.0
        self._ram_pct: float = 0.0
        self._ram_used_mb: float = 0.0
        self._ram_total_mb: float = 0.0
        self.running = True

    def run(self):
        import psutil
        psutil.cpu_percent(percpu=True)   # discard first sample (always 0)
        while self.running:
            time.sleep(0.5)
            cores = psutil.cpu_percent(percpu=True)
            total = sum(cores) / len(cores) if cores else 0.0
            m = psutil.virtual_memory()
            with self._lock:
                self._cpu_per_core = cores
                self._cpu_total = total
                self._ram_pct = m.percent
                self._ram_used_mb  = m.used  / 1024 / 1024
                self._ram_total_mb = m.total / 1024 / 1024

    def snapshot(self):
        with self._lock:
            return (list(self._cpu_per_core), self._cpu_total,
                    self._ram_pct, self._ram_used_mb, self._ram_total_mb)

    def stop(self):
        self.running = False


def _bar(w: int, pct: float, color: tuple) -> np.ndarray:
    """Horizontal meter: indigo track, colored fill."""
    bar = np.empty((_s(8), w, 3), dtype=np.uint8)
    bar[:] = INDIGO_LIFT
    fill = max(1, int(w * min(pct, 100.0) / 100.0))
    bar[:, :fill] = color
    return bar


def draw_stats_panel(canvas: np.ndarray,
                     cpu_cores: list, cpu_total: float,
                     ram_pct: float, ram_used_mb: float, ram_total_mb: float,
                     cpu_fps: float, npu_fps: float,
                     active_backend: str, npu_ready: bool):
    """
    Translucent indigo telemetry panel, bottom-right:
      - CPU / NPU FPS rows (active one highlighted in violet)
      - Per-core CPU bars (load heat is a meter, kept green→red)
      - RAM bar
    """
    PW, PH = _s(262), _s(212)
    px = W - PW - _s(12)
    py = H - PH - _s(34)           # above the hint bar
    pad = _s(8)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (px - pad, py - pad), (px + PW + pad, py + PH + pad),
                  INDIGO_DEEP, -1)
    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)
    cv2.rectangle(canvas, (px - pad, py - pad), (px + PW + pad, py + PH + pad),
                  VIOLET_LO, _th(1), cv2.LINE_AA)

    y = py + _s(16)
    F = cv2.FONT_HERSHEY_PLAIN
    FS = 1.1 * S

    # ── Backend FPS rows ─────────────────────────────────────────────────────
    for label, fps, is_active, ready in [
        ("CPU", cpu_fps, active_backend == "cpu", True),
        ("NPU", npu_fps, active_backend == "npu", npu_ready),
    ]:
        if is_active:
            cv2.rectangle(canvas, (px - _s(4), y - _s(12)), (px + PW + _s(4), y + _s(6)),
                          INDIGO_LIFT, -1)
            cv2.rectangle(canvas, (px - _s(4), y - _s(12)), (px - _s(2), y + _s(6)), VIOLET, -1)

        if not ready:
            fps_str, col = "loading", VIOLET_LO
        else:
            fps_str = f"{fps:5.1f} FPS"
            col = VIOLET_HI if is_active else MIST

        if is_active:
            tri = np.array([(px, y - _s(8)), (px, y), (px + _s(7), y - _s(4))], np.int32)
            cv2.fillConvexPoly(canvas, tri, VIOLET, cv2.LINE_AA)
        cv2.putText(canvas, label, (px + _s(14), y), F, FS, col, _th(1), cv2.LINE_AA)
        (sw, _), _ = cv2.getTextSize(fps_str, F, FS, _th(1))
        cv2.putText(canvas, fps_str, (px + PW - sw, y), F, FS, col, _th(1), cv2.LINE_AA)
        y += _s(20)

    y += _s(6)
    cv2.line(canvas, (px, y), (px + PW, y), VIOLET_LO, _th(1), cv2.LINE_AA)
    y += _s(12)

    # ── Per-core CPU bars ─────────────────────────────────────────────────────
    cv2.putText(canvas, f"CPU CORES   avg {cpu_total:.0f}%",
                (px, y), F, 0.95 * S, MIST, _th(1), cv2.LINE_AA)
    y += _s(12)

    cols_n = 4
    gap = _s(4)
    bar_w = (PW - (cols_n - 1) * gap) // cols_n
    for i, pct in enumerate(cpu_cores):
        bx = px + (i % cols_n) * (bar_w + gap)
        by = y + (i // cols_n) * _s(18)
        heat = min(255, int(pct * 2.55))
        bar_col = (0, 255 - heat, heat)       # green→red as load rises
        canvas[by:by + _s(8), bx:bx + bar_w] = _bar(bar_w, pct, bar_col)
        cv2.putText(canvas, f"C{i}", (bx, by + _s(18)), F, 0.75 * S, VIOLET_LO, _th(1), cv2.LINE_AA)

    row_count = (len(cpu_cores) + cols_n - 1) // cols_n
    y += row_count * _s(18) + _s(10)

    cv2.line(canvas, (px, y), (px + PW, y), VIOLET_LO, _th(1), cv2.LINE_AA)
    y += _s(12)

    # ── RAM bar ───────────────────────────────────────────────────────────────
    cv2.putText(canvas, f"RAM   {ram_used_mb:.0f} / {ram_total_mb:.0f} MB   {ram_pct:.0f}%",
                (px, y), F, 0.95 * S, MIST, _th(1), cv2.LINE_AA)
    y += _s(10)
    ram_col = (0, int(255 * (1 - ram_pct / 100)), int(255 * ram_pct / 100))
    canvas[y:y + _s(8), px:px + PW] = _bar(PW, ram_pct, ram_col)


# ── Main ──────────────────────────────────────────────────────────────────────

INF_W, INF_H = 320, 240  # inference frame size on board (aarch64 CPU)


def main():
    parser = argparse.ArgumentParser(description="Doorbell demo — InsightFace face recognition")
    parser.add_argument("--backend", choices=["cpu", "npu"], default="cpu",
                        help="Starting backend (default: cpu). Press B to toggle live.")
    args = parser.parse_args()

    app = AppState()
    camera = CameraThread()
    camera.start()

    # Track readiness of each engine separately
    _ready = {"cpu": False, "npu": False}

    def on_cpu_ready():
        _ready["cpu"] = True
        with app.lock:
            if app.state == State.LOADING:
                app.state = State.STANDBY
                app.status_msg = "SYSTEM READY"
        print("[CPU] InsightFace ready")

    def on_npu_ready():
        _ready["npu"] = True
        with app.lock:
            if app.state == State.LOADING:
                app.state = State.STANDBY
                app.status_msg = "SYSTEM READY"
        print("[NPU] in-process HTP engine ready")

    cpu_engine = FaceEngine(on_ready=on_cpu_ready)
    npu_engine = NpuFaceEngine(on_ready=on_npu_ready)
    stats = StatsMonitor()
    cpu_engine.start()
    npu_engine.start()
    stats.start()

    # Mutable backend state — toggled by B key
    active = {"backend": args.backend}

    def active_engine():
        return npu_engine if active["backend"] == "npu" else cpu_engine

    def finish_registration(name: str, embeddings: list):
        with app.lock:
            app.add_face(name, embeddings)
            app.state = State.RECOGNIZED
            app.last_match = name
            app.status_msg = f"Registered! Welcome, {name}"
        print(f"[REG] Done — {name} added with {len(embeddings)} embeddings (TTL {app.data_ttl:.0f}s)")

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    t0 = time.time()
    qr_detector = QRDetector()
    inf_frame_cache: Optional[np.ndarray] = None
    verdict_since = {"state": None, "at": 0.0}   # pulse clock for the access banner

    # Fill the HDMI panel. This board's OpenCV uses the Qt5 HighGUI backend,
    # which by default (WINDOW_GUI_EXPANDED) wraps the image in a toolbar with
    # pan/zoom buttons AND makes WND_PROP_FULLSCREEN unreliable. Creating the
    # window with WINDOW_GUI_NORMAL drops that toolbar and lets Qt's own
    # showFullScreen() take effect. A fullscreen request made *before* the
    # window is mapped is dropped, so we map first (one imshow + waitKey pumps
    # the Qt event loop and realizes it), then assert fullscreen, and re-assert
    # once more on the first loop iteration below as a safety net.
    _gui_normal = getattr(cv2, "WINDOW_GUI_NORMAL", 0x10)
    cv2.namedWindow("Dragonwing Doorbell", cv2.WINDOW_NORMAL | _gui_normal)
    cv2.resizeWindow("Dragonwing Doorbell", W, H)
    cv2.moveWindow("Dragonwing Doorbell", 0, 0)
    cv2.imshow("Dragonwing Doorbell", canvas)   # map it
    cv2.waitKey(1)                              # pump the compositor
    cv2.setWindowProperty("Dragonwing Doorbell", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    _fs_frames = 0                              # re-assert over the first frames in the loop

    print("=== DRAGONWING DOORBELL ===")
    print("B=CPU/NPU  G=guest  C=clear  [ ]=idle  - ==store  ESC=standby  Q=quit")
    print("===========================")

    while True:
        t = time.time() - t0
        now = time.time()
        eng = active_engine()
        backend = active["backend"]

        # Feed latest frame to active engine (downscaled for speed)
        raw_frame = camera.get_frame()
        if raw_frame is not None:
            inf_frame = cv2.resize(raw_frame, (INF_W, INF_H))
            eng.submit(inf_frame)
            inf_frame_cache = inf_frame

        faces = eng.get_faces()

        # QR detection — runs on every frame, triggers registration automatically
        qr_name = None
        if raw_frame is not None:
            qr_name = qr_detector.detect(raw_frame)
            if qr_name:
                print(f"[QR] Detected LinkedIn profile: {qr_name}")

        with app.lock:
            state = app.state
            # QR triggers authorized animation from any active state
            if qr_name and state not in (State.LOADING, State.REGISTRATION, State.QR_AUTHORIZED):
                app.state = State.QR_AUTHORIZED
                app.reg_name = qr_name
                app.qr_authorized_at = now
                app.reg_blocked = False
                app.status_msg = f"QR authorized: {qr_name}"
                state = State.QR_AUTHORIZED

            # Auto state transitions
            if state == State.QR_AUTHORIZED:
                elapsed_qr = now - app.qr_authorized_at
                if elapsed_qr >= 10.0:
                    app.state = State.REGISTRATION
                    app.reg_frames = 0
                    app.reg_start_at = now
                    app.reg_blocked = False
                    app.status_msg = f"Registering {app.reg_name} — look at camera"
                    state = State.REGISTRATION
                    reg_thread = RegistrationDriver(app, eng, finish_registration)
                    reg_thread.start()
            elif state not in (State.LOADING, State.REGISTRATION):
                if faces:
                    app.last_face_time = now
                    if state == State.STANDBY:
                        app.state = State.DETECTION
                        app.status_msg = "Face detected — recognizing..."
                        state = State.DETECTION
                    if state in (State.DETECTION, State.RECOGNIZED, State.UNKNOWN):
                        # Recognize every face independently so each box can show
                        # its own verdict (name + green/red). The door decision
                        # still follows the closest (largest) face only.
                        recognize_all(faces, app.face_db)
                        primary = largest_face(faces)
                        match = getattr(primary, 'rec_name', None) if primary else None
                        if match:
                            app.state = State.RECOGNIZED
                            app.last_match = match
                            app.status_msg = f"Welcome, {match}!"
                        else:
                            app.state = State.UNKNOWN
                            app.last_match = None
                            app.status_msg = "Unknown person — access denied"
                else:
                    if state != State.STANDBY and now - app.last_face_time > app.idle_timeout:
                        app.state = State.STANDBY
                        app.last_match = None
                        app.status_msg = "SYSTEM READY"

            state    = app.state
            match    = app.last_match
            status   = app.status_msg
            reg_n    = app.reg_frames
            reg_tot  = app.reg_target
            db_count = len(app.face_db)
            idle_to  = app.idle_timeout
            data_ttl = app.data_ttl
            reg_blocked = app.reg_blocked

        app.prune_expired()

        # Render
        if state == State.LOADING:
            draw_loading(canvas, t)
        elif state == State.STANDBY or raw_frame is None:
            draw_standby(canvas, t, db_count)
        elif state == State.QR_AUTHORIZED:
            with app.lock:
                elapsed_qr = now - app.qr_authorized_at
                qr_name_cur = app.reg_name
            if elapsed_qr < 5.0:
                draw_qr_authorized(canvas, elapsed_qr, qr_name_cur)
            elif raw_frame is not None:
                countdown = max(0, 5 - int(elapsed_qr - 5.0))
                draw_enroll_gate(canvas, raw_frame, faces, INF_W, INF_H, qr_name_cur,
                                 phase="countdown", countdown=countdown, t=t)
            else:
                countdown = max(0, 5 - int(elapsed_qr - 5.0))
                draw_qr_countdown(canvas, elapsed_qr - 5.0, qr_name_cur, countdown)
        else:
            if state == State.DETECTION:
                draw_camera_state(canvas, raw_frame, faces, INF_W, INF_H,
                                  DETECT, "RECOGNIZING", DETECT, per_face=True)
            elif state == State.RECOGNIZED:
                draw_camera_state(canvas, raw_frame, faces, INF_W, INF_H,
                                  UNLOCK, f"UNLOCKED   {match or '?'}", UNLOCK, per_face=True)
            elif state == State.UNKNOWN:
                draw_camera_state(canvas, raw_frame, faces, INF_W, INF_H,
                                  LOCK, "ACCESS DENIED", LOCK, per_face=True)
            elif state == State.REGISTRATION:
                draw_enroll_gate(canvas, raw_frame, faces, INF_W, INF_H, app.reg_name,
                                 phase="capture", reg_progress=(reg_n, reg_tot),
                                 blocked=reg_blocked, t=t)

        npu_tag = "" if _ready["npu"] else " (loading...)"
        draw_hud(canvas, state, status, db_count,
                 fps=eng.fps_counter.fps(),
                 backend=backend + (npu_tag if backend == "npu" else ""),
                 idle_timeout=idle_to, data_ttl=data_ttl)

        cpu_cores, cpu_total, ram_pct, ram_used, ram_total = stats.snapshot()
        draw_stats_panel(canvas,
                         cpu_cores, cpu_total, ram_pct, ram_used, ram_total,
                         cpu_fps=cpu_engine.fps_counter.fps(),
                         npu_fps=npu_engine.fps_counter.fps(),
                         active_backend=active["backend"],
                         npu_ready=_ready["npu"])
        cv2.imshow("Dragonwing Doorbell", canvas)
        if _fs_frames < 10:
            # Mutter can take a few frames to honor the request; re-assert fullscreen
            # over the first ~10 frames so it reliably sticks, then stop.
            cv2.setWindowProperty("Dragonwing Doorbell", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            _fs_frames += 1

        key = cv2.waitKey(33) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break
        elif key == 27:  # ESC
            with app.lock:
                app.state = State.STANDBY
                app.last_match = None
                app.status_msg = "SYSTEM READY"
            print("[STATE] → STANDBY (manual)")
        elif key == ord('b') or key == ord('B'):
            new_be = "npu" if active["backend"] == "cpu" else "cpu"
            if new_be == "npu" and not _ready["npu"]:
                print("[B] NPU still loading, please wait...")
            else:
                active["backend"] = new_be
                print(f"[B] Switched to {new_be.upper()}")
        elif key == ord('g') or key == ord('G'):
            # Authorize a guest without a QR — same flow a LinkedIn QR triggers.
            with app.lock:
                can_reg = app.state in (State.DETECTION, State.UNKNOWN,
                                        State.RECOGNIZED, State.STANDBY)
                if can_reg:
                    app.state = State.QR_AUTHORIZED
                    app.reg_name = f"Guest_{int(time.time()) % 10000}"
                    app.qr_authorized_at = time.time()
                    app.status_msg = f"Authorized: {app.reg_name}"
                    name_copy = app.reg_name
            if can_reg:
                print(f"[G] Guest authorized ({name_copy})")
        elif key == ord('c') or key == ord('C'):
            with app.lock:
                n = len(app.face_db)
                app.face_db.clear()
            print(f"[C] Cleared face database ({n} removed)")
        elif key == ord('['):
            with app.lock:
                app.idle_timeout = max(5.0, app.idle_timeout - 5.0)
                v = app.idle_timeout
            print(f"[IDLE] idle timeout = {v:.0f}s")
        elif key == ord(']'):
            with app.lock:
                app.idle_timeout = min(60.0, app.idle_timeout + 5.0)
                v = app.idle_timeout
            print(f"[IDLE] idle timeout = {v:.0f}s")
        elif key == ord('-'):
            with app.lock:
                app.data_ttl = max(60.0, app.data_ttl - 60.0)
                v = app.data_ttl
            print(f"[TTL] data store = {v:.0f}s")
        elif key == ord('='):   # unshifted '+'
            with app.lock:
                app.data_ttl = min(600.0, app.data_ttl + 60.0)
                v = app.data_ttl
            print(f"[TTL] data store = {v:.0f}s")

    cpu_engine.stop()
    npu_engine.stop()
    stats.stop()
    camera.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
