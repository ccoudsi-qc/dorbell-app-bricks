"""
arduino:face_recognition — app-local custom brick (Ventuno Q / HTP v75)

A face DETECTION + EMBEDDING service, modelled on the stock
`arduino.app_bricks.video_object_detection` client. It mirrors that brick's
two-channel runner contract exactly:

    * results  — websocket  ws://<runner>:4912   (JSON, runner -> client)
    * frames   — plain TCP   tcp://<runner>:5050   (JPEG, client -> runner)

What makes this brick isolated (per the Ventuno Q project constraint) is that it
lives entirely inside the app at
`<app>/python/arduino/app_bricks/face_recognition/`. Because the base image's
`run.sh` does `cd /app && exec python /app/python/main.py`, `/app/python` is on
`sys.path`, so this namespace-merges with the stock `arduino.app_bricks` and is
both importable AND discovered by `arduino-bricks-list-modules` — with zero
platform writes and no base-image rebuild. See the brick README.

Design split (deliberate): this brick returns RAW faces with 512-d ArcFace
embeddings; it does NOT keep an identity database. Identity matching, the
2-minute TTL store, QR-triggered enrollment and the whole state machine live in
the APP (reused verbatim from doorbell_core), exactly as doorbell.py's
NpuFaceEngine returns faces and the main loop runs recognize_all() against its
own face_db. That keeps the brick stateless and the DB where the TTL/QR policy
already is.

The public surface is intentionally identical to doorbell_core.NpuFaceEngine so
the app's main loop is unchanged:

    eng = FaceRecognition(on_ready=cb)
    eng.start()
    eng.submit(inf_frame)          # 320x240 BGR, latest-wins
    faces = eng.get_faces()        # list of Face: .bbox .kps .embedding .det_score
    eng.fps_counter.fps()
    eng.stop()

Face coordinates are in the SAME 320x240 inference space doorbell_core uses
(INF_W x INF_H), so every draw_* helper maps them onto the 1920x1080 canvas
without change.
"""

import os
import io
import json
import time
import socket
import struct
import threading
import collections
from typing import Optional, List

import numpy as np
import cv2

__all__ = ["FaceRecognition"]

# ── runner contract ──────────────────────────────────────────────────────────
WS_PORT = 4912          # results: JSON over websocket (runner -> client)
TCP_PORT = 5050         # frames:  length-prefixed JPEG (client -> runner)
_JPEG_QUALITY = 80


# ── FPS counter (same shape as doorbell_core.FpsCounter) ───────────────────────
class _FpsCounter:
    def __init__(self, window: int = 30):
        self._times: collections.deque = collections.deque(maxlen=window)

    def tick(self):
        self._times.append(time.perf_counter())

    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0


def _resolve_runner_host() -> str:
    """Resolve the runner service's reachable hostname.

    The daemon merges this brick's brick_compose.ventunoq.yaml into the app's
    app-compose.yaml, so the runner joins the app's compose network under its
    service name — which resolves by DNS from the app container. We take that
    single service name from our own co-located compose file (the same thing the
    stock bricks do via load_brick_compose_file()).

    VERIFY-ON-BOARD: the stock video_object_detection client passes the service
    name through arduino.app_internal's resolve_address() before dialing. On
    ventunoq that is expected to be an identity (compose-DNS name), but if the
    runner turns out to be reachable only via HOST_IP, set FACE_RECOGNITION_HOST.
    """
    override = os.environ.get("FACE_RECOGNITION_HOST")
    if override:
        return override

    here = os.path.dirname(os.path.abspath(__file__))
    board = os.environ.get("BOARD_NAME", "")
    candidates = [
        os.path.join(here, f"brick_compose.{board}.yaml") if board else None,
        os.path.join(here, "brick_compose.yaml"),
    ]
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            import yaml
            with open(path) as fh:
                compose = yaml.safe_load(fh) or {}
            services = compose.get("services") or {}
            if services:
                return next(iter(services.keys()))
        except Exception:
            # Fall back to a cheap grep for the first "  <name>:" under services:
            try:
                with open(path) as fh:
                    in_services = False
                    for line in fh:
                        if line.strip().startswith("services:"):
                            in_services = True
                            continue
                        if in_services and line[:2] == "  " and line[2:3] != " " \
                           and line.rstrip().endswith(":"):
                            return line.strip().rstrip(":")
            except Exception:
                pass
    # Last resort: the service name this brick ships.
    return "face-recognition-runner"


class _Face:
    """Duck-typed face identical to doorbell_core's NPU faces.

    Carries .bbox [x1,y1,x2,y2], .kps (5x2), .embedding (512-d L2-normed),
    .det_score. recognize_all() later annotates .rec_name / .rec_sim.
    """
    __slots__ = ("bbox", "kps", "landmark_2d_106", "det_score", "embedding",
                 "rec_name", "rec_sim")

    def __init__(self, bbox, kps, embedding, det_score):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.kps = np.asarray(kps, dtype=np.float32) if kps is not None else None
        self.landmark_2d_106 = None
        self.det_score = float(det_score)
        self.embedding = np.asarray(embedding, dtype=np.float32)
        self.rec_name = None
        self.rec_sim = 0.0


class FaceRecognition:
    """Client for the face-recognition NPU runner.

    Drop-in replacement for doorbell_core.NpuFaceEngine: same submit/get_faces/
    fps_counter/start/stop/on_ready surface, so the app's main loop is unchanged.
    """

    def __init__(self, on_ready=None, host: Optional[str] = None):
        self.on_ready = on_ready
        self._host = host or _resolve_runner_host()
        self.fps_counter = _FpsCounter(window=30)

        self._pending: Optional[np.ndarray] = None
        self._pending_lock = threading.Lock()
        self._faces: List[_Face] = []
        self._faces_lock = threading.Lock()

        self.running = True
        self._ready_fired = False
        self._tcp: Optional[socket.socket] = None
        self._tcp_lock = threading.Lock()

        self._ws_thread = threading.Thread(target=self._results_loop, daemon=True)
        self._tx_thread = threading.Thread(target=self._frames_loop, daemon=True)

    # ── public surface (mirrors NpuFaceEngine) ────────────────────────────────
    def start(self):
        self._ws_thread.start()
        self._tx_thread.start()

    def submit(self, frame: np.ndarray):
        with self._pending_lock:
            self._pending = frame  # latest-wins, drop older

    def get_faces(self) -> list:
        with self._faces_lock:
            return list(self._faces)

    def stop(self):
        self.running = False
        with self._tcp_lock:
            if self._tcp is not None:
                try:
                    self._tcp.close()
                except Exception:
                    pass
                self._tcp = None

    # ── internals ──────────────────────────────────────────────────────────────
    def _fire_ready(self):
        if not self._ready_fired:
            self._ready_fired = True
            if self.on_ready:
                try:
                    self.on_ready()
                except Exception:
                    pass

    def _ensure_tcp(self) -> Optional[socket.socket]:
        with self._tcp_lock:
            if self._tcp is not None:
                return self._tcp
            try:
                s = socket.create_connection((self._host, TCP_PORT), timeout=5.0)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._tcp = s
                return s
            except Exception as e:
                print(f"[face_recognition] frame channel connect failed "
                      f"({self._host}:{TCP_PORT}): {e}", flush=True)
                return None

    def _frames_loop(self):
        """Stream the latest submitted frame as length-prefixed JPEG to the runner."""
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY]
        while self.running:
            with self._pending_lock:
                frame = self._pending
                self._pending = None
            if frame is None:
                time.sleep(0.005)
                continue
            s = self._ensure_tcp()
            if s is None:
                time.sleep(0.5)
                continue
            try:
                ok, buf = cv2.imencode(".jpg", frame, encode_params)
                if not ok:
                    continue
                data = buf.tobytes()
                s.sendall(struct.pack(">I", len(data)) + data)
            except Exception as e:
                print(f"[face_recognition] frame send error: {e}", flush=True)
                with self._tcp_lock:
                    try:
                        if self._tcp:
                            self._tcp.close()
                    except Exception:
                        pass
                    self._tcp = None
                time.sleep(0.2)

    def _results_loop(self):
        """Receive JSON face results from the runner over the websocket."""
        uri = f"ws://{self._host}:{WS_PORT}"
        while self.running:
            try:
                import websocket  # websocket-client
            except Exception as e:
                print(f"[face_recognition] websocket-client missing: {e}", flush=True)
                self._fire_ready()  # don't wedge the app's LOADING->STANDBY gate
                return
            try:
                ws = websocket.create_connection(uri, timeout=5.0)
            except Exception as e:
                print(f"[face_recognition] results connect failed ({uri}): {e}",
                      flush=True)
                time.sleep(1.0)
                continue
            print(f"[face_recognition] results channel up: {uri}", flush=True)
            try:
                while self.running:
                    raw = ws.recv()
                    if not raw:
                        break
                    self._on_message(raw)
            except Exception as e:
                print(f"[face_recognition] results recv error: {e}", flush=True)
            finally:
                try:
                    ws.close()
                except Exception:
                    pass
            time.sleep(0.5)

    def _on_message(self, raw):
        try:
            msg = json.loads(raw)
        except Exception:
            return
        mtype = msg.get("type")
        if mtype == "hello":
            self._fire_ready()
            return
        if mtype != "faces":
            return
        faces = []
        for fd in msg.get("faces", []):
            emb = fd.get("embedding")
            if emb is None:
                continue
            faces.append(_Face(bbox=fd.get("bbox", [0, 0, 0, 0]),
                               kps=fd.get("kps"),
                               embedding=emb,
                               det_score=fd.get("score", 0.0)))
        with self._faces_lock:
            self._faces = faces
        self.fps_counter.tick()
        self._fire_ready()
