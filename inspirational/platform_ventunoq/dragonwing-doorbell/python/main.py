"""
Dragonwing Doorbell — Arduino App Lab app entrypoint (Ventuno Q).

This is the PROTOTYPE-side app. It reuses the production doorbell UI + state
machine *verbatim* from `doorbell_core` (a byte-for-byte snapshot of the
IQ-8275 `doorbell.py`) and changes ONLY the two I/O ends:

  * inference  — the in-process `NpuFaceEngine` is replaced by the app-local
                 `arduino:face_recognition` brick client, which streams frames
                 to the HTP v75 NPU runner and returns faces. Its surface
                 (submit / get_faces / fps_counter / start / stop / on_ready) is
                 identical to NpuFaceEngine, so the main loop below is the same
                 loop as doorbell_core.main().

  * display    — instead of cv2.imshow() on an HDMI/Xwayland window, the
                 composited 1920x1080 canvas is JPEG-encoded and streamed into
                 the `arduino:web_ui` webview via Socket.IO. Key events (O/B/G/
                 C/[/]/-/=/ESC/Q) come BACK from the browser over the same
                 socket, replacing cv2.waitKey().

Everything in between — recognition, the 2-minute TTL store, QR-triggered
enrollment, the enrollment gate, all draw_* compositing — is doorbell_core,
unchanged. See python/arduino/app_bricks/face_recognition/README.md for the
brick/runner contract.
"""

import os
import sys
import time
import base64
import threading
import collections

import cv2
import numpy as np

# doorbell_core is the vendored snapshot of the production doorbell.py. Import it
# from the app's /app/python (sys.path[0] in the container).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doorbell_core as dc

from arduino.app_bricks.face_recognition import FaceRecognition
from arduino.app_bricks.web_ui import WebUI


# JPEG quality + target cadence for the canvas stream. 1920x1080 at q70 is a few
# hundred KB; ~20 fps keeps the webview smooth without saturating the socket.
_STREAM_JPEG_QUALITY = 70
_STREAM_FPS = 20
_STREAM_PERIOD = 1.0 / _STREAM_FPS


def main():
    # ── bricks ────────────────────────────────────────────────────────────────
    ui = WebUI(port=7000)
    # Bind + serve the webview. The stock bricks get this for free from the
    # blocking App.run(), which calls ui.start() (build the uvicorn server) and
    # then runs ui.execute() (the blocking self._server.run()) on a managed
    # thread. The doorbell drives its OWN ~30 Hz render loop below instead of
    # App.run(), so we reproduce that two-step launch explicitly: start() to
    # configure the server + socket.io, then execute() on a daemon thread so it
    # serves without blocking our loop. Without this nothing listens on :7000.
    ui.start()
    threading.Thread(target=ui.execute, name="web_ui", daemon=True).start()

    app = dc.AppState()
    camera = dc.CameraThread()
    camera.start()

    _ready = {"npu": False}

    def on_npu_ready():
        _ready["npu"] = True
        with app.lock:
            if app.state == dc.State.LOADING:
                app.state = dc.State.STANDBY
                app.status_msg = "SYSTEM READY"
        print("[NPU] face_recognition runner ready", flush=True)

    eng = FaceRecognition(on_ready=on_npu_ready)
    stats = dc.StatsMonitor()
    eng.start()
    stats.start()

    # ── key events from the webview (replaces cv2.waitKey) ─────────────────────
    # The web page emits {"key": "<char>"} on keydown; we queue them and drain
    # the queue once per loop, exactly where doorbell_core.main() reads waitKey.
    key_q: collections.deque = collections.deque()

    def on_key(sid, data):
        try:
            k = (data or {}).get("key", "")
        except Exception:
            k = ""
        if k:
            key_q.append(k)

    ui.on_message("key", on_key)

    def next_key():
        return key_q.popleft() if key_q else None

    # ── registration completion (verbatim from doorbell_core.main) ──────────────
    def finish_registration(name: str, embeddings: list):
        with app.lock:
            app.add_face(name, embeddings)
            app.state = dc.State.RECOGNIZED
            app.last_match = name
            app.status_msg = f"Registered! Welcome, {name}"
        print(f"[REG] Done — {name} added with {len(embeddings)} embeddings "
              f"(TTL {app.data_ttl:.0f}s)", flush=True)

    canvas = np.zeros((dc.H, dc.W, 3), dtype=np.uint8)
    t0 = time.time()
    qr_detector = dc.QRDetector()
    backend = "npu"
    last_stream = 0.0

    print("=== DRAGONWING DOORBELL (App Lab / Ventuno Q) ===", flush=True)
    print(f"web_ui: {getattr(ui, 'local_url', 'http://localhost:7000')}", flush=True)
    print("Keys (in the webview): O=enroll G=guest C=clear [ ]=idle - ==store ESC=standby Q=quit",
          flush=True)

    running = True
    while running:
        t = time.time() - t0
        now = time.time()

        # Feed latest frame to the brick (downscaled), same as doorbell_core.
        raw_frame = camera.get_frame()
        if raw_frame is not None:
            inf_frame = cv2.resize(raw_frame, (dc.INF_W, dc.INF_H))
            eng.submit(inf_frame)

        faces = eng.get_faces()

        # QR detection — every frame, triggers registration automatically.
        qr_name = None
        if raw_frame is not None:
            qr_name = qr_detector.detect(raw_frame)
            if qr_name:
                print(f"[QR] Detected LinkedIn profile: {qr_name}", flush=True)

        with app.lock:
            state = app.state
            if qr_name and state not in (dc.State.LOADING, dc.State.REGISTRATION,
                                         dc.State.QR_AUTHORIZED):
                app.state = dc.State.QR_AUTHORIZED
                app.reg_name = qr_name
                app.qr_authorized_at = now
                app.reg_blocked = False
                app.status_msg = f"QR authorized: {qr_name}"
                state = dc.State.QR_AUTHORIZED

            if state == dc.State.QR_AUTHORIZED:
                elapsed_qr = now - app.qr_authorized_at
                if elapsed_qr >= 10.0:
                    app.state = dc.State.REGISTRATION
                    app.reg_frames = 0
                    app.reg_start_at = now
                    app.reg_blocked = False
                    app.status_msg = f"Registering {app.reg_name} — look at camera"
                    state = dc.State.REGISTRATION
                    reg_thread = dc.RegistrationDriver(app, eng, finish_registration)
                    reg_thread.start()
            elif state not in (dc.State.LOADING, dc.State.REGISTRATION):
                if faces:
                    app.last_face_time = now
                    if state == dc.State.STANDBY:
                        app.state = dc.State.DETECTION
                        app.status_msg = "Face detected — recognizing..."
                        state = dc.State.DETECTION
                    if state in (dc.State.DETECTION, dc.State.RECOGNIZED, dc.State.UNKNOWN):
                        dc.recognize_all(faces, app.face_db)
                        primary = dc.largest_face(faces)
                        match = getattr(primary, 'rec_name', None) if primary else None
                        if match:
                            app.state = dc.State.RECOGNIZED
                            app.last_match = match
                            app.status_msg = f"Welcome, {match}!"
                        else:
                            app.state = dc.State.UNKNOWN
                            app.last_match = None
                            app.status_msg = "Unknown person — access denied"
                else:
                    if state != dc.State.STANDBY and now - app.last_face_time > app.idle_timeout:
                        app.state = dc.State.STANDBY
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

        # ── render (verbatim draw_* calls from doorbell_core) ───────────────────
        if state == dc.State.LOADING:
            dc.draw_loading(canvas, t)
        elif state == dc.State.STANDBY or raw_frame is None:
            dc.draw_standby(canvas, t, db_count)
        elif state == dc.State.QR_AUTHORIZED:
            with app.lock:
                elapsed_qr = now - app.qr_authorized_at
                qr_name_cur = app.reg_name
            if elapsed_qr < 5.0:
                dc.draw_qr_authorized(canvas, elapsed_qr, qr_name_cur)
            elif raw_frame is not None:
                countdown = max(0, 5 - int(elapsed_qr - 5.0))
                dc.draw_enroll_gate(canvas, raw_frame, faces, dc.INF_W, dc.INF_H,
                                    qr_name_cur, phase="countdown",
                                    countdown=countdown, t=t)
            else:
                countdown = max(0, 5 - int(elapsed_qr - 5.0))
                dc.draw_qr_countdown(canvas, elapsed_qr - 5.0, qr_name_cur, countdown)
        else:
            if state == dc.State.DETECTION:
                dc.draw_camera_state(canvas, raw_frame, faces, dc.INF_W, dc.INF_H,
                                     dc.DETECT, "RECOGNIZING", dc.DETECT, per_face=True)
            elif state == dc.State.RECOGNIZED:
                dc.draw_camera_state(canvas, raw_frame, faces, dc.INF_W, dc.INF_H,
                                     dc.UNLOCK, f"UNLOCKED   {match or '?'}", dc.UNLOCK,
                                     per_face=True)
            elif state == dc.State.UNKNOWN:
                dc.draw_camera_state(canvas, raw_frame, faces, dc.INF_W, dc.INF_H,
                                     dc.LOCK, "ACCESS DENIED", dc.LOCK, per_face=True)
            elif state == dc.State.REGISTRATION:
                dc.draw_enroll_gate(canvas, raw_frame, faces, dc.INF_W, dc.INF_H,
                                    app.reg_name, phase="capture",
                                    reg_progress=(reg_n, reg_tot),
                                    blocked=reg_blocked, t=t)

        npu_tag = "" if _ready["npu"] else " (loading...)"
        dc.draw_hud(canvas, state, status, db_count,
                    fps=eng.fps_counter.fps(),
                    backend=backend + npu_tag,
                    idle_timeout=idle_to, data_ttl=data_ttl)

        cpu_cores, cpu_total, ram_pct, ram_used, ram_total = stats.snapshot()
        dc.draw_stats_panel(canvas,
                            cpu_cores, cpu_total, ram_pct, ram_used, ram_total,
                            cpu_fps=0.0,
                            npu_fps=eng.fps_counter.fps(),
                            active_backend="npu",
                            npu_ready=_ready["npu"])

        # ── stream the canvas into the webview (replaces cv2.imshow) ────────────
        if now - last_stream >= _STREAM_PERIOD:
            ok, buf = cv2.imencode(".jpg", canvas,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), _STREAM_JPEG_QUALITY])
            if ok:
                b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                try:
                    ui.send_message("frame", {"jpeg": b64})
                except Exception as e:
                    print(f"[web_ui] send_message failed: {e}", flush=True)
            last_stream = now

        # ── keys from the webview (replaces cv2.waitKey dispatch) ───────────────
        key = next_key()
        if key is not None:
            key = key.lower()
            if key in ("q", "escape_quit"):
                running = False
            elif key == "escape":
                with app.lock:
                    app.state = dc.State.STANDBY
                    app.last_match = None
                    app.status_msg = "SYSTEM READY"
                print("[STATE] → STANDBY (manual)", flush=True)
            elif key == "g":
                with app.lock:
                    can_reg = app.state in (dc.State.DETECTION, dc.State.UNKNOWN,
                                            dc.State.RECOGNIZED, dc.State.STANDBY)
                    if can_reg:
                        app.state = dc.State.QR_AUTHORIZED
                        app.reg_name = f"Guest_{int(time.time()) % 10000}"
                        app.qr_authorized_at = time.time()
                        app.status_msg = f"Authorized: {app.reg_name}"
                        name_copy = app.reg_name
                if can_reg:
                    print(f"[G] Guest authorized ({name_copy})", flush=True)
            elif key == "c":
                with app.lock:
                    n = len(app.face_db)
                    app.face_db.clear()
                print(f"[C] Cleared face database ({n} removed)", flush=True)
            elif key == "[":
                with app.lock:
                    app.idle_timeout = max(5.0, app.idle_timeout - 5.0)
                    v = app.idle_timeout
                print(f"[IDLE] idle timeout = {v:.0f}s", flush=True)
            elif key == "]":
                with app.lock:
                    app.idle_timeout = min(60.0, app.idle_timeout + 5.0)
                    v = app.idle_timeout
                print(f"[IDLE] idle timeout = {v:.0f}s", flush=True)
            elif key == "-":
                with app.lock:
                    app.data_ttl = max(60.0, app.data_ttl - 60.0)
                    v = app.data_ttl
                print(f"[TTL] data store = {v:.0f}s", flush=True)
            elif key in ("=", "+"):
                with app.lock:
                    app.data_ttl = min(600.0, app.data_ttl + 60.0)
                    v = app.data_ttl
                print(f"[TTL] data store = {v:.0f}s", flush=True)

        # Pace the loop. doorbell_core used waitKey(33); here the sleep plays that
        # role (~30 Hz state/render), independent of the ~20 Hz stream throttle.
        time.sleep(0.01)

    eng.stop()
    stats.stop()
    camera.stop()
    print("[APP] stopped", flush=True)


if __name__ == "__main__":
    main()
