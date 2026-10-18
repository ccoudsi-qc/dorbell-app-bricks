# face-recognition-runner — NPU inference server for the arduino:face_recognition brick.
#
# Reaches the Ventuno Q HTP v75 through onnxruntime-qnn (a pip wheel that bundles
# its own version-matched QNN runtime), the same "pin host QNN to the board QAIRT"
# discipline the production IQ-8275 container uses — just delivered as a wheel.
#
# BASE: Arduino's ei-qnn-models-runner (verified present on the board, Debian 13
# aarch64, Python 3.13 + pip). We ignore its Edge-Impulse Node.js runner and
# override the entrypoint. We do NOT use its /usr/lib QNN libs — the wheel carries
# its own matched QNN 2.51 skels, and mixing generations makes the HTP session fail
# with QNN_DEVICE_ERROR_INVALID_CONFIG and silently fall back to CPU.
#
# VERIFY-ON-BOARD recap (done 2026-10-18): aihub-onnx-models-runner does NOT exist;
# ei-qnn-models-runner:0.12.1 does, and onnxruntime-qnn==2.7.0 has a real
# linux-aarch64/cp313 wheel whose bundled skels drive the v75 HTP (FP16-rounded
# output + fastrpc/rpcmem DSP calls confirm it, vs exact output on CPU fallback).
ARG BASE=ghcr.io/arduino/app-bricks/ei-qnn-models-runner:0.12.1
FROM ${BASE}

ARG DEBIAN_FRONTEND=noninteractive
USER 0:0

# insightface -> cv2 needs libGL.so.1 (+ glib); not in the EI base. ca-certificates
# lets insightface fetch norm_crop deps if ever needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install order matters: insightface pulls plain onnxruntime as a dep, so install it
# FIRST and let onnxruntime-qnn win the onnxruntime slot LAST. --break-system-packages
# matches the Debian 13 base's PEP 668 posture. insightface is used ONLY for
# insightface.utils.face_align.norm_crop (5-pt align).
COPY runner/requirements.txt /tmp/req.txt
RUN pip3 install --break-system-packages --no-cache-dir \
        insightface opencv-python-headless numpy websockets && \
    pip3 install --break-system-packages --no-cache-dir onnxruntime-qnn==2.7.0

# ── models ────────────────────────────────────────────────────────────────────
# buffalo_sc det + rec ONNX (det_500m.onnx + w600k_mbf.onnx) — the SAME models the
# production app runs. NOT committed (gitignored); stage them into
# runner/models/buffalo_sc/ before building, OR mount them at runtime to
# /opt/models/buffalo_sc and skip this COPY. The COPY tolerates an empty dir so the
# image still builds for the mount-at-runtime path; runner.py errors clearly if the
# ONNX are absent.
#
#   runner/models/buffalo_sc/det_500m.onnx
#   runner/models/buffalo_sc/w600k_mbf.onnx
COPY runner/models/ /opt/models/
ENV FACE_RECOGNITION_MODEL_DIR=/opt/models/buffalo_sc

COPY runner/runner.py /opt/runner/runner.py
WORKDIR /opt/runner

EXPOSE 4912 5050
ENTRYPOINT ["python3", "/opt/runner/runner.py"]
