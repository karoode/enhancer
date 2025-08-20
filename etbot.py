# etbot.py
import os
import urllib.request
from io import BytesIO
from datetime import datetime

# ↓↓↓ Reduce native threads to save RAM/CPU ↓↓↓
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_JIT", "0")  # tiny savings

import cv2
cv2.setNumThreads(1)

import numpy as np
from flask import Flask, request, send_file, jsonify

app = Flask(__name__)

# --------- model path with /tmp fallback (no disk required) ----------
DEFAULT_MODEL_DIR = os.environ.get("MODEL_DIR", "/var/data/models")
MODEL_DIR = DEFAULT_MODEL_DIR
try:
    os.makedirs(MODEL_DIR, exist_ok=True)
except PermissionError:
    MODEL_DIR = "/tmp/models"
    os.makedirs(MODEL_DIR, exist_ok=True)

GFPGAN_WEIGHTS = os.path.join(MODEL_DIR, "GFPGANv1.4.pth")
GFPGAN_MODEL_URL = os.environ.get("GFPGAN_MODEL_URL")  # direct link
RESTORER = None  # lazy-loaded singleton


def ensure_gfpgan_weights():
    if os.path.exists(GFPGAN_WEIGHTS):
        return
    if not GFPGAN_MODEL_URL:
        raise RuntimeError(
            f"GFPGAN weights not found at {GFPGAN_WEIGHTS} and GFPGAN_MODEL_URL not set."
        )
    tmp = GFPGAN_WEIGHTS + ".part"
    urllib.request.urlretrieve(GFPGAN_MODEL_URL, tmp)
    os.replace(tmp, GFPGAN_WEIGHTS)


def get_restorer():
    """Load GFPGAN only when needed, once."""
    global RESTORER
    if RESTORER is not None:
        return RESTORER
    ensure_gfpgan_weights()
    # Import heavy libs here to avoid loading at boot
    from gfpgan import GFPGANer
    RESTORER = GFPGANer(
        model_path=GFPGAN_WEIGHTS,
        upscale=1,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None  # keep None to avoid RealESRGAN load (saves RAM)
    )
    return RESTORER


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_dir": MODEL_DIR,
        "weights_exists": os.path.exists(GFPGAN_WEIGHTS),
        "time": datetime.utcnow().isoformat() + "Z"
    })


@app.route("/enhance", methods=["POST"])
def enhance_image():
    file = request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"error": "Missing image file field 'image'"}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image data"}), 400

    try:
        restorer = get_restorer()
        # lighter settings help on low RAM
        _, _, restored = restorer.enhance(
            img, has_aligned=False, only_center_face=True, paste_back=True
        )
    except Exception as e:
        return jsonify({"error": f"GFPGAN processing failed: {e}"}), 500

    ok, buf = cv2.imencode(".jpg", restored, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return jsonify({"error": "Image encode failed"}), 500

    return send_file(BytesIO(buf.tobytes()), mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
