import os
import urllib.request
from io import BytesIO
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify
from gfpgan import GFPGANer

# ---------------- Config ----------------
# Where to keep big model files on Render (persistent disk)
MODEL_DIR = os.environ.get("MODEL_DIR", "/var/data/models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Expected filename for GFPGAN weights
GFPGAN_WEIGHTS = os.path.join(MODEL_DIR, "GFPGANv1.4.pth")

# Optional: set GFPGAN_MODEL_URL as an environment variable to auto-download the weights at boot
GFPGAN_MODEL_URL = os.environ.get("GFPGAN_MODEL_URL")  # e.g. an S3/HF/Drive *direct* link

# ----------------------------------------


def ensure_gfpgan_weights():
    """Ensure the GFPGAN weights exist locally; download if URL provided."""
    if os.path.exists(GFPGAN_WEIGHTS):
        return

    if not GFPGAN_MODEL_URL:
        raise RuntimeError(
            "GFPGAN weights not found at {} and GFPGAN_MODEL_URL is not set.\n"
            "Upload GFPGANv1.4.pth to /var/data/models or set GFPGAN_MODEL_URL."
            .format(GFPGAN_WEIGHTS)
        )

    tmp_path = GFPGAN_WEIGHTS + ".part"
    try:
        urllib.request.urlretrieve(GFPGAN_MODEL_URL, tmp_path)
        os.replace(tmp_path, GFPGAN_WEIGHTS)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# Initialize Flask
app = Flask(__name__)

# Warm-up GFPGAN on startup
ensure_gfpgan_weights()
restorer = GFPGANer(
    model_path=GFPGAN_WEIGHTS,
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_path": GFPGAN_WEIGHTS,
        "time": datetime.utcnow().isoformat() + "Z"
    })


@app.route("/enhance", methods=["POST"])
def enhance_image():
    # Expect a form field named "image"
    file = request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"error": "Missing image file field 'image'"}), 400

    # Decode image
    data = file.read()
    npimg = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image data"}), 400

    # Run GFPGAN (face restoration; paste_back=True returns full image)
    try:
        _, _, restored = restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )
    except Exception as e:
        return jsonify({"error": f"GFPGAN processing failed: {e}"}), 500

    # Encode back to JPEG
    ok, buf = cv2.imencode(".jpg", restored)
    if not ok:
        return jsonify({"error": "Image encode failed"}), 500

    return send_file(BytesIO(buf.tobytes()), mimetype="image/jpeg")


if __name__ == "__main__":
    # For local dev; on Render use gunicorn:
    # gunicorn app:app --bind 0.0.0.0:$PORT
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
