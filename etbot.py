from flask import Flask, request, send_file
import cv2
import numpy as np
from gfpgan import GFPGANer
from io import BytesIO

app = Flask(__name__)

# Load GFPGAN
restorer = GFPGANer(
    model_path='GFPGANv_path',
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)

@app.route('/enhance', methods=['POST'])
def enhance_image():
    if 'image' not in request.files:
        return "Missing image file", 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Enhance the image
    _, _, output = restorer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)

    # ✅ Save locally with timestamp
    from datetime import datetime
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # save_path = f"enhanced_{timestamp}.jpg"
    # cv2.imwrite(save_path, output)

    # Return response
    _, buffer = cv2.imencode('.jpg', output)
    return send_file(BytesIO(buffer), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
