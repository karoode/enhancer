#!/usr/bin/env bash
set -o errexit

echo "Python version:"
python --version

python -m pip install --upgrade "pip==24.2" "setuptools==69.5.1" wheel

pip install -r requirements.txt

# Install these without dependencies so they do not pull wrong OpenCV / NumPy versions
pip install --no-deps basicsr==1.4.2
pip install --no-deps gfpgan==1.3.8

# Force headless OpenCV and NumPy 1.x after all installs
pip uninstall -y opencv-python opencv-contrib-python || true
pip install --force-reinstall "numpy==1.26.4" "opencv-python-headless==4.8.1.78"

if [ ! -f GFPGANv1.4.pth ]; then
  echo "Downloading GFPGANv1.4.pth..."
  python - <<'PY'
from urllib.request import urlretrieve
url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
urlretrieve(url, "GFPGANv1.4.pth")
print("GFPGANv1.4.pth downloaded")
PY
fi

echo "Testing imports..."
python - <<'PY'
import sys
import numpy
import cv2
import torch
from gfpgan import GFPGANer

print("Python:", sys.version)
print("NumPy:", numpy.__version__)
print("OpenCV:", cv2.__version__)
print("Torch:", torch.__version__)
print("GFPGAN import OK")
PY
