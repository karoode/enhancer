#!/usr/bin/env bash
set -o errexit

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

if [ ! -f GFPGANv1.4.pth ]; then
  echo "Downloading GFPGANv1.4.pth..."
  python - <<'PY'
from urllib.request import urlretrieve
url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
urlretrieve(url, "GFPGANv1.4.pth")
print("GFPGANv1.4.pth downloaded")
PY
fi