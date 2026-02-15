#!/bin/bash
set -e
cd /home/ubuntu/WanAvatar

# 1. Clean up and create directory
rm -rf amt_model
mkdir -p amt_model

# 2. Download AMT repository
echo "Downloading AMT..."
curl -L -o /tmp/amt.zip https://github.com/MCG-NKU/AMT/archive/refs/heads/main.zip
python3 -c "import zipfile; zipfile.ZipFile('/tmp/amt.zip').extractall('/tmp/amt_extract')"
cp -r /tmp/amt_extract/AMT-main/* amt_model/
rm -rf /tmp/amt.zip /tmp/amt_extract

# 2b. Fix filename for Python import (hyphen -> underscore)
if [ -f amt_model/networks/AMT-L.py ]; then
    cp amt_model/networks/AMT-L.py amt_model/networks/AMT_L.py
    echo "Copied AMT-L.py -> AMT_L.py for Python import"
fi
if [ -f amt_model/networks/AMT-S.py ]; then
    cp amt_model/networks/AMT-S.py amt_model/networks/AMT_S.py
fi
if [ -f amt_model/networks/AMT-G.py ]; then
    cp amt_model/networks/AMT-G.py amt_model/networks/AMT_G.py
fi

# 3. Download AMT-G model weights from HuggingFace (primary)
echo "Downloading AMT-G model weights..."
if curl -L -o amt_model/amt-g.pth "https://huggingface.co/lalala125/AMT/resolve/main/amt-g.pth" 2>/dev/null && [ -s amt_model/amt-g.pth ]; then
    echo "Downloaded AMT-G weights: $(ls -lh amt_model/amt-g.pth)"
else
    echo "WARNING: Could not download amt-g.pth, will try AMT-L as fallback."
fi

# 3b. Download AMT-L model weights (fallback)
echo "Downloading AMT-L model weights..."
if curl -L -o amt_model/amt-l.pth "https://huggingface.co/lalala125/AMT/resolve/main/amt-l.pth" 2>/dev/null && [ -s amt_model/amt-l.pth ]; then
    echo "Downloaded AMT-L weights: $(ls -lh amt_model/amt-l.pth)"
else
    echo "WARNING: Could not download amt-l.pth."
fi

# 4. Install dependencies
echo "Installing AMT dependencies..."
/home/ubuntu/WanAvatar/venv/bin/pip install einops 2>/dev/null || true

# 5. Verify
echo ""
echo "=== Verification ==="
ls -la amt_model/networks/AMT_G.py 2>/dev/null && echo "AMT_G.py: OK" || echo "AMT_G.py: MISSING"
ls -la amt_model/networks/AMT_L.py 2>/dev/null && echo "AMT_L.py: OK" || echo "AMT_L.py: MISSING"
ls -lh amt_model/amt-g.pth 2>/dev/null && echo "AMT-G Weights: OK" || echo "AMT-G Weights: MISSING"
ls -lh amt_model/amt-l.pth 2>/dev/null && echo "AMT-L Weights: OK" || echo "AMT-L Weights: MISSING"
echo "Done!"
