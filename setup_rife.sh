#!/bin/bash
set -e
cd /home/ubuntu/WanAvatar

# 1. Clean up and create directory
rm -rf rife_model
mkdir -p rife_model/train_log

# 2. Download Practical-RIFE
echo "Downloading Practical-RIFE..."
curl -L -o /tmp/rife.zip https://github.com/hzwer/Practical-RIFE/archive/refs/heads/main.zip
unzip -o /tmp/rife.zip -d /tmp/rife_extract
cp -r /tmp/rife_extract/Practical-RIFE-main/* rife_model/
rm -rf /tmp/rife.zip /tmp/rife_extract

# 3. Download RIFE v4.26 model weights from HuggingFace mirror
echo "Downloading RIFE v4.26 model weights..."
cd rife_model/train_log

# Try HuggingFace mirror first
if curl -L -o flownet.pkl "https://huggingface.co/laanlabs/rife-v4.6/resolve/main/flownet.pkl" 2>/dev/null && [ -s flownet.pkl ]; then
    echo "Downloaded from HuggingFace: $(ls -la flownet.pkl)"
else
    echo "HuggingFace failed, trying alternative..."
    # Download from the Practical-RIFE releases page
    wget -q "https://drive.google.com/uc?export=download&id=1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_" -O flownet.pkl 2>/dev/null || true
    if [ ! -s flownet.pkl ]; then
        echo "ERROR: Could not download flownet.pkl. Please download manually."
        exit 1
    fi
fi

cd /home/ubuntu/WanAvatar

# 4. Install dependency
echo "Installing sk-video..."
/home/ubuntu/WanAvatar/venv/bin/pip install sk-video 2>/dev/null

# 5. Verify
echo ""
echo "=== Verification ==="
echo "Model file: $(ls -la rife_model/model/RIFE.py 2>/dev/null || echo 'MISSING')"
echo "Weights: $(ls -la rife_model/train_log/flownet.pkl 2>/dev/null || echo 'MISSING')"
echo "Done!"
