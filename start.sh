#!/bin/bash
# WanAvatar Server Start Script

cd /home/ubuntu/WanAvatar
source venv/bin/activate

# Create necessary directories
mkdir -p outputs uploads

# Check if frontend is built
if [ ! -d "frontend/dist" ]; then
    echo "Building frontend..."
    cd frontend
    npm install
    npm run build
    cd ..
fi

echo "============================================"
echo "Starting WanAvatar Server"
echo "============================================"
echo "Frontend: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"

# Detect GPU count for Sequence Parallel
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "GPUs detected: $NUM_GPUS"

if [ "${DISABLE_SP}" = "1" ]; then
    echo "SP: Disabled (DISABLE_SP=1)"
    echo "============================================"
    python server.py
elif [ "$NUM_GPUS" -gt 1 ]; then
    echo "SP: Enabled (torchrun --nproc_per_node=$NUM_GPUS)"
    echo "============================================"
    torchrun --nproc_per_node=$NUM_GPUS server.py
else
    echo "SP: Disabled (single GPU)"
    echo "============================================"
    python server.py
fi
