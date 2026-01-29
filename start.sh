#!/bin/bash
# WanAvatar Server Start Script

cd /home/work/WanAvatar

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
echo "============================================"

# Start the server
python server.py
