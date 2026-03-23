#!/bin/bash
# VoxScript - Startup Script
set -e

echo "============================================"
echo "  VoxScript — Audio Transcription Server"
echo "============================================"

# Check Python
python3 --version || { echo "ERROR: Python 3 required"; exit 1; }

# Check ffmpeg
which ffmpeg > /dev/null 2>&1 || {
  echo "⚠ ffmpeg not found. Installing..."
  sudo apt-get update && sudo apt-get install -y ffmpeg
}

# Create backend dirs
mkdir -p backend/uploads backend/outputs

# Install Python dependencies
cd backend
echo ""
echo "📦 Installing Python dependencies..."
pip install --break-system-packages -q -r requirements.txt

echo ""
echo "✅ Starting server on http://localhost:8000"
echo ""
echo "Frontend UI available at: http://localhost:8000"
echo "API docs available at:    http://localhost:8000/docs"
echo ""

python3 main.py
