#!/bin/bash

set -e  # Exit on error
set -x  # Print commands for debugging

# Ensure script is executable
chmod +x "$0"

# Install Python dependencies
echo "Installing Python dependencies..."
cd adapters/image_processor || { echo "Failed to cd into adapters/image_processor"; exit 1; }
pip install -r requirements.txt || { echo "Failed to install Python dependencies"; exit 1; }

# Download models
echo "Downloading models..."
python setup.py || { echo "Failed to download models"; exit 1; }

# Start Python server in background with better control
echo "Starting Python AI server..."
python main.py > python_server.log 2>&1 &
PYTHON_PID=$!
echo "Python server PID: $PYTHON_PID"

# Wait a bit to ensure Python starts
sleep 5

# Check if Python is still running
if ! ps -p $PYTHON_PID > /dev/null 2>&1; then
    echo "Python server failed to start. Check python_server.log"
    cat python_server.log
    exit 1
fi

# Start Go server in foreground
echo "Starting Go API server..."
cd ../../adapters/api || { echo "Failed to cd into adapters/api"; exit 1; }
go mod tidy || { echo "Failed to run go mod tidy"; exit 1; }
go run main.go || { echo "Failed to run Go server"; exit 1; }

# Keep terminal open (alternative to pause)
read -p "Press Enter to exit..."