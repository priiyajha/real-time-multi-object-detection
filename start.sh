#!/bin/bash

# This script simplifies starting the WebRTC demo in different modes.

# Check if MODE is set, otherwise default to wasm
MODE=${MODE:-wasm}
NGROK_TUNNEL=${NGROK_TUNNEL:-false}

echo "Starting WebRTC VLM demo in $MODE mode..."

# Build and start Docker containers
if [ "$NGROK_TUNNEL" = "true" ]; then
  # Expose a public URL using ngrok
  docker-compose up --build
  # You would need to manually run ngrok and update the URL in your phone
  echo "ngrok tunnel started. Copy the public URL from the ngrok console and paste it in your phone browser."
else
  # Regular start for local network
  docker-compose up --build
fi

# To run in server mode, use: MODE=server ./start.sh
# To run with ngrok tunnel, use: NGROK_TUNNEL=true ./start.sh
# Note: ngrok must be installed and configured on your host machine.
