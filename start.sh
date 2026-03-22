#!/bin/bash

PORT=${PORT:-5001}
PID_FILE=".chatbot.pid"

echo "Starting FAQ Chatbot..."

# Kill anything already on the port
existing=$(lsof -ti tcp:$PORT 2>/dev/null)
if [ -n "$existing" ]; then
  echo "Port $PORT in use — killing existing process (PID $existing)..."
  kill -9 $existing 2>/dev/null
  sleep 1
fi

# Clean up stale PID file
rm -f "$PID_FILE"

# Load env and start Flask in background
set -a && source .env && set +a
python app.py &
APP_PID=$!
echo $APP_PID > "$PID_FILE"

echo ""
echo "Chatbot running at http://localhost:$PORT"
echo "Admin panel at  http://localhost:$PORT/admin"
echo ""
echo "To stop: bash stop.sh"
