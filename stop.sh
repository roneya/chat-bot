#!/bin/bash

PORT=${PORT:-5001}
PID_FILE=".chatbot.pid"

# Kill by saved PID
if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE")
  if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping chatbot (PID $PID)..."
    kill "$PID"
    sleep 1
  fi
  rm -f "$PID_FILE"
fi

# Also kill anything still on the port (catches child processes)
leftover=$(lsof -ti tcp:$PORT 2>/dev/null)
if [ -n "$leftover" ]; then
  echo "Clearing port $PORT (PID $leftover)..."
  kill -9 $leftover 2>/dev/null
fi

echo "Chatbot stopped. Port $PORT is free."
