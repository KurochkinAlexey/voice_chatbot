#!/bin/bash

set -e

echo "Starting Ollama server in background..."
/usr/local/bin/ollama serve &
OLLAMA_PID=$!
echo "Ollama server started with PID $OLLAMA_PID."

echo "Waiting for Ollama server to be ready..."
sleep 5 

while ! curl -s http://localhost:11434/ > /dev/null; do
    echo "Ollama not ready yet, waiting 2 more seconds..."
    sleep 2
    if ! ps -p $OLLAMA_PID > /dev/null; then
        echo "Ollama server process terminated unexpectedly!"
        exit 1
    fi
done
echo "Ollama server is ready."

MODEL_TO_PULL=${OLLAMA_MODEL:-qwen2.5:7b}
echo "Pulling Ollama model: $MODEL_TO_PULL ..."
/usr/local/bin/ollama pull $MODEL_TO_PULL
echo "Model '$MODEL_TO_PULL' pulled successfully."

echo "Starting Python application (main.py)..."

uvicorn main:app --host 0.0.0.0 --port 8000
#python3 main.py

trap 'echo "Stopping Ollama server..."; kill $OLLAMA_PID; exit' SIGINT SIGTERM

wait $OLLAMA_PID
echo "Ollama server stopped."