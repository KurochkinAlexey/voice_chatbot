#!/bin/bash

sudo docker run -it\
  --gpus all \
  -v /models/hf:/root/.cache/huggingface \
  -v /models/torch:/root/.cache/torch \
  -v /models/cache:/models/cache \
  -v /models/ollama:/root/.ollama \
  -v /models/tts:/root/.cache/tts \
  -p 8000:8000 \
  voice-chatbot