#!/bin/bash

mkdir -p models
mkdir -p models/ollama
mkdir -p models/hf
mkdir -p models/torch
mkdir -p models/cache
mkdir -p models/tts

sudo docker build -t voice-chatbot .

