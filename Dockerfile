# Use NVIDIA CUDA 12.2 base image with Ubuntu 22.04
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1-dev \
    ffmpeg \
    espeak \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Clone the voice chatbot repository
RUN git clone https://github.com/KurochkinAlexey/voice_chatbot /app
WORKDIR /app

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Add Ollama to PATH
ENV PATH="$PATH:/usr/local/bin"

# Set up CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Set up environment for voice processing
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Create directories
RUN mkdir -p /app/audio_files
RUN mkdir -p /usr/share/ollama/.ollama

# Set the entry point
ENTRYPOINT ["uvicorn", "main:app"]