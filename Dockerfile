FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    python3 \
    python3-pip \
    git \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1-dev \
    ffmpeg \
    espeak \
    curl \
    sqlite3 \
    libsqlite3-dev \
    systemd \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb    
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update
RUN apt-get -y install cudnn-cuda-12 

RUN git clone https://github.com/KurochkinAlexey/voice_chatbot /app
WORKDIR /app

RUN pip3 install torch
RUN pip3 install fastapi
RUN pip3 install faster_whisper
RUN pip3 install langchain
RUN pip3 install langchain_community
RUN pip3 install langchain_core
RUN pip3 install langchain_ollama
RUN pip3 install librosa
RUN pip3 install pandas
RUN pip3 install soundfile
RUN pip3 install TTS
RUN pip3 install uvicorn
RUN pip3 install 'uvicorn[standard]'
RUN pip3 install websocket
#RUN pip3 install -r requirements.txt
#RUN pip3 install --no-cache-dir -r requirements.txt

RUN curl -fsSL https://ollama.com/install.sh | sh

ENV PATH="$PATH:/usr/local/bin"

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

ENV PYTHONPATH "${PYTHONPATH}:/app"

ENV OLLAMA_MODEL=qwen2.5:7b

RUN mkdir -p \
    /root/.cache/huggingface \
    /root/.cache/torch \
    /root/.cache/tts \
    /root/.ollama

ENV HF_HOME=/root/.cache/huggingface
ENV TORCH_HOME=/root/.cache/torch
ENV XDG_CACHE_HOME=/root/.cache
ENV OLLAMA_HOME=/root/.ollama
ENV TTS_HOME=/root/.cache/tts

VOLUME [ \
    "/root/.cache/huggingface", \
    "/root/.cache/torch", \
    "/root/.cache/tts", \
    "/root/.ollama" \
]

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 11434
EXPOSE 8000

CMD ["/start.sh"]

