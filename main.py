import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import subprocess
from vad_utils import FixedVADIterator
import torch
from collections import deque
import uvicorn
from asr import FasterWhisperASR
from llms import LLMChat
from tts import TextToSpeech
import librosa
import base64
from soundfile import write as sf_write
from io import BytesIO
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

SAMPLE_RATE = 16000
CHUNK_SIZE_MS = 40
SAMPLES_PER_CHUNK = (SAMPLE_RATE * CHUNK_SIZE_MS) // 1000
BYTES_PER_CHUNK = SAMPLES_PER_CHUNK * 4  
MAX_SPEECH_SECONDS = 30
MAX_SPEECH_BYTES = SAMPLE_RATE * MAX_SPEECH_SECONDS * 4
ENERGY_T = 0.004

vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
asr_model = FasterWhisperASR()
llm_chat = LLMChat()
tts_model = TextToSpeech()

@app.on_event("startup")
async def startup_event():
    global ollama_process
    logger.info("Starting ollama server...")
    ollama_process = await asyncio.create_subprocess_exec(
        "ollama",
        "serve",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    # Wait briefly to ensure Ollama is ready
    await asyncio.sleep(2)
    logger.info("Ollama server started")

@app.on_event("shutdown")
async def shutdown_event():
    global ollama_process
    if ollama_process:
        logger.info("Stopping ollama server...")
        ollama_process.terminate()
        try:
            await asyncio.wait_for(ollama_process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            ollama_process.kill()
        logger.info("Ollama server stopped")

@app.get("/")
async def get():
    return HTMLResponse(open("static/mic_recorder.html").read())

async def ffmpeg_process():
    return await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-f", "s16le",            # Input format: 16-bit PCM
        "-ar", "48000",           # Input sample rate (matches browser)
        "-ac", "1",               # Mono audio
        "-i", "pipe:0",           # Input from stdin
        "-acodec", "pcm_f32le",   # Output raw PCM (32-bit float)
        "-ar", "16000",           # Resample to 16kHz
        "-ac", "1",               # Mono audio
        "-f", "f32le",            # Output format: 32-bit float
        "pipe:1",                 # Output to stdout
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info('Client connected')

    process = await ffmpeg_process()
    vad = FixedVADIterator(vad_model)
    
    raw_audio_buffer = bytearray()
    speech_buffer = bytearray()
    was_triggered = False
    to_stt = False

    async def writer():
        try:
            while True:
                data = await websocket.receive_bytes()
                process.stdin.write(data)
                await process.stdin.drain()
        except Exception as e:
            msg = "Writer error: {}".format(e)
            logger.warning(msg)

    async def text_to_speech(text: str, source_sr=22050, target_sr=16000):
        wav = tts_model.to_audio(text)
        wav = np.array(wav, dtype=np.float32)
        wav = librosa.resample(wav, orig_sr=source_sr, target_sr=target_sr)
        
        buffer = BytesIO()

        sf_write(buffer, wav, target_sr, format='WAV', subtype='FLOAT')

        buffer.seek(0)

        encoded_audio = base64.b64encode(buffer.read()).decode('utf-8')
        return encoded_audio
        

    async def reader():
        nonlocal raw_audio_buffer, speech_buffer, was_triggered, to_stt
        
        try:
            while True:
                pcm_data = await process.stdout.read(4096)
                if not pcm_data:
                    break
                
                raw_audio_buffer.extend(pcm_data)
                
                while len(raw_audio_buffer) >= BYTES_PER_CHUNK:
                    chunk = raw_audio_buffer[:BYTES_PER_CHUNK]
                    del raw_audio_buffer[:BYTES_PER_CHUNK]
                    
                    # check if there is voice in audio chunk
                    np_chunk = np.frombuffer(chunk, dtype=np.float32)
                    vad(np_chunk)
                    
                    # process speech start/end 
                    current_triggered = vad.triggered
                    if current_triggered and not was_triggered:
                        logger.info('Speech started')
                        speech_buffer.clear()
                        was_triggered = True
                        
                    elif not current_triggered and was_triggered:
                        logger.info('Speech ended')
                        was_triggered = False
                        to_stt = True
                    
                    if was_triggered:
                        # if speech was detected, start accumulate data
                        
                        remaining_space = MAX_SPEECH_BYTES - len(speech_buffer)
                        if remaining_space >= len(chunk):
                            speech_buffer.extend(chunk)
                        else:
                            # if speech length is longer than 30 sec, stop accumulating
                            if remaining_space > 0:
                                speech_buffer.extend(chunk[:remaining_space])
                            logger.warning('Speech buffer is full - data acumulation is stopped')
                            was_triggered = False
                
                    # STT -> LLM -> TTS
                    if to_stt and len(speech_buffer) > 0:
                        to_stt = False
                        try:
                            # Copy current speech buffer, transform speech to text
                            current_speech_bytes = bytes(speech_buffer)
                            speech = np.frombuffer(current_speech_bytes, dtype=np.float32)
                            logger.debug('Energy: {}'.format(np.mean(np.abs(speech))))
                            if np.mean(np.abs(speech)) < ENERGY_T:
                                #possibly synthetic speech, ignore
                                speech_buffer.clear()
                                vad.reset_states()
                                continue
                                
                            tokens = asr_model.transcribe(speech)                        
                            question = "".join([t.text for t in tokens])
                            logger.info('STT result: {}'.format(question))
                            
                            # Send question to client
                            await websocket.send_json({
                                "type": "question",
                                "text": question
                            })
                            
                            # Detect stop word to interrupt chatbot voice
                            if 'stop' in question.lower():
                                logger.info('Stop command detected') #TODO replace with KWS model
                                await websocket.send_json({
                                    "type": "command",
                                    "action": "stop_playback"
                                })
                                speech_buffer.clear()
                                vad.reset_states()
                                continue
                            
                            # Get LLM response
                            response = llm_chat.respond(question)['response']
                            logger.info('LLM response: {}'.format(response))
                            
                            # Convert response to speech and send
                            encoded_audio = await text_to_speech(response)
                            
                            await websocket.send_json({
                                "type": "response",
                                "text": response,
                                "audio": encoded_audio
                            })
                            
                        except Exception as e:
                            logger.warning('STT/LLM/TTS error : {}'.format(e))
                        finally:
                            speech_buffer.clear()
                            vad.reset_states()
    
        except Exception as e:
            logger.warning('Reader error : {}'.format(e))
            raise e
        finally:
            vad.reset_states()

    try:
        await asyncio.gather(reader(), writer())
    except Exception as e:
        logger.warning('Main loop error: {}'.format(e))
        raise e
    finally:
        if process.stdin:
            process.stdin.close()
            await process.stdin.wait_closed()
        if process.stdout:
            process.stdout.close()
        await process.wait()
        
             