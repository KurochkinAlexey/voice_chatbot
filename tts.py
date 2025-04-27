from TTS.api import TTS
import torch


class TextToSpeech:
    def __init__(self, model_type="tts_models/en/ljspeech/glow-tts"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TTS(model_type).to(device)
    
    def to_audio(self, text):
        wav = self.model.tts(text=text)
        return wav