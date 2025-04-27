from faster_whisper import WhisperModel


class FasterWhisperASR:
    def __init__(self, model_size='medium', lang='en', device="cuda", compute_type="float16", beam_size=5, condition_on_prev=True):
        model_desc = model_size if lang != 'en' else '.'.join([model_size, lang]) 
        self.model = WhisperModel(model_desc, device=device, compute_type=compute_type)
        self.transcribe_params = {
            'language': lang,
            'beam_size': beam_size,
            'condition_on_previous_text': condition_on_prev,
            'vad_filter': False,
        }
    
    def transcribe(self, speech):
        segments, _ = self.model.transcribe(speech, **self.transcribe_params)
        return list(segments)