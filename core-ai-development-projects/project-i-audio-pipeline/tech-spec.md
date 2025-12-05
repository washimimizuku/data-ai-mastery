# Tech Spec: Audio AI Pipeline

## Stack
Whisper, Coqui TTS, Bark, PyTorch, librosa

## Models
```python
# Speech-to-Text
import whisper
model = whisper.load_model("base")

# Text-to-Speech
from TTS.api import TTS
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")

# Bark (generative audio)
from bark import generate_audio
```

## Features
- STT with Whisper (multiple languages)
- TTS with Coqui TTS
- Speaker diarization (pyannote)
- Audio classification
- Real-time processing
- Gradio interface
