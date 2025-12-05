# Project I: Audio AI Pipeline

## Objective

Build a comprehensive audio processing pipeline demonstrating speech-to-text (Whisper), text-to-speech (Coqui TTS, Bark), speaker diarization, audio classification, voice activity detection, and real-time processing to create a production-ready audio AI system.

**What You'll Build**: A complete audio AI toolkit with Whisper transcription (100+ languages), Coqui TTS synthesis, speaker diarization, audio classification, voice cloning, noise reduction, and a Gradio interface for interactive demos.

**What You'll Learn**: Audio processing fundamentals, speech recognition with Whisper, text-to-speech synthesis, speaker diarization, audio classification, voice activity detection (VAD), audio preprocessing, real-time streaming, and production deployment.

## Time Estimate

**2 days (16 hours)** - Following the implementation plan

### Day 1 (8 hours)
- **Hour 1**: Setup & basics (install dependencies, test audio I/O, verify models)
- **Hours 2-3**: Speech-to-text (Whisper models, multilingual, timestamps, word-level)
- **Hours 4-5**: Text-to-speech (Coqui TTS, Bark, voice cloning basics)
- **Hours 6-7**: Audio preprocessing (noise reduction, normalization, VAD)
- **Hour 8**: Audio classification (sound events, music genre, emotion)

### Day 2 (8 hours)
- **Hours 1-2**: Speaker diarization (who spoke when, pyannote.audio)
- **Hours 3-4**: Real-time processing (streaming audio, WebRTC VAD)
- **Hours 5-6**: Pipeline integration (STT → processing → TTS workflow)
- **Hour 7**: Gradio UI (interactive demos, file upload, microphone input)
- **Hour 8**: Documentation & deployment (FastAPI, Docker, optimization)

## Prerequisites

### Required Knowledge
Complete these bootcamp sections first:
- [100 Days Data & AI](https://github.com/washimimizuku/100-days-data-ai) - Days 71-85
  - Days 71-75: Audio AI fundamentals
  - Days 76-80: Speech recognition and synthesis
  - Days 81-85: Audio processing and applications
- [30 Days of Python](https://github.com/washimimizuku/30-days-python-data-ai) - Days 1-20

### Technical Requirements
- Python 3.11+ installed
- 8GB+ RAM (16GB recommended)
- GPU recommended for faster processing (optional)
- Microphone for real-time testing (optional)
- Understanding of audio basics (sampling rate, channels)

### Tools Needed
- Python with whisper, TTS, torch, torchaudio
- FFmpeg for audio processing
- Gradio for UI
- FastAPI for API
- Git for version control

## Getting Started

### Step 1: Review Documentation
Read the project documents in order:
1. `prd.md` - Understand what you're building
2. `tech-spec.md` - Review technical architecture
3. `implementation-plan.md` - Follow the implementation steps

### Step 2: Install FFmpeg
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html

# Verify installation
ffmpeg -version
```

### Step 3: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install openai-whisper torch torchaudio

# Install TTS libraries
pip install TTS bark

# Install audio processing
pip install librosa soundfile pydub noisereduce

# Install speaker diarization
pip install pyannote.audio

# Install API and UI
pip install fastapi uvicorn gradio

# Install utilities
pip install datasets tqdm

# Verify installation
python -c "import whisper; print('✓ Whisper installed')"
python -c "from TTS.api import TTS; print('✓ TTS installed')"
```

### Step 4: Test Audio I/O
```python
# test_audio.py
import librosa
import soundfile as sf
import numpy as np

# Load audio file
audio, sr = librosa.load('test.wav', sr=16000)
print(f"Audio shape: {audio.shape}")
print(f"Sample rate: {sr}")
print(f"Duration: {len(audio)/sr:.2f} seconds")

# Save audio file
sf.write('output.wav', audio, sr)
print("✓ Audio I/O working")

# Generate test audio (sine wave)
duration = 3  # seconds
frequency = 440  # Hz (A4 note)
t = np.linspace(0, duration, int(sr * duration))
test_audio = np.sin(2 * np.pi * frequency * t)
sf.write('test_sine.wav', test_audio, sr)
print("✓ Generated test audio")
```


### Step 5: Implement Speech-to-Text with Whisper
```python
# src/stt/whisper_transcriber.py
import whisper
import torch
from typing import Dict, List
import time

class WhisperTranscriber:
    """Speech-to-text using OpenAI Whisper"""
    
    def __init__(self, model_size: str = "base", device: str = None):
        """
        Initialize Whisper model
        
        Args:
            model_size: tiny, base, small, medium, large
            device: cuda, cpu, or None (auto-detect)
        """
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading Whisper {model_size} model on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)
        print("✓ Model loaded")
    
    def transcribe(
        self,
        audio_path: str,
        language: str = None,
        task: str = "transcribe",
        word_timestamps: bool = False
    ) -> Dict:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'es', 'fr') or None for auto-detect
            task: 'transcribe' or 'translate' (to English)
            word_timestamps: Include word-level timestamps
        """
        start = time.time()
        
        result = self.model.transcribe(
            audio_path,
            language=language,
            task=task,
            word_timestamps=word_timestamps,
            verbose=False
        )
        
        elapsed = time.time() - start
        
        return {
            'text': result['text'],
            'language': result['language'],
            'segments': result['segments'],
            'duration': elapsed
        }

    
    def transcribe_with_timestamps(self, audio_path: str) -> List[Dict]:
        """Get detailed timestamps for each segment"""
        result = self.transcribe(audio_path, word_timestamps=True)
        
        segments = []
        for segment in result['segments']:
            segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip(),
                'words': segment.get('words', [])
            })
        
        return segments
    
    def transcribe_multilingual(self, audio_path: str) -> Dict:
        """Auto-detect language and transcribe"""
        result = self.transcribe(audio_path, language=None)
        
        return {
            'text': result['text'],
            'detected_language': result['language'],
            'confidence': 'high' if len(result['text']) > 50 else 'medium'
        }

# Usage
if __name__ == "__main__":
    transcriber = WhisperTranscriber(model_size="base")
    
    # Basic transcription
    result = transcriber.transcribe("audio.mp3")
    print(f"Transcription: {result['text']}")
    print(f"Language: {result['language']}")
    print(f"Duration: {result['duration']:.2f}s")
    
    # With timestamps
    segments = transcriber.transcribe_with_timestamps("audio.mp3")
    for seg in segments:
        print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")
```


### Step 6: Implement Text-to-Speech
```python
# src/tts/tts_synthesizer.py
from TTS.api import TTS
import torch
import numpy as np
import soundfile as sf

class TTSSynthesizer:
    """Text-to-speech using Coqui TTS"""
    
    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading TTS model: {model_name}")
        self.tts = TTS(model_name).to(self.device)
        print("✓ TTS model loaded")
    
    def synthesize(
        self,
        text: str,
        output_path: str = "output.wav",
        speaker: str = None
    ):
        """Synthesize speech from text"""
        
        # Generate audio
        if speaker:
            self.tts.tts_to_file(text=text, file_path=output_path, speaker=speaker)
        else:
            self.tts.tts_to_file(text=text, file_path=output_path)
        
        print(f"✓ Audio saved to {output_path}")
        return output_path
    
    def list_speakers(self) -> List[str]:
        """List available speakers (for multi-speaker models)"""
        if hasattr(self.tts, 'speakers'):
            return self.tts.speakers
        return []

# Bark for more natural speech
from bark import SAMPLE_RATE, generate_audio, preload_models

class BarkSynthesizer:
    """Text-to-speech using Bark (more natural, slower)"""
    
    def __init__(self):
        print("Loading Bark models...")
        preload_models()
        print("✓ Bark loaded")
    
    def synthesize(self, text: str, output_path: str = "output.wav"):
        """Generate natural speech with Bark"""
        
        audio_array = generate_audio(text)
        sf.write(output_path, audio_array, SAMPLE_RATE)
        
        print(f"✓ Audio saved to {output_path}")
        return output_path

# Usage
if __name__ == "__main__":
    # Coqui TTS
    tts = TTSSynthesizer()
    tts.synthesize("Hello, this is a test of text to speech synthesis.")
    
    # Bark (more natural)
    bark = BarkSynthesizer()
    bark.synthesize("Hello! [laughs] This is Bark, a more natural sounding voice.")
```


### Step 7: Implement Speaker Diarization
```python
# src/diarization/speaker_diarization.py
from pyannote.audio import Pipeline
import torch

class SpeakerDiarizer:
    """Speaker diarization - who spoke when"""
    
    def __init__(self, auth_token: str = None):
        """
        Initialize diarization pipeline
        Requires HuggingFace token for pyannote models
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if auth_token:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=auth_token
            )
            self.pipeline.to(self.device)
            print("✓ Diarization pipeline loaded")
        else:
            print("⚠️  No auth token provided. Get one from https://huggingface.co/pyannote/speaker-diarization")
            self.pipeline = None
    
    def diarize(self, audio_path: str) -> Dict:
        """Perform speaker diarization"""
        
        if not self.pipeline:
            return {'error': 'Pipeline not initialized'}
        
        diarization = self.pipeline(audio_path)
        
        # Extract speaker segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker,
                'duration': turn.end - turn.start
            })
        
        # Get speaker statistics
        speakers = {}
        for seg in segments:
            speaker = seg['speaker']
            if speaker not in speakers:
                speakers[speaker] = {'total_time': 0, 'segments': 0}
            speakers[speaker]['total_time'] += seg['duration']
            speakers[speaker]['segments'] += 1
        
        return {
            'segments': segments,
            'speakers': speakers,
            'num_speakers': len(speakers)
        }

# Usage
if __name__ == "__main__":
    # Requires HuggingFace token
    diarizer = SpeakerDiarizer(auth_token="your_token_here")
    result = diarizer.diarize("conversation.wav")
    
    print(f"Number of speakers: {result['num_speakers']}")
    for speaker, stats in result['speakers'].items():
        print(f"{speaker}: {stats['total_time']:.1f}s ({stats['segments']} segments)")
```


### Step 8: Implement Audio Classification
```python
# src/classification/audio_classifier.py
from transformers import pipeline
import librosa

class AudioClassifier:
    """Classify audio into categories"""
    
    def __init__(self, task: str = "audio-classification"):
        print("Loading audio classification model...")
        self.classifier = pipeline(
            task,
            model="MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        print("✓ Classifier loaded")
    
    def classify(self, audio_path: str, top_k: int = 5) -> List[Dict]:
        """Classify audio file"""
        
        results = self.classifier(audio_path, top_k=top_k)
        
        return [
            {
                'label': r['label'],
                'score': r['score']
            }
            for r in results
        ]
    
    def classify_emotion(self, audio_path: str) -> Dict:
        """Classify emotion in speech"""
        # Load emotion classification model
        emotion_classifier = pipeline(
            "audio-classification",
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        )
        
        results = emotion_classifier(audio_path)
        return results[0]

# Usage
if __name__ == "__main__":
    classifier = AudioClassifier()
    
    # Classify sound
    results = classifier.classify("audio.wav")
    print("Top predictions:")
    for r in results:
        print(f"  {r['label']}: {r['score']:.3f}")
```

## Key Features to Implement

### 1. Speech-to-Text (Whisper)
- **Multiple models**: tiny, base, small, medium, large
- **100+ languages**: Auto-detection or specified
- **Timestamps**: Word-level and segment-level
- **Translation**: Translate to English
- **Batch processing**: Multiple files

### 2. Text-to-Speech
- **Coqui TTS**: Fast, multiple voices
- **Bark**: Natural, expressive speech
- **Voice cloning**: Basic speaker adaptation
- **Multi-speaker**: Different voices
- **Prosody control**: Speed, pitch

### 3. Speaker Diarization
- **Who spoke when**: Timeline of speakers
- **Speaker counting**: Auto-detect number
- **Speaker statistics**: Talk time per speaker
- **Overlap detection**: Simultaneous speech

### 4. Audio Classification
- **Sound events**: Dog bark, car horn, music
- **Music genre**: Rock, jazz, classical
- **Emotion**: Happy, sad, angry, neutral
- **Language detection**: Identify language

### 5. Audio Preprocessing
- **Noise reduction**: Remove background noise
- **Normalization**: Consistent volume
- **VAD**: Voice activity detection
- **Resampling**: Convert sample rates
- **Format conversion**: WAV, MP3, FLAC


### 6. Real-Time Processing
- **Streaming audio**: Process as it arrives
- **WebRTC VAD**: Real-time voice detection
- **Low latency**: Sub-second processing
- **Microphone input**: Live transcription

### 7. Pipeline Integration
- **STT → TTS**: Transcribe and re-synthesize
- **Translation pipeline**: Speech translation
- **Dubbing**: Replace audio track
- **Subtitles**: Generate SRT files

## Success Criteria

By the end of this project, you should have:

- [ ] Whisper transcription working (multiple languages)
- [ ] TTS synthesis with Coqui TTS and Bark
- [ ] Speaker diarization functional
- [ ] Audio classification accurate
- [ ] Noise reduction implemented
- [ ] VAD working
- [ ] Real-time streaming (optional)
- [ ] Gradio UI with all features
- [ ] FastAPI endpoints deployed
- [ ] Batch processing capability
- [ ] Performance benchmarks documented
- [ ] GitHub repository with examples

## Learning Outcomes

After completing this project, you'll be able to:

- Process audio files programmatically
- Use Whisper for speech recognition
- Synthesize speech with TTS models
- Perform speaker diarization
- Classify audio content
- Preprocess and clean audio
- Detect voice activity
- Build real-time audio pipelines
- Deploy audio AI APIs
- Optimize audio processing performance
- Handle multiple audio formats
- Understand audio ML model trade-offs

## Expected Performance

Based on typical results:

**Whisper Transcription** (base model, CPU):
- Speed: 0.5-1x realtime (1 min audio = 1-2 min processing)
- Accuracy: 95%+ for clear English audio
- Languages: 100+ supported
- GPU speedup: 5-10x faster

**TTS Synthesis**:
- Coqui TTS: 0.1-0.2x realtime (very fast)
- Bark: 0.01-0.05x realtime (slower, more natural)
- Quality: Near-human for both

**Speaker Diarization**:
- Accuracy: 85-95% (depends on audio quality)
- Processing: 0.2-0.5x realtime
- Min speakers: 2, Max: 10+

**Audio Classification**:
- Accuracy: 80-90% for common sounds
- Latency: 50-200ms per file
- Categories: 500+ sound events

## Project Structure

```
project-i-audio-pipeline/
├── src/
│   ├── stt/
│   │   ├── whisper_transcriber.py
│   │   └── realtime_stt.py
│   ├── tts/
│   │   ├── tts_synthesizer.py
│   │   ├── bark_synthesizer.py
│   │   └── voice_cloning.py
│   ├── diarization/
│   │   └── speaker_diarization.py
│   ├── classification/
│   │   ├── audio_classifier.py
│   │   └── emotion_classifier.py
│   ├── preprocessing/
│   │   ├── noise_reduction.py
│   │   ├── vad.py
│   │   └── audio_utils.py
│   ├── pipeline/
│   │   └── audio_pipeline.py
│   ├── api/
│   │   └── fastapi_server.py
│   └── ui/
│       └── gradio_app.py
├── data/
│   ├── audio_samples/
│   └── outputs/
├── models/
│   └── whisper/
├── notebooks/
│   ├── 01_whisper_tutorial.ipynb
│   ├── 02_tts_examples.ipynb
│   ├── 03_diarization.ipynb
│   └── 04_full_pipeline.ipynb
├── tests/
│   └── test_pipeline.py
├── prd.md
├── tech-spec.md
├── implementation-plan.md
└── README.md
```


## Common Challenges & Solutions

### Challenge 1: Slow Whisper Transcription
**Problem**: Transcription takes too long on CPU
**Solution**: Use smaller model, GPU acceleration, or batch processing
```python
# Use smaller model
model = whisper.load_model("tiny")  # Faster, less accurate

# Use GPU
model = whisper.load_model("base", device="cuda")

# Batch process with multiprocessing
from multiprocessing import Pool
with Pool(4) as p:
    results = p.map(transcribe_file, audio_files)
```

### Challenge 2: Poor Audio Quality
**Problem**: Transcription accuracy is low
**Solution**: Preprocess audio (noise reduction, normalization)
```python
import noisereduce as nr
import librosa

# Load audio
audio, sr = librosa.load("noisy.wav", sr=16000)

# Reduce noise
audio_clean = nr.reduce_noise(y=audio, sr=sr)

# Normalize
audio_norm = librosa.util.normalize(audio_clean)

# Save and transcribe
sf.write("clean.wav", audio_norm, sr)
result = model.transcribe("clean.wav")
```

### Challenge 3: TTS Sounds Robotic
**Problem**: Synthesized speech lacks naturalness
**Solution**: Use Bark or fine-tune TTS model
```python
# Use Bark for more natural speech
from bark import generate_audio
audio = generate_audio(
    "Hello! [laughs] This sounds much more natural.",
    history_prompt="v2/en_speaker_6"  # Choose speaker
)
```

### Challenge 4: Speaker Diarization Errors
**Problem**: Speakers not correctly identified
**Solution**: Improve audio quality, adjust parameters
```python
# Use better audio preprocessing
from pydub import AudioSegment
from pydub.effects import normalize

audio = AudioSegment.from_wav("audio.wav")
audio = normalize(audio)
audio.export("normalized.wav", format="wav")

# Adjust diarization parameters
diarization = pipeline(
    "normalized.wav",
    min_speakers=2,
    max_speakers=5
)
```

## Advanced Techniques

### 1. Real-Time Streaming
```python
# src/realtime/streaming_stt.py
import pyaudio
import numpy as np
from faster_whisper import WhisperModel

class RealtimeTranscriber:
    def __init__(self):
        self.model = WhisperModel("base", device="cuda")
        self.audio = pyaudio.PyAudio()
        
    def stream_transcribe(self):
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )
        
        print("Listening...")
        audio_buffer = []
        
        try:
            while True:
                data = stream.read(1024)
                audio_buffer.append(np.frombuffer(data, dtype=np.int16))
                
                # Transcribe every 3 seconds
                if len(audio_buffer) >= 48:  # 3 seconds at 16kHz
                    audio_chunk = np.concatenate(audio_buffer)
                    segments, _ = self.model.transcribe(audio_chunk)
                    
                    for segment in segments:
                        print(f"[{segment.start:.1f}s] {segment.text}")
                    
                    audio_buffer = []
        
        except KeyboardInterrupt:
            stream.stop_stream()
            stream.close()
```

### 2. Voice Cloning
```python
# Basic voice cloning with Coqui TTS
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

# Clone voice from reference audio
tts.tts_to_file(
    text="This is cloned speech",
    file_path="output.wav",
    speaker_wav="reference_voice.wav",  # Reference audio
    language="en"
)
```

### 3. Audio Translation Pipeline
```python
# Translate speech to another language
def translate_audio(input_audio, target_lang="es"):
    # 1. Transcribe to English
    transcriber = WhisperTranscriber()
    result = transcriber.transcribe(input_audio, task="translate")
    english_text = result['text']
    
    # 2. Translate text (using translation model)
    from transformers import pipeline
    translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-{target_lang}")
    translated = translator(english_text)[0]['translation_text']
    
    # 3. Synthesize in target language
    tts = TTS(f"tts_models/{target_lang}/...")
    tts.tts_to_file(translated, "translated.wav")
    
    return translated
```

### 4. Subtitle Generation
```python
# Generate SRT subtitles
def generate_subtitles(audio_path, output_srt="subtitles.srt"):
    transcriber = WhisperTranscriber()
    segments = transcriber.transcribe_with_timestamps(audio_path)
    
    with open(output_srt, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            start = format_timestamp(seg['start'])
            end = format_timestamp(seg['end'])
            
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{seg['text']}\n\n")

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
```


## Troubleshooting

### Installation Issues

**Issue**: FFmpeg not found
```bash
# Solution: Install FFmpeg
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
# Windows: Download from ffmpeg.org

# Verify
ffmpeg -version
```

**Issue**: Whisper installation fails
```bash
# Solution: Install with specific torch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install openai-whisper
```

**Issue**: TTS model download fails
```python
# Solution: Download manually
from TTS.utils.manage import ModelManager
manager = ModelManager()
manager.download_model("tts_models/en/ljspeech/tacotron2-DDC")
```

### Runtime Issues

**Issue**: "CUDA out of memory"
```python
# Solution: Use smaller model or CPU
model = whisper.load_model("tiny", device="cpu")

# Or reduce batch size
result = model.transcribe(audio, batch_size=8)
```

**Issue**: Poor transcription quality
```python
# Solution: Preprocess audio
import noisereduce as nr
audio_clean = nr.reduce_noise(y=audio, sr=sr)

# Use larger model
model = whisper.load_model("medium")

# Specify language
result = model.transcribe(audio, language="en")
```

**Issue**: TTS output is distorted
```python
# Solution: Check sample rate and normalization
import librosa
audio, sr = librosa.load("output.wav")
audio_norm = librosa.util.normalize(audio)
sf.write("fixed.wav", audio_norm, sr)
```

## Production Deployment

### FastAPI Server
```python
# src/api/fastapi_server.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import whisper
import tempfile
import os

app = FastAPI(title="Audio AI API")

# Load models at startup
@app.on_event("startup")
async def startup_event():
    global stt_model, tts_model
    stt_model = whisper.load_model("base")
    tts_model = TTS("tts_models/en/ljspeech/tacotron2-DDC")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    # Transcribe
    result = stt_model.transcribe(tmp_path)
    
    # Cleanup
    os.unlink(tmp_path)
    
    return {"text": result["text"], "language": result["language"]}

@app.post("/synthesize")
async def synthesize(text: str = Form(...)):
    output_path = tempfile.mktemp(suffix=".wav")
    tts_model.tts_to_file(text=text, file_path=output_path)
    
    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename="speech.wav"
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Run: uvicorn fastapi_server:app --host 0.0.0.0 --port 8000
```

### Gradio UI
```python
# src/ui/gradio_app.py
import gradio as gr
import whisper
from TTS.api import TTS

# Load models
stt_model = whisper.load_model("base")
tts_model = TTS("tts_models/en/ljspeech/tacotron2-DDC")

def transcribe_audio(audio_file):
    result = stt_model.transcribe(audio_file)
    return result["text"]

def synthesize_speech(text):
    output_path = "output.wav"
    tts_model.tts_to_file(text=text, file_path=output_path)
    return output_path

# Create interface
with gr.Blocks() as demo:
    gr.Markdown("# Audio AI Pipeline")
    
    with gr.Tab("Speech-to-Text"):
        audio_input = gr.Audio(type="filepath", label="Upload Audio")
        transcribe_btn = gr.Button("Transcribe")
        transcription_output = gr.Textbox(label="Transcription")
        
        transcribe_btn.click(
            transcribe_audio,
            inputs=audio_input,
            outputs=transcription_output
        )
    
    with gr.Tab("Text-to-Speech"):
        text_input = gr.Textbox(label="Enter Text", lines=3)
        synthesize_btn = gr.Button("Synthesize")
        audio_output = gr.Audio(label="Generated Speech")
        
        synthesize_btn.click(
            synthesize_speech,
            inputs=text_input,
            outputs=audio_output
        )

demo.launch(share=True)
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim

# Install FFmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download models at build time
RUN python -c "import whisper; whisper.load_model('base')"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Resources

### Documentation
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [Coqui TTS](https://github.com/coqui-ai/TTS) - Text-to-speech
- [Bark](https://github.com/suno-ai/bark) - Generative audio
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- [Librosa](https://librosa.org/) - Audio processing

### Papers
- [Whisper: Robust Speech Recognition](https://arxiv.org/abs/2212.04356)
- [Bark: Text-to-Audio Model](https://github.com/suno-ai/bark)
- [Speaker Diarization](https://arxiv.org/abs/2012.01477)

### Tutorials
- [Whisper Tutorial](https://github.com/openai/whisper#python-usage)
- [TTS Guide](https://tts.readthedocs.io/)
- [Audio Processing](https://librosa.org/doc/latest/tutorial.html)

## Questions?

If you get stuck:
1. Review the `tech-spec.md` for detailed architecture
2. Check model documentation for API details
3. Test with short audio clips first (< 30 seconds)
4. Review the 100 Days bootcamp materials on audio AI
5. Verify FFmpeg is installed and working
6. Use smaller models for faster iteration

## Related Projects

After completing this project, consider:
- **Project A**: Local RAG - Add voice interface to RAG
- **Project F**: Inference Optimization - Optimize Whisper inference
- **Project G**: Prompt Engineering - Generate better TTS prompts
- Build a voice assistant with STT + LLM + TTS
- Create a podcast transcription service
- Build an audio translation app
