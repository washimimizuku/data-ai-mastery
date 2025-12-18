# Project B: Species Audio Classification Pipeline

**Time**: 2-3 days | **Difficulty**: Intermediate | **Focus**: Bioacoustics ML for wildlife monitoring

Create an ML pipeline to identify bird/animal species from audio recordings. This project demonstrates practical applications of audio processing and machine learning for biodiversity monitoring and conservation biology.

## Learning Objectives

- Process and analyze environmental audio data
- Extract meaningful features from bioacoustic recordings
- Train machine learning models for species classification
- Build real-time audio analysis pipeline
- Create biodiversity monitoring dashboard

## Tech Stack

- **Python**: Core programming language
- **librosa**: Audio processing and feature extraction
- **PyTorch**: Deep learning framework
- **Hugging Face**: Pre-trained audio models
- **scikit-learn**: Traditional ML algorithms
- **Streamlit**: Interactive dashboard
- **NumPy/Pandas**: Data manipulation
- **Matplotlib/Plotly**: Audio visualization

## Project Structure

```
project-b-species-audio-classification/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Audio data download and loading
│   ├── preprocessing.py        # Audio preprocessing and feature extraction
│   ├── models.py              # ML models for classification
│   ├── inference.py           # Real-time audio analysis
│   └── dashboard.py           # Streamlit biodiversity dashboard
├── notebooks/
│   ├── 01_audio_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation_analysis.ipynb
├── data/
│   ├── raw/                   # Original audio recordings
│   ├── processed/             # Preprocessed audio features
│   └── models/                # Trained model artifacts
├── config/
│   └── config.yaml            # Project configuration
└── tests/
    └── test_classification.py
```

## Implementation Steps

### Day 1: Data Acquisition and Audio Processing
1. **Setup Environment**
   - Install librosa, PyTorch, and audio dependencies
   - Configure access to Xeno-canto API

2. **Data Collection**
   - Download bird recordings from Xeno-canto
   - Focus on common species with sufficient samples
   - Collect background noise and other sounds

3. **Audio Preprocessing**
   - Noise reduction and filtering
   - Segmentation and windowing
   - Normalization and augmentation

### Day 2: Feature Engineering and Model Training
1. **Feature Extraction**
   - Mel-frequency cepstral coefficients (MFCCs)
   - Spectrograms and mel-spectrograms
   - Chroma features and spectral features

2. **Model Development**
   - Traditional ML: Random Forest, SVM
   - Deep Learning: CNN for spectrogram classification
   - Pre-trained models: Wav2Vec2, Whisper fine-tuning

3. **Training Pipeline**
   - Data splitting and validation
   - Hyperparameter tuning
   - Model evaluation and comparison

### Day 3: Real-time Analysis and Dashboard
1. **Inference Pipeline**
   - Real-time audio capture and processing
   - Sliding window classification
   - Confidence scoring and filtering

2. **Biodiversity Dashboard**
   - Species detection visualization
   - Audio playback and spectrogram display
   - Biodiversity metrics and statistics

3. **Deployment**
   - Streamlit web application
   - Docker containerization
   - Performance optimization

## Key Features

### Audio Processing
- **Noise Reduction**: Spectral subtraction and filtering
- **Feature Extraction**: MFCCs, spectrograms, chroma features
- **Data Augmentation**: Time stretching, pitch shifting, noise addition
- **Segmentation**: Automatic detection of vocalizations

### Machine Learning Models
- **Traditional ML**: Random Forest, SVM with audio features
- **Deep Learning**: CNN for spectrogram classification
- **Transfer Learning**: Fine-tuned pre-trained audio models
- **Ensemble Methods**: Combining multiple model predictions

### Real-time Analysis
- **Live Audio Capture**: Microphone input processing
- **Sliding Window**: Continuous classification
- **Confidence Filtering**: Threshold-based detection
- **Species Tracking**: Temporal consistency checking

### Biodiversity Dashboard
- **Species Detection**: Real-time identification results
- **Audio Visualization**: Waveforms and spectrograms
- **Biodiversity Metrics**: Species richness, abundance
- **Historical Analysis**: Trends over time

## Sample Data

The project includes sample data from:
- **Xeno-canto**: 10,000+ bird recordings, 50+ species
- **Common Species**: Robin, Blackbird, Wren, Sparrow
- **Background Sounds**: Wind, rain, traffic, silence
- **Quality Ratings**: High-quality recordings only

## Expected Outcomes

1. **Trained classification models** achieving >85% accuracy
2. **Real-time audio analysis** with <1 second latency
3. **Interactive dashboard** for biodiversity monitoring
4. **Feature extraction pipeline** for new audio data

## Real-World Applications

- **Biodiversity Surveys**: Automated species counting
- **Conservation Monitoring**: Track endangered species
- **Citizen Science**: Mobile apps for bird identification
- **Research**: Large-scale bioacoustic analysis

## Model Performance Targets

- **Accuracy**: >85% on test set
- **Precision**: >80% per species (top 20 species)
- **Recall**: >75% for common species
- **Inference Speed**: <500ms per 5-second clip

## Extensions

- **Mobile App**: React Native or Flutter implementation
- **Edge Deployment**: Raspberry Pi field deployment
- **Multi-species Detection**: Simultaneous species identification
- **Temporal Modeling**: RNN/LSTM for sequence analysis

## Getting Started

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download sample data
python src/data_loader.py --species robin,blackbird,wren --samples 100

# Start with exploration notebook
jupyter notebook notebooks/01_audio_exploration.ipynb

# Run dashboard
streamlit run src/dashboard.py
```

## Technical Challenges

1. **Audio Quality**: Handling noisy field recordings
2. **Class Imbalance**: Uneven species representation
3. **Real-time Processing**: Low-latency inference requirements
4. **Generalization**: Performance across different environments

## Resources

- [Xeno-canto API](https://www.xeno-canto.org/explore/api)
- [librosa Documentation](https://librosa.org/doc/latest/index.html)
- [BirdNET Research](https://birdnet.cornell.edu/)
- [Bioacoustics Analysis](https://www.birds.cornell.edu/home/bring-birds-back/bird-sounds/)
- [Audio Classification Tutorial](https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html)
