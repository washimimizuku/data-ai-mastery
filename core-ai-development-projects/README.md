# AI Development Projects

Focused, standalone AI projects showcasing GenAI, Agentic AI, Computer Vision, NLP, and Reinforcement Learning. All projects run locally using open-source models and frameworks without cloud dependencies.

## Projects Overview

### Project A: Local RAG System with Open Models
**Time**: 2 days | **Focus**: RAG fundamentals with local LLMs

Build a complete RAG pipeline using local models, vector databases, and evaluation frameworks.

**Stack**: Python, LangChain, Ollama, ChromaDB, sentence-transformers, RAGAS

---

### Project B: Fine-Tuning Small LLM
**Time**: 2 days | **Focus**: Efficient fine-tuning techniques

Fine-tune open-source LLMs using LoRA/QLoRA for parameter-efficient training.

**Stack**: Python, Hugging Face Transformers, PEFT, LoRA, PyTorch

---

### Project C: Multi-Agent System
**Time**: 2 days | **Focus**: Agent orchestration and tool use

Build a multi-agent system with specialized agents that communicate and use tools.

**Stack**: Python, LangGraph, Ollama, local LLMs

---

### Project D: Computer Vision Pipeline
**Time**: 1-2 days | **Focus**: Modern CV with open models

Implement multiple CV tasks using state-of-the-art open models.

**Stack**: Python, PyTorch, Hugging Face, OpenCV, Gradio

---

### Project E: NLP Multi-Task System
**Time**: 1-2 days | **Focus**: Core NLP tasks

Build a comprehensive NLP system covering classification, NER, summarization, and more.

**Stack**: Python, Hugging Face Transformers, spaCy

---

### Project F: LLM Inference Optimization
**Time**: 1-2 days | **Focus**: Fast local inference

Optimize LLM inference with quantization and compare different inference engines.

**Stack**: Python, Rust (optional), llama.cpp, GGUF, vLLM, ONNX

---

### Project G: Prompt Engineering Framework
**Time**: 1 day | **Focus**: Systematic prompt optimization

Build a framework for prompt engineering with evaluation and optimization.

**Stack**: Python, Ollama, DSPy, local LLMs

---

### Project H: Embedding & Similarity Search
**Time**: 1 day | **Focus**: Embeddings and search

Implement semantic search, clustering, and similarity algorithms.

**Stack**: Python, Rust (optional), sentence-transformers, FAISS, Qdrant

---

### Project I: Audio AI Pipeline
**Time**: 1-2 days | **Focus**: Speech and audio processing

Build speech-to-text, text-to-speech, and audio classification systems.

**Stack**: Python, Whisper, Coqui TTS, Bark, PyTorch

---

### Project J: Reinforcement Learning Agent
**Time**: 2 days | **Focus**: RL fundamentals

Train RL agents using multiple algorithms and custom environments.

**Stack**: Python, Gymnasium, Stable-Baselines3, PyTorch

---

## Recommended Learning Paths

### Path 1: GenAI Focus (1 week)
1. Project A - RAG system (2 days)
2. Project B - Fine-tuning (2 days)
3. Project C - Multi-agent (2 days)
4. Project G - Prompt engineering (1 day)

**Covers**: RAG, fine-tuning, agents, prompting

---

### Path 2: Practical AI (1 week)
1. Project D - Computer vision (1-2 days)
2. Project E - NLP tasks (1-2 days)
3. Project I - Audio AI (1-2 days)
4. Project H - Embeddings (1 day)

**Covers**: CV, NLP, audio, embeddings

---

### Path 3: Performance & Optimization (1 week)
1. Project F - Inference optimization (1-2 days)
2. Project H - Embeddings & search (1 day)
3. Project A - RAG system (2 days)
4. Project G - Prompt engineering (1 day)

**Covers**: Optimization, efficiency, production-ready

---

## Technology Coverage

| Project | LLMs | RAG | Agents | CV | NLP | Audio | RL | Fine-tune | Rust |
|---------|------|-----|--------|----|----|-------|-------|-----------|------|
| A. RAG | ✅✅ | ✅✅ | - | - | - | - | - | - | - |
| B. Fine-tune | ✅✅ | - | - | - | - | - | - | ✅✅ | - |
| C. Multi-Agent | ✅✅ | - | ✅✅ | - | - | - | - | - | - |
| D. Computer Vision | - | - | - | ✅✅ | - | - | - | - | - |
| E. NLP | ✅ | - | - | - | ✅✅ | - | - | - | - |
| F. Optimization | ✅✅ | - | - | - | - | - | - | - | ✅ |
| G. Prompting | ✅✅ | - | - | - | - | - | - | - | - |
| H. Embeddings | - | ✅ | - | - | ✅ | - | - | - | ✅ |
| I. Audio | ✅ | - | - | - | - | ✅✅ | - | - | - |
| J. RL | - | - | ✅ | - | - | - | ✅✅ | - | - |

---

## Key Open Source Models

### LLMs
- **Llama 3 / 3.1** (8B, 70B) - Meta's flagship models
- **Mistral 7B / Mixtral 8x7B** - High-quality open models
- **Phi-3** (3.8B) - Microsoft's efficient small model
- **Gemma 2** (9B, 27B) - Google's open models

### Embeddings
- **sentence-transformers** (all-MiniLM, BGE)
- **nomic-embed-text** - High-quality embeddings
- **jina-embeddings-v2** - Multilingual embeddings

### Computer Vision
- **YOLO** (v8, v10) - Object detection
- **SAM** (Segment Anything) - Image segmentation
- **DETR, ViT, CLIP** - Various CV tasks

### Audio
- **Whisper** (OpenAI) - Speech-to-text
- **Coqui TTS** - Text-to-speech
- **Bark** (Suno AI) - Generative audio

### NLP
- **BERT variants** - Classification, NER
- **T5, FLAN-T5** - Text generation
- **DistilBERT** - Efficient NLP

---

## Key Libraries & Frameworks

### GenAI
- **LangChain** - RAG and chains
- **LangGraph** - Agent orchestration
- **LlamaIndex** - RAG alternative
- **DSPy** - Prompt optimization
- **Ollama** - Local LLM serving

### ML/DL
- **PyTorch** - Deep learning framework
- **Hugging Face Transformers** - Pre-trained models
- **PEFT** - Parameter-efficient fine-tuning
- **sentence-transformers** - Embeddings

### Vector Databases
- **ChromaDB** - Simple, embedded
- **FAISS** - Fast similarity search
- **Qdrant** - Production-ready

### Inference Optimization
- **llama.cpp** - Fast CPU inference
- **vLLM** - Fast GPU inference
- **ONNX Runtime** - Optimized inference

### UI
- **Gradio** - Quick ML interfaces
- **Streamlit** - Dashboards

### RL
- **Gymnasium** - RL environments
- **Stable-Baselines3** - RL algorithms

---

## Programming Languages

### Primary: Python 3.11+
All projects can be completed entirely in Python. This is the industry standard for AI/ML development.

### Optional: Rust 1.75+
For performance-critical components in:
- **Project F**: Custom inference engine optimizations
- **Project H**: Fast similarity search implementations

Rust is optional but recommended to showcase performance optimization skills alongside the data engineering projects.

---

## Data Sources

**See [DATA_SOURCES.md](./DATA_SOURCES.md) for comprehensive data source guide with setup scripts.**

### Quick Reference

#### Hugging Face Datasets (Primary)
```python
from datasets import load_dataset
dataset = load_dataset("imdb")  # Auto-downloads and caches
```

#### Computer Vision
- COCO, ImageNet, Open Images

#### Audio
- LibriSpeech, Common Voice

#### Documents (RAG)
- Wikipedia, arXiv, technical docs

### By Project

| Project | Recommended Source | Size | Notes |
|---------|-------------------|------|-------|
| A. RAG | Wikipedia, arXiv, docs | 1K-100K docs | PDF, TXT, MD |
| B. Fine-tune | Alpaca, Dolly, SQL datasets | 10K-50K examples | Instruction format |
| C. Multi-Agent | Synthetic task descriptions | N/A | Generate queries |
| D. Computer Vision | COCO, ImageNet | 10K-100K images | Pre-download |
| E. NLP | IMDB, CoNLL, CNN/DailyMail | 10K-1M texts | Hugging Face |
| F. Optimization | WikiText, any text | 1M tokens | For benchmarking |
| G. Prompting | MMLU, GSM8K | 1K-10K examples | Multi-task |
| H. Embeddings | Quora pairs, news articles | 100K-1M texts | Similarity tasks |
| I. Audio | LibriSpeech, Common Voice | Hours of audio | Pre-download |
| J. RL | Gymnasium environments | N/A | Built-in |

### Quick Setup

```python
# Install Hugging Face datasets
pip install datasets

# Download and cache datasets
from datasets import load_dataset

# This downloads and caches locally
dataset = load_dataset("imdb")
```

---

## Prerequisites

### For All Projects
- Python 3.11+
- pip or conda
- 16GB+ RAM recommended
- GPU optional (CPU works for all projects)

### For Rust Projects (F, H - Optional)
- Rust 1.75+
- Cargo

### Model Storage
- 10-50GB disk space for models
- Models downloaded via Ollama, Hugging Face

---

## Key Features

**All projects include**:
- Working code (< 800 lines per project)
- README with setup and examples
- Model evaluation metrics
- Demo interface (Gradio/Streamlit)
- Jupyter notebooks (where applicable)

**All projects are**:
- Runnable locally (no cloud required)
- Using open-source models only
- Completable in 1-2 days
- Production-quality code
- Well-documented

---

## Why These Projects Stand Out

1. **Fully local** - No API keys or cloud services
2. **Open source** - All models and tools are free
3. **2025-relevant** - Covers current AI landscape
4. **Comprehensive** - GenAI, CV, NLP, Audio, RL
5. **Practical** - Real-world applications
6. **Quick** - 1-2 days each
7. **Diverse** - Shows breadth of AI skills
8. **Performance-focused** - Optional Rust optimizations

---

## Getting Started

Each project folder contains:
- `prd.md` - Product requirements
- `tech-spec.md` - Technical specification
- `implementation-plan.md` - Step-by-step guide

Start with any project based on your learning goals or follow one of the recommended paths above.

---

## Model Download Guide

### Using Ollama (Projects A, C, G)
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Download models
ollama pull llama3
ollama pull mistral
ollama pull phi3
```

### Using Hugging Face (Projects B, D, E, I)
```python
from transformers import AutoModel, AutoTokenizer

# Models download automatically on first use
model = AutoModel.from_pretrained("meta-llama/Llama-3-8B")
```

### Storage Requirements
- Small models (Phi-3): ~2GB
- Medium models (Llama-3-8B, Mistral-7B): ~4-5GB
- Large models (Llama-3-70B): ~40GB
- Vision/Audio models: 1-5GB each

---

## Hardware Recommendations

### Minimum
- CPU: 4+ cores
- RAM: 16GB
- Storage: 50GB free
- GPU: Not required (CPU inference works)

### Recommended
- CPU: 8+ cores
- RAM: 32GB
- Storage: 100GB+ free
- GPU: NVIDIA with 8GB+ VRAM (for faster training/inference)

### Optimal
- CPU: 16+ cores
- RAM: 64GB
- Storage: 500GB+ SSD
- GPU: NVIDIA RTX 4090 or A100 (for large models)

All projects work on minimum specs, just slower for training/inference.
