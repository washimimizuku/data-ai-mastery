# Data Sources for AI Development Projects

## Quick Start: Hugging Face Datasets

### Installation
```bash
pip install datasets transformers
```

### Basic Usage
```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("imdb")

# Access data
print(dataset["train"][0])

# Save locally
dataset.save_to_disk("./data/imdb")
```

---

## Project-Specific Data Sources

### Project A: Local RAG System

#### Option 1: Wikipedia (Recommended)
```python
from datasets import load_dataset

# Load Wikipedia subset
wiki = load_dataset("wikipedia", "20220301.en", split="train[:10000]")

# Save as documents
import os
os.makedirs("data/documents", exist_ok=True)

for i, article in enumerate(wiki):
    with open(f"data/documents/wiki_{i}.txt", "w") as f:
        f.write(f"# {article['title']}\n\n{article['text']}")
```

#### Option 2: arXiv Papers
- **Source**: [Kaggle arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)
- **Format**: JSON with abstracts and full text
- **Size**: 1.7M papers

#### Option 3: Technical Documentation
```python
import requests

docs_urls = [
    "https://raw.githubusercontent.com/python/cpython/main/README.rst",
    "https://raw.githubusercontent.com/tiangolo/fastapi/master/README.md",
    "https://raw.githubusercontent.com/pytorch/pytorch/master/README.md",
]

for i, url in enumerate(docs_urls):
    response = requests.get(url)
    with open(f"data/documents/doc_{i}.md", "w") as f:
        f.write(response.text)
```

---

### Project B: LLM Fine-Tuning

#### Instruction Datasets

**1. Alpaca Dataset**
```python
from datasets import load_dataset

alpaca = load_dataset("tatsu-lab/alpaca")
# 52K instruction-following examples
```

**2. Dolly Dataset**
```python
dolly = load_dataset("databricks/databricks-dolly-15k")
# 15K high-quality human-generated examples
```

**3. SQL Generation**
```python
sql_dataset = load_dataset("b-mc2/sql-create-context")
# Text-to-SQL examples
```

#### Custom Dataset Format
```json
{
  "instruction": "Generate a SQL query for the following request",
  "input": "Get all users who signed up in the last 30 days",
  "output": "SELECT * FROM users WHERE signup_date >= CURRENT_DATE - INTERVAL '30 days'"
}
```

---

### Project C: Multi-Agent System

#### Synthetic Task Descriptions
```python
tasks = [
    "Research the latest trends in AI and summarize the top 3",
    "Write a Python function to calculate fibonacci numbers and test it",
    "Analyze the sales data and create a visualization",
    "Find information about climate change and write a report"
]
```

#### APIs for Tools
- **DuckDuckGo**: Free search API
- **Wikipedia**: `pip install wikipedia`
- **Python REPL**: Built-in

---

### Project D: Computer Vision Pipeline

#### COCO Dataset
```python
from pycocotools.coco import COCO
import requests

# Download COCO
# Train: http://images.cocodataset.org/zips/train2017.zip
# Val: http://images.cocodataset.org/zips/val2017.zip
# Annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

#### Hugging Face Vision Datasets
```python
from datasets import load_dataset

# Image classification
cifar10 = load_dataset("cifar10")

# Object detection
coco = load_dataset("detection-datasets/coco")
```

#### Quick Test Images
```python
from PIL import Image
import requests
from io import BytesIO

# Download sample images
urls = [
    "https://images.unsplash.com/photo-1506905925346-21bda4d32df4",
    "https://images.unsplash.com/photo-1441974231531-c6227db76b6e",
]

for i, url in enumerate(urls):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.save(f"data/images/sample_{i}.jpg")
```

---

### Project E: NLP Multi-Task System

#### Sentiment Analysis
```python
imdb = load_dataset("imdb")  # Movie reviews
```

#### Named Entity Recognition
```python
conll = load_dataset("conll2003")  # NER dataset
```

#### Summarization
```python
cnn_dailymail = load_dataset("cnn_dailymail", "3.0.0")
```

#### Question Answering
```python
squad = load_dataset("squad")  # Stanford QA dataset
```

#### Translation
```python
wmt = load_dataset("wmt14", "de-en")  # German-English
```

---

### Project F: LLM Inference Optimization

#### Benchmark Text
```python
wikitext = load_dataset("wikitext", "wikitext-2-raw-v1")
# Use for perplexity benchmarks
```

---

### Project G: Prompt Engineering Framework

#### Multi-Task Benchmarks
```python
# MMLU - Massive Multitask Language Understanding
mmlu = load_dataset("cais/mmlu", "all")

# GSM8K - Math reasoning
gsm8k = load_dataset("gsm8k", "main")
```

---

### Project H: Embedding & Similarity Search

#### Quora Question Pairs
```python
quora = load_dataset("quora")
# Duplicate question detection
```

#### News Articles
```python
ag_news = load_dataset("ag_news")
# News categorization and clustering
```

---

### Project I: Audio AI Pipeline

#### LibriSpeech (Speech Recognition)
```python
from datasets import load_dataset

librispeech = load_dataset("librispeech_asr", "clean", split="test")

# Save audio files
for i, example in enumerate(librispeech):
    audio = example["audio"]
    # Process audio
```

#### Common Voice
- **Source**: [Mozilla Common Voice](https://commonvoice.mozilla.org/)
- **Languages**: 100+
- **Format**: MP3 + transcriptions

#### Quick Test Audio
```python
import requests

# Download sample audio
url = "https://www2.cs.uic.edu/~i101/SoundFiles/StarWars60.wav"
response = requests.get(url)
with open("data/audio/sample.wav", "wb") as f:
    f.write(response.content)
```

---

### Project J: Reinforcement Learning Agent

#### Built-in Environments
```python
import gymnasium as gym

# No external data needed
env = gym.make("CartPole-v1")
env = gym.make("LunarLander-v2")
```

---

## Complete Setup Script

```python
# setup_data.py
from datasets import load_dataset
import os

def setup_all_datasets():
    """Download and prepare all datasets"""
    
    os.makedirs("data", exist_ok=True)
    
    # NLP datasets
    print("Downloading NLP datasets...")
    load_dataset("imdb").save_to_disk("data/imdb")
    load_dataset("squad").save_to_disk("data/squad")
    
    # Instruction datasets
    print("Downloading instruction datasets...")
    load_dataset("tatsu-lab/alpaca").save_to_disk("data/alpaca")
    load_dataset("databricks/databricks-dolly-15k").save_to_disk("data/dolly")
    
    # Vision datasets (small subset)
    print("Downloading vision datasets...")
    load_dataset("cifar10").save_to_disk("data/cifar10")
    
    # Audio datasets (small subset)
    print("Downloading audio datasets...")
    load_dataset("librispeech_asr", "clean", split="test[:100]").save_to_disk("data/librispeech_sample")
    
    print("All datasets downloaded!")

if __name__ == "__main__":
    setup_all_datasets()
```

---

## Storage Requirements

| Project | Dataset | Size | Download Time |
|---------|---------|------|---------------|
| A. RAG | Wikipedia (10K articles) | ~500MB | 5-10 min |
| B. Fine-tune | Alpaca + Dolly | ~100MB | 2-5 min |
| C. Multi-Agent | Synthetic | <1MB | Instant |
| D. Computer Vision | COCO (subset) | 1-5GB | 30-60 min |
| E. NLP | IMDB + SQuAD + CNN | ~500MB | 10-15 min |
| F. Optimization | WikiText | ~10MB | 1 min |
| G. Prompting | MMLU + GSM8K | ~100MB | 5 min |
| H. Embeddings | Quora + AG News | ~200MB | 5 min |
| I. Audio | LibriSpeech (subset) | ~500MB | 10-15 min |
| J. RL | Built-in | 0MB | Instant |

**Total**: ~3-8GB depending on subsets

---

## Data Storage Structure

```
ai-development/
├── data/
│   ├── documents/          # RAG documents
│   │   ├── wiki_*.txt
│   │   └── docs_*.md
│   ├── datasets/           # Hugging Face datasets
│   │   ├── imdb/
│   │   ├── squad/
│   │   ├── alpaca/
│   │   └── dolly/
│   ├── images/            # Computer vision
│   │   ├── train/
│   │   └── val/
│   ├── audio/             # Audio files
│   │   └── samples/
│   └── models/            # Fine-tuned models
│       └── checkpoints/
```

---

## Tips

1. **Use Hugging Face cache**: Datasets are cached in `~/.cache/huggingface/`
2. **Start with subsets**: Use `split="train[:1000]"` for testing
3. **Offline mode**: Set `HF_DATASETS_OFFLINE=1` after downloading
4. **Storage**: Keep 10-20GB free for models and datasets
5. **Git LFS**: Use for large files if committing to Git

---

## Hugging Face Hub Setup

```bash
# Install
pip install huggingface_hub

# Login (optional, for private datasets)
huggingface-cli login

# Download specific dataset
huggingface-cli download dataset_name
```

---

## Alternative: Local Document Collection

For RAG projects, create your own document collection:

```python
import os
import glob

def collect_local_docs(directory="./docs"):
    """Collect all markdown and text files"""
    docs = []
    for ext in ["*.md", "*.txt", "*.rst"]:
        docs.extend(glob.glob(f"{directory}/**/{ext}", recursive=True))
    return docs

# Use your own code documentation, notes, etc.
```
