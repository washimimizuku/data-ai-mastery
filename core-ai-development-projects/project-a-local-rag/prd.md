# Product Requirements Document: Local RAG System with Open Models

## Overview
Build a complete Retrieval-Augmented Generation (RAG) system using local open-source models, demonstrating document ingestion, embedding, retrieval, and generation without external APIs.

## Goals
- Demonstrate RAG fundamentals
- Use only local/open-source models
- Show evaluation and optimization
- Create production-ready pipeline

## Core Features

### 1. Document Ingestion
- Support multiple formats (PDF, TXT, Markdown, DOCX)
- Text extraction and cleaning
- Metadata extraction
- Batch processing

### 2. Chunking Strategies
- Recursive character splitting
- Semantic chunking
- Sliding window with overlap
- Custom chunking logic
- Chunk size optimization

### 3. Embedding Generation
- Local embedding models (sentence-transformers)
- Multiple model comparison
- Batch embedding generation
- Embedding caching

### 4. Vector Storage
- ChromaDB for vector storage
- Metadata filtering
- Hybrid search (vector + keyword)
- Persistence and loading

### 5. Retrieval
- Similarity search (cosine, dot product)
- Top-k retrieval
- Re-ranking strategies
- Context window management

### 6. Generation
- Local LLM inference (Ollama)
- Multiple models (Llama 3, Mistral, Phi-3)
- Prompt engineering
- Streaming responses

### 7. Evaluation
- RAGAS metrics (faithfulness, relevance, context recall)
- Answer quality assessment
- Retrieval accuracy
- End-to-end evaluation

### 8. User Interface
- Gradio chat interface
- Document upload
- Source citation display
- Configuration options

## Technical Requirements

### Functionality
- End-to-end RAG pipeline
- Configurable components
- Evaluation framework
- Production-ready code

### Performance
- Fast retrieval (< 1 second)
- Reasonable generation time
- Efficient memory usage

### Usability
- Simple setup
- Clear documentation
- Interactive UI

## Success Metrics
- RAG pipeline working end-to-end
- RAGAS scores documented
- Multiple models compared
- Clean, reusable code
- < 800 lines of code

## Timeline
2 days implementation

---

## Data Sources

### Recommended Documents

#### Option 1: Technical Documentation
- **Python Documentation**: [Download](https://docs.python.org/3/download.html)
- **AWS Documentation**: Scrape or use existing docs
- **Framework Docs**: FastAPI, Django, React

#### Option 2: Wikipedia Articles
```python
from datasets import load_dataset

# Load Wikipedia dataset
wiki = load_dataset("wikipedia", "20220301.en", split="train[:10000]")

# Save as text files
for i, article in enumerate(wiki):
    with open(f"docs/wiki_{i}.txt", "w") as f:
        f.write(article["text"])
```

#### Option 3: arXiv Papers
- **arXiv Dataset**: [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)
- Focus on specific domain (ML, Physics, etc.)

#### Option 4: Custom Documents
- Create your own documentation
- Use company docs (if available)
- Technical blog posts

### Quick Setup
```python
# Download sample documents
import requests

docs = [
    "https://raw.githubusercontent.com/python/cpython/main/README.rst",
    "https://raw.githubusercontent.com/tiangolo/fastapi/master/README.md",
    # Add more URLs
]

for i, url in enumerate(docs):
    response = requests.get(url)
    with open(f"data/documents/doc_{i}.md", "w") as f:
        f.write(response.text)
```

### Recommended Size
- **Minimum**: 50-100 documents
- **Optimal**: 500-1000 documents
- **Total size**: 10-50 MB of text
