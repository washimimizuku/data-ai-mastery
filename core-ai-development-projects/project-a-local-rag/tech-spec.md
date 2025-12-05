# Technical Specification: Local RAG System with Open Models

## Architecture
```
Documents → Loader → Chunker → Embeddings → ChromaDB
                                                ↓
User Query → Embeddings → Retrieval → Context + Query → LLM → Response
```

## Technology Stack
- **Python**: 3.11+
- **LLM Serving**: Ollama (Llama 3, Mistral, Phi-3)
- **Embeddings**: sentence-transformers
- **Vector DB**: ChromaDB
- **Framework**: LangChain
- **Evaluation**: RAGAS
- **UI**: Gradio

## Core Components

### Document Loader
```python
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader
)

class DocumentLoader:
    def load_documents(self, file_paths: List[str]) -> List[Document]
    def extract_metadata(self, doc: Document) -> dict
```

### Text Chunker
```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SemanticChunker
)

class TextChunker:
    def __init__(self, strategy: str, chunk_size: int, overlap: int)
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]
    def optimize_chunk_size(self, documents: List[Document]) -> int
```

### Embedding Generator
```python
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray
    def compare_models(self, texts: List[str]) -> dict
```

### Vector Store
```python
import chromadb
from langchain.vectorstores import Chroma

class VectorStore:
    def __init__(self, collection_name: str, persist_directory: str)
    
    def add_documents(self, documents: List[Document], embeddings: List)
    def similarity_search(self, query: str, k: int = 5) -> List[Document]
    def hybrid_search(self, query: str, k: int = 5) -> List[Document]
```

### LLM Interface
```python
from langchain.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class LocalLLM:
    def __init__(self, model: str = "llama3")
    
    def generate(self, prompt: str, stream: bool = False) -> str
    def generate_with_context(self, query: str, context: List[Document]) -> str
```

### RAG Pipeline
```python
class RAGPipeline:
    def __init__(
        self,
        vector_store: VectorStore,
        llm: LocalLLM,
        retriever_k: int = 5
    )
    
    def query(self, question: str) -> dict:
        # Returns: {
        #   "answer": str,
        #   "sources": List[Document],
        #   "metadata": dict
        # }
    
    def evaluate(self, test_questions: List[dict]) -> dict
```

## Embedding Models Comparison

### Models to Test
1. **all-MiniLM-L6-v2** (384 dim, 80MB)
   - Fast, good quality
   - Best for general use

2. **all-mpnet-base-v2** (768 dim, 420MB)
   - Higher quality
   - Slower but more accurate

3. **bge-small-en-v1.5** (384 dim, 130MB)
   - SOTA performance
   - Good balance

4. **nomic-embed-text-v1** (768 dim, 550MB)
   - Long context (8192 tokens)
   - Excellent for documents

## LLM Models

### Ollama Models
```bash
# Download models
ollama pull llama3:8b      # 4.7GB, best quality
ollama pull mistral:7b     # 4.1GB, fast
ollama pull phi3:mini      # 2.3GB, efficient
```

### Model Comparison
| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| Llama 3 8B | 4.7GB | Medium | High | Best answers |
| Mistral 7B | 4.1GB | Fast | High | Balanced |
| Phi-3 Mini | 2.3GB | Very Fast | Good | Quick responses |

## Chunking Strategies

### Strategy 1: Recursive Character Splitting
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
```

### Strategy 2: Semantic Chunking
```python
splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile"
)
```

### Strategy 3: Custom Sliding Window
```python
def sliding_window_chunk(text: str, window_size: int, overlap: int):
    # Custom implementation
    pass
```

## Prompt Templates

### RAG Prompt
```python
RAG_TEMPLATE = """Use the following context to answer the question.
If you cannot answer based on the context, say so.

Context:
{context}

Question: {question}

Answer:"""
```

### With Citations
```python
RAG_WITH_CITATIONS = """Use the following context to answer the question.
Cite your sources using [1], [2], etc.

Context:
{context}

Question: {question}

Answer with citations:"""
```

## Evaluation with RAGAS

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)

def evaluate_rag(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str]
) -> dict:
    
    result = evaluate(
        dataset={
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        },
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision
        ]
    )
    
    return result
```

## Gradio Interface

```python
import gradio as gr

def create_ui(rag_pipeline: RAGPipeline):
    def chat(message, history):
        result = rag_pipeline.query(message)
        
        # Format response with sources
        response = result["answer"]
        sources = "\n\n**Sources:**\n"
        for i, doc in enumerate(result["sources"], 1):
            sources += f"{i}. {doc.metadata.get('source', 'Unknown')}\n"
        
        return response + sources
    
    interface = gr.ChatInterface(
        fn=chat,
        title="Local RAG System",
        description="Ask questions about your documents",
        examples=[
            "What is the main topic of the documents?",
            "Summarize the key points",
            "What are the conclusions?"
        ]
    )
    
    return interface
```

## Project Structure
```
project-a-local-rag/
├── src/
│   ├── loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── llm.py
│   ├── rag_pipeline.py
│   └── evaluation.py
├── notebooks/
│   ├── 01_document_processing.ipynb
│   ├── 02_embedding_comparison.ipynb
│   ├── 03_rag_pipeline.ipynb
│   └── 04_evaluation.ipynb
├── data/
│   ├── documents/
│   └── chroma_db/
├── app.py              # Gradio UI
├── requirements.txt
└── README.md
```

## Performance Targets
- Document ingestion: 100 pages/minute
- Embedding generation: 1000 chunks/minute
- Retrieval: < 1 second
- Generation: 20-50 tokens/second (model dependent)
- End-to-end query: < 10 seconds

## Testing Strategy
- Unit tests for each component
- Integration tests for pipeline
- Evaluation on test dataset
- Compare different configurations
- Benchmark performance
