# Project A: Local RAG System with Open Models

## Objective

Build a complete Retrieval-Augmented Generation (RAG) system using local open-source models, demonstrating document ingestion, embedding generation, vector storage, retrieval, and generation without any external APIs.

**What You'll Build**: A production-ready RAG system that ingests documents (PDF, TXT, Markdown), chunks them intelligently, generates embeddings, stores them in ChromaDB, retrieves relevant context, and generates answers using local LLMs (Ollama) with a Gradio web interface.

**What You'll Learn**: RAG architecture, document processing, embedding models, vector databases, retrieval strategies, prompt engineering, LLM integration, evaluation metrics (RAGAS), and building complete GenAI applications.

## Time Estimate

**2 days (16 hours)** - Following the implementation plan

### Day 1 (8 hours)
- **Hour 1**: Setup & document loading (install dependencies, Ollama models, loaders)
- **Hours 2-3**: Chunking & embeddings (recursive + semantic chunking, embedding models)
- **Hours 3-4**: Vector store (ChromaDB setup, ingestion, similarity search)
- **Hours 5-6**: LLM integration (Ollama client, prompt templates, context formatting)
- **Hours 7-8**: RAG pipeline (end-to-end flow, source tracking, testing)

### Day 2 (8 hours)
- **Hours 1-2**: Evaluation framework (RAGAS setup, test dataset, metrics)
- **Hours 3-4**: Model comparison (embedding models, LLMs, chunk sizes, benchmarks)
- **Hours 5-6**: Gradio UI (chat interface, document upload, source display)
- **Hour 7**: Jupyter notebooks (4 notebooks: processing, comparison, pipeline, evaluation)
- **Hour 8**: Documentation & polish (README, setup instructions, examples)

## Prerequisites

### Required Knowledge
Complete these bootcamp sections first:
- [100 Days Data & AI](https://github.com/washimimizuku/100-days-data-ai) - Days 71-85
  - Days 71-75: LLM fundamentals
  - Days 76-80: RAG basics
  - Days 81-85: Vector databases
- [30 Days of Python](https://github.com/washimimizuku/30-days-python-data-ai) - Days 1-20

### Technical Requirements
- Python 3.11+ installed
- 8GB+ RAM (16GB recommended)
- 10GB+ free disk space for models
- Understanding of embeddings and vector search
- Basic knowledge of LLMs

### Tools Needed
- Python with langchain, chromadb, sentence-transformers
- Ollama for local LLM serving
- Gradio for UI
- Sample documents (PDFs, text files)
- Git for version control

## Getting Started

### Step 1: Review Documentation
Read the project documents in order:
1. `prd.md` - Understand what you're building
2. `tech-spec.md` - Review technical architecture
3. `implementation-plan.md` - Follow the implementation steps

### Step 2: Install Ollama and Models
```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Or download from https://ollama.com for Windows

# Pull LLM models
ollama pull llama3:8b      # 4.7GB - Best quality
ollama pull mistral:7b     # 4.1GB - Fast and good
ollama pull phi3:mini      # 2.3GB - Efficient

# Verify installation
ollama list
```

### Step 3: Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install langchain langchain-community chromadb \
    sentence-transformers pypdf python-docx \
    gradio ragas ollama

# Create project structure
mkdir -p local-rag/{src,data/documents,data/chroma_db,notebooks}
cd local-rag
```

### Step 4: Build Document Loader
```python
# src/loader.py
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from pathlib import Path
from typing import List
from langchain.schema import Document

class DocumentLoader:
    """Load documents from various formats"""
    
    def __init__(self, documents_dir: str = "./data/documents"):
        self.documents_dir = Path(documents_dir)
    
    def load_documents(self) -> List[Document]:
        """Load all documents from directory"""
        documents = []
        
        for file_path in self.documents_dir.rglob("*"):
            if file_path.is_file():
                try:
                    docs = self._load_file(file_path)
                    documents.extend(docs)
                    print(f"✓ Loaded: {file_path.name} ({len(docs)} chunks)")
                except Exception as e:
                    print(f"✗ Failed to load {file_path.name}: {e}")
        
        print(f"\n✓ Total documents loaded: {len(documents)}")
        return documents
    
    def _load_file(self, file_path: Path) -> List[Document]:
        """Load single file based on extension"""
        suffix = file_path.suffix.lower()
        
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif suffix == ".txt":
            loader = TextLoader(str(file_path))
        elif suffix == ".md":
            loader = UnstructuredMarkdownLoader(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        return loader.load()

# Usage
if __name__ == "__main__":
    loader = DocumentLoader()
    documents = loader.load_documents()
    print(f"Loaded {len(documents)} document chunks")
```

### Step 5: Implement Chunking Strategy
```python
# src/chunker.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

class TextChunker:
    """Chunk documents for optimal retrieval"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        chunks = self.splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
        
        print(f"✓ Created {len(chunks)} chunks")
        print(f"  Avg chunk size: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} chars")
        
        return chunks

# Usage
if __name__ == "__main__":
    from loader import DocumentLoader
    
    loader = DocumentLoader()
    documents = loader.load_documents()
    
    chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.chunk_documents(documents)
```

### Step 6: Generate Embeddings and Store in ChromaDB
```python
# src/vector_store.py
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from typing import List

class VectorStore:
    """Manage vector storage with ChromaDB"""
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./data/chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        print(f"Initializing embeddings: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.vectorstore = None
    
    def create_vectorstore(self, documents: List[Document]):
        """Create vector store from documents"""
        print(f"Creating vector store with {len(documents)} documents...")
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        
        print(f"✓ Vector store created and persisted")
    
    def load_vectorstore(self):
        """Load existing vector store"""
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        print(f"✓ Vector store loaded")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def similarity_search_with_score(self, query: str, k: int = 5):
        """Search with relevance scores"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results

# Usage
if __name__ == "__main__":
    from loader import DocumentLoader
    from chunker import TextChunker
    
    # Load and chunk documents
    loader = DocumentLoader()
    documents = loader.load_documents()
    
    chunker = TextChunker()
    chunks = chunker.chunk_documents(documents)
    
    # Create vector store
    vector_store = VectorStore()
    vector_store.create_vectorstore(chunks)
    
    # Test retrieval
    results = vector_store.similarity_search("What is RAG?", k=3)
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.page_content[:200]}...")
```

### Step 7: Build RAG Pipeline
```python
# src/rag_pipeline.py
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from vector_store import VectorStore
from typing import Dict

class RAGPipeline:
    """Complete RAG pipeline"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        model_name: str = "llama3:8b",
        retriever_k: int = 5
    ):
        self.vector_store = vector_store
        self.retriever_k = retriever_k
        
        # Initialize LLM
        print(f"Initializing LLM: {model_name}")
        self.llm = Ollama(
            model=model_name,
            temperature=0.7,
        )
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            template="""Use the following context to answer the question.
If you cannot answer based on the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )
        
        # Create retrieval chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.vectorstore.as_retriever(
                search_kwargs={"k": retriever_k}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt_template}
        )
    
    def query(self, question: str) -> Dict:
        """Query the RAG system"""
        print(f"\nQuery: {question}")
        print("Retrieving relevant documents...")
        
        result = self.qa_chain({"query": question})
        
        response = {
            "question": question,
            "answer": result["result"],
            "sources": result["source_documents"],
            "num_sources": len(result["source_documents"])
        }
        
        return response
    
    def query_with_sources(self, question: str) -> str:
        """Query and format response with sources"""
        result = self.query(question)
        
        response = f"**Answer:**\n{result['answer']}\n\n"
        response += f"**Sources ({result['num_sources']}):**\n"
        
        for i, doc in enumerate(result['sources'], 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            response += f"{i}. {source} (Page {page})\n"
            response += f"   {doc.page_content[:150]}...\n\n"
        
        return response

# Usage
if __name__ == "__main__":
    # Load vector store
    vector_store = VectorStore()
    vector_store.load_vectorstore()
    
    # Create RAG pipeline
    rag = RAGPipeline(vector_store, model_name="llama3:8b")
    
    # Test queries
    questions = [
        "What is the main topic of the documents?",
        "Summarize the key points",
        "What are the conclusions?"
    ]
    
    for question in questions:
        response = rag.query_with_sources(question)
        print(response)
        print("-" * 80)
```

### Step 8: Create Gradio Interface
```python
# app.py
import gradio as gr
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline

# Initialize RAG system
print("Loading RAG system...")
vector_store = VectorStore()
vector_store.load_vectorstore()

rag = RAGPipeline(vector_store, model_name="llama3:8b")
print("✓ RAG system ready!")

def chat(message, history):
    """Chat function for Gradio"""
    result = rag.query(message)
    
    # Format response
    response = result["answer"]
    
    # Add sources
    if result["sources"]:
        response += "\n\n**Sources:**\n"
        for i, doc in enumerate(result["sources"][:3], 1):
            source = doc.metadata.get('source', 'Unknown')
            response += f"{i}. {source}\n"
    
    return response

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat,
    title="🤖 Local RAG System",
    description="Ask questions about your documents using local LLMs",
    examples=[
        "What is the main topic?",
        "Summarize the key points",
        "What are the conclusions?",
    ],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(share=False)
```

## Key Features to Implement

### 1. Document Ingestion
- Support multiple formats (PDF, TXT, Markdown, DOCX)
- Text extraction and cleaning
- Metadata extraction
- Batch processing

### 2. Chunking Strategies
- **Recursive character splitting** (primary method)
- **Semantic chunking** (content-aware splitting)
- Sliding window with overlap
- Custom chunking logic
- Chunk size optimization (test 500, 1000, 1500 tokens)

### 3. Embedding Generation
- Local embedding models (sentence-transformers)
- **Multiple model comparison**:
  - all-MiniLM-L6-v2 (384 dim, 80MB) - Fast, general use
  - all-mpnet-base-v2 (768 dim, 420MB) - Higher quality
  - bge-small-en-v1.5 (384 dim, 130MB) - SOTA performance
  - nomic-embed-text-v1 (768 dim, 550MB) - Long context
- Batch embedding generation
- Embedding caching

### 4. Vector Storage
- ChromaDB for vector storage
- Metadata filtering
- **Hybrid search** (vector + keyword)
- Persistence and loading

### 5. Retrieval
- Similarity search (cosine, dot product)
- Top-k retrieval (configurable k)
- **Re-ranking strategies** (optional)
- Context window management

### 6. Generation
- Local LLM inference (Ollama)
- **Multiple models**:
  - Llama 3 8B (4.7GB) - Best quality
  - Mistral 7B (4.1GB) - Fast and balanced
  - Phi-3 Mini (2.3GB) - Efficient
- Prompt engineering (with/without citations)
- Streaming responses

### 7. Evaluation
- **RAGAS metrics**:
  - Faithfulness (answer grounded in context)
  - Answer relevancy (answer addresses question)
  - Context recall (retrieved relevant context)
  - Context precision (retrieved context is precise)
- Answer quality assessment
- Retrieval accuracy
- End-to-end evaluation

### 8. User Interface
- Gradio chat interface
- Document upload capability
- Source citation display
- Configuration options (model selection, k value)

## Success Criteria

By the end of this project, you should have:

### Functionality
- [ ] Document loader supporting multiple formats (PDF, TXT, Markdown, DOCX)
- [ ] Chunking strategies implemented (recursive + semantic)
- [ ] Embeddings generated and stored in ChromaDB
- [ ] Vector search working (similarity + hybrid)
- [ ] Local LLM integrated (Ollama with 3 models)
- [ ] RAG pipeline functional end-to-end
- [ ] Gradio web interface with document upload
- [ ] Evaluation with RAGAS (all 4 metrics)

### Quality Metrics
- [ ] **RAGAS scores documented**:
  - Faithfulness > 0.7
  - Answer relevancy > 0.7
  - Context recall > 0.6
  - Context precision > 0.6
- [ ] **Code quality**: < 800 lines of code
- [ ] **Performance**: End-to-end query < 10 seconds
- [ ] **Retrieval speed**: < 1 second

### Deliverables
- [ ] Complete RAG pipeline code
- [ ] 4 Jupyter notebooks (processing, comparison, pipeline, evaluation)
- [ ] Model comparison results with charts
- [ ] Comprehensive documentation
- [ ] GitHub repository with examples
- [ ] Working Gradio interface

## Learning Outcomes

After completing this project, you'll be able to:

- Build end-to-end RAG systems
- Process and chunk documents effectively
- Generate and store embeddings
- Implement vector search
- Integrate local LLMs
- Design effective prompts
- Evaluate RAG performance
- Create user interfaces for GenAI apps
- Explain RAG architecture and trade-offs

## Expected Performance

**Processing Speed**:
```
Document ingestion: 100 pages/minute
Embedding generation: 1000 chunks/minute
Retrieval: <1 second
Generation: 20-50 tokens/second (model dependent)
End-to-end query: <10 seconds
```

**Model Comparison**:
```
Llama 3 8B:
  Quality: ★★★★★
  Speed: ★★★☆☆
  Size: 4.7GB

Mistral 7B:
  Quality: ★★★★☆
  Speed: ★★★★☆
  Size: 4.1GB

Phi-3 Mini:
  Quality: ★★★☆☆
  Speed: ★★★★★
  Size: 2.3GB
```

## Project Structure

```
project-a-local-rag/
├── src/
│   ├── loader.py            # Document loading (PDF, TXT, MD, DOCX)
│   ├── chunker.py           # Text chunking (recursive + semantic)
│   ├── embeddings.py        # Embedding generation
│   ├── vector_store.py      # ChromaDB integration
│   ├── llm.py               # Ollama LLM interface
│   ├── rag_pipeline.py      # End-to-end RAG pipeline
│   └── evaluation.py        # RAGAS evaluation
├── notebooks/
│   ├── 01_document_processing.ipynb    # Loading, chunking, metadata
│   ├── 02_embedding_comparison.ipynb   # Compare embedding models
│   ├── 03_rag_pipeline.ipynb           # RAG demo and testing
│   └── 04_evaluation.ipynb             # RAGAS metrics and results
├── data/
│   ├── documents/           # Input documents (50-1000 files)
│   └── chroma_db/           # Vector database (persisted)
├── results/
│   ├── model_comparison.png # Embedding model comparison
│   ├── ragas_scores.json    # Evaluation results
│   └── benchmarks.md        # Performance benchmarks
├── app.py                   # Gradio UI
├── requirements.txt
├── prd.md
├── tech-spec.md
├── implementation-plan.md
└── README.md
```

## Common Challenges & Solutions

### Challenge 1: Ollama Connection Issues
**Problem**: Cannot connect to Ollama
**Solution**: Ensure Ollama is running: `ollama serve`

### Challenge 2: Slow Embedding Generation
**Problem**: Embedding large documents takes too long
**Solution**: Use batch processing, smaller embedding model, or GPU

### Challenge 3: Poor Retrieval Quality
**Problem**: Retrieved chunks not relevant
**Solution**: Tune chunk size, overlap, and retriever_k parameter

### Challenge 4: LLM Hallucinations
**Problem**: LLM generates information not in documents
**Solution**: Improve prompt, lower temperature, add "stick to context" instruction

## Next Steps

After completing this project:

1. **Add to Portfolio**: Document on GitHub with examples
2. **Write Blog Post**: "Building a Local RAG System with Open Models"
3. **Extend Features**: Add re-ranking, hybrid search, multi-query
4. **Build Project B**: Continue with LLM Fine-Tuning
5. **Production Use**: Deploy with FastAPI backend

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Documentation](https://ollama.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [RAGAS Documentation](https://docs.ragas.io/)
- [Sentence Transformers](https://www.sbert.net/)

## Questions?

If you get stuck:
1. Review the tech-spec.md for detailed architecture
2. Check Ollama and LangChain documentation
3. Search LangChain community forums
4. Review the 100 Days bootcamp materials on RAG
