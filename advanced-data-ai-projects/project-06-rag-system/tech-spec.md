# Technical Specification: Production RAG System

## Architecture
```
Documents → Chunking → Embeddings → Vector DB
                                        ↓
User Query → Embedding → Retrieval → LLM → Response
                            ↓
                      Conversation Memory (Redis)
```

## Technology Stack
- **Framework**: LangChain, LlamaIndex
- **LLM**: OpenAI GPT-4, Anthropic Claude
- **Embeddings**: OpenAI, Cohere, sentence-transformers
- **Vector DB**: Pinecone, Weaviate, or pgvector
- **API**: FastAPI
- **Cache**: Redis
- **Frontend**: React or Streamlit
- **Evaluation**: RAGAS, LangSmith

## Document Processing

### Chunking Strategies
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Semantic chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]
)

chunks = splitter.split_documents(documents)
```

### Embedding Pipeline
```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Pinecone.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="documents"
)
```

## RAG Implementation

### Basic RAG Chain
```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

result = qa_chain({"query": "What is...?"})
```

### Advanced RAG with Citations
```python
from langchain.chains import RetrievalQAWithSourcesChain

qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)
```

### Hybrid Search
```python
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(documents)
vector_retriever = vectorstore.as_retriever()

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]
)
```

## FastAPI Service

### Endpoints
```python
from fastapi import FastAPI, Depends
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    question: str
    conversation_id: Optional[str] = None

@app.post("/api/v1/query")
async def query(q: Query, user: User = Depends(get_current_user)):
    # Retrieve conversation history
    history = get_conversation_history(q.conversation_id)
    
    # Run RAG chain
    result = await qa_chain.arun(
        question=q.question,
        chat_history=history
    )
    
    # Store in conversation history
    save_to_history(q.conversation_id, q.question, result)
    
    return {
        "answer": result["answer"],
        "sources": result["source_documents"],
        "conversation_id": q.conversation_id
    }

@app.post("/api/v1/documents")
async def upload_document(file: UploadFile):
    # Process and index document
    chunks = process_document(file)
    vectorstore.add_documents(chunks)
    return {"status": "indexed", "chunks": len(chunks)}
```

### Authentication
```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    return payload["user_id"]
```

### Rate Limiting
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/v1/query")
@limiter.limit("10/minute")
async def query(request: Request, q: Query):
    # Query logic
    pass
```

## Conversation Memory

### Redis Integration
```python
from langchain.memory import RedisChatMessageHistory

def get_conversation_history(conversation_id: str):
    history = RedisChatMessageHistory(
        session_id=conversation_id,
        url="redis://localhost:6379"
    )
    return history.messages
```

## Evaluation Framework

### RAGAS Evaluation
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

results = evaluate(
    dataset=test_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)

print(f"Faithfulness: {results['faithfulness']}")
print(f"Answer Relevancy: {results['answer_relevancy']}")
```

### Custom Evaluation
```python
def evaluate_citations(answer: str, sources: List[str]) -> float:
    # Check if citations are present and accurate
    citations = extract_citations(answer)
    accuracy = verify_citations(citations, sources)
    return accuracy
```

## Prompt Engineering

### System Prompt
```python
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Always cite your sources using [Source: document_name].
If you cannot answer based on the context, say so clearly.
Be concise and accurate."""
```

### Few-Shot Examples
```python
FEW_SHOT_EXAMPLES = [
    {
        "question": "What is X?",
        "context": "...",
        "answer": "X is... [Source: doc1.pdf]"
    }
]
```

## Performance Optimization
- Embedding caching
- Prompt caching (Anthropic)
- Async processing
- Connection pooling
- Response streaming

## Monitoring
- Query latency
- Retrieval accuracy
- LLM token usage
- Error rates
- User satisfaction scores
