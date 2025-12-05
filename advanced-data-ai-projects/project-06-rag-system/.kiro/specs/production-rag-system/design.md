# Design Document: Production RAG System

## Overview

The Production RAG System is an enterprise-grade application that enables users to query documents using natural language and receive accurate, cited responses. The system architecture follows a pipeline pattern: documents are ingested, chunked, embedded, and stored in a vector database; user queries are embedded and used to retrieve relevant context; retrieved chunks are passed to a large language model (LLM) to generate responses with citations. The system maintains conversation history in Redis and exposes functionality through a FastAPI REST service with authentication and rate limiting.

The design emphasizes modularity, with clear separation between document processing, embedding generation, retrieval, response generation, and API layers. This enables independent testing, optimization, and replacement of components.

## Architecture

```
┌─────────────┐
│  Documents  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Document        │
│ Processor       │
│ (Chunking)      │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Embedding       │
│ Pipeline        │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Vector Database │◄────────┐
│ (Pinecone/      │         │
│  pgvector)      │         │
└─────────────────┘         │
                            │
┌─────────────┐             │
│ User Query  │             │
└──────┬──────┘             │
       │                    │
       ▼                    │
┌─────────────────┐         │
│ Query Embedding │         │
└──────┬──────────┘         │
       │                    │
       ▼                    │
┌─────────────────┐         │
│ Retrieval       ├─────────┘
│ (Hybrid Search) │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐      ┌──────────────┐
│ RAG Chain       │◄─────┤ Conversation │
│ (LLM)           │      │ Manager      │
└──────┬──────────┘      │ (Redis)      │
       │                 └──────────────┘
       ▼
┌─────────────────┐
│ Response with   │
│ Citations       │
└─────────────────┘
```

The architecture is layered:
1. **Ingestion Layer**: Document processing and chunking
2. **Storage Layer**: Vector database and conversation memory
3. **Retrieval Layer**: Hybrid search combining vector and keyword matching
4. **Generation Layer**: LLM-based response generation with citations
5. **API Layer**: FastAPI service with authentication and rate limiting

## Components and Interfaces

### Document Processor

**Responsibility**: Ingest documents in multiple formats, extract text, and create chunks with metadata.

**Interface**:
```python
class DocumentProcessor:
    def process_document(self, file: UploadFile, file_type: str) -> List[Chunk]:
        """Process a document and return chunks with metadata."""
        pass
    
    def extract_text(self, file: UploadFile, file_type: str) -> str:
        """Extract raw text from document."""
        pass
    
    def create_chunks(self, text: str, metadata: dict) -> List[Chunk]:
        """Split text into chunks with overlap."""
        pass
```

**Implementation Details**:
- Uses format-specific parsers (PyPDF2 for PDF, python-docx for DOCX, BeautifulSoup for HTML)
- Implements RecursiveCharacterTextSplitter with separators: ["\n\n", "\n", ".", " "]
- Chunk size: 1000 characters (±200 tolerance)
- Chunk overlap: 200 characters (±50 tolerance)
- Preserves metadata: source filename, chunk index, document type

### Embedding Pipeline

**Responsibility**: Generate vector embeddings for text chunks and queries.

**Interface**:
```python
class EmbeddingPipeline:
    def __init__(self, model_name: str):
        """Initialize with embedding model (e.g., 'text-embedding-3-small')."""
        pass
    
    def embed_chunks(self, chunks: List[Chunk]) -> List[Tuple[Chunk, np.ndarray]]:
        """Generate embeddings for document chunks."""
        pass
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query."""
        pass
    
    def store_embeddings(self, chunks_with_embeddings: List[Tuple[Chunk, np.ndarray]]) -> None:
        """Store embeddings in vector database."""
        pass
```

**Implementation Details**:
- Uses OpenAI text-embedding-3-small model (1536 dimensions)
- Batches embedding requests for efficiency
- Caches embeddings to avoid redundant API calls
- Handles rate limiting and retries

### Vector Database

**Responsibility**: Store and retrieve document embeddings with metadata.

**Interface**:
```python
class VectorStore:
    def add_documents(self, chunks: List[Chunk], embeddings: List[np.ndarray]) -> None:
        """Add document chunks with embeddings to the store."""
        pass
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> List[Chunk]:
        """Retrieve top k most similar chunks."""
        pass
    
    def delete_document(self, document_id: str) -> None:
        """Remove all chunks for a document."""
        pass
```

**Implementation Details**:
- Uses Pinecone or pgvector as backend
- Indexes embeddings with cosine similarity metric
- Stores metadata alongside vectors
- Supports filtering by document metadata

### Retrieval System

**Responsibility**: Implement hybrid search combining vector and keyword matching.

**Interface**:
```python
class HybridRetriever:
    def __init__(self, vector_store: VectorStore, documents: List[Chunk]):
        """Initialize with vector store and document corpus."""
        pass
    
    def retrieve(self, query: str, k: int = 5) -> List[Chunk]:
        """Retrieve top k chunks using hybrid search."""
        pass
    
    def vector_search(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        """Perform vector similarity search."""
        pass
    
    def keyword_search(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        """Perform BM25 keyword search."""
        pass
    
    def merge_results(self, vector_results: List[Tuple[Chunk, float]], 
                     keyword_results: List[Tuple[Chunk, float]]) -> List[Chunk]:
        """Merge and rank results with weights (0.7 vector, 0.3 keyword)."""
        pass
```

**Implementation Details**:
- Uses LangChain's EnsembleRetriever
- Vector search weight: 0.7
- Keyword search (BM25) weight: 0.3
- Returns top k chunks after merging and ranking

### RAG Chain

**Responsibility**: Generate responses using retrieved context and LLM.

**Interface**:
```python
class RAGChain:
    def __init__(self, llm: ChatModel, retriever: HybridRetriever):
        """Initialize with LLM and retriever."""
        pass
    
    def generate_response(self, query: str, conversation_history: List[Message] = None) -> Response:
        """Generate response with citations."""
        pass
    
    def format_context(self, chunks: List[Chunk]) -> str:
        """Format retrieved chunks as context for LLM."""
        pass
    
    def extract_citations(self, response: str, sources: List[Chunk]) -> List[Citation]:
        """Extract and validate citations from response."""
        pass
```

**Implementation Details**:
- Uses GPT-4 or Claude as LLM
- Temperature: 0 for consistent responses
- System prompt instructs model to cite sources and admit when information is unavailable
- Formats context with source identifiers
- Validates that citations reference actual retrieved documents
- Implements streaming for real-time response delivery

### Conversation Manager

**Responsibility**: Maintain conversation state across multiple turns.

**Interface**:
```python
class ConversationManager:
    def __init__(self, redis_url: str):
        """Initialize with Redis connection."""
        pass
    
    def create_conversation(self) -> str:
        """Create new conversation and return ID."""
        pass
    
    def get_history(self, conversation_id: str) -> List[Message]:
        """Retrieve conversation history."""
        pass
    
    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        """Add message to conversation history."""
        pass
    
    def clear_conversation(self, conversation_id: str) -> None:
        """Delete conversation history."""
        pass
```

**Implementation Details**:
- Uses Redis for fast in-memory storage
- Stores messages as JSON with timestamps
- Implements TTL (time-to-live) for automatic cleanup
- Maintains message order using Redis lists

### API Service

**Responsibility**: Expose RAG functionality through REST endpoints with authentication and rate limiting.

**Interface**:
```python
@app.post("/api/v1/query")
async def query_endpoint(query: QueryRequest, user: User = Depends(authenticate)) -> QueryResponse:
    """Process user query and return response with citations."""
    pass

@app.post("/api/v1/documents")
async def upload_document(file: UploadFile, user: User = Depends(authenticate)) -> UploadResponse:
    """Upload and index a document."""
    pass

@app.get("/api/v1/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, user: User = Depends(authenticate)) -> ConversationResponse:
    """Retrieve conversation history."""
    pass
```

**Implementation Details**:
- FastAPI with async/await for concurrent request handling
- JWT-based authentication using HTTPBearer security
- Rate limiting: 10 requests per minute per IP using slowapi
- OpenAPI documentation auto-generated at /docs
- CORS middleware for frontend integration
- Request/response validation using Pydantic models

### Evaluation Framework

**Responsibility**: Measure RAG system performance using standard metrics.

**Interface**:
```python
class EvaluationFramework:
    def evaluate_faithfulness(self, response: str, context: List[Chunk]) -> float:
        """Measure if response is grounded in context."""
        pass
    
    def evaluate_relevancy(self, response: str, query: str) -> float:
        """Measure if response addresses the query."""
        pass
    
    def evaluate_context_precision(self, retrieved_chunks: List[Chunk], query: str) -> float:
        """Measure quality of retrieved context."""
        pass
    
    def run_evaluation(self, test_dataset: List[TestCase]) -> EvaluationResults:
        """Run full evaluation suite."""
        pass
```

**Implementation Details**:
- Uses RAGAS library for standard metrics
- Faithfulness: checks if response claims are supported by context
- Answer relevancy: measures how well response addresses query
- Context precision: evaluates quality of retrieved chunks
- Stores evaluation results for tracking over time

## Data Models

### Chunk
```python
@dataclass
class Chunk:
    id: str
    content: str
    metadata: ChunkMetadata
    embedding: Optional[np.ndarray] = None

@dataclass
class ChunkMetadata:
    source_document: str
    chunk_index: int
    document_type: str
    created_at: datetime
```

### Message
```python
@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
```

### QueryRequest
```python
class QueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    max_sources: int = 5
```

### QueryResponse
```python
class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    conversation_id: str
    latency_ms: int

class Source(BaseModel):
    document_name: str
    chunk_content: str
    relevance_score: float
```

### Response
```python
@dataclass
class Response:
    answer: str
    sources: List[Chunk]
    citations: List[Citation]
    latency: float

@dataclass
class Citation:
    text: str
    source_document: str
    chunk_index: int
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


### Property 1: Chunk size bounds

*For any* document processed by the system, all resulting chunks should have content length between 800 and 1200 characters.

**Validates: Requirements 2.2**

### Property 2: Chunk overlap consistency

*For any* document with multiple chunks, consecutive chunks should have overlap between 150 and 250 characters.

**Validates: Requirements 2.3**

### Property 3: Chunk metadata preservation

*For any* chunk created by the Document Processor, the chunk should contain metadata with source document name, chunk index, and document type.

**Validates: Requirements 2.4**

### Property 4: Embedding and storage round-trip

*For any* document uploaded to the system, if chunks are created and embedded, then querying the vector database should retrieve those chunks with their metadata intact.

**Validates: Requirements 3.1, 3.2**

### Property 5: Query embedding consistency

*For any* user query, the system should generate an embedding using the same model used for document embeddings.

**Validates: Requirements 3.3**

### Property 6: Retrieval count

*For any* query embedding, the retrieval system should return exactly 5 chunks (or fewer if fewer than 5 chunks exist in the database).

**Validates: Requirements 3.4**

### Property 7: Response contains citations

*For any* generated response, the response should include at least one citation referencing a source document.

**Validates: Requirements 4.1**

### Property 8: Response faithfulness

*For any* generated response, all claims in the response should be supported by the retrieved context (faithfulness score > 0.8).

**Validates: Requirements 4.2**

### Property 9: Response latency

*For any* query, the system should generate and return a response within 3 seconds.

**Validates: Requirements 4.4**

### Property 10: Conversation ID uniqueness

*For any* two conversations created by the system, they should have different conversation identifiers.

**Validates: Requirements 5.1**

### Property 11: Conversation history round-trip

*For any* conversation, if a query and response are stored, then retrieving the conversation history should include that query-response pair.

**Validates: Requirements 5.2, 5.3**

### Property 12: Message chronological order

*For any* conversation with multiple messages, retrieving the history should return messages in the same chronological order they were added.

**Validates: Requirements 5.4**

### Property 13: Query API response structure

*For any* valid query request to /api/v1/query, the response should contain both an "answer" field and a "sources" field.

**Validates: Requirements 6.1**

### Property 14: Document indexing completeness

*For any* document uploaded via /api/v1/documents, the document's chunks should be retrievable through subsequent queries.

**Validates: Requirements 6.2**

### Property 15: Authentication rejection

*For any* API request without a valid JWT token, the API should return a 401 status code.

**Validates: Requirements 7.1**

### Property 16: Authentication acceptance

*For any* API request with a valid JWT token, the API should process the request and return a non-401 status code.

**Validates: Requirements 7.2**

### Property 17: Rate limit independence

*For any* two clients with different IP addresses, their rate limit counters should be tracked independently.

**Validates: Requirements 7.4**

### Property 18: Hybrid search execution

*For any* query using hybrid search, the system should execute both vector similarity search and keyword (BM25) search.

**Validates: Requirements 8.1**

### Property 19: Hybrid search weighting

*For any* hybrid search result, the final score should be calculated as (0.7 × vector_score + 0.3 × keyword_score).

**Validates: Requirements 8.2**

### Property 20: Result ranking by score

*For any* set of hybrid search results, the returned chunks should be ordered by descending combined score.

**Validates: Requirements 8.3**

### Property 21: Faithfulness metric calculation

*For any* evaluation run with a test dataset, the framework should calculate and return faithfulness scores for all responses.

**Validates: Requirements 9.1**

### Property 22: Relevancy metric calculation

*For any* evaluation run with a test dataset, the framework should calculate and return answer relevancy scores for all responses.

**Validates: Requirements 9.2**

### Property 23: Context precision metric calculation

*For any* evaluation run with a test dataset, the framework should calculate and return context precision scores for all retrievals.

**Validates: Requirements 9.3**

## Error Handling

### Document Processing Errors

- **Unsupported file format**: Return 400 Bad Request with error message specifying supported formats
- **Corrupted file**: Return 400 Bad Request with error message indicating file cannot be parsed
- **Empty document**: Return 400 Bad Request with error message indicating document has no extractable text
- **File too large**: Return 413 Payload Too Large with maximum file size information

### Embedding Errors

- **API rate limit exceeded**: Implement exponential backoff and retry up to 3 times
- **API authentication failure**: Log error and return 500 Internal Server Error to client
- **Network timeout**: Retry up to 3 times with exponential backoff, then return 503 Service Unavailable

### Vector Database Errors

- **Connection failure**: Implement connection pooling with automatic reconnection
- **Index not found**: Create index automatically on first document upload
- **Query timeout**: Return 504 Gateway Timeout after 5 seconds

### LLM Errors

- **API rate limit exceeded**: Implement exponential backoff and retry up to 3 times
- **Context length exceeded**: Reduce number of retrieved chunks and retry
- **Content policy violation**: Return 400 Bad Request with sanitized error message
- **Network timeout**: Retry once, then return 503 Service Unavailable

### Conversation Manager Errors

- **Redis connection failure**: Implement connection pooling with automatic reconnection
- **Conversation not found**: Return 404 Not Found
- **Redis memory full**: Implement TTL-based cleanup and return 507 Insufficient Storage if still full

### API Errors

- **Invalid JWT token**: Return 401 Unauthorized with WWW-Authenticate header
- **Expired JWT token**: Return 401 Unauthorized with error message indicating token expiration
- **Rate limit exceeded**: Return 429 Too Many Requests with Retry-After header
- **Invalid request body**: Return 422 Unprocessable Entity with validation error details
- **Internal server error**: Log full error details, return 500 with generic error message to client

## Testing Strategy

The Production RAG System will employ a comprehensive testing strategy combining unit tests, property-based tests, and integration tests to ensure correctness, reliability, and performance.

### Unit Testing

Unit tests will verify specific examples, edge cases, and component integration points:

- **Document Processing**: Test each file format parser (PDF, DOCX, HTML, Markdown, code) with sample files
- **Chunking Edge Cases**: Test empty documents, single-character documents, documents smaller than chunk size
- **Authentication**: Test JWT token validation, expiration, and malformed tokens
- **Rate Limiting**: Test exact threshold behavior (10th vs 11th request)
- **Error Handling**: Test each error condition with specific inputs that trigger the error

Unit tests will use pytest as the testing framework and will be co-located with source files using the `.test.py` suffix.

### Property-Based Testing

Property-based tests will verify universal properties that should hold across all inputs. We will use **Hypothesis** as the property-based testing library for Python.

**Configuration**:
- Each property-based test will run a minimum of 100 iterations
- Tests will use Hypothesis strategies to generate random but valid inputs
- Each test will be tagged with a comment explicitly referencing the correctness property

**Tag Format**: `# Feature: production-rag-system, Property {number}: {property_text}`

**Property Test Coverage**:

Each of the 23 correctness properties listed above will be implemented as a property-based test. Key property tests include:

- **Chunk Properties** (Properties 1-3): Generate random documents and verify chunk size, overlap, and metadata
- **Embedding Round-Trip** (Property 4): Generate random documents, embed and store, then retrieve and verify
- **Retrieval Properties** (Properties 5-6): Generate random queries and verify embedding model consistency and result count
- **Response Properties** (Properties 7-9): Generate random queries and verify citations, faithfulness, and latency
- **Conversation Properties** (Properties 10-12): Generate random conversations and verify ID uniqueness, round-trip, and ordering
- **API Properties** (Properties 13-17): Generate random API requests and verify response structure, authentication, and rate limiting
- **Hybrid Search Properties** (Properties 18-20): Generate random queries and verify search execution, weighting, and ranking
- **Evaluation Properties** (Properties 21-23): Generate random test datasets and verify metric calculations

**Example Property Test**:
```python
from hypothesis import given, strategies as st

# Feature: production-rag-system, Property 1: Chunk size bounds
@given(st.text(min_size=2000))
def test_chunk_size_bounds(document_text):
    processor = DocumentProcessor()
    chunks = processor.create_chunks(document_text, metadata={})
    
    for chunk in chunks:
        assert 800 <= len(chunk.content) <= 1200
```

### Integration Testing

Integration tests will verify end-to-end workflows:

- **Document Upload to Query**: Upload document, wait for indexing, query for content, verify response
- **Multi-Turn Conversation**: Create conversation, send multiple queries, verify context is maintained
- **Authentication Flow**: Obtain JWT token, make authenticated requests, verify access control
- **Hybrid Search Pipeline**: Upload documents, query with hybrid search, verify both search methods contribute

### Performance Testing

Performance tests will verify latency and throughput requirements:

- **Response Latency**: Measure response time for 100 random queries, verify 95th percentile < 3 seconds
- **Concurrent Users**: Simulate 100 concurrent users, verify no errors and acceptable latency
- **Embedding Throughput**: Measure time to process and index 1000 documents

### Evaluation Testing

Evaluation tests will verify RAG quality metrics:

- **RAGAS Metrics**: Run evaluation framework on curated test dataset, verify faithfulness > 0.8, relevancy > 0.95
- **Citation Accuracy**: Manually verify that citations in responses reference correct source documents
- **Hallucination Detection**: Test with queries that have no answer in corpus, verify system admits lack of information

### Test Execution Strategy

1. **Implementation-First Development**: Implement features before writing corresponding tests
2. **Incremental Testing**: Write and run tests after each component is implemented
3. **Continuous Testing**: Run unit and property tests on every code change
4. **Periodic Evaluation**: Run full evaluation suite weekly to track quality metrics
5. **Pre-Deployment Testing**: Run full test suite including integration and performance tests before deployment

This dual approach of unit tests (for specific cases) and property tests (for universal properties) provides comprehensive coverage: unit tests catch concrete bugs, property tests verify general correctness.
