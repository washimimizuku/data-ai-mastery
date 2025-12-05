# Implementation Plan

- [ ] 1. Set up project structure and dependencies
  - Create Python project with poetry or pip requirements
  - Install core dependencies: langchain, openai, fastapi, redis, pinecone-client, hypothesis, pytest, ragas
  - Set up directory structure: src/, tests/, config/
  - Create configuration management for API keys and environment variables
  - _Requirements: All_

- [ ] 2. Implement document processing and chunking
  - Create DocumentProcessor class with file type detection
  - Implement text extraction for PDF (PyPDF2), DOCX (python-docx), HTML (BeautifulSoup), Markdown, and code files
  - Implement RecursiveCharacterTextSplitter with separators ["\n\n", "\n", ".", " "]
  - Create Chunk and ChunkMetadata data models
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.4_

- [ ] 2.1 Write property test for chunk size bounds
  - **Property 1: Chunk size bounds**
  - **Validates: Requirements 2.2**

- [ ] 2.2 Write property test for chunk overlap
  - **Property 2: Chunk overlap consistency**
  - **Validates: Requirements 2.3**

- [ ] 2.3 Write property test for chunk metadata
  - **Property 3: Chunk metadata preservation**
  - **Validates: Requirements 2.4**

- [ ] 2.4 Write unit tests for document format parsers
  - Test PDF extraction with sample file
  - Test DOCX extraction with sample file
  - Test HTML extraction with sample file
  - Test Markdown extraction with sample file
  - Test code file extraction with sample file
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 3. Implement embedding pipeline and vector storage
  - Create EmbeddingPipeline class with OpenAI text-embedding-3-small integration
  - Implement batch embedding with rate limiting and retry logic
  - Create VectorStore class with Pinecone or pgvector backend
  - Implement add_documents, similarity_search, and delete_document methods
  - Add embedding caching to avoid redundant API calls
  - _Requirements: 3.1, 3.2_

- [ ] 3.1 Write property test for embedding and storage round-trip
  - **Property 4: Embedding and storage round-trip**
  - **Validates: Requirements 3.1, 3.2**

- [ ] 3.2 Write property test for query embedding consistency
  - **Property 5: Query embedding consistency**
  - **Validates: Requirements 3.3**

- [ ] 3.3 Write property test for retrieval count
  - **Property 6: Retrieval count**
  - **Validates: Requirements 3.4**

- [ ] 4. Implement hybrid retrieval system
  - Create HybridRetriever class combining vector and keyword search
  - Implement vector_search using VectorStore
  - Implement keyword_search using BM25Retriever from LangChain
  - Implement merge_results with weights (0.7 vector, 0.3 keyword)
  - _Requirements: 8.1, 8.2, 8.3_

- [ ] 4.1 Write property test for hybrid search execution
  - **Property 18: Hybrid search execution**
  - **Validates: Requirements 8.1**

- [ ] 4.2 Write property test for hybrid search weighting
  - **Property 19: Hybrid search weighting**
  - **Validates: Requirements 8.2**

- [ ] 4.3 Write property test for result ranking
  - **Property 20: Result ranking by score**
  - **Validates: Requirements 8.3**

- [ ] 5. Implement RAG chain with citations
  - Create RAGChain class with LLM integration (GPT-4 or Claude)
  - Implement system prompt instructing model to cite sources
  - Implement format_context to prepare retrieved chunks for LLM
  - Implement generate_response with citation extraction
  - Create Response and Citation data models
  - Add response streaming support
  - _Requirements: 4.1, 4.2, 4.3_

- [ ] 5.1 Write property test for response citations
  - **Property 7: Response contains citations**
  - **Validates: Requirements 4.1**

- [ ] 5.2 Write property test for response faithfulness
  - **Property 8: Response faithfulness**
  - **Validates: Requirements 4.2**

- [ ] 5.3 Write property test for response latency
  - **Property 9: Response latency**
  - **Validates: Requirements 4.4**

- [ ] 5.4 Write unit test for insufficient context handling
  - Test that system admits when it cannot answer based on context
  - _Requirements: 4.3_

- [ ] 6. Implement conversation management with Redis
  - Create ConversationManager class with Redis integration
  - Implement create_conversation with UUID generation
  - Implement get_history to retrieve messages from Redis
  - Implement add_message to store messages with timestamps
  - Create Message data model
  - Add TTL-based cleanup for old conversations
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [ ] 6.1 Write property test for conversation ID uniqueness
  - **Property 10: Conversation ID uniqueness**
  - **Validates: Requirements 5.1**

- [ ] 6.2 Write property test for conversation history round-trip
  - **Property 11: Conversation history round-trip**
  - **Validates: Requirements 5.2, 5.3**

- [ ] 6.3 Write property test for message chronological order
  - **Property 12: Message chronological order**
  - **Validates: Requirements 5.4**

- [ ] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Implement FastAPI service with core endpoints
  - Create FastAPI application with CORS middleware
  - Create Pydantic models: QueryRequest, QueryResponse, UploadResponse, Source
  - Implement POST /api/v1/query endpoint with async processing
  - Implement POST /api/v1/documents endpoint for file uploads
  - Implement GET /api/v1/conversations/{conversation_id} endpoint
  - Add OpenAPI documentation configuration
  - _Requirements: 6.1, 6.2, 6.4_

- [ ] 8.1 Write property test for query API response structure
  - **Property 13: Query API response structure**
  - **Validates: Requirements 6.1**

- [ ] 8.2 Write property test for document indexing completeness
  - **Property 14: Document indexing completeness**
  - **Validates: Requirements 6.2**

- [ ] 8.3 Write unit test for OpenAPI documentation endpoint
  - Test that /docs endpoint is accessible
  - _Requirements: 6.4_

- [ ] 9. Implement authentication and rate limiting
  - Create JWT token generation and validation utilities
  - Implement HTTPBearer security scheme
  - Create authenticate dependency for protected endpoints
  - Integrate slowapi for rate limiting (10 requests/minute per IP)
  - Add rate limit headers to responses
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [ ] 9.1 Write property test for authentication rejection
  - **Property 15: Authentication rejection**
  - **Validates: Requirements 7.1**

- [ ] 9.2 Write property test for authentication acceptance
  - **Property 16: Authentication acceptance**
  - **Validates: Requirements 7.2**

- [ ] 9.3 Write unit test for rate limiting threshold
  - Test that 11th request within a minute is rejected with 429
  - _Requirements: 7.3_

- [ ] 9.4 Write property test for rate limit independence
  - **Property 17: Rate limit independence**
  - **Validates: Requirements 7.4**

- [ ] 10. Implement error handling across all components
  - Add error handling for document processing (unsupported format, corrupted file, empty document, file too large)
  - Add error handling for embedding pipeline (rate limits, auth failures, timeouts)
  - Add error handling for vector database (connection failures, query timeouts)
  - Add error handling for LLM (rate limits, context length, content policy, timeouts)
  - Add error handling for conversation manager (Redis connection, conversation not found)
  - Add error handling for API (invalid/expired JWT, rate limits, invalid request body)
  - Implement exponential backoff for retryable errors
  - _Requirements: All error scenarios_

- [ ] 10.1 Write unit tests for error conditions
  - Test unsupported file format returns 400
  - Test invalid JWT returns 401
  - Test expired JWT returns 401
  - Test conversation not found returns 404

- [ ] 11. Implement evaluation framework
  - Create EvaluationFramework class with RAGAS integration
  - Implement evaluate_faithfulness using RAGAS faithfulness metric
  - Implement evaluate_relevancy using RAGAS answer_relevancy metric
  - Implement evaluate_context_precision using RAGAS context_precision metric
  - Implement run_evaluation to process test datasets
  - Create EvaluationResults data model
  - _Requirements: 9.1, 9.2, 9.3_

- [ ] 11.1 Write property test for faithfulness metric calculation
  - **Property 21: Faithfulness metric calculation**
  - **Validates: Requirements 9.1**

- [ ] 11.2 Write property test for relevancy metric calculation
  - **Property 22: Relevancy metric calculation**
  - **Validates: Requirements 9.2**

- [ ] 11.3 Write property test for context precision metric calculation
  - **Property 23: Context precision metric calculation**
  - **Validates: Requirements 9.3**

- [ ] 12. Create integration tests for end-to-end workflows
  - Test document upload to query workflow
  - Test multi-turn conversation workflow
  - Test authentication flow with token generation
  - Test hybrid search pipeline with real documents
  - _Requirements: All_

- [ ] 13. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 14. Create deployment configuration and documentation
  - Create Docker configuration for containerization
  - Create docker-compose.yml with Redis and application services
  - Write README with setup instructions
  - Write API usage documentation with examples
  - Create environment variable configuration guide
  - Document evaluation metrics and how to run evaluation
  - _Requirements: All_
