# Project H: Embedding & Similarity Search

## Objective

Build a comprehensive semantic search and similarity system demonstrating embedding generation, vector databases (FAISS, Qdrant), similarity algorithms, clustering, duplicate detection, and recommendation engines to achieve sub-100ms search latency on 1M+ documents.

**What You'll Build**: A production-ready semantic search engine with multiple embedding models, vector database comparison, advanced similarity algorithms, clustering visualization, duplicate detection, and a recommendation system with FastAPI and Gradio interfaces.

**What You'll Learn**: Embedding models and selection, vector databases (FAISS vs Qdrant), similarity metrics (cosine, dot product, L2), approximate nearest neighbors (ANN), HNSW indexing, clustering (UMAP, t-SNE), duplicate detection, recommendation algorithms, and production deployment.

## Time Estimate

**2 days (16 hours)** - Following the implementation plan

### Day 1 (8 hours)
- **Hour 1**: Setup & basics (install dependencies, download models, test embeddings)
- **Hours 2-3**: Embedding generation (multiple models, batch processing, caching)
- **Hours 4-5**: FAISS integration (flat index, IVF, HNSW, benchmarking)
- **Hours 6-7**: Qdrant integration (collections, filtering, hybrid search)
- **Hour 8**: Similarity algorithms (cosine, dot product, L2, comparison)

### Day 2 (8 hours)
- **Hours 1-2**: Clustering & visualization (UMAP, t-SNE, K-means, DBSCAN)
- **Hours 3-4**: Duplicate detection (fuzzy matching, semantic similarity)
- **Hours 5-6**: Recommendation system (content-based, collaborative filtering)
- **Hour 7**: FastAPI & Gradio UI (search API, interactive demo)
- **Hour 8**: Documentation & optimization (performance tuning, deployment guide)

## Prerequisites

### Required Knowledge
Complete these bootcamp sections first:
- [100 Days Data & AI](https://github.com/washimimizuku/100-days-data-ai) - Days 71-85
  - Days 71-75: Embeddings fundamentals
  - Days 76-80: Vector databases and similarity search
  - Days 81-85: Semantic search applications
- [30 Days of Python](https://github.com/washimimizuku/30-days-python-data-ai) - Days 1-20

### Technical Requirements
- Python 3.11+ installed
- 8GB+ RAM (16GB recommended for large datasets)
- Understanding of vector spaces and similarity
- Basic knowledge of embeddings

### Tools Needed
- Python with sentence-transformers, faiss, qdrant-client
- Gradio for UI
- FastAPI for API
- Git for version control

## Getting Started

### Step 1: Review Documentation
Read the project documents in order:
1. `prd.md` - Understand what you're building
2. `tech-spec.md` - Review technical architecture
3. `implementation-plan.md` - Follow the implementation steps

### Step 2: Set Up Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install sentence-transformers faiss-cpu numpy pandas

# Install vector databases
pip install qdrant-client

# Install visualization
pip install umap-learn scikit-learn matplotlib plotly

# Install API and UI
pip install fastapi uvicorn gradio

# Install utilities
pip install datasets tqdm

# Verify installation
python -c "from sentence_transformers import SentenceTransformer; print('✓ Setup complete')"
```

### Step 3: Download and Test Embedding Models

```python
# test_embeddings.py
from sentence_transformers import SentenceTransformer
import time

# Test different embedding models
models = [
    'all-MiniLM-L6-v2',      # 384 dim, 80MB, fast
    'all-mpnet-base-v2',     # 768 dim, 420MB, best quality
    'bge-small-en-v1.5',     # 384 dim, 130MB, good balance
]

for model_name in models:
    print(f"\nTesting {model_name}...")
    
    # Load model
    start = time.time()
    model = SentenceTransformer(model_name)
    load_time = time.time() - start
    
    # Generate embeddings
    texts = ["This is a test sentence"] * 100
    start = time.time()
    embeddings = model.encode(texts, show_progress_bar=False)
    encode_time = time.time() - start
    
    print(f"  Load time: {load_time:.2f}s")
    print(f"  Encode time: {encode_time:.2f}s for 100 sentences")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"  Throughput: {len(texts)/encode_time:.0f} sentences/sec")
```

### Step 4: Build Embedding Manager
```python
# src/embeddings/embedding_manager.py
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np
from pathlib import Path
import pickle

class EmbeddingManager:
    """Manage embedding generation and caching"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = './cache'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.embedding_cache = {}
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """Generate embeddings for texts"""
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize
        )
        
        return embeddings
    
    def encode_with_cache(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings with caching"""
        
        # Check cache
        cache_key = self._get_cache_key(texts)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            print(f"Loading from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Generate embeddings
        embeddings = self.encode(texts)
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
        
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode single query"""
        return self.model.encode([query], normalize_embeddings=True)[0]
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()
    
    def _get_cache_key(self, texts: List[str]) -> str:
        """Generate cache key from texts"""
        import hashlib
        text_hash = hashlib.md5(''.join(texts).encode()).hexdigest()
        return f"{self.model_name}_{text_hash}"
    
    def compare_models(self, texts: List[str]) -> Dict:
        """Compare different embedding models"""
        models_to_test = [
            'all-MiniLM-L6-v2',
            'all-mpnet-base-v2',
            'bge-small-en-v1.5'
        ]
        
        results = {}
        
        for model_name in models_to_test:
            print(f"\nTesting {model_name}...")
            model = SentenceTransformer(model_name)
            
            import time
            start = time.time()
            embeddings = model.encode(texts, show_progress_bar=False)
            elapsed = time.time() - start
            
            results[model_name] = {
                'dimension': embeddings.shape[1],
                'time': elapsed,
                'throughput': len(texts) / elapsed,
                'embeddings': embeddings
            }
        
        return results

# Usage
if __name__ == "__main__":
    manager = EmbeddingManager()
    
    texts = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing deals with text and speech"
    ]
    
    embeddings = manager.encode(texts)
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {embeddings.shape[1]}")
```

### Step 5: Implement FAISS Search
```python
# src/search/faiss_search.py
import faiss
import numpy as np
from typing import List, Tuple
import time

class FAISSSearch:
    """FAISS-based similarity search"""
    
    def __init__(self, dimension: int, index_type: str = 'flat'):
        self.dimension = dimension
        self.index_type = index_type
        self.index = self._create_index(index_type)
        self.documents = []
    
    def _create_index(self, index_type: str):
        """Create FAISS index"""
        
        if index_type == 'flat':
            # Exact search (L2 distance)
            return faiss.IndexFlatL2(self.dimension)
        
        elif index_type == 'flat_ip':
            # Exact search (inner product / cosine similarity)
            return faiss.IndexFlatIP(self.dimension)
        
        elif index_type == 'ivf':
            # Inverted file index (faster, approximate)
            quantizer = faiss.IndexFlatL2(self.dimension)
            nlist = 100  # number of clusters
            return faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        
        elif index_type == 'hnsw':
            # Hierarchical Navigable Small World (fast, accurate)
            M = 32  # number of connections
            return faiss.IndexHNSWFlat(self.dimension, M)
        
        else:
            raise ValueError(f"Unknown index type: {index_type}")
    
    def add_documents(self, embeddings: np.ndarray, documents: List[str]):
        """Add documents to index"""
        
        # Ensure float32
        embeddings = embeddings.astype('float32')
        
        # Train index if needed (for IVF)
        if self.index_type == 'ivf' and not self.index.is_trained:
            print("Training IVF index...")
            self.index.train(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        self.documents.extend(documents)
        
        print(f"Added {len(documents)} documents to index")
        print(f"Total documents: {self.index.ntotal}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> Tuple[List[float], List[int]]:
        """Search for similar documents"""
        
        # Ensure correct shape and type
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        return distances[0].tolist(), indices[0].tolist()
    
    def search_with_documents(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Dict]:
        """Search and return documents with scores"""
        
        distances, indices = self.search(query_embedding, k)
        
        results = []
        for dist, idx in zip(distances, indices):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'distance': float(dist),
                    'similarity': 1 / (1 + dist),  # Convert distance to similarity
                    'index': int(idx)
                })
        
        return results
    
    def benchmark(self, query_embeddings: np.ndarray, k: int = 10) -> Dict:
        """Benchmark search performance"""
        
        query_embeddings = query_embeddings.astype('float32')
        
        # Warm-up
        self.index.search(query_embeddings[:1], k)
        
        # Benchmark
        start = time.time()
        distances, indices = self.index.search(query_embeddings, k)
        elapsed = time.time() - start
        
        qps = len(query_embeddings) / elapsed
        latency = elapsed / len(query_embeddings) * 1000  # ms
        
        return {
            'queries': len(query_embeddings),
            'time': elapsed,
            'qps': qps,
            'latency_ms': latency
        }
    
    def save(self, filepath: str):
        """Save index to disk"""
        faiss.write_index(self.index, filepath)
        print(f"Index saved to {filepath}")
    
    def load(self, filepath: str):
        """Load index from disk"""
        self.index = faiss.read_index(filepath)
        print(f"Index loaded from {filepath}")

# Usage
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    
    # Generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    documents = [
        "Python is a programming language",
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "Natural language processing deals with text",
        "Computer vision processes images"
    ]
    embeddings = model.encode(documents)
    
    # Create FAISS index
    search = FAISSSearch(dimension=embeddings.shape[1], index_type='flat')
    search.add_documents(embeddings, documents)
    
    # Search
    query = "What is machine learning?"
    query_embedding = model.encode([query])[0]
    results = search.search_with_documents(query_embedding, k=3)
    
    print(f"\nQuery: {query}")
    print("\nTop results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['document']}")
        print(f"   Similarity: {result['similarity']:.3f}")
```

### Step 6: Implement Qdrant Search
```python
# src/search/qdrant_search.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict
import numpy as np
import uuid

class QdrantSearch:
    """Qdrant-based similarity search with filtering"""
    
    def __init__(self, collection_name: str = "documents", dimension: int = 384):
        self.client = QdrantClient(":memory:")  # In-memory for development
        self.collection_name = collection_name
        self.dimension = dimension
        self._create_collection()
    
    def _create_collection(self):
        """Create Qdrant collection"""
        
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.dimension,
                distance=Distance.COSINE
            )
        )
        print(f"Created collection: {self.collection_name}")
    
    def add_documents(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        metadata: List[Dict] = None
    ):
        """Add documents with metadata to Qdrant"""
        
        points = []
        for i, (embedding, document) in enumerate(zip(embeddings, documents)):
            payload = {
                'text': document,
                'index': i
            }
            
            # Add metadata if provided
            if metadata and i < len(metadata):
                payload.update(metadata[i])
            
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload=payload
                )
            )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"Added {len(points)} documents to Qdrant")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_dict: Dict = None
    ) -> List[Dict]:
        """Search with optional filtering"""
        
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=k,
            query_filter=filter_dict
        )
        
        results = []
        for hit in search_result:
            results.append({
                'document': hit.payload['text'],
                'score': hit.score,
                'metadata': {k: v for k, v in hit.payload.items() if k != 'text'},
                'id': hit.id
            })
        
        return results
    
    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        k: int = 5
    ) -> List[Dict]:
        """Hybrid search combining vector and text search"""
        
        # Vector search
        vector_results = self.search(query_embedding, k=k*2)
        
        # Simple text matching (in production, use full-text search)
        text_scores = {}
        query_words = set(query_text.lower().split())
        
        for result in vector_results:
            doc_words = set(result['document'].lower().split())
            overlap = len(query_words & doc_words)
            text_scores[result['id']] = overlap / len(query_words) if query_words else 0
        
        # Combine scores
        for result in vector_results:
            vector_score = result['score']
            text_score = text_scores.get(result['id'], 0)
            result['hybrid_score'] = 0.7 * vector_score + 0.3 * text_score
        
        # Sort by hybrid score
        vector_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return vector_results[:k]

# Usage
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    documents = [
        "Python programming language",
        "Machine learning algorithms",
        "Deep learning neural networks",
        "Natural language processing",
        "Computer vision applications"
    ]
    
    metadata = [
        {'category': 'programming', 'year': 2023},
        {'category': 'ai', 'year': 2023},
        {'category': 'ai', 'year': 2024},
        {'category': 'nlp', 'year': 2024},
        {'category': 'cv', 'year': 2024}
    ]
    
    embeddings = model.encode(documents)
    
    # Create Qdrant search
    search = QdrantSearch(dimension=embeddings.shape[1])
    search.add_documents(embeddings, documents, metadata)
    
    # Search with filter
    query = "artificial intelligence"
    query_embedding = model.encode([query])[0]
    
    results = search.search(
        query_embedding,
        k=3,
        filter_dict={'category': 'ai'}
    )
    
    print(f"\nQuery: {query}")
    print("Filtered results (category='ai'):")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['document']}")
        print(f"   Score: {result['score']:.3f}")
        print(f"   Metadata: {result['metadata']}")
```

### Step 7: Implement Clustering & Visualization
```python
# src/analysis/clustering.py
from umap import UMAP
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from typing import List, Dict

class EmbeddingClusterer:
    """Cluster and visualize embeddings"""
    
    def __init__(self, embeddings: np.ndarray, labels: List[str] = None):
        self.embeddings = embeddings
        self.labels = labels or [f"Doc {i}" for i in range(len(embeddings))]
        self.reduced_embeddings = None
        self.clusters = None
    
    def reduce_dimensions(self, method: str = 'umap', n_components: int = 2):
        """Reduce embeddings to 2D/3D for visualization"""
        
        if method == 'umap':
            reducer = UMAP(
                n_components=n_components,
                random_state=42,
                n_neighbors=15,
                min_dist=0.1
            )
        elif method == 'tsne':
            reducer = TSNE(
                n_components=n_components,
                random_state=42,
                perplexity=30
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"Reducing dimensions with {method}...")
        self.reduced_embeddings = reducer.fit_transform(self.embeddings)
        
        return self.reduced_embeddings
    
    def cluster(self, method: str = 'kmeans', n_clusters: int = 5):
        """Cluster embeddings"""
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=2)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"Clustering with {method}...")
        self.clusters = clusterer.fit_predict(self.embeddings)
        
        return self.clusters
    
    def visualize_2d(self, title: str = "Embedding Visualization"):
        """Visualize embeddings in 2D"""
        
        if self.reduced_embeddings is None:
            self.reduce_dimensions(n_components=2)
        
        if self.clusters is None:
            self.cluster()
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            self.reduced_embeddings[:, 0],
            self.reduced_embeddings[:, 1],
            c=self.clusters,
            cmap='viridis',
            alpha=0.6,
            s=100
        )
        
        # Add labels for some points
        for i in range(min(20, len(self.labels))):
            plt.annotate(
                self.labels[i][:30],
                (self.reduced_embeddings[i, 0], self.reduced_embeddings[i, 1]),
                fontsize=8,
                alpha=0.7
            )
        
        plt.colorbar(scatter, label='Cluster')
        plt.title(title)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.tight_layout()
        plt.savefig('embedding_visualization.png', dpi=300)
        print("Visualization saved to embedding_visualization.png")
    
    def visualize_interactive(self, title: str = "Interactive Embedding Visualization"):
        """Create interactive 3D visualization"""
        
        if self.reduced_embeddings is None or self.reduced_embeddings.shape[1] != 3:
            self.reduce_dimensions(n_components=3)
        
        if self.clusters is None:
            self.cluster()
        
        import pandas as pd
        df = pd.DataFrame({
            'x': self.reduced_embeddings[:, 0],
            'y': self.reduced_embeddings[:, 1],
            'z': self.reduced_embeddings[:, 2],
            'cluster': self.clusters,
            'label': self.labels
        })
        
        fig = px.scatter_3d(
            df,
            x='x',
            y='y',
            z='z',
            color='cluster',
            hover_data=['label'],
            title=title
        )
        
        fig.write_html('embedding_visualization_3d.html')
        print("Interactive visualization saved to embedding_visualization_3d.html")
        
        return fig

# Usage
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset
    
    # Load sample data
    dataset = load_dataset("ag_news", split="train[:1000]")
    texts = dataset['text']
    
    # Generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Cluster and visualize
    clusterer = EmbeddingClusterer(embeddings, labels=texts)
    clusterer.reduce_dimensions(method='umap')
    clusterer.cluster(method='kmeans', n_clusters=4)
    clusterer.visualize_2d(title="News Articles Clustering")
```

### Step 8: Implement Duplicate Detection

```python
# src/analysis/duplicate_detection.py
from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class DuplicateDetector:
    """Detect duplicate and near-duplicate documents"""
    
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
    
    def find_duplicates(
        self,
        embeddings: np.ndarray,
        documents: List[str]
    ) -> List[Tuple[int, int, float]]:
        """Find duplicate pairs based on similarity threshold"""
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        duplicates = []
        n = len(embeddings)
        
        for i in range(n):
            for j in range(i + 1, n):
                if similarities[i, j] >= self.threshold:
                    duplicates.append((i, j, similarities[i, j]))
        
        return duplicates
    
    def find_duplicate_groups(
        self,
        embeddings: np.ndarray,
        documents: List[str]
    ) -> List[List[int]]:
        """Group duplicates together"""
        
        duplicates = self.find_duplicates(embeddings, documents)
        
        # Build graph of duplicates
        from collections import defaultdict
        graph = defaultdict(set)
        
        for i, j, sim in duplicates:
            graph[i].add(j)
            graph[j].add(i)
        
        # Find connected components
        visited = set()
        groups = []
        
        def dfs(node, group):
            visited.add(node)
            group.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, group)
        
        for node in graph:
            if node not in visited:
                group = []
                dfs(node, group)
                groups.append(sorted(group))
        
        return groups
    
    def deduplicate(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        keep: str = 'first'
    ) -> Tuple[np.ndarray, List[str], List[int]]:
        """Remove duplicates, keeping one from each group"""
        
        groups = self.find_duplicate_groups(embeddings, documents)
        
        # Indices to keep
        to_keep = set(range(len(documents)))
        
        for group in groups:
            if keep == 'first':
                # Keep first in group
                to_remove = group[1:]
            elif keep == 'longest':
                # Keep longest document
                longest_idx = max(group, key=lambda i: len(documents[i]))
                to_remove = [i for i in group if i != longest_idx]
            else:
                to_remove = group[1:]
            
            to_keep -= set(to_remove)
        
        # Filter
        kept_indices = sorted(list(to_keep))
        deduplicated_embeddings = embeddings[kept_indices]
        deduplicated_documents = [documents[i] for i in kept_indices]
        
        print(f"Original: {len(documents)} documents")
        print(f"After deduplication: {len(deduplicated_documents)} documents")
        print(f"Removed: {len(documents) - len(deduplicated_documents)} duplicates")
        
        return deduplicated_embeddings, deduplicated_documents, kept_indices

# Usage
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    documents = [
        "Machine learning is a subset of AI",
        "Machine learning is a subset of artificial intelligence",  # Near duplicate
        "Deep learning uses neural networks",
        "Python is a programming language",
        "Machine learning is a subset of AI",  # Exact duplicate
    ]
    
    embeddings = model.encode(documents)
    
    detector = DuplicateDetector(threshold=0.90)
    duplicates = detector.find_duplicates(embeddings, documents)
    
    print("Duplicate pairs:")
    for i, j, sim in duplicates:
        print(f"\nSimilarity: {sim:.3f}")
        print(f"  Doc {i}: {documents[i]}")
        print(f"  Doc {j}: {documents[j]}")
    
    # Deduplicate
    dedup_emb, dedup_docs, kept = detector.deduplicate(embeddings, documents)
```

## Key Features to Implement

### 1. Embedding Generation
- **Multiple models**: MiniLM, MPNet, BGE
- **Batch processing**: Efficient encoding
- **Caching**: Save/load embeddings
- **Normalization**: L2 normalization for cosine similarity

### 2. Vector Databases
- **FAISS**: Flat, IVF, HNSW indices
- **Qdrant**: Collections, filtering, hybrid search
- **Comparison**: Performance benchmarks

### 3. Similarity Search
- **Metrics**: Cosine, dot product, L2 distance
- **K-NN search**: Top-k similar documents
- **Filtering**: Metadata-based filtering
- **Hybrid search**: Vector + text matching

### 4. Clustering
- **Dimensionality reduction**: UMAP, t-SNE
- **Clustering algorithms**: K-means, DBSCAN
- **Visualization**: 2D/3D plots
- **Interactive**: Plotly visualizations

### 5. Duplicate Detection
- **Similarity threshold**: Configurable threshold
- **Grouping**: Connected components
- **Deduplication**: Remove near-duplicates
- **Fuzzy matching**: Semantic similarity

### 6. Recommendation System
- **Content-based**: Similar items
- **Collaborative filtering**: User-item matrix
- **Hybrid**: Combined approaches

### 7. API & UI
- **FastAPI**: REST API endpoints
- **Gradio**: Interactive search interface
- **Batch processing**: Bulk operations

## Success Criteria

By the end of this project, you should have:

- [ ] Multiple embedding models tested and compared
- [ ] FAISS index with flat, IVF, and HNSW variants
- [ ] Qdrant integration with filtering
- [ ] Similarity search with sub-100ms latency
- [ ] Clustering with UMAP/t-SNE visualization
- [ ] Duplicate detection working
- [ ] Recommendation system implemented
- [ ] FastAPI endpoints deployed
- [ ] Gradio UI for interactive search
- [ ] Performance benchmarks documented
- [ ] 1M+ documents indexed and searchable
- [ ] GitHub repository with examples

## Learning Outcomes

After completing this project, you'll be able to:

- Generate and work with embeddings
- Choose appropriate embedding models
- Implement vector similarity search
- Use FAISS and Qdrant effectively
- Understand similarity metrics
- Cluster and visualize embeddings
- Detect duplicates semantically
- Build recommendation systems
- Deploy search APIs
- Optimize search performance
- Explain ANN algorithms (HNSW, IVF)

## Expected Performance

Based on typical results:

**Embedding Generation** (all-MiniLM-L6-v2):
- Throughput: 500-1000 sentences/sec (CPU)
- Throughput: 5000-10000 sentences/sec (GPU)
- Dimension: 384

**Search Performance** (1M documents):
- FAISS Flat: 50-100ms (exact)
- FAISS IVF: 5-10ms (approximate, 95% recall)
- FAISS HNSW: 1-5ms (approximate, 98% recall)
- Qdrant: 10-20ms (with filtering)

**Clustering** (10K documents):
- UMAP reduction: 10-30 seconds
- K-means clustering: 1-5 seconds
- Visualization: 2-5 seconds

## Project Structure

```
project-h-embeddings-search/
├── src/
│   ├── embeddings/
│   │   ├── embedding_manager.py
│   │   └── model_comparison.py
│   ├── search/
│   │   ├── faiss_search.py
│   │   ├── qdrant_search.py
│   │   └── hybrid_search.py
│   ├── analysis/
│   │   ├── clustering.py
│   │   ├── duplicate_detection.py
│   │   └── visualization.py
│   ├── recommendation/
│   │   └── recommender.py
│   ├── api/
│   │   └── fastapi_server.py
│   └── ui/
│       └── gradio_app.py
├── data/
│   ├── documents.json
│   └── embeddings.npy
├── cache/
│   └── embeddings/
├── notebooks/
│   ├── 01_embedding_basics.ipynb
│   ├── 02_faiss_tutorial.ipynb
│   ├── 03_clustering.ipynb
│   └── 04_recommendations.ipynb
├── benchmarks/
│   ├── search_benchmark.py
│   └── results.csv
├── prd.md
├── tech-spec.md
├── implementation-plan.md
└── README.md
```

## Common Challenges & Solutions

### Challenge 1: Slow Embedding Generation
**Problem**: Encoding large datasets takes too long
**Solution**: Use GPU, batch processing, and caching
```python
# Use GPU if available
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

# Batch processing
embeddings = model.encode(texts, batch_size=128, show_progress_bar=True)

# Cache embeddings
np.save('embeddings.npy', embeddings)
```

### Challenge 2: High Memory Usage
**Problem**: Large embedding matrices don't fit in RAM
**Solution**: Use memory-mapped arrays or process in chunks
```python
# Memory-mapped array
embeddings = np.memmap('embeddings.dat', dtype='float32', 
                       mode='w+', shape=(n_docs, dim))

# Process in chunks
for i in range(0, len(texts), chunk_size):
    chunk = texts[i:i+chunk_size]
    embeddings[i:i+chunk_size] = model.encode(chunk)
```

### Challenge 3: Poor Search Quality
**Problem**: Search returns irrelevant results
**Solution**: Try different embedding models, normalize embeddings, tune similarity threshold
```python
# Normalize embeddings for cosine similarity
from sklearn.preprocessing import normalize
embeddings = normalize(embeddings, norm='l2')

# Try better embedding model
model = SentenceTransformer('all-mpnet-base-v2')  # Higher quality
```

### Challenge 4: Slow Search on Large Datasets
**Problem**: Search latency too high
**Solution**: Use approximate nearest neighbors (HNSW, IVF)
```python
# Use HNSW for fast approximate search
index = faiss.IndexHNSWFlat(dimension, 32)  # M=32 connections
index.add(embeddings)

# 10-100x faster than flat index with minimal accuracy loss
```

## Advanced Techniques

### 1. Multi-Vector Search
```python
# Search with multiple query vectors
query_embeddings = model.encode([query1, query2, query3])
all_results = []
for qe in query_embeddings:
    results = search.search(qe, k=10)
    all_results.extend(results)

# Deduplicate and re-rank
unique_results = deduplicate_results(all_results)
```

### 2. Reranking
```python
# Initial retrieval with fast index
candidates = faiss_search.search(query_embedding, k=100)

# Rerank with cross-encoder
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

pairs = [[query, doc] for doc in candidates]
scores = reranker.predict(pairs)

# Sort by reranker scores
reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
```

### 3. Query Expansion
```python
# Expand query with similar terms
similar_queries = search.search(query_embedding, k=5)
expanded_query = query + " " + " ".join([q['document'] for q in similar_queries])

# Search with expanded query
expanded_embedding = model.encode([expanded_query])[0]
results = search.search(expanded_embedding, k=10)
```

### 4. Negative Sampling
```python
# Exclude certain results
positive_embedding = model.encode([positive_query])[0]
negative_embedding = model.encode([negative_query])[0]

# Adjust query embedding
adjusted_embedding = positive_embedding - 0.3 * negative_embedding
results = search.search(adjusted_embedding, k=10)
```

## Troubleshooting

### Installation Issues

**Issue**: FAISS installation fails
```bash
# Solution: Install CPU version
pip install faiss-cpu

# Or GPU version (requires CUDA)
pip install faiss-gpu
```

**Issue**: UMAP installation fails
```bash
# Solution: Install with conda
conda install -c conda-forge umap-learn

# Or pip with specific version
pip install umap-learn==0.5.3
```

### Runtime Issues

**Issue**: "Index not trained" error with IVF
```python
# Solution: Train index before adding vectors
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist=100)
index.train(embeddings)  # Must train first
index.add(embeddings)
```

**Issue**: Qdrant connection errors
```python
# Solution: Use in-memory mode for development
client = QdrantClient(":memory:")

# Or connect to server
client = QdrantClient(host="localhost", port=6333)
```

### Performance Issues

**Issue**: Slow UMAP reduction
```python
# Solution: Reduce n_neighbors or use PCA first
from sklearn.decomposition import PCA

# Pre-reduce with PCA
pca = PCA(n_components=50)
embeddings_pca = pca.fit_transform(embeddings)

# Then UMAP
reducer = UMAP(n_neighbors=5)  # Smaller n_neighbors
embeddings_2d = reducer.fit_transform(embeddings_pca)
```

## Production Deployment

### FastAPI Server
```python
# src/api/fastapi_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Semantic Search API")

# Load model and index at startup
@app.on_event("startup")
async def startup_event():
    global model, search_engine
    model = SentenceTransformer('all-MiniLM-L6-v2')
    search_engine = FAISSSearch(dimension=384)
    # Load pre-built index
    search_engine.load('index.faiss')

class SearchRequest(BaseModel):
    query: str
    k: int = 10

class SearchResponse(BaseModel):
    results: List[Dict]
    query_time_ms: float

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    import time
    start = time.time()
    
    # Generate query embedding
    query_embedding = model.encode([request.query])[0]
    
    # Search
    results = search_engine.search_with_documents(query_embedding, k=request.k)
    
    query_time = (time.time() - start) * 1000
    
    return SearchResponse(results=results, query_time_ms=query_time)

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Run: uvicorn fastapi_server:app --host 0.0.0.0 --port 8000
```

### Gradio UI
```python
# src/ui/gradio_app.py
import gradio as gr
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
search_engine = FAISSSearch(dimension=384)
search_engine.load('index.faiss')

def search_interface(query, k):
    query_embedding = model.encode([query])[0]
    results = search_engine.search_with_documents(query_embedding, k=k)
    
    output = []
    for i, result in enumerate(results, 1):
        output.append(f"{i}. {result['document']}")
        output.append(f"   Similarity: {result['similarity']:.3f}\n")
    
    return "\n".join(output)

demo = gr.Interface(
    fn=search_interface,
    inputs=[
        gr.Textbox(label="Search Query", placeholder="Enter your search query..."),
        gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Number of Results")
    ],
    outputs=gr.Textbox(label="Search Results", lines=15),
    title="Semantic Search Engine",
    description="Search through documents using semantic similarity"
)

demo.launch(share=True)
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

CMD ["uvicorn", "src.api.fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Resources

### Documentation
- [Sentence Transformers](https://www.sbert.net/) - Embedding models
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [Qdrant](https://qdrant.tech/documentation/) - Vector database
- [UMAP](https://umap-learn.readthedocs.io/) - Dimensionality reduction

### Papers
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) - Sentence embeddings
- [HNSW](https://arxiv.org/abs/1603.09320) - Approximate nearest neighbors
- [UMAP](https://arxiv.org/abs/1802.03426) - Dimensionality reduction

### Tutorials
- [FAISS Tutorial](https://github.com/facebookresearch/faiss/wiki/Getting-started) - Getting started
- [Semantic Search Guide](https://www.sbert.net/examples/applications/semantic-search/README.html) - Applications
- [Vector Database Comparison](https://benchmark.vectorview.ai/) - Benchmarks

## Questions?

If you get stuck:
1. Review the `tech-spec.md` for detailed architecture
2. Check FAISS/Qdrant documentation for API details
3. Test with small datasets first (100-1000 documents)
4. Review the 100 Days bootcamp materials on embeddings
5. Compare different embedding models
6. Use visualization to debug clustering issues

## Related Projects

After completing this project, consider:
- **Project A**: Local RAG - Use embeddings for retrieval
- **Project F**: Inference Optimization - Optimize embedding generation
- **Project G**: Prompt Engineering - Optimize search prompts
- Build a production search engine for your domain
- Create a recommendation system for e-commerce
