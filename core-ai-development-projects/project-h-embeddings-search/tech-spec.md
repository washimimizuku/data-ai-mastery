# Tech Spec: Embedding & Similarity Search

## Stack
sentence-transformers, FAISS, Qdrant, scikit-learn, Rust (optional)

## Embedding Models
- all-MiniLM-L6-v2
- all-mpnet-base-v2
- bge-small-en-v1.5

## Vector Databases
```python
# FAISS
import faiss
index = faiss.IndexFlatL2(dimension)

# Qdrant
from qdrant_client import QdrantClient
client = QdrantClient(":memory:")
```

## Clustering
```python
from umap import UMAP
from sklearn.cluster import KMeans

reducer = UMAP(n_components=2)
embeddings_2d = reducer.fit_transform(embeddings)
```

## Optional Rust
Fast cosine similarity, approximate nearest neighbors in Rust for performance comparison.
