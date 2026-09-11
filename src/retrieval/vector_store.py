import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

class VectorStore:
    """Vector index store supporting cosine similarity ranking and metadata association."""
    
    def __init__(self):
        self.vectors: np.ndarray = None  # shape (N, D), normalized
        self.metadata: List[Dict[str, Any]] = []
        
    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.metadata.extend(metadata)
        
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Computes cosine similarity of query_vector (1, D) against all stored vectors (N, D).
        Returns top-k (metadata, similarity_score) pairs.
        """
        if self.vectors is None or len(self.metadata) == 0:
            return []
            
        # Ensure query vector is shape (1, D)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
            
        # Dot product of L2-normalized vectors is exact cosine similarity in [-1, 1]
        sims = np.dot(self.vectors, query_vector.T).flatten()
        # Clip to [0, 1]
        sims = np.clip(sims, 0.0, 1.0)
        
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.metadata[idx], float(sims[idx])) for idx in top_indices]
        
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"vectors": self.vectors, "metadata": self.metadata}, f)
            
    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.vectors = data["vectors"]
            self.metadata = data["metadata"]
