import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.embeddings import DenseTextEncoder
from src.retrieval.vector_store import VectorStore
from src.utils.io import get_project_root, load_config
from src.utils.logging import get_logger

logger = get_logger("retrieve")

class HybridEvidenceRetriever:
    """
    Hybrid retriever combining lexical BM25 matching with dense vector similarity.
    Indexes only training historical conversations to guarantee zero leakage into evaluation.
    """
    
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha  # weight for BM25 score vs dense score
        self.bm25_retriever = BM25Retriever()
        self.encoder = DenseTextEncoder()
        self.vector_store = VectorStore()
        self.evidence_documents: List[Dict[str, Any]] = []
        self.is_indexed = False
        
    def build_index(self, pairs_df: pd.DataFrame) -> None:
        """Indexes historical customer message and brand reply pairs."""
        logger.info(f"Building hybrid retrieval index from {len(pairs_df)} historical training pairs...")
        docs = []
        texts = []
        
        for idx, row in pairs_df.iterrows():
            cust_text = str(row["customer_message"]).strip()
            brand_reply = str(row["brand_reply"]).strip()
            conv_id = str(row.get("conversation_id", f"hist_{idx:04d}"))
            
            doc = {
                "evidence_id": f"ev_{idx:04d}",
                "conversation_id": conv_id,
                "customer_message": cust_text,
                "brand_response": brand_reply,
                "resolution_pattern": "Historical verified agent resolution via DM/Public"
            }
            docs.append(doc)
            texts.append(cust_text)
            
        self.evidence_documents = docs
        
        # 1. Fit BM25
        self.bm25_retriever.fit(docs)
        
        # 2. Fit and encode dense vectors
        self.encoder.fit(texts)
        dense_vectors = self.encoder.encode(texts)
        self.vector_store.add(dense_vectors, docs)
        
        self.is_indexed = True
        logger.info(f"Hybrid index successfully built with {len(docs)} documents.")
        
    def retrieve(self, query: str, top_k: int = 3, intent_hint: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Performs hybrid retrieval scoring:
        hybrid_score = alpha * norm_bm25 + (1 - alpha) * cosine_sim
        """
        if not self.is_indexed or len(self.evidence_documents) == 0:
            return []
            
        # 1. Lexical BM25 scores
        raw_bm25 = np.array(self.bm25_retriever.get_scores(query))
        max_bm25 = raw_bm25.max() if len(raw_bm25) > 0 and raw_bm25.max() > 0 else 1.0
        norm_bm25 = raw_bm25 / max_bm25 if max_bm25 > 0 else raw_bm25
        
        # 2. Dense semantic similarity scores
        query_vec = self.encoder.encode([query])[0]
        dense_sims = np.dot(self.vector_store.vectors, query_vec.T).flatten()
        dense_sims = np.clip(dense_sims, 0.0, 1.0)
        
        # 3. Hybrid fusion score
        hybrid_scores = (self.alpha * norm_bm25) + ((1.0 - self.alpha) * dense_sims)
        
        # Rank by hybrid score descending
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        results = []
        for rank_idx in top_indices:
            doc = self.evidence_documents[rank_idx]
            sim_score = round(float(hybrid_scores[rank_idx]), 4)
            results.append({
                "evidence_id": doc["evidence_id"],
                "similarity_score": sim_score,
                "customer_message": doc["customer_message"],
                "brand_response": doc["brand_response"],
                "conversation_id": doc["conversation_id"]
            })
            
        return results

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = index_dir / "hybrid_retriever.pkl"
        with open(artifact_path, "wb") as f:
            pickle.dump({
                "alpha": self.alpha,
                "bm25": self.bm25_retriever,
                "encoder": self.encoder,
                "vector_store": self.vector_store,
                "evidence_documents": self.evidence_documents
            }, f)
        logger.info(f"Saved hybrid retriever index to {artifact_path}")
        
    @classmethod
    def load(cls, index_dir: Path) -> "HybridEvidenceRetriever":
        artifact_path = index_dir / "hybrid_retriever.pkl"
        if not artifact_path.exists():
            raise FileNotFoundError(f"Retrieval index not found at {artifact_path}. Run build_index.py first.")
        with open(artifact_path, "rb") as f:
            data = pickle.load(f)
            retriever = cls(alpha=data["alpha"])
            retriever.bm25_retriever = data["bm25"]
            retriever.encoder = data["encoder"]
            retriever.vector_store = data["vector_store"]
            retriever.evidence_documents = data["evidence_documents"]
            retriever.is_indexed = True
            return retriever
