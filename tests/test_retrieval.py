from pathlib import Path
import pytest
import pandas as pd

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.embeddings import DenseTextEncoder
from src.retrieval.retrieve import HybridEvidenceRetriever
from src.retrieval.vector_store import VectorStore
from src.utils.io import get_project_root

def test_bm25_retriever():
    docs = [
        {"customer_message": "Where is my package? Tracking says delivered."},
        {"customer_message": "I was charged twice for my Prime membership."},
        {"customer_message": "How do I cancel an order?"}
    ]
    bm25 = BM25Retriever()
    bm25.fit(docs)
    scores = bm25.get_scores("package delivered")
    assert len(scores) == 3
    assert scores[0] > scores[1]

def test_dense_encoder_and_vector_store():
    encoder = DenseTextEncoder()
    texts = ["late package tracking", "double charge refund"]
    encoder.fit(texts)
    vecs = encoder.encode(texts)
    assert vecs.shape[0] == 2
    
    store = VectorStore()
    store.add(vecs, [{"id": 1}, {"id": 2}])
    hits = store.search(vecs[0], top_k=1)
    assert len(hits) == 1
    assert hits[0][0]["id"] == 1

def test_hybrid_retriever_load_and_retrieve():
    root = get_project_root()
    index_dir = root / "artifacts" / "retrieval_index"
    retriever = HybridEvidenceRetriever.load(index_dir)
    hits = retriever.retrieve("Where is my package?", top_k=2)
    assert len(hits) > 0
    assert "evidence_id" in hits[0]
    assert "similarity_score" in hits[0]
    assert "customer_message" in hits[0]
    assert "brand_response" in hits[0]
