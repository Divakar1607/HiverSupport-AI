import math
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi

class BM25Retriever:
    """Lexical BM25 retriever for historical customer support evidence."""
    
    def __init__(self):
        self.corpus_tokens: List[List[str]] = []
        self.documents: List[Dict] = []
        self.bm25: BM25Okapi = None
        
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple, fast domain-specific tokenizer."""
        return [
            w.strip(".,!?\"':;()[]{}")
            for w in text.lower().split()
            if len(w.strip(".,!?\"':;()[]{}")) > 1
        ]
        
    def fit(self, documents: List[Dict]) -> None:
        """
        Fits BM25 index on a list of document dicts.
        Each doc should have 'customer_message' or 'text'.
        """
        self.documents = documents
        self.corpus_tokens = [
            self.tokenize(doc.get("customer_message", doc.get("text", "")))
            for doc in documents
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        
    def get_scores(self, query: str) -> List[float]:
        """Returns raw BM25 scores for all documents in the corpus."""
        if self.bm25 is None or len(self.documents) == 0:
            return []
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return [0.0] * len(self.documents)
        return list(self.bm25.get_scores(query_tokens))
