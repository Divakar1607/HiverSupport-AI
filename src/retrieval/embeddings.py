from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

class DenseTextEncoder:
    """
    Fast, reproducible dense embedding encoder using character-word n-gram features
    with L2-normalization for exact cosine similarity calculation.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=1,
            analyzer="word"
        )
        self.is_fit = False
        
    def fit(self, texts: List[str]) -> None:
        self.vectorizer.fit(texts)
        self.is_fit = True
        
    def encode(self, texts: List[str]) -> np.ndarray:
        if not self.is_fit:
            raise RuntimeError("Encoder must be fit before encoding.")
        sparse_vecs = self.vectorizer.transform(texts)
        # Convert to dense normalized array for fast dot-product cosine similarity
        dense_vecs = sparse_vecs.toarray().astype(np.float32)
        return normalize(dense_vecs, norm="l2", axis=1)
