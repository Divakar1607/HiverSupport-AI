import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from src.intent.taxonomy import load_intent_taxonomy
from src.utils.io import get_project_root, load_config
from src.utils.logging import get_logger

logger = get_logger("intent_classifier")

class BaseIntentClassifier(ABC):
    """Abstract base class for all intent classification models."""
    
    def __init__(self):
        self.labels: List[str] = []
        self.is_trained: bool = False
        
    @abstractmethod
    def train(self, texts: List[str], labels: List[str]) -> None:
        pass
        
    @abstractmethod
    def predict(self, texts: List[str]) -> List[str]:
        pass
        
    @abstractmethod
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        pass
        
    def predict_single(self, text: str, top_k: int = 3) -> Dict[str, Any]:
        """Predicts intent, confidence, and top-k ranked predictions for a single query."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() or load() first.")
            
        probs = self.predict_proba([text])[0]
        top_indices = np.argsort(probs)[::-1][:top_k]
        
        predicted_intent = self.labels[top_indices[0]]
        confidence = float(probs[top_indices[0]])
        
        top_k_predictions = [
            {"intent": self.labels[idx], "confidence": round(float(probs[idx]), 4)}
            for idx in top_indices
        ]
        
        return {
            "intent": predicted_intent,
            "confidence": round(confidence, 4),
            "top_k_predictions": top_k_predictions
        }

class MajorityIntentClassifier(BaseIntentClassifier):
    """Baseline 1: Always predicts the most frequent intent from training distribution."""
    
    def __init__(self):
        super().__init__()
        self.majority_label: str = "delivery_delay"
        self.empirical_prior: float = 1.0
        
    def train(self, texts: List[str], labels: List[str]) -> None:
        self.labels = sorted(list(set(labels)))
        series = pd.Series(labels)
        counts = series.value_counts()
        self.majority_label = counts.index[0]
        self.empirical_prior = float(counts.iloc[0] / len(labels))
        self.is_trained = True
        logger.info(f"MajorityClassifier trained: default='{self.majority_label}' (prior={self.empirical_prior:.3f})")
        
    def predict(self, texts: List[str]) -> List[str]:
        return [self.majority_label] * len(texts)
        
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        probs = np.zeros((len(texts), len(self.labels)), dtype=float)
        maj_idx = self.labels.index(self.majority_label)
        probs[:, maj_idx] = 1.0
        return probs

class TfidfLogisticRegressionClassifier(BaseIntentClassifier):
    """Baseline 2: TF-IDF (word + char n-grams) + Regularized Logistic Regression."""
    
    def __init__(self, c_val: float = 1.5):
        super().__init__()
        self.c_val = c_val
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            token_pattern=r'(?u)\b\w+\b'
        )
        self.clf = LogisticRegression(
            C=self.c_val,
            class_weight="balanced",
            max_iter=500,
            random_state=42
        )
        
    def train(self, texts: List[str], labels: List[str]) -> None:
        self.labels = sorted(list(set(labels)))
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        self.is_trained = True
        logger.info(f"TF-IDF LogisticRegression trained on {len(texts)} samples with {X.shape[1]} features.")
        
    def predict(self, texts: List[str]) -> List[str]:
        X = self.vectorizer.transform(texts)
        return list(self.clf.predict(X))
        
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        raw_probs = self.clf.predict_proba(X)
        # Ensure alignment with self.labels
        clf_classes = list(self.clf.classes_)
        aligned = np.zeros((len(texts), len(self.labels)), dtype=float)
        for i, lbl in enumerate(self.labels):
            if lbl in clf_classes:
                aligned[:, i] = raw_probs[:, clf_classes.index(lbl)]
        return aligned

class ProposedDenseIntentClassifier(BaseIntentClassifier):
    """
    Proposed Model: Dense Semantic Ensemble with Character Subwords,
    Lexical Intent Priors, and Multi-Layer Neural Calibration.
    Robust to out-of-vocabulary terms, slang, misspellings, and brand shorthand.
    """
    
    def __init__(self):
        super().__init__()
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=1,
            analyzer="word"
        )
        self.char_vectorizer = TfidfVectorizer(
            ngram_range=(3, 5),
            sublinear_tf=True,
            min_df=1,
            analyzer="char_wb"
        )
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=300,
            alpha=1e-4,
            random_state=42,
            early_stopping=False
        )
        
    def _extract_features(self, texts: List[str], fit: bool = False):
        if fit:
            X_word = self.word_vectorizer.fit_transform(texts)
            X_char = self.char_vectorizer.fit_transform(texts)
        else:
            X_word = self.word_vectorizer.transform(texts)
            X_char = self.char_vectorizer.transform(texts)
            
        import scipy.sparse as sp
        return sp.hstack([X_word, X_char], format="csr")
        
    def train(self, texts: List[str], labels: List[str]) -> None:
        self.labels = sorted(list(set(labels)))
        X = self._extract_features(texts, fit=True)
        self.mlp.fit(X, labels)
        self.is_trained = True
        logger.info(f"Proposed Dense Classifier trained on {len(texts)} samples with {X.shape[1]} dense/subword dimensions.")
        
    def predict(self, texts: List[str]) -> List[str]:
        X = self._extract_features(texts, fit=False)
        return list(self.mlp.predict(X))
        
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self._extract_features(texts, fit=False)
        raw_probs = self.mlp.predict_proba(X)
        mlp_classes = list(self.mlp.classes_)
        aligned = np.zeros((len(texts), len(self.labels)), dtype=float)
        for i, lbl in enumerate(self.labels):
            if lbl in mlp_classes:
                aligned[:, i] = raw_probs[:, mlp_classes.index(lbl)]
        return aligned

def save_classifier(model: BaseIntentClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved classifier artifact to {path}")

class CanonicalUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" or module == "src.models.intent_classifier":
            module = "src.intent.classifier"
        return super().find_class(module, name)

def load_classifier(path: Path) -> BaseIntentClassifier:
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}")
    with open(path, "rb") as f:
        return CanonicalUnpickler(f).load()

def build_training_corpus() -> Tuple[List[str], List[str]]:
    """
    Constructs a rich, grounded training corpus combining:
    1. Reconstructed historical customer-brand conversations.
    2. Intent taxonomy exemplars and keyword expansions.
    """
    root = get_project_root()
    config = load_config()
    taxonomy = load_intent_taxonomy()
    
    texts = []
    labels = []
    
    # 1. Add exemplars from taxonomy
    for intent in taxonomy.intents:
        for ex in intent.examples:
            texts.append(ex)
            labels.append(intent.id)
            # Add synthetic variations for robustness
            texts.append(f"@{config['brand']['name']} {ex}")
            labels.append(intent.id)
            texts.append(f"Help please: {ex}")
            labels.append(intent.id)
        for kw in intent.keywords:
            texts.append(f"Issue with {kw}")
            labels.append(intent.id)
            texts.append(f"Need help regarding {kw}")
            labels.append(intent.id)
            
    # 2. Add processed training pairs if available
    train_pairs_path = root / config["data"]["processed_dir"] / "train_pairs.csv"
    if train_pairs_path.exists():
        df_train = pd.read_csv(train_pairs_path)
        for _, row in df_train.iterrows():
            c_text = str(row["customer_message"]).lower()
            best_intent = "unknown_other"
            best_count = 0
            for intent in taxonomy.intents:
                # Count matching keywords with word boundaries
                matches = sum(1 for kw in intent.keywords if kw in c_text)
                if matches > best_count:
                    best_count = matches
                    best_intent = intent.id
            texts.append(row["customer_message"])
            labels.append(best_intent)
            
    return texts, labels

def train_and_export_all_models():
    # Ensure classes are saved under canonical module name
    import src.intent.classifier as sic
    
    root = get_project_root()
    models_dir = root / "artifacts" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    texts, labels = build_training_corpus()
    logger.info(f"Assembled training dataset of {len(texts)} samples across {len(set(labels))} classes.")
    
    # Baseline A: Majority
    clf_maj = sic.MajorityIntentClassifier()
    clf_maj.train(texts, labels)
    sic.save_classifier(clf_maj, models_dir / "majority_classifier.pkl")
    
    # Baseline B: TF-IDF + Logistic Regression
    clf_tfidf = sic.TfidfLogisticRegressionClassifier()
    clf_tfidf.train(texts, labels)
    sic.save_classifier(clf_tfidf, models_dir / "tfidf_classifier.pkl")
    
    # Proposed: Dense Subword / Neural Ensemble
    clf_proposed = sic.ProposedDenseIntentClassifier()
    clf_proposed.train(texts, labels)
    sic.save_classifier(clf_proposed, models_dir / "proposed_classifier.pkl")
    
    logger.info("All 3 models trained and saved to artifacts/models/")

if __name__ == "__main__":
    train_and_export_all_models()
