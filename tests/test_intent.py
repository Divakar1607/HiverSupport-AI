from pathlib import Path
import pytest
import numpy as np

from src.intent.classifier import (
    MajorityIntentClassifier,
    TfidfLogisticRegressionClassifier,
    ProposedDenseIntentClassifier,
    load_classifier
)
from src.intent.taxonomy import load_intent_taxonomy
from src.utils.io import get_project_root

def test_intent_taxonomy_loads():
    taxonomy = load_intent_taxonomy()
    assert taxonomy.brand != ""
    assert len(taxonomy.intents) >= 8
    intent_ids = [i.id for i in taxonomy.intents]
    assert "delivery_delay" in intent_ids
    assert "refund_request" in intent_ids
    assert "unknown_other" in intent_ids

def test_majority_classifier():
    clf = MajorityIntentClassifier()
    texts = ["a", "b", "c", "d"]
    labels = ["refund_request", "refund_request", "delivery_delay", "refund_request"]
    clf.train(texts, labels)
    preds = clf.predict(["x", "y"])
    assert preds == ["refund_request", "refund_request"]

def test_tfidf_classifier():
    clf = TfidfLogisticRegressionClassifier()
    texts = ["where is my package", "cancel my order", "i want a refund"]
    labels = ["delivery_delay", "cancellation_request", "refund_request"]
    clf.train(texts, labels)
    preds = clf.predict(["where is my late package"])
    assert len(preds) == 1
    assert preds[0] in labels

def test_proposed_classifier_prediction_shape():
    root = get_project_root()
    model_path = root / "artifacts" / "models" / "proposed_classifier.pkl"
    if model_path.exists():
        clf = load_classifier(model_path)
        res = clf.predict_single("Where is my package? Tracking says delivered", top_k=3)
        assert "intent" in res
        assert "confidence" in res
        assert 0.0 <= res["confidence"] <= 1.0
        assert len(res["top_k_predictions"]) == 3
