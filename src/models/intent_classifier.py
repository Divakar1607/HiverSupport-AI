"""
Compatibility wrapper exposing the intent classifiers under src.models.intent_classifier
as required by the project specification.
"""
from src.intent.classifier import (
    BaseIntentClassifier,
    MajorityIntentClassifier,
    TfidfLogisticRegressionClassifier,
    ProposedDenseIntentClassifier,
    load_classifier,
    save_classifier,
    build_training_corpus,
    train_and_export_all_models
)

__all__ = [
    "BaseIntentClassifier",
    "MajorityIntentClassifier",
    "TfidfLogisticRegressionClassifier",
    "ProposedDenseIntentClassifier",
    "load_classifier",
    "save_classifier",
    "build_training_corpus",
    "train_and_export_all_models"
]

if __name__ == "__main__":
    train_and_export_all_models()
