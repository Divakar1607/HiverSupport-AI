import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from src.intent.classifier import load_classifier, train_and_export_all_models
from src.utils.io import get_project_root, load_config, save_json
from src.utils.logging import get_logger
from src.utils.metrics import compute_classification_metrics

logger = get_logger("evaluate_intent")

def evaluate_intent_classifiers(golden_df: pd.DataFrame) -> Dict[str, Any]:
    root = get_project_root()
    models_dir = root / "artifacts" / "models"
    
    if not (models_dir / "proposed_classifier.pkl").exists():
        logger.warning("Models not found. Training all models...")
        train_and_export_all_models()
        
    y_true = list(golden_df["intent"])
    texts = list(golden_df["message"])
    
    models = {
        "Majority": load_classifier(models_dir / "majority_classifier.pkl"),
        "TF-IDF + LogReg": load_classifier(models_dir / "tfidf_classifier.pkl"),
        "Proposed (Dense Subword)": load_classifier(models_dir / "proposed_classifier.pkl")
    }
    
    results = {}
    summary_rows = []
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name} on {len(texts)} golden test samples...")
        y_pred = model.predict(texts)
        metrics = compute_classification_metrics(y_true, y_pred)
        results[model_name] = metrics
        
        summary_rows.append({
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Macro F1": metrics["macro_f1"],
            "Weighted F1": metrics["weighted_f1"],
            "Macro Precision": metrics["macro_precision"],
            "Macro Recall": metrics["macro_recall"]
        })
        
    summary_df = pd.DataFrame(summary_rows)
    
    # Save to reports
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    summary_csv_path = reports_dir / "intent_results.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info(f"Saved intent comparison results to {summary_csv_path}")
    
    # Save detailed metrics JSON
    save_json(results, reports_dir / "intent_detailed_metrics.json")
    
    print("\n" + "="*60)
    print("INTENT CLASSIFIER BENCHMARK RESULTS")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60 + "\n")
    
    return results

def main():
    root = get_project_root()
    config = load_config()
    golden_path = root / config["data"]["golden_set_path"]
    
    if not golden_path.exists():
        logger.warning(f"Golden dataset not found at {golden_path}. Building now...")
        from scripts.label_golden import build_golden_set
        build_golden_set()
        
    golden_df = pd.read_csv(golden_path)
    evaluate_intent_classifiers(golden_df)

if __name__ == "__main__":
    main()
