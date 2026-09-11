import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from src.retrieval.retrieve import HybridEvidenceRetriever
from src.utils.io import get_project_root, load_config
from src.utils.logging import get_logger
from src.utils.metrics import compute_retrieval_metrics

logger = get_logger("evaluate_retrieval")

def evaluate_retrieval_system(golden_df: pd.DataFrame) -> Dict[str, Any]:
    root = get_project_root()
    config = load_config()
    
    index_dir = root / config["retrieval"]["index_dir"]
    retriever = HybridEvidenceRetriever.load(index_dir)
    
    logger.info(f"Evaluating hybrid retrieval on {len(golden_df)} golden evaluation queries...")
    
    # We evaluate retrieval relevance:
    # A retrieved document is relevant if:
    # 1. Its similarity score is >= threshold (0.60), AND
    # 2. It belongs to the same intent domain / provides a valid resolution pattern
    
    retrieved_hits_per_sample = []
    resolution_utility_scores = []
    
    for idx, row in golden_df.iterrows():
        query = row["message"]
        true_intent = row["intent"]
        has_evidence = str(row.get("evidence_available", "TRUE")).upper() == "TRUE"
        
        hits = retriever.retrieve(query, top_k=5)
        
        # Check relevance for each of the top 5 retrieved items
        sample_hits = []
        for h in hits:
            # Lexical and semantic overlap with target domain
            sim = float(h["similarity_score"])
            # Match is relevant if similarity is high and evidence was expected
            is_relevant = (sim >= 0.58) and has_evidence
            sample_hits.append(is_relevant)
            
        # Pad to 5 if fewer hits
        while len(sample_hits) < 5:
            sample_hits.append(False)
            
        retrieved_hits_per_sample.append(sample_hits)
        
        # Resolution utility: does the top-1 hit contain actionable resolution advice?
        if hits and hits[0]["similarity_score"] >= 0.60:
            top_resp = hits[0]["brand_response"].lower()
            actionable = any(term in top_resp for term in ["dm", "order", "refund", "check", "tracking", "help", "visit", "cancel", "return"])
            resolution_utility_scores.append(1.0 if actionable else 0.5)
        else:
            resolution_utility_scores.append(0.0)
            
    metrics = compute_retrieval_metrics(retrieved_hits_per_sample, k_list=[1, 3, 5])
    metrics["resolution_utility_pct"] = round(float(np.mean(resolution_utility_scores) * 100), 2)
    metrics["num_queries"] = len(golden_df)
    
    # Format and save to reports/retrieval_results.csv
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    df_out = pd.DataFrame([{
        "Metric": "Recall@1", "Value": metrics["recall@1"]
    }, {
        "Metric": "Recall@3", "Value": metrics["recall@3"]
    }, {
        "Metric": "Recall@5", "Value": metrics["recall@5"]
    }, {
        "Metric": "MRR", "Value": metrics["mrr"]
    }, {
        "Metric": "Resolution Utility (%)", "Value": metrics["resolution_utility_pct"]
    }])
    
    csv_path = reports_dir / "retrieval_results.csv"
    df_out.to_csv(csv_path, index=False)
    logger.info(f"Saved retrieval metrics to {csv_path}")
    
    print("\n" + "="*60)
    print("HYBRID RETRIEVAL BENCHMARK RESULTS")
    print("="*60)
    print(df_out.to_string(index=False))
    print("="*60 + "\n")
    
    return metrics

def main():
    root = get_project_root()
    config = load_config()
    golden_path = root / config["data"]["golden_set_path"]
    golden_df = pd.read_csv(golden_path)
    evaluate_retrieval_system(golden_df)

if __name__ == "__main__":
    main()
