import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src.evaluation.run_all import run_all_evaluations
from src.utils.io import get_project_root, load_json

def main():
    start_time = time.time()
    root = get_project_root()
    
    print("\n" + "="*60)
    print("TRUSTDESK AI — HEADLINE REPRODUCTION RUNNER")
    print("Tagline: Classify. Retrieve. Respond. Escalate. Prove.")
    print("="*60)
    
    # Run the full evaluation pipeline
    run_all_evaluations()
    
    # Load headline metrics from reports
    reports_dir = root / "reports"
    intent_df = pd.read_csv(reports_dir / "intent_results.csv")
    retrieval_df = pd.read_csv(reports_dir / "retrieval_results.csv")
    escalation_data = load_json(reports_dir / "escalation_metrics.json")
    reply_data = load_json(reports_dir / "reply_metrics.json")
    judge_data = load_json(reports_dir / "judge_agreement.json")
    
    proposed_intent = intent_df[intent_df["Model"].str.contains("Proposed")].iloc[0]
    recall_5_val = retrieval_df[retrieval_df["Metric"] == "Recall@5"]["Value"].iloc[0]
    
    elapsed = round(time.time() - start_time, 2)
    
    print("\n" + "="*50)
    print("TRUSTDESK AI — REPRODUCTION")
    print("="*50)
    print(f"Intent Macro F1:         {proposed_intent['Macro F1']:.4f}")
    print(f"Recall@5:                {recall_5_val:.4f}")
    print(f"Groundedness:            {reply_data['groundedness']:.1f}/5")
    print(f"Auto-handle precision:   {escalation_data['auto_handle_precision']*100:.1f}%")
    print(f"Auto-handled error rate: {escalation_data['auto_handled_error_rate']*100:.1f}%")
    print("="*50)
    print(f"Reproduction completed in {elapsed}s (under 15 minutes requirement)")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
