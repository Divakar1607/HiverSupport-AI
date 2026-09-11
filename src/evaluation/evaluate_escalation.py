import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from src.pipeline.run_agent import TrustDeskAgent
from src.utils.io import get_project_root, load_config, save_json
from src.utils.logging import get_logger
from src.utils.metrics import compute_escalation_metrics

logger = get_logger("evaluate_escalation")

def evaluate_escalation_system(golden_df: pd.DataFrame) -> Dict[str, Any]:
    agent = TrustDeskAgent(classifier_type="proposed")
    
    logger.info(f"Evaluating escalation policy on {len(golden_df)} golden samples...")
    
    y_true_action = list(golden_df["expected_action"])
    y_true_intent = list(golden_df["intent"])
    
    y_pred_action = []
    y_pred_intent = []
    agent_outputs = []
    
    for idx, row in golden_df.iterrows():
        res = agent.process_message(row["message"])
        y_pred_action.append(res["decision"])
        y_pred_intent.append(res["intent"])
        agent_outputs.append(res)
        
    metrics = compute_escalation_metrics(
        y_true_action=y_true_action,
        y_pred_action=y_pred_action,
        y_true_intent=y_true_intent,
        y_pred_intent=y_pred_intent
    )
    
    root = get_project_root()
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Save escalation metrics JSON
    save_json(metrics, reports_dir / "escalation_metrics.json")
    
    # Coverage vs Quality Curve Analysis:
    # Vary the auto-handling confidence threshold from 0.40 to 0.90
    thresholds = [0.40, 0.50, 0.60, 0.65, 0.70, 0.80, 0.85]
    curve_rows = []
    
    for th in thresholds:
        simulated_actions = []
        errors = 0
        auto_count = 0
        
        for i, out in enumerate(agent_outputs):
            # Check if reason codes exist other than LOW_INTENT_CONFIDENCE
            reasons = [r for r in out["reason_codes"] if r != "LOW_INTENT_CONFIDENCE"]
            conf = out["intent_confidence"]
            
            # Simulated decision with threshold th
            if conf >= th and not reasons and out["evidence"]:
                sim_action = "AUTO_HANDLE"
                auto_count += 1
                # Check error
                is_wrong_action = (y_true_action[i] != "AUTO_HANDLE")
                is_wrong_intent = (y_pred_intent[i] != y_true_intent[i])
                if is_wrong_action or is_wrong_intent:
                    errors += 1
            else:
                sim_action = "ESCALATE"
            simulated_actions.append(sim_action)
            
        coverage_pct = round(float(auto_count / len(golden_df) * 100), 1)
        err_rate_pct = round(float((errors / auto_count * 100) if auto_count > 0 else 0.0), 1)
        auto_accuracy_pct = round(100.0 - err_rate_pct, 1)
        
        curve_rows.append({
            "Confidence Threshold": th,
            "Auto-Handle Coverage (%)": coverage_pct,
            "Auto-Handle Accuracy (%)": auto_accuracy_pct,
            "Auto-Handled Error Rate (%)": err_rate_pct
        })
        
    curve_df = pd.DataFrame(curve_rows)
    curve_csv_path = reports_dir / "coverage_vs_quality.csv"
    curve_df.to_csv(curve_csv_path, index=False)
    logger.info(f"Saved Coverage vs Quality trade-off table to {curve_csv_path}")
    
    print("\n" + "="*60)
    print("ESCALATION POLICY PERFORMANCE")
    print("="*60)
    print(f"Total Test Samples:        {metrics['total_samples']}")
    print(f"Auto-Handle Rate:          {metrics['auto_handle_rate']*100:.1f}% ({metrics['auto_handled_count']} samples)")
    print(f"Escalation Rate:           {metrics['escalation_rate']*100:.1f}% ({metrics['escalated_count']} samples)")
    print(f"Auto-Handle Precision:     {metrics['auto_handle_precision']*100:.1f}%")
    print(f"Auto-Handle Recall:        {metrics['auto_handle_recall']*100:.1f}%")
    print(f"Escalate Precision:        {metrics['escalate_precision']*100:.1f}%")
    print(f"Escalate Recall:           {metrics['escalate_recall']*100:.1f}%")
    print(f"CRITICAL Auto-Handled Error Rate: {metrics['auto_handled_error_rate']*100:.1f}%")
    print("="*60)
    print("\nCOVERAGE VS QUALITY TRADEOFF TABLE")
    print(curve_df.to_string(index=False))
    print("="*60 + "\n")
    
    return metrics

def main():
    root = get_project_root()
    golden_df = pd.read_csv(root / "data" / "golden" / "golden_set.csv")
    evaluate_escalation_system(golden_df)

if __name__ == "__main__":
    main()
