import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.evaluation.evaluate_intent import evaluate_intent_classifiers
from src.evaluation.evaluate_retrieval import evaluate_retrieval_system
from src.evaluation.evaluate_escalation import evaluate_escalation_system
from src.evaluation.evaluate_reply import evaluate_reply_quality
from src.evaluation.llm_judge import run_judge_agreement_experiment
from src.pipeline.run_agent import TrustDeskAgent
from src.utils.io import get_project_root, load_config, save_json
from src.utils.logging import get_logger

logger = get_logger("run_all_eval")

def generate_error_analysis_table(golden_df: pd.DataFrame, agent: TrustDeskAgent) -> pd.DataFrame:
    """
    Builds the detailed error analysis table required by Section 26:
    id, message, expected_intent, predicted_intent, confidence, decision,
    expected_action, retrieved_evidence, generated_reply, failure_type, notes
    """
    logger.info("Generating comprehensive error analysis table across golden benchmark...")
    rows = []
    
    for idx, r in golden_df.iterrows():
        msg = r["message"]
        res = agent.process_message(msg)
        
        expected_intent = r["intent"]
        pred_intent = res["intent"]
        conf = res["intent_confidence"]
        decision = res["decision"]
        expected_action = r["expected_action"]
        
        # Categorize failure type
        failure_type = "NONE"
        intent_mismatch = (expected_intent != pred_intent)
        action_mismatch = (expected_action != decision)
        
        if intent_mismatch and action_mismatch:
            failure_type = "INTENT_AND_ACTION_MISMATCH"
        elif intent_mismatch:
            failure_type = "MISCLASSIFIED_INTENT"
        elif action_mismatch:
            if decision == "ESCALATE" and expected_action == "AUTO_HANDLE":
                failure_type = "OVERCONSERVATIVE_ESCALATION"
            else:
                failure_type = "UNSAFE_AUTO_HANDLE"
                
        ev_summary = "None"
        if res["evidence"]:
            ev_summary = f"[{res['evidence'][0]['evidence_id']}] sim={res['evidence'][0]['similarity_score']:.2f}"
            
        rows.append({
            "id": r["id"],
            "message": msg,
            "expected_intent": expected_intent,
            "predicted_intent": pred_intent,
            "confidence": conf,
            "decision": decision,
            "expected_action": expected_action,
            "retrieved_evidence": ev_summary,
            "generated_reply": res["draft_reply"],
            "failure_type": failure_type,
            "notes": r["notes"]
        })
        
    error_df = pd.DataFrame(rows)
    root = get_project_root()
    err_path = root / "reports" / "error_analysis.csv"
    error_df.to_csv(err_path, index=False)
    logger.info(f"Saved error analysis to {err_path} ({len(error_df)} rows, {(error_df['failure_type'] != 'NONE').sum()} failure cases)")
    return error_df

def run_all_evaluations():
    root = get_project_root()
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    golden_path = root / "data" / "golden" / "golden_set.csv"
    if not golden_path.exists():
        from scripts.label_golden import build_golden_set
        build_golden_set()
    golden_df = pd.read_csv(golden_path)
    
    agent = TrustDeskAgent(classifier_type="proposed")
    
    print("\n" + "#"*70)
    print("TRUSTDESK AI — EXECUTING FULL EVALUATION BENCHMARK SUITE")
    print("#"*70 + "\n")
    
    # 1. Intent Benchmarks
    intent_metrics = evaluate_intent_classifiers(golden_df)
    
    # 2. Retrieval Benchmarks
    retrieval_metrics = evaluate_retrieval_system(golden_df)
    
    # 3. Escalation & Coverage Benchmarks
    escalation_metrics = evaluate_escalation_system(golden_df)
    
    # 4. Reply Quality Benchmarks
    reply_metrics = evaluate_reply_quality(golden_df)
    
    # 5. Judge vs Human Agreement
    judge_metrics = run_judge_agreement_experiment()
    
    # 6. Comprehensive Error Analysis Table
    generate_error_analysis_table(golden_df, agent)
    
    # 7. Final Consolidated Trust Metrics Table (Section 22)
    proposed_intent = intent_metrics["Proposed (Dense Subword)"]
    consolidated_metrics = [
        {"Metric": "Intent Macro F1 (Proposed)", "Value": proposed_intent["macro_f1"], "Target / Benchmark": ">= 0.70"},
        {"Metric": "Intent Accuracy (Proposed)", "Value": proposed_intent["accuracy"], "Target / Benchmark": ">= 0.70"},
        {"Metric": "Retrieval Recall@5", "Value": retrieval_metrics["recall@5"], "Target / Benchmark": ">= 0.60"},
        {"Metric": "Retrieval MRR", "Value": retrieval_metrics["mrr"], "Target / Benchmark": ">= 0.55"},
        {"Metric": "Mean Groundedness (1-5)", "Value": reply_metrics["groundedness"], "Target / Benchmark": ">= 4.0"},
        {"Metric": "Mean Helpfulness (1-5)", "Value": reply_metrics["helpfulness"], "Target / Benchmark": ">= 4.0"},
        {"Metric": "Auto-Handle Precision", "Value": f"{escalation_metrics['auto_handle_precision']*100:.1f}%", "Target / Benchmark": ">= 85%"},
        {"Metric": "Auto-Handled Error Rate", "Value": f"{escalation_metrics['auto_handled_error_rate']*100:.1f}%", "Target / Benchmark": "<= 15%"},
        {"Metric": "Overall Escalation Rate", "Value": f"{escalation_metrics['escalation_rate']*100:.1f}%", "Target / Benchmark": "Inspectable"},
        {"Metric": "Judge/Human +/-1 Agreement", "Value": f"{judge_metrics['plus_minus_one_agreement']*100:.1f}%", "Target / Benchmark": ">= 90%"}
    ]
    
    eval_df = pd.DataFrame(consolidated_metrics)
    eval_csv_path = reports_dir / "evaluation_results.csv"
    eval_df.to_csv(eval_csv_path, index=False)
    
    print("\n" + "="*70)
    print("CONSOLIDATED TRUST & EVALUATION DASHBOARD")
    print("="*70)
    print(eval_df.to_string(index=False))
    print("="*70 + "\n")

if __name__ == "__main__":
    run_all_evaluations()
