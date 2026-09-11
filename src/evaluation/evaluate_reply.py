import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from src.evaluation.llm_judge import LLMJudge
from src.pipeline.run_agent import TrustDeskAgent
from src.utils.io import get_project_root, save_json
from src.utils.logging import get_logger

logger = get_logger("evaluate_reply")

def evaluate_reply_quality(golden_df: pd.DataFrame) -> Dict[str, Any]:
    agent = TrustDeskAgent(classifier_type="proposed")
    judge = LLMJudge()
    
    logger.info(f"Evaluating grounded reply quality across {len(golden_df)} benchmark examples...")
    
    dimension_scores = {
        "groundedness": [],
        "helpfulness": [],
        "relevance": [],
        "resolution_quality": [],
        "brand_consistency": [],
        "safety": [],
        "overall_score": []
    }
    
    unsupported_claims_count = 0
    
    for idx, row in golden_df.iterrows():
        msg = row["message"]
        res = agent.process_message(msg)
        
        j_eval = judge.judge_reply(
            customer_message=msg,
            retrieved_evidence=res["evidence"],
            draft_reply=res["draft_reply"],
            escalation_decision=res["decision"]
        )
        
        dimension_scores["groundedness"].append(j_eval.groundedness)
        dimension_scores["helpfulness"].append(j_eval.helpfulness)
        dimension_scores["relevance"].append(j_eval.relevance)
        dimension_scores["resolution_quality"].append(j_eval.resolution_quality)
        dimension_scores["brand_consistency"].append(j_eval.brand_consistency)
        dimension_scores["safety"].append(j_eval.safety)
        dimension_scores["overall_score"].append(j_eval.overall_score)
        
        if "UNSUPPORTED_CLAIM" in res["reason_codes"]:
            unsupported_claims_count += 1
            
    summary = {
        k: round(float(np.mean(vals)), 2)
        for k, vals in dimension_scores.items()
    }
    summary["unsupported_claim_rate_pct"] = round(float(unsupported_claims_count / len(golden_df) * 100), 2)
    summary["total_evaluated"] = len(golden_df)
    
    root = get_project_root()
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    out_df = pd.DataFrame([
        {"Dimension (1-5 Scale)": k.replace("_", " ").title(), "Mean Score": v}
        for k, v in summary.items() if k not in ["total_evaluated", "unsupported_claim_rate_pct"]
    ])
    out_df.to_csv(reports_dir / "reply_results.csv", index=False)
    save_json(summary, reports_dir / "reply_metrics.json")
    
    print("\n" + "="*60)
    print("GROUNDED REPLY QUALITY EVALUATION (1-5 SCALE)")
    print("="*60)
    print(out_df.to_string(index=False))
    print("="*60)
    print(f"Unsupported Claims Detected: {summary['unsupported_claim_rate_pct']}%")
    print("="*60 + "\n")
    
    return summary

def main():
    root = get_project_root()
    golden_df = pd.read_csv(root / "data" / "golden" / "golden_set.csv")
    evaluate_reply_quality(golden_df)

if __name__ == "__main__":
    main()
