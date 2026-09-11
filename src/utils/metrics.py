from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from scipy.stats import pearsonr, spearmanr

def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = None
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics with emphasis on Macro F1."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    
    unique_labels = labels if labels else sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels).tolist()
    report = classification_report(y_true, y_pred, labels=unique_labels, output_dict=True, zero_division=0)
    
    per_intent_f1 = {}
    for lbl in unique_labels:
        if lbl in report:
            per_intent_f1[lbl] = {
                "precision": round(report[lbl]["precision"], 4),
                "recall": round(report[lbl]["recall"], 4),
                "f1": round(report[lbl]["f1-score"], 4),
                "support": report[lbl]["support"]
            }
            
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "labels": unique_labels,
        "confusion_matrix": cm,
        "per_intent": per_intent_f1
    }

def compute_retrieval_metrics(
    retrieved_intent_matches: List[List[bool]],
    k_list: List[int] = [1, 3, 5]
) -> Dict[str, float]:
    """
    Computes Recall@K and MRR given binary relevance indicators for retrieved ranks.
    retrieved_intent_matches: list of boolean lists, e.g. [[True, False, False], [False, True, False]]
    """
    if not retrieved_intent_matches:
        return {f"recall@{k}": 0.0 for k in k_list} | {"mrr": 0.0}
        
    metrics = {}
    for k in k_list:
        recalls = []
        for hits in retrieved_intent_matches:
            top_k_hits = hits[:k]
            recalls.append(1.0 if any(top_k_hits) else 0.0)
        metrics[f"recall@{k}"] = round(float(np.mean(recalls)), 4)
        
    # MRR (Mean Reciprocal Rank)
    reciprocal_ranks = []
    for hits in retrieved_intent_matches:
        rr = 0.0
        for rank, is_hit in enumerate(hits, start=1):
            if is_hit:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    metrics["mrr"] = round(float(np.mean(reciprocal_ranks)), 4)
    
    return metrics

def compute_escalation_metrics(
    y_true_action: List[str],  # AUTO_HANDLE or ESCALATE
    y_pred_action: List[str],  # AUTO_HANDLE or ESCALATE
    y_true_intent: List[str] = None,
    y_pred_intent: List[str] = None
) -> Dict[str, Any]:
    """
    Computes policy decision metrics and the critical Auto-Handled Error Rate:
    Auto-Handled Error Rate = % of items model chose to AUTO_HANDLE that were actually ESCALATE or incorrect intent.
    """
    total = len(y_true_action)
    if total == 0:
        return {}
        
    auto_handle_true = [a == "AUTO_HANDLE" for a in y_true_action]
    auto_handle_pred = [a == "AUTO_HANDLE" for a in y_pred_action]
    
    auto_handled_count = sum(auto_handle_pred)
    escalated_count = total - auto_handled_count
    
    auto_handle_rate = round(auto_handled_count / total, 4)
    escalation_rate = round(escalated_count / total, 4)
    
    # Precision and recall for AUTO_HANDLE
    auto_tp = sum(1 for yt, yp in zip(y_true_action, y_pred_action) if yt == "AUTO_HANDLE" and yp == "AUTO_HANDLE")
    auto_precision = round(auto_tp / auto_handled_count, 4) if auto_handled_count > 0 else 0.0
    auto_recall = round(auto_tp / sum(auto_handle_true), 4) if sum(auto_handle_true) > 0 else 0.0
    
    # Precision and recall for ESCALATE
    esc_tp = sum(1 for yt, yp in zip(y_true_action, y_pred_action) if yt == "ESCALATE" and yp == "ESCALATE")
    esc_precision = round(esc_tp / escalated_count, 4) if escalated_count > 0 else 0.0
    esc_recall = round(esc_tp / (total - sum(auto_handle_true)), 4) if (total - sum(auto_handle_true)) > 0 else 0.0
    
    # Auto-handled error rate:
    # An auto-handled item is an error if either its true action was ESCALATE, or its intent prediction was wrong
    errors_in_auto = 0
    if auto_handled_count > 0:
        for i in range(total):
            if y_pred_action[i] == "AUTO_HANDLE":
                action_wrong = (y_true_action[i] != "AUTO_HANDLE")
                intent_wrong = False
                if y_true_intent and y_pred_intent:
                    intent_wrong = (y_true_intent[i] != y_pred_intent[i])
                if action_wrong or intent_wrong:
                    errors_in_auto += 1
        auto_handled_error_rate = round(errors_in_auto / auto_handled_count, 4)
    else:
        auto_handled_error_rate = 0.0
        
    return {
        "total_samples": total,
        "auto_handled_count": auto_handled_count,
        "escalated_count": escalated_count,
        "auto_handle_rate": auto_handle_rate,
        "escalation_rate": escalation_rate,
        "auto_handle_precision": auto_precision,
        "auto_handle_recall": auto_recall,
        "escalate_precision": esc_precision,
        "escalate_recall": esc_recall,
        "auto_handled_error_rate": auto_handled_error_rate
    }

def compute_judge_human_agreement(
    human_scores: List[float],
    llm_scores: List[float]
) -> Dict[str, Any]:
    """Computes exact agreement, +/-1 agreement, Pearson & Spearman correlation."""
    n = len(human_scores)
    if n == 0:
        return {}
        
    diffs = [abs(h - l) for h, l in zip(human_scores, llm_scores)]
    exact_agree = sum(1 for d in diffs if d < 0.01) / n
    plus_minus_one = sum(1 for d in diffs if d <= 1.01) / n
    
    # Pearson and Spearman
    try:
        pearson_val, _ = pearsonr(human_scores, llm_scores)
        spearman_val, _ = spearmanr(human_scores, llm_scores)
        pearson_corr = round(float(pearson_val), 4) if not np.isnan(pearson_val) else 0.0
        spearman_corr = round(float(spearman_val), 4) if not np.isnan(spearman_val) else 0.0
    except Exception:
        pearson_corr = 0.0
        spearman_corr = 0.0
        
    return {
        "num_samples": n,
        "exact_agreement": round(exact_agree, 4),
        "plus_minus_one_agreement": round(plus_minus_one, 4),
        "pearson_correlation": pearson_corr,
        "spearman_correlation": spearman_corr
    }
