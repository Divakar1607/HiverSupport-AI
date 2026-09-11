from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.policy.risk import scan_for_risk_factors
from src.utils.io import load_config
from src.utils.logging import get_logger

logger = get_logger("escalation_policy")

class EscalationDecision(BaseModel):
    decision: str  # AUTO_HANDLE or ESCALATE
    reason_codes: List[str] = Field(default_factory=list)
    explanation: str
    confidence: float
    evidence_score: float

class EscalationEngine:
    """
    Transparent, deterministic, inspectable escalation policy engine.
    Never relies on opaque LLM judgment to decide whether human review is needed.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.65,
        evidence_threshold: float = 0.60
    ):
        self.confidence_threshold = confidence_threshold
        self.evidence_threshold = evidence_threshold
        
    def evaluate(
        self,
        message: str,
        predicted_intent: str,
        intent_confidence: float,
        evidence_items: List[Dict[str, Any]],
        unsupported_claims: Optional[List[str]] = None
    ) -> EscalationDecision:
        """
        Evaluates signals deterministically and produces an inspectable decision.
        """
        reason_codes = []
        explanations = []
        
        # 1. Check for sensitive/high-risk language (legal threats, fraud, injury, safety)
        risk_profile = scan_for_risk_factors(message)
        if risk_profile["is_high_risk"]:
            reason_codes.append("HIGH_RISK")
            if risk_profile["is_legal"]:
                explanations.append("Detected legal threat or regulatory complaint language (attorney/lawsuit/court/FTC/BBB).")
            elif risk_profile["is_fraud"]:
                explanations.append("Detected potential financial fraud, chargeback warning, or compromised account indicator.")
            elif risk_profile["is_hazard"]:
                explanations.append("Detected physical safety hazard, battery risk, or medical urgency requiring immediate human handling.")
                
        # 2. Check for repeated agent failure or supervisor demand
        if risk_profile["is_repeated_failure"]:
            reason_codes.append("REPEATED_FAILURE")
            explanations.append("Customer reports repeated unresolved interactions, agent hang-up, or explicitly demands supervisory intervention.")
            
        # 3. Check for account-specific credentials or private data
        if risk_profile["is_account_specific"]:
            reason_codes.append("ACCOUNT_SPECIFIC")
            explanations.append("Inquiry involves sensitive account credentials, banking tokens, or PII requiring secure authenticated agent workflow.")
            
        # 4. Check for low intent confidence
        if intent_confidence < self.confidence_threshold:
            reason_codes.append("LOW_INTENT_CONFIDENCE")
            explanations.append(f"Classifier intent confidence ({intent_confidence:.2f}) is below policy threshold ({self.confidence_threshold:.2f}).")
            
        # 5. Check for unknown/unclassified intent
        if predicted_intent == "unknown_other":
            reason_codes.append("UNKNOWN_INTENT")
            explanations.append("Customer issue could not be mapped to any established brand support intent.")
            
        # 6. Check retrieval evidence sufficiency
        max_evidence_score = 0.0
        if evidence_items:
            max_evidence_score = max(float(e.get("similarity_score", 0.0)) for e in evidence_items)
            
        if not evidence_items or max_evidence_score < self.evidence_threshold:
            reason_codes.append("INSUFFICIENT_EVIDENCE")
            explanations.append(f"No sufficiently verified historical brand resolution retrieved (top similarity {max_evidence_score:.2f} < threshold {self.evidence_threshold:.2f}).")
            
        # 7. Check for unsupported generated claims
        if unsupported_claims and len(unsupported_claims) > 0:
            reason_codes.append("UNSUPPORTED_CLAIM")
            explanations.append(f"Draft generation proposed claims not grounded in verified evidence: {', '.join(unsupported_claims)}")
            
        # Deterministic Decision Synthesis
        if reason_codes:
            decision = "ESCALATE"
            full_explanation = " | ".join(explanations)
        else:
            decision = "AUTO_HANDLE"
            full_explanation = "Verified historical brand evidence strongly supports autonomous handling with calibrated confidence."
            
        return EscalationDecision(
            decision=decision,
            reason_codes=reason_codes,
            explanation=full_explanation,
            confidence=round(intent_confidence, 4),
            evidence_score=round(max_evidence_score, 4)
        )
