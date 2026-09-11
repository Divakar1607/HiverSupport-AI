import pytest

from src.policy.escalation import EscalationEngine
from src.policy.risk import scan_for_risk_factors

def test_risk_scanning():
    # Legal threat
    r1 = scan_for_risk_factors("My attorney is preparing a lawsuit against your company!")
    assert r1["is_high_risk"] is True
    assert r1["is_legal"] is True
    
    # Fraud / chargeback
    r2 = scan_for_risk_factors("I am reporting this unauthorized charge to my bank as fraud and filing a chargeback.")
    assert r2["is_high_risk"] is True
    assert r2["is_fraud"] is True
    
    # Benign inquiry
    r3 = scan_for_risk_factors("Where is my package? Tracking says delivered.")
    assert r3["is_high_risk"] is False

def test_escalation_engine_triggers():
    engine = EscalationEngine(confidence_threshold=0.65, evidence_threshold=0.60)
    
    # Low confidence trigger
    dec1 = engine.evaluate(
        message="Where is my stuff",
        predicted_intent="delivery_delay",
        intent_confidence=0.50,
        evidence_items=[{"similarity_score": 0.80}]
    )
    assert dec1.decision == "ESCALATE"
    assert "LOW_INTENT_CONFIDENCE" in dec1.reason_codes
    
    # Insufficient evidence trigger
    dec2 = engine.evaluate(
        message="Where is my stuff",
        predicted_intent="delivery_delay",
        intent_confidence=0.90,
        evidence_items=[{"similarity_score": 0.45}]
    )
    assert dec2.decision == "ESCALATE"
    assert "INSUFFICIENT_EVIDENCE" in dec2.reason_codes
    
    # Legal risk trigger
    dec3 = engine.evaluate(
        message="I will sue you in small claims court!",
        predicted_intent="complaint_escalation",
        intent_confidence=0.95,
        evidence_items=[{"similarity_score": 0.85}]
    )
    assert dec3.decision == "ESCALATE"
    assert "HIGH_RISK" in dec3.reason_codes
    
    # Clean auto-handle case
    dec4 = engine.evaluate(
        message="Where is my package? Tracking says delivered but nothing is here!",
        predicted_intent="delivery_delay",
        intent_confidence=0.92,
        evidence_items=[{"similarity_score": 0.78}]
    )
    assert dec4.decision == "AUTO_HANDLE"
    assert len(dec4.reason_codes) == 0
