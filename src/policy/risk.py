import re
from typing import Dict, List, Tuple

# Compiled risk patterns
LEGAL_KEYWORDS = [
    r"\blawsuit\b", r"\blattorney\b", r"\blawyer\b", r"\blegal action\b",
    r"\bcourt\b", r"\bsue\b", r"\bpolice\b", r"\bconsumer court\b",
    r"\bbbb\b", r"\bbetter business bureau\b", r"\bftc\b", r"\bfederal trade commission\b"
]

FRAUD_KEYWORDS = [
    r"\bfraud\b", r"\bscam\b", r"\bstolen card\b", r"\bunauthorized charge\b",
    r"\bchargeback\b", r"\bhacked\b", r"\baccount takeover\b", r"\bcompromised\b"
]

HAZARD_KEYWORDS = [
    r"\bfire\b", r"\bsmoke\b", r"\bexploded\b", r"\bshattered glass cut\b",
    r"\bhospital\b", r"\bmedical emergency\b", r"\binjury\b", r"\bpoison\b"
]

ACCOUNT_SPECIFIC_KEYWORDS = [
    r"\bmy password\b", r"\bmy bank statement\b", r"\bssn\b", r"\bcvv\b",
    r"\brouting number\b", r"\bpin code\b"
]

REPEATED_FAILURE_KEYWORDS = [
    r"\bthird time\b", r"\b5 different (agents|people|reps)\b", r"\bhung up on me\b",
    r"\btransferred (me )?to 4\b", r"\btransferred (me )?to 5\b", r"\bwaiting (for )?3 weeks\b",
    r"\bmanager immediately\b", r"\bsupervisor\b"
]

def scan_for_risk_factors(text: str) -> Dict[str, bool]:
    """Scans incoming text for sensitive, high-risk, regulatory, or safety factors."""
    t = text.lower()
    
    is_legal = any(re.search(p, t) for p in LEGAL_KEYWORDS)
    is_fraud = any(re.search(p, t) for p in FRAUD_KEYWORDS)
    is_hazard = any(re.search(p, t) for p in HAZARD_KEYWORDS)
    is_account_specific = any(re.search(p, t) for p in ACCOUNT_SPECIFIC_KEYWORDS)
    is_repeated_failure = any(re.search(p, t) for p in REPEATED_FAILURE_KEYWORDS)
    
    is_high_risk = is_legal or is_fraud or is_hazard
    
    return {
        "is_high_risk": is_high_risk,
        "is_legal": is_legal,
        "is_fraud": is_fraud,
        "is_hazard": is_hazard,
        "is_account_specific": is_account_specific,
        "is_repeated_failure": is_repeated_failure
    }
