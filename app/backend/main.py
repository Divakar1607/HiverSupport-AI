import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from src.intent.taxonomy import load_intent_taxonomy
from src.pipeline.run_agent import TrustDeskAgent
from src.utils.io import get_project_root, load_config, load_json
from src.utils.logging import get_logger

logger = get_logger("fastapi_backend")

app = FastAPI(
    title="TrustDesk AI — Customer Support Agent API",
    description="Evidence-Grounded Customer Support Agent (Hiver SDE Intern Evaluation)",
    version="1.0.0"
)

# Enable CORS for React Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global lazy agent instance
_agent: Optional[TrustDeskAgent] = None

def get_agent() -> TrustDeskAgent:
    global _agent
    if _agent is None:
        _agent = TrustDeskAgent(classifier_type="proposed")
    return _agent

# Pydantic Request/Response Models
class PredictRequest(BaseModel):
    message: str
    model: Optional[str] = "proposed"

class PredictResponse(BaseModel):
    message: str
    intent: str
    confidence: float
    decision: str
    reason_codes: List[str]
    evidence: List[Dict[str, Any]]
    draft_reply: str
    reply_confidence: float

@app.get("/api/health")
def health_check():
    config = load_config()
    return {
        "status": "HEALTHY",
        "brand": config["brand"]["name"],
        "tagline": "Classify. Retrieve. Respond. Escalate. Prove.",
        "llm_provider": config["generation"].get("provider", "LOCAL"),
        "version": "1.0.0"
    }

@app.post("/api/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Customer message must not be empty.")
    agent = get_agent()
    result = agent.process_message(req.message)
    return PredictResponse(
        message=result["message"],
        intent=result["intent"],
        confidence=result["intent_confidence"],
        decision=result["decision"],
        reason_codes=result["reason_codes"],
        evidence=result["evidence"],
        draft_reply=result["draft_reply"],
        reply_confidence=result["reply_confidence"]
    )

@app.get("/api/intents")
def get_intents_endpoint():
    try:
        taxonomy = load_intent_taxonomy()
        return taxonomy.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/evidence/{evidence_id}")
def get_evidence_endpoint(evidence_id: str):
    agent = get_agent()
    for doc in agent.retriever.evidence_documents:
        if doc.get("evidence_id") == evidence_id:
            return doc
    raise HTTPException(status_code=404, detail=f"Evidence ID '{evidence_id}' not found.")

@app.get("/api/metrics")
def get_metrics_endpoint():
    root = get_project_root()
    reports_dir = root / "reports"
    
    intent_csv = reports_dir / "intent_results.csv"
    retrieval_csv = reports_dir / "retrieval_results.csv"
    coverage_csv = reports_dir / "coverage_vs_quality.csv"
    eval_csv = reports_dir / "evaluation_results.csv"
    escalation_json = reports_dir / "escalation_metrics.json"
    reply_json = reports_dir / "reply_metrics.json"
    judge_json = reports_dir / "judge_agreement.json"
    detailed_intent_json = reports_dir / "intent_detailed_metrics.json"
    
    if not eval_csv.exists():
        raise HTTPException(status_code=404, detail="Evaluation reports not found. Run 'python scripts/reproduce_results.py' first.")
        
    return {
        "summary": pd.read_csv(eval_csv).to_dict(orient="records"),
        "intent_models": pd.read_csv(intent_csv).to_dict(orient="records") if intent_csv.exists() else [],
        "retrieval": pd.read_csv(retrieval_csv).to_dict(orient="records") if retrieval_csv.exists() else [],
        "coverage_vs_quality": pd.read_csv(coverage_csv).to_dict(orient="records") if coverage_csv.exists() else [],
        "escalation": load_json(escalation_json) if escalation_json.exists() else {},
        "reply": load_json(reply_json) if reply_json.exists() else {},
        "judge_agreement": load_json(judge_json) if judge_json.exists() else {},
        "intent_details": load_json(detailed_intent_json) if detailed_intent_json.exists() else {}
    }

@app.post("/api/evaluate")
def run_evaluation_endpoint():
    from src.evaluation.run_all import run_all_evaluations
    run_all_evaluations()
    return {"status": "SUCCESS", "message": "Evaluation suite executed and reports refreshed."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.backend.main:app", host="127.0.0.1", port=8000, reload=True)
