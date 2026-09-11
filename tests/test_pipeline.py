import pytest
from fastapi.testclient import TestClient

from app.backend.main import app
from src.pipeline.run_agent import TrustDeskAgent

client = TestClient(app)

def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert "AmazonHelp" in data["brand"]

def test_api_predict():
    res = client.post("/api/predict", json={"message": "My refund has not arrived yet in my account"})
    assert res.status_code == 200
    data = res.json()
    assert "intent" in data
    assert "confidence" in data
    assert data["decision"] in ["AUTO_HANDLE", "ESCALATE"]
    assert "draft_reply" in data
    assert len(data["evidence"]) > 0

def test_api_intents():
    res = client.get("/api/intents")
    assert res.status_code == 200
    data = res.json()
    assert len(data["intents"]) >= 8

def test_pipeline_unsupported_claim_detection():
    agent = TrustDeskAgent(classifier_type="proposed")
    # Benign message
    res = agent.process_message("Where is my package? Tracking says delivered but nothing is here!")
    assert res["decision"] in ["AUTO_HANDLE", "ESCALATE"]
    assert isinstance(res["reason_codes"], list)
    assert res["reply_confidence"] >= 0.0
