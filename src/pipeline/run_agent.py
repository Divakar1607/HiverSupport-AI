import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.reply_generator import ReplyGenerator
from src.intent.classifier import load_classifier
from src.policy.escalation import EscalationEngine
from src.retrieval.retrieve import HybridEvidenceRetriever
from src.utils.io import get_project_root, load_config
from src.utils.logging import get_logger
from src.utils.privacy import anonymize_text

logger = get_logger("run_agent")

class TrustDeskAgent:
    """End-to-End Evidence-Grounded Customer Support Agent."""
    
    def __init__(self, classifier_type: str = "proposed"):
        self.root = get_project_root()
        self.config = load_config()
        
        # 1. Load Intent Classifier
        model_filename = {
            "majority": "majority_classifier.pkl",
            "tfidf": "tfidf_classifier.pkl",
            "proposed": "proposed_classifier.pkl"
        }.get(classifier_type, "proposed_classifier.pkl")
        
        model_path = self.root / "artifacts" / "models" / model_filename
        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}. Training intent classifiers first...")
            from src.intent.classifier import train_and_export_all_models
            train_and_export_all_models()
        self.classifier = load_classifier(model_path)
        
        # 2. Load Hybrid Retriever
        index_dir = self.root / self.config["retrieval"]["index_dir"]
        if not (index_dir / "hybrid_retriever.pkl").exists():
            logger.warning(f"Retriever index not found at {index_dir}. Building index now...")
            from scripts.build_index import main as run_build_index
            run_build_index()
        self.retriever = HybridEvidenceRetriever.load(index_dir)
        
        # 3. Escalation Engine
        self.escalation_engine = EscalationEngine(
            confidence_threshold=float(self.config["intent"].get("confidence_threshold", 0.65)),
            evidence_threshold=float(self.config["retrieval"].get("similarity_threshold", 0.60))
        )
        
        # 4. Reply Generator
        self.generator = ReplyGenerator()
        
    def process_message(self, raw_message: str) -> Dict[str, Any]:
        """
        Executes the full pipeline:
        Sanitize -> Classify Intent -> Retrieve Evidence -> Check Escalation -> Draft Grounded Reply
        """
        # Step 1: Sanitize and anonymize customer PII
        sanitized_msg = anonymize_text(raw_message)
        
        # Step 2: Intent Classification
        pred_res = self.classifier.predict_single(sanitized_msg, top_k=3)
        intent = pred_res["intent"]
        intent_confidence = pred_res["confidence"]
        
        # Step 3: Hybrid Retrieval of Historical Brand Evidence
        retrieved_evidence = self.retriever.retrieve(
            query=sanitized_msg,
            top_k=self.config["retrieval"].get("top_k", 3),
            intent_hint=intent
        )
        
        # Step 4: Evaluate Deterministic Escalation Policy
        escalation_eval = self.escalation_engine.evaluate(
            message=sanitized_msg,
            predicted_intent=intent,
            intent_confidence=intent_confidence,
            evidence_items=retrieved_evidence
        )
        
        # Step 5: Draft Grounded Reply (or escalate if policy dictates)
        if escalation_eval.decision == "AUTO_HANDLE":
            draft_res = self.generator.generate_reply(
                customer_message=sanitized_msg,
                intent=intent,
                intent_confidence=intent_confidence,
                evidence_items=retrieved_evidence
            )
            # Re-verify if generator identified unsupported claims
            if draft_res.should_escalate or len(draft_res.unsupported_claims) > 0:
                final_decision = "ESCALATE"
                reason_codes = ["UNSUPPORTED_CLAIM"]
                draft_reply = "Your request has been routed to a human specialist to ensure accurate account verification."
                reply_conf = 0.50
            else:
                final_decision = "AUTO_HANDLE"
                reason_codes = []
                draft_reply = draft_res.reply
                reply_conf = draft_res.confidence
        else:
            final_decision = "ESCALATE"
            reason_codes = escalation_eval.reason_codes
            draft_reply = (
                f"We've routed your inquiry to a specialized human support agent. "
                f"Reason: {escalation_eval.explanation}"
            )
            reply_conf = 0.50
            
        return {
            "message": raw_message,
            "intent": intent,
            "intent_confidence": round(intent_confidence, 4),
            "decision": final_decision,
            "reason_codes": reason_codes,
            "evidence": retrieved_evidence,
            "draft_reply": draft_reply,
            "reply_confidence": round(reply_conf, 4)
        }

def main():
    parser = argparse.ArgumentParser(description="TrustDesk AI Agent CLI")
    parser.add_argument("--message", type=str, required=True, help="Incoming customer inquiry text.")
    parser.add_argument("--model", type=str, default="proposed", choices=["majority", "tfidf", "proposed"])
    args = parser.parse_args()
    
    agent = TrustDeskAgent(classifier_type=args.model)
    result = agent.process_message(args.message)
    
    print("\n" + "="*60)
    print("TRUSTDESK AI — AGENT INFERENCE RESULT")
    print("="*60)
    print(json.dumps(result, indent=2))
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
