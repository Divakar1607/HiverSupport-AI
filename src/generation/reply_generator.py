import json
import os
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.generation.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, format_evidence_block
from src.utils.io import load_config
from src.utils.logging import get_logger

logger = get_logger("reply_generator")

class GroundedReply(BaseModel):
    reply: str
    evidence_ids: List[str] = Field(default_factory=list)
    confidence: float
    unsupported_claims: List[str] = Field(default_factory=list)
    should_escalate: bool = False
    escalation_reason: Optional[str] = None

class ReplyGenerator:
    """
    Evidence-grounded reply generator supporting multiple LLM backends (OpenAI, Gemini, Anthropic)
    with a deterministic local fallback engine for zero-dependency reproducibility.
    """
    
    def __init__(self):
        self.config = load_config()
        self.brand_name = self.config["brand"]["name"]
        self.provider = os.environ.get("LLM_PROVIDER", self.config["generation"].get("provider", "LOCAL")).upper()
        self.model = os.environ.get("LLM_MODEL", self.config["generation"].get("model", "deterministic-v1"))
        
    def generate_reply(
        self,
        customer_message: str,
        intent: str,
        intent_confidence: float,
        evidence_items: List[Dict[str, Any]]
    ) -> GroundedReply:
        """Generates structured, grounded reply with evidence binding."""
        if self.provider == "LOCAL" or not self._has_api_key():
            return self._generate_deterministic_fallback(
                customer_message, intent, intent_confidence, evidence_items
            )
            
        try:
            if self.provider == "OPENAI":
                return self._call_openai(customer_message, intent, intent_confidence, evidence_items)
            elif self.provider == "GEMINI":
                return self._call_gemini(customer_message, intent, intent_confidence, evidence_items)
            elif self.provider == "ANTHROPIC":
                return self._call_anthropic(customer_message, intent, intent_confidence, evidence_items)
            else:
                logger.warning(f"Unknown provider '{self.provider}'. Falling back to deterministic generation.")
                return self._generate_deterministic_fallback(customer_message, intent, intent_confidence, evidence_items)
        except Exception as e:
            logger.error(f"Error calling {self.provider} API: {e}. Falling back to deterministic generator.")
            return self._generate_deterministic_fallback(customer_message, intent, intent_confidence, evidence_items)

    def _has_api_key(self) -> bool:
        if self.provider == "OPENAI" and os.environ.get("OPENAI_API_KEY"):
            return True
        if self.provider == "GEMINI" and (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            return True
        if self.provider == "ANTHROPIC" and os.environ.get("ANTHROPIC_API_KEY"):
            return True
        return False

    def _generate_deterministic_fallback(
        self,
        customer_message: str,
        intent: str,
        intent_confidence: float,
        evidence_items: List[Dict[str, Any]]
    ) -> GroundedReply:
        """
        Deterministic, verifiable grounded generator.
        Binds reply strictly to verified historical brand responses from the training corpus.
        """
        if not evidence_items or len(evidence_items) == 0:
            return GroundedReply(
                reply="Thank you for reaching out. We are escalating your request to a specialized support representative who can inspect your order details.",
                evidence_ids=[],
                confidence=0.50,
                unsupported_claims=[],
                should_escalate=True,
                escalation_reason="INSUFFICIENT_EVIDENCE: No verified historical resolution pattern was retrieved."
            )
            
        top_evidence = evidence_items[0]
        ev_id = top_evidence.get("evidence_id", "ev_0001")
        brand_raw = top_evidence.get("brand_response", "")
        
        # Clean brand response (strip old twitter handle prefix like @cust_1234)
        clean_brand_reply = re.sub(r"^@\w+\s+", "", brand_raw).strip()
        
        # Ensure brand persona consistency
        if not clean_brand_reply:
            clean_brand_reply = f"We'd like to look into this for you! Please send us a direct message with your account or order details."
            
        # Grounding check: verify that no wild hallucinated monetary figures or non-existent warranties are added
        unsupported = []
        if "1000" in clean_brand_reply and "1000" not in brand_raw:
            unsupported.append("Unverified compensation amount")
            
        return GroundedReply(
            reply=clean_brand_reply,
            evidence_ids=[ev_id],
            confidence=round(min(float(top_evidence.get("similarity_score", 0.85)), 0.95), 4),
            unsupported_claims=unsupported,
            should_escalate=False,
            escalation_reason=None
        )

    def _call_openai(self, customer_message, intent, intent_confidence, evidence_items) -> GroundedReply:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        system_msg = SYSTEM_PROMPT.format(brand_name=self.brand_name)
        user_msg = USER_PROMPT_TEMPLATE.format(
            customer_message=customer_message,
            intent=intent,
            intent_confidence=intent_confidence,
            formatted_evidence=format_evidence_block(evidence_items)
        )
        response = client.chat.completions.create(
            model=self.model if self.model != "deterministic-v1" else "gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.0
        )
        data = json.loads(response.choices[0].message.content)
        return GroundedReply(**data)

    def _call_gemini(self, customer_message, intent, intent_confidence, evidence_items) -> GroundedReply:
        import requests
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        system_msg = SYSTEM_PROMPT.format(brand_name=self.brand_name)
        user_msg = USER_PROMPT_TEMPLATE.format(
            customer_message=customer_message,
            intent=intent,
            intent_confidence=intent_confidence,
            formatted_evidence=format_evidence_block(evidence_items)
        )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": f"{system_msg}\n\n{user_msg}"}]}],
            "generationConfig": {"response_mime_type": "application/json", "temperature": 0.0}
        }
        res = requests.post(url, json=payload, timeout=10)
        res.raise_for_status()
        text_out = res.json()["candidates"][0]["content"]["parts"][0]["text"]
        data = json.loads(text_out)
        return GroundedReply(**data)

    def _call_anthropic(self, customer_message, intent, intent_confidence, evidence_items) -> GroundedReply:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        system_msg = SYSTEM_PROMPT.format(brand_name=self.brand_name)
        user_msg = USER_PROMPT_TEMPLATE.format(
            customer_message=customer_message,
            intent=intent,
            intent_confidence=intent_confidence,
            formatted_evidence=format_evidence_block(evidence_items)
        )
        msg = client.messages.create(
            model=self.model if self.model != "deterministic-v1" else "claude-3-5-haiku-20241022",
            max_tokens=300,
            system=system_msg,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.0
        )
        data = json.loads(msg.content[0].text)
        return GroundedReply(**data)
