from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field

from src.utils.io import get_project_root, load_yaml, save_yaml
from src.utils.logging import get_logger

logger = get_logger("taxonomy")

class IntentDefinition(BaseModel):
    id: str
    description: str
    examples: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)

class IntentTaxonomy(BaseModel):
    brand: str
    version: str
    intents: List[IntentDefinition]

def load_intent_taxonomy(path: Optional[str | Path] = None) -> IntentTaxonomy:
    """Loads and validates the frozen intent taxonomy YAML."""
    root = get_project_root()
    if path is None:
        path = root / "artifacts" / "intent_taxonomy.yaml"
    else:
        path = Path(path)
        
    if not path.exists():
        raise FileNotFoundError(f"Intent taxonomy not found at {path}. Run discover_intents.py first.")
        
    data = load_yaml(path)
    return IntentTaxonomy(**data)

def save_intent_taxonomy(taxonomy: IntentTaxonomy, path: Optional[str | Path] = None) -> None:
    """Serializes the intent taxonomy to YAML."""
    root = get_project_root()
    if path is None:
        path = root / "artifacts" / "intent_taxonomy.yaml"
    else:
        path = Path(path)
        
    path.parent.mkdir(parents=True, exist_ok=True)
    save_yaml(taxonomy.model_dump(), path)
    logger.info(f"Saved frozen intent taxonomy to {path}")
