import json
import os
from pathlib import Path
from typing import Any, Dict
import yaml

def get_project_root() -> Path:
    """Returns the root directory of the project."""
    return Path(__file__).resolve().parent.parent.parent

def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(data: Any, path: str | Path) -> None:
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def load_config() -> Dict[str, Any]:
    root = get_project_root()
    config_path = root / "configs" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    config = load_yaml(config_path)
    
    # Environment variable overrides
    brand_override = os.environ.get("BRAND_NAME")
    if brand_override:
        config["brand"]["name"] = brand_override
        config["brand"]["twitter_handle"] = f"@{brand_override}"
        
    return config
