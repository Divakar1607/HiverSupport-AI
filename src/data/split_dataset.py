import random
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import pandas as pd
import numpy as np

from src.utils.logging import get_logger
from src.utils.io import get_project_root, load_config, save_json, load_json

logger = get_logger("split_dataset")

def tokenize_simple(text: str) -> Set[str]:
    """Tokenizes text into a set of lowercased alphanumeric words for duplicate checking."""
    return set(text.lower().replace("@", " ").replace("#", " ").split())

def is_near_duplicate(text1: str, text2: str, threshold: float = 0.85) -> bool:
    """Computes Jaccard similarity of token sets to flag near-duplicate customer inquiries."""
    s1, s2 = tokenize_simple(text1), tokenize_simple(text2)
    if not s1 or not s2:
        return False
    jaccard = len(s1 & s2) / len(s1 | s2)
    return jaccard >= threshold

def split_conversations(
    conversations: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Performs a strict conversation-level split, ensuring that all turns of a conversation
    stay in exactly one split, and checks for near-duplicate customer message leakage.
    """
    logger.info(f"Splitting {len(conversations)} conversations: {train_ratio*100}% Train, {val_ratio*100}% Val, {test_ratio*100}% Test...")
    random.seed(seed)
    shuffled = list(conversations)
    random.shuffle(shuffled)
    
    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_convs = shuffled[:n_train]
    val_convs = shuffled[n_train:n_train + n_val]
    test_convs = shuffled[n_train + n_val:]
    
    # Audit for potential near-duplicate leakage between train and test
    train_texts = [
        m["text"] for c in train_convs for m in c["messages"] if m["author_type"] == "CUSTOMER"
    ]
    test_texts = [
        m["text"] for c in test_convs for m in c["messages"] if m["author_type"] == "CUSTOMER"
    ]
    
    leakage_count = 0
    for test_t in test_texts:
        for train_t in train_texts:
            if is_near_duplicate(test_t, train_t, threshold=0.92):
                leakage_count += 1
                break
                
    logger.info(f"Leakage audit: {leakage_count} near-duplicate customer questions detected across train/test boundaries.")
    logger.info(f"Train conversations: {len(train_convs)} | Val: {len(val_convs)} | Test: {len(test_convs)}")
    
    return train_convs, val_convs, test_convs

def convs_to_pairs_df(convs: List[Dict[str, Any]]) -> pd.DataFrame:
    """Flattens conversations into customer-message to brand-reply pairs for model training & retrieval."""
    rows = []
    for conv in convs:
        cust_msg = None
        for msg in conv["messages"]:
            if msg["author_type"] == "CUSTOMER" and not cust_msg:
                cust_msg = msg["text"]
            elif msg["author_type"] == "BRAND" and cust_msg:
                rows.append({
                    "conversation_id": conv["conversation_id"],
                    "brand": conv["brand"],
                    "customer_message": cust_msg,
                    "brand_reply": msg["text"]
                })
                cust_msg = None
    return pd.DataFrame(rows)

def main():
    root = get_project_root()
    config = load_config()
    target_brand = config["brand"]["name"]
    
    interim_dir = root / config["data"]["interim_dir"]
    convs_file = interim_dir / f"conversations_{target_brand}.json"
    
    if not convs_file.exists():
        logger.warning(f"Reconstructed conversations file not found at {convs_file}. Running reconstruct_threads first...")
        from src.data.reconstruct_threads import main as run_reconstruct
        run_reconstruct()
        
    conversations = load_json(convs_file)
    train_convs, val_convs, test_convs = split_conversations(
        conversations,
        train_ratio=config["data"]["conversation_split"]["train"],
        val_ratio=config["data"]["conversation_split"]["val"],
        test_ratio=config["data"]["conversation_split"]["test"],
        seed=config["data"]["conversation_split"]["seed"]
    )
    
    processed_dir = root / config["data"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save conversation splits
    save_json(train_convs, processed_dir / "train_conversations.json")
    save_json(val_convs, processed_dir / "val_conversations.json")
    save_json(test_convs, processed_dir / "test_conversations.json")
    
    # Save paired CSVs for retrieval and intent models
    train_pairs = convs_to_pairs_df(train_convs)
    val_pairs = convs_to_pairs_df(val_convs)
    test_pairs = convs_to_pairs_df(test_convs)
    
    train_pairs.to_csv(processed_dir / "train_pairs.csv", index=False)
    val_pairs.to_csv(processed_dir / "val_pairs.csv", index=False)
    test_pairs.to_csv(processed_dir / "test_pairs.csv", index=False)
    
    logger.info(f"Saved processed splits to {processed_dir}")

if __name__ == "__main__":
    main()
