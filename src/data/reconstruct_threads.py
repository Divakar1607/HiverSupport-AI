from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set
import pandas as pd
import numpy as np

from src.data.load_dataset import load_raw_dataset
from src.utils.logging import get_logger
from src.utils.io import get_project_root, load_config, save_json

logger = get_logger("reconstruct_threads")

def reconstruct_conversations(df: pd.DataFrame, target_brand: str = None) -> List[Dict[str, Any]]:
    """
    Reconstructs multi-turn conversational trees into ordered threads.
    Each conversation:
    - conversation_id
    - brand
    - customer_id
    - messages: list of {message_id, author_type, text, timestamp, in_reply_to, position}
    - timestamps
    """
    logger.info("Reconstructing conversation threads from message linkage graph...")
    
    # Map tweet_id -> row dictionary
    tweet_lookup = {}
    reply_children = defaultdict(list)
    
    for _, row in df.iterrows():
        t_id = str(row["tweet_id"])
        in_reply = str(row.get("in_response_to_tweet_id", "")) if pd.notna(row.get("in_response_to_tweet_id")) else ""
        tweet_lookup[t_id] = {
            "message_id": t_id,
            "author_id": str(row["author_id"]),
            "author_type": str(row["author_type"]),
            "text": str(row.get("text_clean", row.get("text", ""))),
            "timestamp": str(row.get("created_at", "")),
            "in_reply_to": in_reply if in_reply != "nan" else "",
            "response_tweet_id": str(row.get("response_tweet_id", ""))
        }
        if in_reply and in_reply != "nan" and in_reply != "":
            reply_children[in_reply].append(t_id)
            
    # Find root tweets: tweets where in_reply_to is empty or not in tweet_lookup
    # Filter for threads relevant to the target brand if specified
    conversations = []
    visited: Set[str] = set()
    
    root_tweet_ids = [
        t_id for t_id, data in tweet_lookup.items()
        if not data["in_reply_to"] or data["in_reply_to"] not in tweet_lookup
    ]
    
    logger.info(f"Found {len(root_tweet_ids)} candidate root tweets.")
    
    conv_id_counter = 1
    for root_id in root_tweet_ids:
        if root_id in visited:
            continue
            
        # Traverse descendants via BFS/DFS
        thread_messages = []
        queue = [root_id]
        
        while queue:
            curr_id = queue.pop(0)
            if curr_id in visited:
                continue
            visited.add(curr_id)
            
            if curr_id in tweet_lookup:
                msg = tweet_lookup[curr_id]
                thread_messages.append(msg)
                
                # Add known children
                for child_id in reply_children.get(curr_id, []):
                    if child_id not in visited:
                        queue.append(child_id)
                        
        if not thread_messages:
            continue
            
        # Identify brand and customer in this thread
        brands_in_thread = [m["author_id"] for m in thread_messages if m["author_type"] == "BRAND"]
        customers_in_thread = [m["author_id"] for m in thread_messages if m["author_type"] == "CUSTOMER"]
        
        assigned_brand = brands_in_thread[0] if brands_in_thread else "Unknown"
        assigned_customer = customers_in_thread[0] if customers_in_thread else "Anonymous"
        
        # If filtering by target brand, skip if doesn't match
        if target_brand and assigned_brand != target_brand:
            continue
            
        # Sort messages by position
        ordered_messages = []
        for pos, msg in enumerate(thread_messages):
            ordered_messages.append({
                "message_id": msg["message_id"],
                "author_type": msg["author_type"],
                "author_id": msg["author_id"],
                "text": msg["text"],
                "timestamp": msg["timestamp"],
                "in_reply_to": msg["in_reply_to"],
                "position": pos
            })
            
        conv = {
            "conversation_id": f"conv_{conv_id_counter:06d}",
            "brand": assigned_brand,
            "customer_id": assigned_customer,
            "message_count": len(ordered_messages),
            "timestamps": [m["timestamp"] for m in ordered_messages if m["timestamp"]],
            "messages": ordered_messages
        }
        conversations.append(conv)
        conv_id_counter += 1
        
    logger.info(f"Reconstructed {len(conversations)} conversations for brand '{target_brand or 'ALL'}'.")
    return conversations

def main():
    root = get_project_root()
    config = load_config()
    target_brand = config["brand"]["name"]
    
    df = load_raw_dataset()
    conversations = reconstruct_conversations(df, target_brand=target_brand)
    
    # Save to interim directory
    interim_dir = root / config["data"]["interim_dir"]
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = interim_dir / f"conversations_{target_brand}.json"
    save_json(conversations, out_path)
    logger.info(f"Saved reconstructed conversations to {out_path}")
    
    # Also create flat CSV of customer-brand exchange pairs for easy inspection
    pairs = []
    for conv in conversations:
        cust_msg = None
        for msg in conv["messages"]:
            if msg["author_type"] == "CUSTOMER" and not cust_msg:
                cust_msg = msg["text"]
            elif msg["author_type"] == "BRAND" and cust_msg:
                pairs.append({
                    "conversation_id": conv["conversation_id"],
                    "customer_message": cust_msg,
                    "brand_reply": msg["text"],
                    "brand": conv["brand"]
                })
                cust_msg = None # reset for next turn
                
    pairs_df = pd.DataFrame(pairs)
    pairs_csv_path = interim_dir / f"pairs_{target_brand}.csv"
    pairs_df.to_csv(pairs_csv_path, index=False)
    logger.info(f"Saved {len(pairs_df)} exchange pairs to {pairs_csv_path}")

if __name__ == "__main__":
    main()
