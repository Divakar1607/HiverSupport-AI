import os
import sys
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import numpy as np

from src.data.load_dataset import load_raw_dataset, KNOWN_BRANDS
from src.utils.logging import get_logger
from src.utils.io import get_project_root, save_json, save_yaml, load_yaml

logger = get_logger("analyze_dataset")

def analyze_and_rank_brands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies brands, aggregates conversation and reply statistics,
    and produces a ranked table with brand recommendation score.
    """
    brand_df = df[df["author_type"] == "BRAND"]
    brand_counts = brand_df["author_id"].value_counts()
    
    brand_stats = []
    
    for brand, tweet_count in brand_counts.items():
        # Customer tweets directed to this brand (in_response_to or mentioning)
        # In TWCS, customer root tweets have in_response_to_tweet_id empty or point to brand
        # Find replies by brand:
        replies = brand_df[brand_df["author_id"] == brand]
        num_brand_replies = len(replies)
        
        # Approximate threads initiated or handled by this brand
        parent_tweet_ids = set(replies["in_response_to_tweet_id"].dropna().unique())
        customer_msgs = df[(df["tweet_id"].isin(parent_tweet_ids)) & (df["author_type"] == "CUSTOMER")]
        num_customer_msgs = len(customer_msgs)
        
        num_conversations = max(len(parent_tweet_ids), 1)
        # Average conversation length in tweets:
        avg_thread_len = round(float((num_brand_replies + num_customer_msgs) / num_conversations), 2)
        reply_pct = round(float((num_brand_replies / max(num_conversations, 1)) * 100), 1)
        
        # Composite suitability score for customer support agent:
        # High volume, balanced customer-brand exchange, good response coverage
        suitability_score = round(
            (np.log1p(tweet_count) * 0.4) + 
            (min(avg_thread_len, 4.0) * 1.5) + 
            (min(reply_pct, 100.0) * 0.05),
            2
        )
        
        brand_stats.append({
            "brand": brand,
            "total_tweets": int(tweet_count),
            "brand_replies": int(num_brand_replies),
            "customer_messages": int(num_customer_msgs),
            "estimated_conversations": int(num_conversations),
            "avg_thread_length": avg_thread_len,
            "reply_coverage_pct": min(reply_pct, 100.0),
            "suitability_score": suitability_score
        })
        
    ranking_df = pd.DataFrame(brand_stats).sort_values(by="suitability_score", ascending=False).reset_index(drop=True)
    return ranking_df

def profile_dataset(df: pd.DataFrame, ranking_df: pd.DataFrame) -> Dict[str, Any]:
    """Computes comprehensive dataset profile metrics."""
    total_rows = len(df)
    unique_users = int(df[df["author_type"] == "CUSTOMER"]["author_id"].nunique())
    unique_brands = int(df[df["author_type"] == "BRAND"]["author_id"].nunique())
    missing_values = {col: int(df[col].isna().sum()) for col in df.columns}
    duplicate_count = int(df.duplicated(subset=["text"]).sum())
    
    num_customer_tweets = int((df["author_type"] == "CUSTOMER").sum())
    num_brand_tweets = int((df["author_type"] == "BRAND").sum())
    ratio = round(num_customer_tweets / max(num_brand_tweets, 1), 2)
    
    profile = {
        "total_row_count": total_rows,
        "unique_customers": unique_users,
        "unique_brands": unique_brands,
        "customer_messages_count": num_customer_tweets,
        "brand_messages_count": num_brand_tweets,
        "customer_brand_ratio": ratio,
        "duplicate_text_count": duplicate_count,
        "missing_values": missing_values,
        "recommended_brand": str(ranking_df.iloc[0]["brand"]) if len(ranking_df) > 0 else "AmazonHelp",
        "top_brands_ranked": ranking_df.to_dict(orient="records")
    }
    return profile

def main():
    root = get_project_root()
    df = load_raw_dataset()
    
    logger.info("Analyzing dataset and ranking brands...")
    ranking_df = analyze_and_rank_brands(df)
    profile = profile_dataset(df, ranking_df)
    
    # Save reports
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    profile_csv_path = reports_dir / "data_profile.csv"
    ranking_df.to_csv(profile_csv_path, index=False)
    
    profile_json_path = reports_dir / "data_profile.json"
    save_json(profile, profile_json_path)
    
    logger.info(f"Saved brand ranking to {profile_csv_path}")
    logger.info(f"Saved dataset profile to {profile_json_path}")
    
    # Check for user override or recommend top brand
    env_brand = os.environ.get("BRAND_NAME")
    selected_brand = env_brand if env_brand else profile["recommended_brand"]
    logger.info(f"Selected brand: {selected_brand} (Source: {'ENV override' if env_brand else 'Recommendation engine'})")
    
    # Update config.yaml with selected brand
    config_path = root / "configs" / "config.yaml"
    if config_path.exists():
        cfg = load_yaml(config_path)
        cfg["brand"]["name"] = selected_brand
        cfg["brand"]["twitter_handle"] = f"@{selected_brand}"
        save_yaml(cfg, config_path)
        logger.info(f"Updated {config_path} with selected brand '{selected_brand}'")
        
    print("\n" + "="*60)
    print("DATASET BRAND RANKING TABLE")
    print("="*60)
    print(ranking_df.to_string(index=False))
    print("="*60)
    print(f"AUTOMATIC RECOMMENDATION: {selected_brand}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
