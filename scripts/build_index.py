import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.retrieval.retrieve import HybridEvidenceRetriever
from src.utils.io import get_project_root, load_config
from src.utils.logging import get_logger

logger = get_logger("build_index")

def main():
    root = get_project_root()
    config = load_config()
    
    train_pairs_path = root / config["data"]["processed_dir"] / "train_pairs.csv"
    if not train_pairs_path.exists():
        logger.error(f"Training pairs not found at {train_pairs_path}. Run split_dataset.py first.")
        sys.exit(1)
        
    df_train = pd.read_csv(train_pairs_path)
    logger.info(f"Loaded {len(df_train)} training pairs for retrieval index.")
    
    retriever = HybridEvidenceRetriever(alpha=config["retrieval"].get("hybrid_alpha", 0.5))
    retriever.build_index(df_train)
    
    index_dir = root / config["retrieval"]["index_dir"]
    retriever.save(index_dir)
    logger.info(f"Retrieval index built and stored at {index_dir}")
    
    # Run sanity test retrieval
    test_query = "Where is my late package?"
    hits = retriever.retrieve(test_query, top_k=2)
    print("\n" + "="*60)
    print(f"SANITY CHECK RETRIEVAL: '{test_query}'")
    print("="*60)
    for h in hits:
        print(f"[{h['evidence_id']}] Sim: {h['similarity_score']} | Match: {h['customer_message'][:50]}... -> Reply: {h['brand_response'][:50]}...")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
