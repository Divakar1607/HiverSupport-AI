import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_dataset import load_raw_dataset
from src.data.analyze_dataset import main as run_analyze
from src.data.reconstruct_threads import main as run_reconstruct
from src.data.split_dataset import main as run_split
from src.utils.logging import get_logger

logger = get_logger("preprocess")

def main():
    logger.info("Starting complete preprocessing pipeline...")
    logger.info("Step 1: Dataset profiling & Brand ranking...")
    run_analyze()
    
    logger.info("Step 2: Conversation reconstruction...")
    run_reconstruct()
    
    logger.info("Step 3: Leakage-free dataset splitting...")
    run_split()
    
    logger.info("Preprocessing complete! Processed files ready in data/processed/")

if __name__ == "__main__":
    main()
