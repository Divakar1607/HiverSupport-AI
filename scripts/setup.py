import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import get_project_root
from src.utils.logging import get_logger

logger = get_logger("setup")

def main():
    root = get_project_root()
    logger.info("Initializing TrustDesk AI environment and directories...")
    
    dirs = [
        root / "data" / "raw",
        root / "data" / "interim",
        root / "data" / "processed",
        root / "data" / "golden",
        root / "artifacts" / "models",
        root / "artifacts" / "retrieval_index",
        root / "reports",
        root / "docs",
        root / "tests"
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        
    logger.info("Created directory trees. Checking dataset status...")
    from src.data.load_dataset import generate_sample_dataset_if_missing
    raw_path = root / "data" / "raw" / "twcs.csv"
    generate_sample_dataset_if_missing(raw_path)
    
    logger.info("Setup complete. You can now run 'python scripts/preprocess.py'.")

if __name__ == "__main__":
    main()
