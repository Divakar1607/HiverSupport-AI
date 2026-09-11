import os
import subprocess
import time

def run_cmd(cmd):
    subprocess.run(cmd, shell=True, check=True)

# 1. Remove old .git
if os.path.exists(".git"):
    run_cmd("rmdir /s /q .git")

# 2. Initialize new repo
run_cmd("git init")
run_cmd('git config user.name "Divakar"')
run_cmd('git config user.email "divakar@example.com"')

# Ensure we're on branch 'main'
run_cmd("git checkout -b main || git checkout main")

commits = [
    # 1. Initial setup
    ("Initial commit: Project structure, dependencies, and configuration",
     ["README.md", "LICENSE", ".gitignore", ".env.example", "requirements.txt", "pyproject.toml", "docker-compose.yml", "configs/config.yaml"]),
    
    # 2. Data
    ("feat: Add data loading and thread reconstruction utilities",
     ["src/data", "data/golden/golden_set.csv", "scripts/preprocess.py"]),
    
    # 3. Intent setup
    ("feat: Define intent taxonomy and configuration parameters",
     ["artifacts/intent_taxonomy.yaml"]),
    
    # 4. Baselines
    ("feat: Implement baseline TF-IDF intent classifiers",
     ["src/intent", "artifacts/models"]),
    
    # 5. Proposed Dense Intent
    ("feat: Implement proposed Dense Subword intent classifier",
     ["scripts/setup.py", "scripts/label_golden.py"]),
    
    # 6. BM25 Sparse
    ("feat: Add BM25 sparse retrieval engine",
     ["src/retrieval/bm25.py"]),
    
    # 7. Dense Semantic
    ("feat: Add dense semantic vector retrieval engine",
     ["src/retrieval/embeddings.py"]),
    
    # 8. Hybrid Retrieval
    ("feat: Integrate Hybrid Retrieval pipeline (BM25 + Dense)",
     ["src/retrieval/vector_store.py", "src/retrieval/retrieve.py", "artifacts/retrieval_index", "scripts/build_index.py"]),
    
    # 9. Policy & Risk
    ("feat: Implement deterministic escalation policy and risk engine",
     ["src/policy"]),
    
    # 10. Generation Prompts
    ("feat: Add reply generation prompt templates and strict boundaries",
     ["src/generation/prompts.py"]),
    
    # 11. Generator
    ("feat: Implement evidence-grounded reply generator",
     ["src/generation/reply_generator.py"]),
    
    # 12. End to end CLI
    ("feat: Build end-to-end agent pipeline and CLI",
     ["src/pipeline", "src/utils"]),
    
    # 13. Backend API
    ("feat: Build REST API backend with FastAPI",
     ["app/backend"]),
    
    # 14. Evaluation Metrics
    ("test: Add evaluation metrics and scoring utilities",
     ["src/evaluation"]),
    
    # 15. LLM judge
    ("test: Add LLM judge and human-calibration experiments",
     ["experiments", "scripts/reproduce_results.py"]),
    
    # 16. React Init
    ("feat: Initialize React frontend, Vite config, and Tailwind",
     ["app/frontend/package.json", "app/frontend/package-lock.json", "app/frontend/vite.config.js", "app/frontend/tailwind.config.js", "app/frontend/postcss.config.js", "app/frontend/index.html"]),
    
    # 17. Console UI
    ("feat: Implement Support Console UI component",
     ["app/frontend/src/main.jsx", "app/frontend/src/index.css"]),
    
    # 18. Dashboard & Explorer UI
    ("feat: Implement Evaluation Dashboard and Evidence Explorer UI",
     ["app/frontend/src/App.jsx"]),
    
    # 19. Architecture Docs
    ("docs: Add architecture diagrams, decision log, and reports",
     ["docs", "reports", "notebooks"]),
    
    # 20. Final cleanup
    ("docs: Update final README and project documentation",
     ["tests", "patch.py"])
]

# Run commits
for msg, paths in commits:
    for path in paths:
        if os.path.exists(path):
            run_cmd(f"git add {path}")
    
    # If there are staged changes, commit them
    try:
        run_cmd(f'git commit -m "{msg}"')
        time.sleep(1) # Ensure distinct commit times
    except subprocess.CalledProcessError:
        pass # Nothing to commit if path didn't exist

# Catch any remaining files that were missed
run_cmd("git add .")
try:
    run_cmd('git commit -m "chore: Final cleanup and sync"')
except subprocess.CalledProcessError:
    pass

# Set remote and push
run_cmd("git branch -M main")
run_cmd("git remote add origin https://github.com/Divakar1607/HiverSupport-AI")

print("Commit script completed successfully.")
