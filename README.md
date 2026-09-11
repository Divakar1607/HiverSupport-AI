# TrustDesk AI — Evidence-Grounded Customer Support Agent

> **Tagline:** *Classify. Retrieve. Respond. Escalate. Prove.*  
> **Author:** Kathirvel ([@Kathirvel005](https://github.com/Kathirvel005))  
> **Evaluation:** Hiver SDE Intern Evaluation Benchmark  
> **Target Brand:** AmazonHelp (`@AmazonHelp`)

TrustDesk AI is a production-grade, evidence-grounded AI customer support agent built on the Twitter Customer Support dataset. Designed around the principle that **trustworthy AI is fundamentally an evaluation and systems problem rather than a prompt-engineering problem**, TrustDesk AI classifies incoming inquiries into brand-specific support intents, retrieves verified historical agent resolutions via hybrid BM25 and dense semantic vector search, enforces a transparent deterministic escalation policy with structured reason codes, and drafts grounded replies bound strictly to empirical evidence.

---

## 1. Problem

Modern language models deployed in enterprise customer service suffer from catastrophic hallucinations: they invent non-existent refund policies, guess warranty procedures, and display uncalibrated overconfidence during hazardous or legal incidents. Most existing commercial chatbots optimize for vanity "automation rates" without measuring how many automated answers are misleading, unsafe, or harmful.

---

## 2. Why This Matters

In customer service operations, a false auto-handled resolution is **$10\times$ more damaging than a safe human escalation**. When a bot promises an unauthorized \$50 credit or misdiagnoses an exploding lithium battery as a standard return, it creates immense legal liability and destroys brand trust. TrustDesk AI inverts this paradigm: every automated action requires cryptographic-like evidence provenance, and uncertain or high-risk cases are deterministically routed to human support teams with plain-language explanations.

---

## 3. Architecture

```
Customer Message
       |
       v
Preprocessing & PII Masking ([EMAIL], [PHONE], [ORDER_ID])
       |
       v
Intent Classifier (Dense Subword Ensemble)
       |
       v
Confidence Calibration (Softmax + Temperature)
       |
       +--------------------------------+
       |                                |
       v                                v
Hybrid Retrieval                Risk & Policy Scanning
(BM25 + Semantic Cosine)        (Legal, Fraud, Hazard, Repeat)
       |                                |
       v                                |
Historical Brand Evidence               |
       |                                |
       +---------------+----------------+
                       |
                       v
         Deterministic Escalation Policy
                       |
        +--------------+--------------+
        |                             |
        v                             v
   AUTO_HANDLE                     ESCALATE
        |                             |
Grounded Reply Generator         Route to Human Queue
(Strict Evidence Boundary)       (Structured Reason Codes)
        |
        v
Post-Gen Unsupported Claim Audit
        |
        v
Draft Reply + Evidence Provenance
```

---

## 4. Dataset Setup

The system is developed on the **Customer Support on Twitter** dataset ([Kaggle: thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)).

### Local Placement
1. Download `twcs.csv` (or `twcs.csv.zip`) from Kaggle.
2. Place the uncompressed CSV at:
   ```
   data/raw/twcs.csv
   ```
3. *Note on Zero-Configuration Reproducibility:* If `data/raw/twcs.csv` is not present, TrustDesk AI automatically generates a representative, stratified benchmark dataset matching the exact schema so all tests and evaluations run out of the box in **under 15 minutes**.

---

## 5. Installation

```bash
# 1. Clone the repository and enter directory
cd "d:\Project work\Unwanted"

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 6. Environment Variables

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Key configuration options:
```ini
BRAND_NAME=AmazonHelp
LLM_PROVIDER=LOCAL          # LOCAL (deterministic offline generator), OPENAI, GEMINI, ANTHROPIC
LLM_MODEL=deterministic-v1
CONFIDENCE_THRESHOLD=0.65
RETRIEVAL_SIMILARITY_THRESHOLD=0.60
```
*Zero API keys are required to reproduce headline results.*

---

## 7. Quick Start

Run the entire pipeline on any customer inquiry via the CLI:
```bash
python -m src.pipeline.run_agent --message "My refund has not arrived in my bank account yet."
```
Example Output:
```json
{
  "message": "My refund has not arrived in my bank account yet.",
  "intent": "refund_request",
  "intent_confidence": 0.9929,
  "decision": "AUTO_HANDLE",
  "reason_codes": [],
  "evidence": [
    {
      "evidence_id": "ev_0000",
      "similarity_score": 0.7899,
      "customer_message": "My refund has not arrived in my bank account yet. It has been 10 days.",
      "brand_response": "@AmazonHelp Refunds typically take 3-5 business days depending on your bank...",
      "conversation_id": "conv_000026"
    }
  ],
  "draft_reply": "Refunds typically take 3-5 business days depending on your bank. Please DM us your order details so we can trace the transaction.",
  "reply_confidence": 0.7899
}
```

---

## 8. Reproduce Results (< 15 Minutes)

To execute the entire evaluation suite, benchmark all models against the golden set, and print headline results:
```bash
python scripts/reproduce_results.py
```
Expected output:
```
==================================================
TRUSTDESK AI — REPRODUCTION
==================================================
Intent Macro F1:         0.7372
Recall@5:                0.6050
Groundedness:            3.8/5
Auto-handle precision:   89.6%
Auto-handled error rate: 12.5%
==================================================
Reproduction completed in 2.57s (under 15 minutes requirement)
==================================================
```

---

## 9. Run API (FastAPI Backend)

Launch the REST backend:
```bash
python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
Key Endpoints:
- `POST /api/predict` — Processes customer messages through the full pipeline.
- `GET /api/metrics` — Returns consolidated dashboard benchmarks.
- `GET /api/intents` — Returns the frozen intent taxonomy.
- `GET /api/health` — System status and active brand.

---

## 10. Run Frontend (React + Vite + Tailwind CSS)

```bash
cd app/frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) to access:
1. **Support Console:** Interactive message evaluation, intent badge, top-3 evidence cards, grounding audit, and prominent `⚠ HUMAN REVIEW REQUIRED` escalation banners.
2. **Evaluation Dashboard:** Multi-model comparison, confusion matrix metrics, coverage vs quality curves, and judge calibration results.
3. **Evidence Explorer:** Semantic search interface into historical agent resolutions.
4. **Decision Log:** Interactive viewer for all 15 architectural decisions.

---

## 11. Results vs. Baselines

Evaluated on the 200-example curated golden test set:

| Model | Accuracy | Macro F1 | Weighted F1 | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline 1 (Majority Class)** | 12.5% | 0.0222 | 0.0278 | 0.0125 | 0.1000 |
| **Baseline 2 (TF-IDF + LogReg)** | 68.5% | 0.6708 | 0.6778 | 0.7428 | 0.6727 |
| **Proposed (Dense Subword Ensemble)** | **74.5%** | **0.7372** | **0.7420** | **0.7642** | **0.7353** |

- **Macro F1 Emphasis:** Evaluated on unweighted Macro F1 to prevent frequent classes (`delivery_delay`, `refund_request`) from masking collapse on rare intents.
- **Auxiliary Banking77 Experiment:** Running `python experiments/banking77_transfer.py` proves that Banking77 has 0% concept coverage for physical delivery and damaged goods, collapsing on e-commerce support.

---

## 12. Retrieval Evaluation

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Recall@1** | 0.6050 | Hybrid BM25 + dense semantic vector fusion |
| **Recall@3** | 0.6050 | Top-3 ranked evidence candidates |
| **Recall@5** | 0.6050 | Threshold-calibrated retrieval |
| **MRR** | 0.6050 | Mean Reciprocal Rank |
| **Resolution Utility (%)** | **42.0%** | % of hits containing actionable self-service advice vs. generic DM deflections |

---

## 13. Escalation Evaluation & Trust Metrics

- **Auto-Handle Precision:** **89.6%**
- **Overall Escalation Rate:** **76.0%**
- **CRITICAL Auto-Handled Error Rate:** **12.5%** (6 errors / 48 auto-handled messages)

### Coverage vs. Quality Trade-Off
| Threshold | Auto-Handle Coverage (%) | Auto-Handle Accuracy (%) | Auto-Handled Error Rate (%) |
| :--- | :--- | :--- | :--- |
| 0.40 (Aggressive) | 35.5% | 78.9% | **21.1% (High Risk)** |
| 0.50 | 30.5% | 86.9% | 13.1% |
| **0.65 (Baseline)** | **24.0%** | **87.5%** | **12.5% (Target Balanced)** |
| 0.85 (Conservative) | 18.5% | 89.2% | 10.8% |

---

## 14. Golden Set (`data/golden/golden_set.csv`)

Curated benchmark of **200 examples** stratified across 10 brand intents and 3 difficulty tiers:
- **EASY (98 / 49%):** Standard inquiries with unambiguous keywords.
- **MEDIUM (73 / 36.5%):** Conversational slang, overlapping intent boundaries.
- **HARD (29 / 14.5%):** Adversarial stress cases, physical hazards, legal notices.
- Full methodology: [`docs/golden_set.md`](docs/golden_set.md).

---

## 15. Failure Analysis (Top 5 Real Failures)

Documented in detail in [`reports/failure_analysis.md`](reports/failure_analysis.md):
1. **Semantic Delivery Anomaly (`gold_013`):** *"Driver threw package onto the roof..."* Auto-handled and advised checking neighbors.
2. **Shared Temporal Bigrams (`gold_009`):** *"Expedite shipment after dispatch"* misclassified as cancellation.
3. **Overconservative Hyperbole (`gold_007`):** Customer venting *"for the third time"* triggered `REPEATED_FAILURE` escalation rule.
4. **Vocabulary Sparsity (`gold_005`):** UK terms (*"parcel"*, *"sort facility"*) caused confidence collapse to 0.20.
5. **Urgency Keyword Distortion (`gold_012`):** Medical emergency words (*"hospitalized"*) diverted routing to complaint escalation (intercepted safely by risk filters).

---

## 16. What Is Misleading About Our Headline Numbers?

Documented in detail in [`reports/headline_number.md`](reports/headline_number.md):
- **89.6% Precision** is achieved by escalating 76% of all messages.
- **0.7372 Macro F1** hides performance below 0.47 on rare intents like `unknown_other`.
- **60.5% Recall@5** includes generic DM deflections; actual resolution utility is 42%.
- **95.0% Judge Agreement** conceals a near-zero Pearson correlation ($-0.0436$), showing the judge cannot rank high-scoring nuance.

---

## 17. Decision Log

Documented in [`docs/decision_log.md`](docs/decision_log.md) covering 15 non-obvious engineering decisions:
1. Conversation-level splitting (prevents 15–25% turn leakage).
2. Prioritizing Macro F1 over accuracy.
3. Historical replies as evidence instead of static FAQs.
4. Mandatory retrieval evidence for auto-handling.
5. Deterministic escalation engine over LLM self-judgment.
6. Bespoke taxonomy over Banking77.
7. Hybrid BM25 + dense vectors.
8. Human agreement calibration for LLM judge.
9. Near-duplicate leakage auditing.
10. Hard escalation for legal, fraud, and hazard signals.
11. Excluding multi-GB raw data from Git.
12. 25% Hard stress cases in golden set.
13. Forefronting Auto-Handled Error Rate.
14. Post-generation unsupported claim audits.
15. Optimizing trust over maximum automation volume.

---

## 18. Project Structure

```
trustdesk-ai/
├── README.md                     # Comprehensive documentation
├── LICENSE                       # MIT License
├── .gitignore                    # Excludes raw data, embeddings, .env
├── .env.example                  # Environment configuration template
├── requirements.txt              # Core Python dependencies
├── pyproject.toml                # Build & pytest configuration
├── docker-compose.yml            # Container orchestration
│
├── configs/
│   └── config.yaml               # Selected brand, thresholds, paths
│
├── data/
│   ├── raw/                      # Excluded from git (twcs.csv)
│   ├── interim/                  # Reconstructed conversations
│   ├── processed/                # Train/Val/Test splits (leakage-free)
│   └── golden/
│       └── golden_set.csv        # 200 curated benchmark samples
│
├── artifacts/
│   ├── intent_taxonomy.yaml      # Frozen 10-intent taxonomy
│   ├── models/                   # Serialized classifiers (Majority, TF-IDF, Proposed)
│   └── retrieval_index/          # Hybrid BM25 + dense vector store
│
├── src/
│   ├── data/                     # load_dataset, analyze_dataset, reconstruct_threads, split_dataset
│   ├── intent/                   # discover_intents, taxonomy, classifier
│   ├── retrieval/                # bm25, embeddings, vector_store, retrieve
│   ├── generation/               # prompts, reply_generator
│   ├── policy/                   # escalation, risk
│   ├── pipeline/                 # run_agent (End-to-End CLI & class)
│   ├── evaluation/               # evaluate_intent, evaluate_retrieval, evaluate_reply, evaluate_escalation, llm_judge, run_all
│   └── utils/                    # logging, io, metrics, privacy
│
├── scripts/
│   ├── setup.py                  # Directory and environment initializer
│   ├── preprocess.py             # Data profiling and thread reconstruction
│   ├── build_index.py            # Hybrid retrieval index builder
│   ├── label_golden.py           # Golden set builder and labeling tool
│   └── reproduce_results.py      # Instant <15 min headline reproduction runner
│
├── app/
│   ├── backend/
│   │   └── main.py               # FastAPI backend with REST endpoints
│   └── frontend/                 # React + Vite + Tailwind CSS Console
│
├── reports/
│   ├── data_profile.csv          # Brand ranking table
│   ├── intent_results.csv        # Model comparison benchmark
│   ├── retrieval_results.csv     # Recall@K and MRR metrics
│   ├── coverage_vs_quality.csv   # Automation vs error rate trade-off
│   ├── error_analysis.csv        # Complete sample-by-sample error audit
│   ├── headline_number.md        # "What is misleading about my headline number?"
│   ├── failure_analysis.md       # Top 5 real failure modes
│   └── final_report.md           # 6-page comprehensive technical report
│
├── docs/
│   ├── golden_set.md             # Golden set documentation
│   ├── decision_log.md           # 15 engineering trade-offs
│   └── architecture.mmd          # Mermaid architecture diagram
│
├── tests/                        # 16 unit tests (100% passing)
├── experiments/                  # banking77_transfer.py
└── notebooks/                    # exploration.ipynb
```

---

## 19. Testing

Execute the comprehensive test suite:
```bash
pytest
```
Results: **16 passed in 6.80s** covering PII anonymization, thread reconstruction, intent classification, retrieval ranking, policy triggers, unsupported claim detection, and FastAPI endpoints.

---

## 20. License

MIT License — Copyright (c) 2026 Kathirvel.
