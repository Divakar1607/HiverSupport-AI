# Decision Log: 15 Core Engineering & Architectural Decisions

**Project:** TrustDesk AI — Evidence-Grounded Customer Support Agent  
**Purpose:** Documenting non-obvious engineering decisions, rationale, rejected alternatives, and trade-offs made during system design and implementation.

---

### Decision 1: Conversation-Level Splitting (Not Tweet-Level)
- **Context:** The raw Twitter customer support dataset contains millions of interconnected tweets linked via `in_response_to_tweet_id`.
- **Decision:** Split data strictly by conversational threads/dialogue graphs, ensuring that all turns of a conversation reside in exactly one split (Train 70% / Val 15% / Test 15%).
- **Trade-off / Why Not Alternative:** Random tweet-level splitting causes catastrophic data leakage: the model trains on a customer's initial tweet and evaluates on the follow-up reply in the exact same conversation, artificially inflating evaluation accuracy by 15–25%.

---

### Decision 2: Prioritizing Macro F1 Over Accuracy as the Primary Metric
- **Context:** Real-world customer support requests exhibit heavy power-law class imbalance (`delivery_delay` and `refund_request` make up over 50% of volume).
- **Decision:** Benchmark models primarily on unweighted Macro F1 rather than overall accuracy.
- **Trade-off / Why Not Alternative:** In a 10-class dataset where two classes represent 60% of data, a model predicting only the top 2 classes can achieve 60% accuracy while having a 0.00 F1 on critical safety/complaint categories. Macro F1 weights all intents equally, exposing model failure on rare classes.

---

### Decision 3: Using Historical Brand Replies as Verifiable Evidence (Not Static FAQ Docs)
- **Context:** Generating grounded support replies requires authoritative reference information.
- **Decision:** Use historical, verified customer service agent replies from the selected brand (`AmazonHelp`) as the retrieval corpus.
- **Trade-off / Why Not Alternative:** Static documentation is often abstract, outdated, and lacks the conversational empathy, tone, and practical troubleshooting sequences used by frontline human agents on social platforms.

---

### Decision 4: Mandatory Retrieval Evidence Requirement for Auto-Handling
- **Context:** An agent might classify intent with high confidence (e.g., 95% `cancellation_request`) even when no relevant historical resolution is retrieved.
- **Decision:** Enforce that `AUTO_HANDLE` requires **both** intent confidence $\ge 0.65$ **and** retrieval evidence similarity $\ge 0.60$.
- **Trade-off / Why Not Alternative:** High classifier confidence only indicates what the customer is asking about; it does not mean the system knows the answer. Without retrieved evidence, LLMs hallucinate non-existent cancellation policies.

---

### Decision 5: Deterministic, Inspectable Escalation Policy (Not LLM Self-Judgment)
- **Context:** Modern agents often ask the LLM: *"Should you escalate this message? Respond yes or no."*
- **Decision:** Built a deterministic, rule-based escalation policy engine evaluated on explicit signals (confidence, evidence similarity, risk keywords, account credentials).
- **Trade-off / Why Not Alternative:** LLMs suffer from severe overconfidence and sycophancy, frequently claiming they can handle legal threats or hazardous product incidents. A deterministic engine is 100% auditable, reproducible, and provides explicit reason codes.

---

### Decision 6: Deriving Brand-Specific Taxonomy (Rejecting Blind Use of Banking77)
- **Context:** Banking77 is a popular benchmark in academic intent classification.
- **Decision:** Discovered a bespoke 10-intent taxonomy grounded in Twitter e-commerce support data rather than forcing Banking77.
- **Trade-off / Why Not Alternative:** Banking77 categories (e.g., `card_arrival`, `pin_blocked`, `exchange_rate`) have zero relevance to Amazon's core support operations (packages, broken goods, prime video buffering, returns). Benchmarking on the wrong domain is useless in production.

---

### Decision 7: Hybrid Retrieval (BM25 + Dense Semantic Vector Space)
- **Context:** Lexical search fails on paraphrases; dense embeddings fail on exact order numbers and specific technical keywords.
- **Decision:** Implemented a hybrid retriever: $\text{Score} = 0.5 \times \text{BM25}_{\text{norm}} + 0.5 \times \text{DenseCosine}$.
- **Trade-off / Why Not Alternative:** If a customer submits order ID `112-984712-441` or exact error `500`, BM25 matches immediately. When a customer uses slang like *"my package went poof"*, dense embeddings bridge the vocabulary gap.

---

### Decision 8: Mandatory Human Agreement Calibration for the LLM Judge
- **Context:** LLM-as-a-judge is widely used for automated reply evaluation.
- **Decision:** Explicitly calibrated judge scores against 40 human-annotated ratings, calculating Pearson/Spearman correlation and $\pm 1$ agreement.
- **Trade-off / Why Not Alternative:** Without human calibration, teams blindly trust LLM judges. Our empirical experiment revealed that while $\pm 1$ agreement was 95%, Pearson correlation was near-zero ($-0.04$), proving that the judge cannot rank fine-grained response quality within high-scoring bands.

---

### Decision 9: Active Near-Duplicate Leakage Auditing Across Splits
- **Context:** Twitter users often tweet identical copy-pasted corporate complaints or automated bot pings across different thread IDs.
- **Decision:** Added an automated Jaccard token audit in `split_dataset.py` flagging customer inquiries with $>0.90$ similarity across train and test sets.
- **Trade-off / Why Not Alternative:** Thread-level splitting alone does not protect against identical duplicate text generated by different accounts. Auditing eliminates synthetic benchmark inflation.

---

### Decision 10: Immediate Hard Escalation for Legal, Regulatory, and Safety Signals
- **Context:** Customers frequently mention lawyers, chargebacks, or injuries during support interactions.
- **Decision:** Hard-coded keyword and regex heuristics that trigger immediate `ESCALATE` with reason codes `HIGH_RISK` or `REPEATED_FAILURE`, bypassing the generation stage entirely.
- **Trade-off / Why Not Alternative:** Permitting an LLM to generate automated replies to customers threatening lawsuits or reporting exploding batteries creates massive legal and brand liability.

---

### Decision 11: Excluding Raw Multi-GB Datasets and Vector Blobs from Git
- **Context:** Kaggle's Twitter customer support dataset is ~1.5GB uncompressed.
- **Decision:** Enforced `.gitignore` exclusion of `data/raw/*`, `data/interim/*`, and binary vector indices. Built an automated realistic sample generator so the test suite runs out of the box in $< 15$ minutes.
- **Trade-off / Why Not Alternative:** Committing multi-gigabyte raw files clogs Git history and violates Kaggle distribution agreements.

---

### Decision 12: Intentionally Structuring Golden Set with 25% Hard & Adversarial Cases
- **Context:** Standard evaluation sets often sample uniformly from clean historical data.
- **Decision:** Curated 200 benchmark examples with 98 Easy (49%), 73 Medium (36.5%), and 29 Hard (14.5%) cases, specifically injecting sarcasm, multi-intent ambiguity, and hazard edge cases.
- **Trade-off / Why Not Alternative:** Clean benchmarks yield deceptively high accuracy scores (90%+) that collapse upon production deployment. Challenging benchmarks reveal the true operational boundaries of the system.

---

### Decision 13: Rejecting Raw Accuracy as the Primary Executive Metric
- **Context:** Stakeholders often ask for a single "Accuracy" percentage.
- **Decision:** Foregrounded **Auto-Handled Error Rate** and **Macro F1** in all executive summaries.
- **Trade-off / Why Not Alternative:** A system with 85% accuracy could have a 25% error rate among the decisions it actively automates. In customer support, failing silently on an automated answer is 10x worse than escalating to a human.

---

### Decision 14: Automated Re-Verification of Unsupported Claims
- **Context:** Prompting LLMs with *"do not invent policies"* reduces, but does not completely eliminate, hallucinations.
- **Decision:** The generator scans draft replies for claims not present in the retrieved evidence (e.g. invented cash figures, unwarranted guarantees). If detected, the system forces escalation with reason code `UNSUPPORTED_CLAIM`.
- **Trade-off / Why Not Alternative:** Relying solely on prompt instructions is inadequate for enterprise deployment; post-generation verification acts as a hard safety guardrail.

---

### Decision 15: Optimizing for Trust and Precision over Maximum Automation Volume
- **Context:** Businesses frequently demand: *"Can we auto-resolve 80% of our tickets?"*
- **Decision:** Tuned policy thresholds to achieve $\ge 89\%$ auto-handle precision at a 24% automation rate, rather than maximizing automation volume at the cost of errors.
- **Trade-off / Why Not Alternative:** Maximizing automation to 60% causes the auto-handled error rate to jump to $>21\%$ (as proven by our Coverage vs Quality curve). Unreliable automation destroys customer trust and overwhelms support teams with angry re-contacts.
