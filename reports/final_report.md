# TrustDesk AI — Technical Evaluation Report
**Project Title:** TrustDesk AI — Evidence-Grounded Customer Support Agent  
**Tagline:** *Classify. Retrieve. Respond. Escalate. Prove.*  
**Author:** Kathirvel (Kathir)  
**Date:** September 2026  
**Target Brand:** AmazonHelp (`@AmazonHelp`)

---

## 1. Executive Summary

Autonomous AI customer support systems face a critical trust crisis: when language models generate ungrounded answers or guess company policies, they introduce severe regulatory, legal, and financial liability. TrustDesk AI was built around a singular thesis: **proof of grounding and calibrated safety are fundamentally systems and evaluation problems, not prompt-engineering problems.**

We built an end-to-end support agent that:
1. Classifies customer queries across a brand-specific 10-intent taxonomy derived from real Twitter support data.
2. Retrieves verified historical agent resolutions via a hybrid lexical (BM25) and dense semantic vector retriever.
3. Executes a deterministic, fully inspectable escalation policy based on explicit confidence, retrieval similarity, and risk thresholds.
4. Generates grounded responses strictly bound to retrieved evidence, enforcing an automated post-generation audit for unsupported claims.

### Key Benchmark Findings
- **Intent Macro F1:** **0.7372** (Proposed Dense Subword Ensemble) vs. **0.0222** (Majority Baseline) and **0.6708** (TF-IDF Baseline).
- **Retrieval Recall@5:** **60.5%** with **MRR of 0.605**.
- **Auto-Handle Precision:** **89.6%** at a **24.0% automation coverage rate**.
- **Auto-Handled Error Rate:** **12.5%** (a core trust metric measuring false auto-completions).
- **Judge-Human Agreement:** **95.0%** within $\pm 1$ point, though showing near-zero rank correlation ($-0.0436$), exposing critical limitations in automated evaluators.

---

## 2. Problem Framing & Principles

In high-volume customer service, maximizing the "automation rate" without strict error containment is counter-productive: a bot that confidently hallucinated a non-existent refund or ignored a physical safety threat damages brand trust far more than routing the customer to a human queue.

### Core Engineering Principles
1. **Never optimize automation volume at the cost of precision.**
2. **Every autonomous claim must cite verifiable historical evidence.**
3. **Escalation policies must be deterministic and auditable—never hidden behind an LLM's self-assessment.**
4. **Conversation-level data isolation is non-negotiable to prevent turn leakage.**
5. **Report failure modes transparently.**

---

## 3. Brand Selection

Rather than hardcoding an arbitrary company, we implemented a data-driven profiling engine (`src/data/analyze_dataset.py`) that scanned the Kaggle Twitter Customer Support corpus across 13 major enterprises.

```
       brand  total_tweets  brand_replies  customer_msgs  est_conversations  avg_thread_len  reply_coverage  suitability_score
  AmazonHelp            50             50             50                 50             2.0          100.0%               9.57
AppleSupport            20             20             20                 20             2.0          100.0%               9.22
Uber_Support            15             15             15                 15             2.0          100.0%               9.11
SpotifyCares            10             10             10                 10             2.0          100.0%               8.96
       Delta            10             10             10                 10             2.0          100.0%               8.96
```

**Selection Rationale:** `AmazonHelp` ranked #1 with the highest composite suitability score (**9.57**). It features a balanced 1:1 customer-to-brand exchange ratio, diverse support workflows (shipping delays, damaged goods, digital streaming glitches, account takeovers, billing reconciliations), and consistent public-to-DM transition procedures. The pipeline allows user overrides via `BRAND_NAME` or `configs/config.yaml`.

---

## 4. Intent Taxonomy

Standard academic taxonomies like Banking77 fail catastrophically when applied to retail customer service (as proven in Section 7). Using semantic clustering over customer inquiries, we discovered and froze a 10-intent taxonomy in `artifacts/intent_taxonomy.yaml`:

1. **`delivery_delay`**: Inquiries regarding transit scans, late packages, or misdelivered shipments.
2. **`refund_request`**: Demands for monetary return, double subscriptions, or missing bank credits.
3. **`cancellation_request`**: Requests to revoke orders, preorders, or recurring subscriptions.
4. **`damaged_item`**: Broken fragile goods, leaking liquids, defective electronics, or missing accessories.
5. **`account_access`**: OTP delivery failures, password resets, locked accounts, and 2FA recovery.
6. **`return_policy_inquiry`**: Questions on return windows, label-free drop-off, and gift returns.
7. **`technical_problem`**: App crashes, HTTP 500 errors, Prime Video buffering, and device boot loops.
8. **`product_question`**: Invoices, VAT certificates, warranty terms, and electrical specifications.
9. **`complaint_escalation`**: Severe dissatisfaction with agent conduct, supervisor demands, and threats.
10. **`unknown_other`**: Vague chatter, greetings, jokes, and out-of-domain messages.

---

## 5. System Architecture

```
Customer Message -> Sanitization/PII Masking -> Intent Classifier (Dense Subword)
                            |
           +----------------+----------------+
           |                                 |
           v                                 v
   Hybrid Retrieval                   Risk Scanning
(BM25 + Semantic Cosine)      (Legal, Fraud, Hazard, Repeat)
           |                                 |
           +----------------+----------------+
                            |
                            v
               Deterministic Escalation Policy
                            |
             +--------------+--------------+
             |                             |
             v                             v
        AUTO_HANDLE                     ESCALATE
             |                             |
   Grounded Reply Gen             Route to Human Agent
   (Evidence Bound)               (Reason Codes & Note)
             |
   Post-Gen Claim Audit
```

1. **Privacy Layer (`src/utils/privacy.py`):** Masks emails, phone numbers, order IDs, and account handles into tokens (`[EMAIL]`, `[ORDER_ID]`, `[PHONE]`).
2. **Classifier (`src/intent/classifier.py`):** Dense subword n-gram vectorizer with an MLP neural head and temperature scaling.
3. **Hybrid Retriever (`src/retrieval/retrieve.py`):** Lexical BM25 fused with normalized dense vector dot-product cosine similarity ($\alpha = 0.5$).
4. **Escalation Engine (`src/policy/escalation.py`):** Inspectable rule-based decision tree returning structured reason codes (`HIGH_RISK`, `LOW_INTENT_CONFIDENCE`, `INSUFFICIENT_EVIDENCE`, `UNSUPPORTED_CLAIM`).
5. **Grounded Generator (`src/generation/reply_generator.py`):** Strict prompt-enforced LLM with deterministic local fallback synthesizer ensuring 100% offline reproducibility.

---

## 6. Evaluation Methodology

### Stratified Golden Evaluation Dataset (`data/golden/golden_set.csv`)
We curated a balanced, 200-sample golden evaluation dataset stratified across:
- **Intents:** 15 to 25 examples per class.
- **Difficulty Tiers:** 98 Easy (49%), 73 Medium (36.5%), and 29 Hard (14.5%) stress cases.
- **Expected Actions:** 172 `AUTO_HANDLE` (86%) and 28 `ESCALATE` (14%).
- **Edge Cases:** Specific injection of legal threats, hazardous products (exploding batteries, shattered glass injuries), ambiguous slang, and multi-intent queries.

### Leakage Prevention
Data was split strictly at the conversation graph level (70% Train / 15% Val / 15% Test). The retrieval index contains **only** training split items, verified by automated Jaccard token audits ensuring 0% near-duplicate leakage.

---

## 7. Intent Results vs. Baselines

```
                   Model  Accuracy  Macro F1  Weighted F1  Macro Precision  Macro Recall
                Majority     0.125    0.0222       0.0278           0.0125        0.1000
         TF-IDF + LogReg     0.685    0.6708       0.6778           0.7428        0.6727
Proposed (Dense Subword)     0.745    0.7372       0.7420           0.7642        0.7353
```

- **Majority Baseline** scored 0.0222 Macro F1, reflecting complete vulnerability to class imbalance.
- **TF-IDF + Logistic Regression** achieved 0.6708 Macro F1, establishing a strong linear baseline.
- **Proposed Dense Subword Ensemble** outperformed both baselines across all metrics, achieving **0.7372 Macro F1** and **74.5% Accuracy**. Subword n-grams granted robust resistance to social media typos and shorthand (`pkg`, `trkg`, `deliv`).

### Auxiliary Banking77 Transfer Experiment
Testing a classifier trained on Banking77 directly on Amazon support queries yielded a severe collapse: 42.5% of queries had zero conceptual overlap. Banking77 classified package tracking as `pending_transfer` and gift returns as `fee_inquiry`, proving that generic financial taxonomies cannot transfer to retail operations.

---

## 8. Retrieval Evaluation

```
                Metric   Value
              Recall@1   0.605
              Recall@3   0.605
              Recall@5   0.605
                   MRR   0.605
Resolution Utility (%)  42.00%
```

- Hybrid retrieval achieved **Recall@5 = 0.605** and **MRR = 0.605**.
- **Resolution Utility Analysis:** In only **42.0%** of cases did the retrieved historical reply contain concrete, actionable self-service advice. In 18.5% of cases, the retrieved text was a generic deflection (*"Please DM us your order ID"*). This distinction is critical: retrieving a syntactically similar tweet is not equivalent to solving the customer's problem.

---

## 9. Reply Quality Evaluation & LLM-as-a-Judge

Generated replies were scored on a 1–5 scale across 6 dimensions:
```
Dimension (1-5 Scale)  Mean Score
         Groundedness        3.81
          Helpfulness        4.35
            Relevance        3.97
   Resolution Quality        4.35
    Brand Consistency        3.39
               Safety        4.89
        Overall Score        4.10
```
- **Safety scored highest (4.89/5)** due to aggressive deterministic interception of high-risk cases.
- **Unsupported Claims:** 0.0% detected, as the deterministic fallback generator synthesizes text directly from verified brand replies.

### Human Agreement Calibration (40 Samples)
We benchmarked the automated judge against 40 human ratings:
- **Within $\pm 1$ Point Agreement:** **95.0%** (38/40).
- **Pearson Correlation ($r$):** **-0.0436**
- **Spearman Correlation ($\rho$):** **-0.1203**
- **Insight:** The high agreement percentage is an artifact of low score variance (scores cluster in the 4.0–5.0 band). The judge cannot reliably rank subtle quality differences within that band, demonstrating why automated judges must not be accepted without calibration.

---

## 10. Escalation Evaluation & Trust Metrics

```
Total Test Samples:        200
Auto-Handle Rate:          24.0% (48 samples)
Escalation Rate:           76.0% (152 samples)
Auto-Handle Precision:     89.6%
Auto-Handle Recall:        25.0%
Escalate Precision:        15.1%
Escalate Recall:           82.1%
Auto-Handled Error Rate:   12.5% (6 errors / 48 auto-handled)
```

### Coverage vs. Quality Trade-Off
```
 Confidence Threshold  Auto-Handle Coverage (%)  Auto-Handle Accuracy (%)  Auto-Handled Error Rate (%)
                 0.40                      35.5                      78.9                         21.1
                 0.50                      30.5                      86.9                         13.1
                 0.60                      26.5                      86.8                         13.2
                 0.65                      24.0                      87.5                         12.5
                 0.70                      23.5                      89.4                         10.6
                 0.80                      21.0                      88.1                         11.9
                 0.85                      18.5                      89.2                         10.8
```
- Lowering the confidence threshold to 0.40 expands automation coverage to 35.5%, but spikes the error rate to **21.1%** (1 in 5 automated messages is incorrect).
- Our default 0.65 threshold achieves an optimal balance: 87.5% accuracy and 12.5% error rate.

---

## 11. Failure Mode Analysis (Top 5 Real Incidents)

1. **Semantic Anomaly / Bizarre Delivery (`gold_013`):** *"Driver literally threw my package onto the roof..."* Model predicted `delivery_delay` and auto-handled, advising the customer to *"check with neighbors"*. **Fix:** Add a spatial anomaly detector.
2. **Shared Temporal Bigrams (`gold_009`):** *"Is there any way to expedite my shipment after dispatch?"* Misclassified as `cancellation_request` because `"after dispatch"` heavily correlated with cancellation questions. **Fix:** Sentence transformer encoder capturing verb-object semantics.
3. **Overconservative Hyperbole (`gold_007`):** *"Guaranteed 1-day shipping failed for the third time this week!"* Customer venting triggered the `REPEATED_FAILURE` regex rule. **Fix:** Require agent/ticket context words for repeated failure triggers.
4. **Vocabulary Sparsity (`gold_005`):** *"My parcel is still stuck at the regional sort facility."* Confidence collapsed to 0.20 due to UK terminology (`parcel`, `sort facility`). **Fix:** Synonym canonicalization layer.
5. **Urgency Keyword Distortion (`gold_012`):** *"My meds were inside... or I will be hospitalized."* Urgency words biased intent to `complaint_escalation`, though safety heuristics correctly triggered escalation. **Fix:** Decouple urgency from domain intent.

---

## 12. What Is Misleading About Our Headline Numbers?

1. **89.6% Auto-Handle Precision** is flattered by a **76% Escalation Rate**. The system only acts on 1 in 4 inquiries.
2. **0.7372 Macro F1** hides intent collapse on ambiguous categories (`unknown_other` F1 = 0.462).
3. **60.5% Retrieval Recall@5** counts generic DM deflections as successful hits, even when resolution utility is only 42%.
4. **95.0% Judge Agreement** conceals a near-zero Pearson correlation ($-0.04$), rendering the judge incapable of ranking fine quality nuances.

---

## 13. Limitations

1. **Static Retrieval Index:** The current FAISS/vector store does not dynamically incorporate newly resolved agent conversations without re-indexing.
2. **Zero Multi-Turn State Tracking:** Each message is processed independently without threading previous turns in the dialogue tree.
3. **Bilingual & Slang Constraints:** Support queries containing non-English phrases or heavy regional slang suffer lower intent confidence.

---

## 14. What I Would Do With One More Week

- **Day 1 (Adversarial Stress Testing & Hard-Negative Mining):** Mine hard negatives from the 3M Kaggle dataset where customer queries share 80%+ lexical overlap but require opposite actions (e.g. *"cancel before ship"* vs. *"cancel after ship"*).
- **Day 2 (Hierarchical Multi-Intent Detection):** Implement a two-stage classifier: Stage 1 predicts high-level domain (Logistics, Billing, Account, Feedback); Stage 2 predicts fine-grained leaf action.
- **Day 3 (Conversation-Level Reranking):** Upgrade retrieval from single-turn customer pairs to full dialogue subtrees, scoring evidence with a cross-encoder reranker (e.g. `bge-reranker-large`).
- **Day 4 (Calibrated Temperature Scaling):** Train isotonic regression or Platt scaling on the validation split to align softmax confidence with empirical auto-handle error probabilities.
- **Day 5 (Human-in-the-Loop Active Learning Interface):** Build an agent review UI where support leads can accept, edit, or reject draft responses with 1-click feedback updating the retrieval index.
- **Day 6 (Automated Regression Test Suite):** Set up continuous evaluation in CI/CD that blocks model deployments if the Auto-Handled Error Rate exceeds 15% on the frozen golden set.
- **Day 7 (Production Latency & Load Testing):** Benchmark p95 latency under 500 concurrent requests; optimize vector retrieval with quantized FAISS (IVFPQ) to achieve $< 25\text{ms}$ execution.
