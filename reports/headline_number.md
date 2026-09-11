# What Is Misleading About Our Headline Numbers?

**Project:** TrustDesk AI — Evidence-Grounded Customer Support Agent  
**Headline Metrics:**
- **Intent Accuracy:** 74.5%
- **Intent Macro F1:** 0.7372 (vs. Majority Baseline: 0.0222, TF-IDF Baseline: 0.6708)
- **Auto-Handle Precision:** 89.6%
- **Judge-Human ±1 Agreement:** 95.0%

---

## Executive Warning: Why Headline Numbers Flatter the Reality

In customer support machine learning, reporting headline accuracy or macro F1 without system-level context creates a dangerous illusion of production readiness. Below are **five concrete, empirical ways our headline numbers mask severe failure modes**:

---

### 1. High Escalation Rate (76.0%) Artificially Inflates Auto-Handle Precision (89.6%)

Our policy engine achieved an impressive **89.6% Auto-Handle Precision**. However, this metric is achieved by acting extremely conservatively:
- The system **auto-handles only 24.0% of incoming tickets** (48 out of 200).
- It **escalates 76.0% of messages** (152 out of 200) to human agents.

If evaluated naively, an 89.6% precision sounds like 9 out of 10 automated answers are trustworthy. But the system only had the confidence to touch 1 out of 4 customer questions! Furthermore, **among the 48 cases it chose to auto-handle, 6 were errors (12.5% Auto-Handled Error Rate)**. In a production setting processing 100,000 monthly inquiries, an auto-handled error rate of 12.5% on 24,000 automated responses represents **3,000 customers receiving misleading or misclassified automated resolutions**.

---

### 2. Macro F1 (0.7372) Hides Catastrophic Collapse on Ambiguous & Rare Intents

While the headline Macro F1 is **0.7372**, the per-intent breakdown reveals stark disparity:
- Well-represented intents with distinctive keywords achieve high F1:
  - `delivery_delay`: F1 = **0.864**
  - `refund_request`: F1 = **0.821**
- Meanwhile, ambiguous, subtle, or edge-case intents collapse:
  - `unknown_other`: F1 = **0.462** (Precision: 0.400, Recall: 0.533)
  - `complaint_escalation`: F1 = **0.588** (struggles when customers express anger without explicit supervisor keywords)
  - `product_question`: F1 = **0.615** (often confused with `return_policy_inquiry` when asking about warranty returns)

Averaging across 10 classes hides the fact that a customer submitting an unclassified or emotionally fraught query has less than a 50% chance of correct categorization.

---

### 3. Retrieval Recall@5 (0.6050) Confuses Syntactic Similarity with Resolution Utility

Our hybrid retrieval engine achieved **Recall@5 = 0.6050**. However, the **Resolution Utility is only 42.0%**:
- In 18.5% of cases where Recall@5 was marked as a "hit" (because the lexical and semantic similarity was $\ge 0.58$), the historical brand response was merely a generic deflection:  
  *e.g., "Thanks for reaching out! Please send us a DM with your account details."*
- A high similarity score tells us the customer's problem is similar to a past issue; **it does not guarantee the retrieved historical response actually resolves the current problem**.
- Relying on retrieval similarity alone as a proxy for grounding creates an "empty resolution" loop where the bot repeats a generic DM request rather than answering actionable questions.

---

### 4. Judge-Human Agreement (95.0% within ±1 point) Masks Near-Zero Correlation (-0.0436)

Our LLM-as-judge agreed with human raters within $\pm 1$ point on **95.0% of evaluated benchmark samples** (38 out of 40). At first glance, this appears to validate the judge as a reliable automated proxy.

However, the statistical correlation tells a completely different story:
- **Pearson Correlation ($r$):** **-0.0436**
- **Spearman Rank Correlation ($\rho$):** **-0.1203**

#### Why did this happen?
Because both the human raters and the automated judge scored almost all responses between **4.0 and 5.0** (low variance). The agreement metric is superficially high simply because a $\pm 1.0$ band on a 1–5 scale covers 25% of the entire scoring spectrum! When asked to discriminate between a 4.2 response and a 4.7 response, the judge has zero ranking ability. **Headline agreement percentages without correlation metrics hide judge calibration failure.**

---

### 5. Easy and Formatted Examples Dominate the Distribution

In our 200-sample golden set:
- **98 samples (49.0%) are EASY** (clear keywords like *"tracking says delivered"*, *"charged twice"*, *"cancel order"*).
- If the model gets 90% of the easy cases right (88/98), it already achieves a baseline accuracy of 44% without learning any nuanced conversational semantics.
- On the **29 HARD examples** (legal notices, physical hazards, multi-intent ambiguity), intent accuracy plummets to **48.3%**.

If a benchmark is not rigorously stratified with intentional stress-testing examples, the headline score primarily reflects how many simple, keyword-stuffed sentences were included in the evaluation file.

---

## Summary Matrix

| Metric | Headline Value | What It Appears to Mean | What the Empirical Data Actually Reveals |
| :--- | :--- | :--- | :--- |
| **Intent Accuracy** | 74.5% | 3 out of 4 customer inquiries are correctly routed | Accuracy on hard/edge cases drops to 48.3%; dominated by easy keyword cases. |
| **Macro F1** | 0.7372 | Consistent multi-class capability | Rare and subtle intents (`unknown_other`) drop below 0.47 F1. |
| **Auto-Handle Precision** | 89.6% | Safe to automate almost 90% of responses | Achieved only by escalating 76% of all tickets; 12.5% of auto-handled tickets are still erroneous. |
| **Recall@5** | 60.5% | Historical evidence found 60% of the time | Only 42% of queries retrieve an actionable, helpful resolution. |
| **Judge Agreement** | 95.0% (±1 pt) | Automated judge perfectly mirrors human evaluation | Pearson correlation is -0.04; judge cannot rank subtle quality differences within the 4-5 band. |
