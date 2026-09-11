# Failure Mode Analysis: Top 5 Real Failures in TrustDesk AI

**Source:** `reports/error_analysis.csv`  
**Dataset:** Stratified Golden Evaluation Set (`data/golden/golden_set.csv`)  
**Methodology:** Automated failure isolation followed by manual diagnostic review. None of the examples below are fabricated.

---

## Failure Mode 1: Semantic Anomaly & Bizarre Physical Exception Leading to Unsafe Auto-Handling

- **Benchmark ID:** `gold_013`
- **Customer Query:** *"Driver literally threw my package onto the roof, how do I get it down?"*
- **Model Prediction:**
  - Intent: `delivery_delay` (Confidence: **0.8739**)
  - Policy Decision: **`AUTO_HANDLE`**
- **Expected Ground Truth:**
  - Intent: `delivery_delay`
  - Action: **`ESCALATE`** (Difficulty: **HARD**)
- **Retrieved Evidence:**
  - ID: `ev_0019` (Similarity: 0.62)
  - Historical Resolution: *"We're sorry to hear that! Please check with neighbors or front desk. If still not found, send us a DM with your order ID."*
- **Why the System Failed:**
  - The lexical tokens `"driver"` and `"package"` triggered high activation for standard delivery issues.
  - The hybrid retriever retrieved an everyday misdelivery resolution advising the customer to *"check with neighbors or front desk"*.
  - Telling a customer whose package is on their roof to check with their neighbors is nonsensical and damages customer trust.
- **Root Cause Hypothesis:**
  - Heuristic risk filters capture threats, lawsuits, and injuries, but lack an **out-of-distribution physical anomaly detector** for absurd delivery events.
- **Proposed Engineering Fix:**
  - Introduce an **anomalous scene / physical obstruction classifier** or zero-shot NLI contradiction check between the customer's spatial query (roof, tree, storm drain) and standard delivery location options (porch, locker, mailbox).

---

## Failure Mode 2: Overconfident Misclassification Caused by Shared Temporal Post-Dispatch N-Grams

- **Benchmark ID:** `gold_009`
- **Customer Query:** *"Is there any way to expedite my shipment after dispatch?"*
- **Model Prediction:**
  - Intent: **`cancellation_request`** (Confidence: **0.7430**)
  - Policy Decision: `ESCALATE` (due to low retrieval match)
- **Expected Ground Truth:**
  - Intent: **`delivery_delay`** (Expedited shipping inquiry)
  - Action: `AUTO_HANDLE`
- **Retrieved Evidence:**
  - ID: `ev_0010` (Similarity: 0.54)
  - Historical Response: *"You can cancel un-shipped items via Your Orders. If it has already dispatched, please refuse delivery..."*
- **Why the System Failed:**
  - The dense model weighted the n-gram subword `"after dispatch"` heavily toward `cancellation_request`, where customers frequently ask *"can I cancel after dispatch?"*.
  - The query was about accelerating shipping speed, not canceling.
- **Root Cause Hypothesis:**
  - Bag-of-words and n-gram subwords failed to preserve the grammatical role of the verb `"expedite"` when competing against the frequent bigram `"after dispatch"`.
- **Proposed Engineering Fix:**
  - Add dependency-parsing or a lightweight Transformer encoder (e.g. ModernBERT / MiniLM) that captures semantic verb-object relationships rather than surface bigrams.

---

## Failure Mode 3: Overconservative Escalation Triggered by Conversational Hyperbole

- **Benchmark ID:** `gold_007`
- **Customer Query:** *"Guaranteed 1-day shipping failed for the third time this week!"*
- **Model Prediction:**
  - Intent: `delivery_delay` (Confidence: **0.8618**)
  - Policy Decision: **`ESCALATE`** (Reason: `REPEATED_FAILURE`)
- **Expected Ground Truth:**
  - Intent: `delivery_delay`
  - Action: **`AUTO_HANDLE`** (Informational shipping delay check)
- **Retrieved Evidence:**
  - ID: `ev_0000` (Similarity: 0.65)
  - Historical Response: *"We apologize for the delay! Please DM us your order ID so we can verify the shipping status."*
- **Why the System Failed:**
  - The risk heuristic regex matched `\bthird time\b`, automatically triggering the `REPEATED_FAILURE` code and forcing human escalation.
  - The customer was merely expressing frustration about multiple past orders, not referencing a stuck open support ticket with an agent.
- **Root Cause Hypothesis:**
  - The rule-based risk regex cannot distinguish between **ticket-level process failure** (an agent failing to resolve a specific open ticket) and **conversational venting about historical service reliability**.
- **Proposed Engineering Fix:**
  - Condition `REPEATED_FAILURE` triggers on the presence of ticket/agent linkage words (e.g. *"third agent"*, *"transferred three times"*, *"ticket reopened"*) rather than bare frequency counters like *"third time"*.

---

## Failure Mode 4: Vocabulary Sparsity Leading to Low-Confidence Classification Collapse

- **Benchmark ID:** `gold_005`
- **Customer Query:** *"It's been two weeks and my parcel is still stuck at the regional sort facility."*
- **Model Prediction:**
  - Intent: **`product_question`** (Confidence: **0.2099**)
  - Policy Decision: `ESCALATE` (Reason: `LOW_INTENT_CONFIDENCE`, `INSUFFICIENT_EVIDENCE`)
- **Expected Ground Truth:**
  - Intent: **`delivery_delay`**
  - Action: `AUTO_HANDLE`
- **Retrieved Evidence:**
  - ID: `ev_0023` (Similarity: 0.51)
- **Why the System Failed:**
  - The customer used UK/formal terminology: `"parcel"` and `"regional sort facility"`, whereas the training taxonomy had higher frequency for US terms like `"package"`, `"tracking"`, and `"box"`.
  - Confidence collapsed to 0.2099, spreading probabilities across all classes and falling into `product_question`.
- **Root Cause Hypothesis:**
  - Regional terminology mismatch (US vs. UK/International English support vocabulary).
- **Proposed Engineering Fix:**
  - Incorporate a synonym normalization map during preprocessing (mapping `parcel` $\to$ `package`, `post` $\to$ `mail`, `courier` $\to$ `carrier`).

---

## Failure Mode 5: Urgent Medical Keywords Diverting Routing from Delivery to Complaint

- **Benchmark ID:** `gold_012`
- **Customer Query:** *"My meds were inside that package and temperature sensitive. Need this immediately or I will be hospitalized."*
- **Model Prediction:**
  - Intent: **`complaint_escalation`** (Confidence: **0.3200**)
  - Policy Decision: **`ESCALATE`** (Reason: `HIGH_RISK`, `LOW_INTENT_CONFIDENCE`)
- **Expected Ground Truth:**
  - Intent: **`delivery_delay`** (High-Risk Urgency)
  - Action: **`ESCALATE`** (Difficulty: **HARD**)
- **Retrieved Evidence:**
  - ID: `ev_0019` (Similarity: 0.53)
- **Why the System Failed:**
  - The intent was misclassified as `complaint_escalation` rather than `delivery_delay`.
  - *Note on System Safety:* Although the intent was misclassified, the **deterministic escalation policy successfully intercepted the message** due to the `HIGH_RISK` medical keyword match (`"hospitalized"`), ensuring no unsafe bot reply was sent.
- **Root Cause Hypothesis:**
  - High emotional valence and extreme urgency words overpower domain nouns in dense representation space.
- **Proposed Engineering Fix:**
  - Decouple **urgency/sentiment classification** from **domain intent classification**. Intent should represent *what the customer needs* (`delivery_delay`), while an orthogonal urgency detector handles *how fast they need it* (High / Emergency).
