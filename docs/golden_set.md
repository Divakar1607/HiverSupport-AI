# Golden Evaluation Dataset Documentation

**Dataset Path:** `data/golden/golden_set.csv`  
**Total Examples:** 200  
**Target Brand:** AmazonHelp  
**Schema:** `id`, `conversation_id`, `message`, `intent`, `expected_action`, `difficulty`, `evidence_available`, `notes`

---

## 1. Sampling & Stratification Methodology

The golden evaluation benchmark was designed using a **stratified multi-tier sampling protocol** across 10 brand-specific customer support intents, rather than random sampling which over-indexes on easy, repetitive questions.

### Distribution by Intent
| Intent | Count | % of Benchmark | Rationale |
| :--- | :--- | :--- | :--- |
| `delivery_delay` | 25 | 12.5% | Dominant volume driver; tests late deliveries, missing packages, and carrier discrepancies. |
| `refund_request` | 25 | 12.5% | High financial impact; tests double charges, pending credits, and chargeback risks. |
| `cancellation_request` | 20 | 10.0% | Time-sensitive; tests pre-dispatch, post-dispatch, and partial cancellations. |
| `damaged_item` | 20 | 10.0% | Quality issue; tests broken items, missing parts, and safety hazards (leaks, batteries). |
| `account_access` | 20 | 10.0% | Security sensitive; tests OTP failures, password reset loops, and account takeover threats. |
| `return_policy_inquiry` | 20 | 10.0% | Informational/procedural; tests drop-off locations, return windows, and gift returns. |
| `technical_problem` | 20 | 10.0% | Platform usability; tests app crashes, video buffering, and checkout gateway failures. |
| `complaint_escalation` | 20 | 10.0% | High emotional escalation; tests agent conduct complaints, supervisor requests, and legal threats. |
| `product_question` | 15 | 7.5% | Specifications, invoice downloads, compatibility, and warranties. |
| `unknown_other` | 15 | 7.5% | Out-of-domain inquiries, conversational pleasantries, gibberish, and single-character prompts. |

### Distribution by Difficulty Tier
- **EASY (98 examples / 49%)**: Clear syntax, unambiguous domain keywords, standard support workflows.
- **MEDIUM (73 examples / 36.5%)**: Moderate ambiguity, complex phrasing, overlapping intent boundaries (e.g. asking for a refund while reporting a damaged item), or conversational slang.
- **HARD (29 examples / 14.5%)**: Edge cases, adversarial phrasing, urgent medical emergencies, severe property damage, legal attorney threats, chargeback warnings, or unresolvable multi-intent conflicts requiring strict human escalation.

### Distribution by Expected Action
- **`AUTO_HANDLE` (172 examples / 86%)**: Messages where historical evidence provides a verifiable, safe, and actionable resolution pattern that an automated agent can draft without human intervention.
- **`ESCALATE` (28 examples / 14%)**: Messages where automation is unsafe, policy forbids autonomous resolution, legal/fraud/threat keywords are present, or insufficient evidence exists.

---

## 2. Intent Taxonomy & Semantic Definitions

1. **`delivery_delay`**: Queries regarding transit progress, carrier tracking scans, late arrival, missed delivery notices, or packages marked delivered that have not arrived.
2. **`refund_request`**: Demands for monetary reimbursement, duplicate charges, pending bank credits, gift card balance reversals, or restocking fee disputes.
3. **`cancellation_request`**: Inquiries or direct commands to halt fulfillment, revoke orders, terminate recurring subscriptions, or cancel preorders.
4. **`damaged_item`**: Reports of defective hardware, cracked displays, leaked liquids, broken fragile items, or soiled products upon unboxing.
5. **`account_access`**: Authentication failures, OTP latency, locked accounts, password reset loops, 2FA device loss, or suspicious login activity.
6. **`return_policy_inquiry`**: Questions regarding return windows, packaging requirements, drop-off locations (UPS/Kohl's/Lockers), and printable return labels.
7. **`technical_problem`**: Software glitches, mobile app crashes, video streaming buffering, HTTP 500 cart errors, and device boot loops.
8. **`product_question`**: Inquiries regarding product specifications, voltage/compatibility, official VAT invoice generation, and manufacturer warranties.
9. **`complaint_escalation`**: Severe dissatisfaction with support agents, requests for supervisory involvement, broken agent promises, or regulatory complaints (FTC/BBB).
10. **`unknown_other`**: Vague or context-free utterances, greetings, non-support chatter, humor, and gibberish.

---

## 3. Disagreement Handling & Annotation Guidelines

When annotating ambiguous customer messages, the following deterministic rules were applied:
1. **Financial or Physical Risk Precedence**: If a message mentions a physical hazard (e.g. *fire, shattered glass injury*) or a financial fraud threat (e.g. *attorney, lawsuit, chargeback*), it is strictly classified into `ESCALATE` under `complaint_escalation` or the respective physical failure class.
2. **Action Intent over Inquiry**: If a customer asks "Why is my package damaged and can I get my money back?", the primary intent is annotated as `refund_request` if monetary return is the stated goal, or `damaged_item` if physical product exchange/proof is highlighted.
3. **Short / OOD Inputs**: Single words ("Hey", "?", "Help") are annotated as `unknown_other` with `AUTO_HANDLE` (directing the user to specify their inquiry).

---

## 4. Benchmark Limitations

1. **Brand-Specific Grounding**: The golden set reflects Amazon's e-commerce customer support workflows on social media (Twitter/X). While generalizable to retail, airline or telecom brands would require different intent taxonomies.
2. **Static Snapshot**: Social media support phrasing evolves with platform changes (e.g., changes in Twitter DM policy, character limit expansions).
3. **Synthetic / Real-World Blend**: To safeguard customer privacy and inject challenging boundary cases (legal threats, medical urgency), edge cases were synthesized to mirror empirical incident patterns rather than exposing unredacted customer PII.
