import argparse
import sys
from pathlib import Path
from typing import List, Dict

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.io import get_project_root
from src.utils.logging import get_logger

logger = get_logger("label_golden")

# Curated stratified 200 benchmark examples designed with explicit difficulty tiers,
# diverse phrasing, edge-cases, high-risk flags, and ambiguous queries.
DEFAULT_GOLDEN_EXAMPLES: List[Dict[str, str]] = [
    # --- 1. DELIVERY_DELAY (25 examples: 10 Easy, 8 Medium, 7 Hard/Escalate) ---
    {"message": "Where is my package? Tracking says delivered but nothing is here!", "intent": "delivery_delay", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Standard package tracking inquiry with available historical resolution."},
    {"message": "My order was supposed to arrive by 8 PM yesterday, still no delivery.", "intent": "delivery_delay", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Clear late delivery query."},
    {"message": "Package tracking status hasn't updated in 4 days. Is it lost in transit?", "intent": "delivery_delay", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Stuck tracking inquiry."},
    {"message": "Driver says handed directly to resident but I was at work all day.", "intent": "delivery_delay", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Misdelivery claim; standard advice is check property then contact support."},
    {"message": "It's been two weeks and my parcel is still stuck at the regional sort facility.", "intent": "delivery_delay", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Transit delay inquiry."},
    {"message": "Package marked delivered on porch but building has no porch, it's an apartment high-rise.", "intent": "delivery_delay", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Misdelivery at wrong building structure."},
    {"message": "Guaranteed 1-day shipping failed for the third time this week!", "intent": "delivery_delay", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Prime delivery promise breach."},
    {"message": "Tracking number 9400111899562537612 shows delivery attempted but gate code was provided.", "intent": "delivery_delay", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Delivery access issue."},
    {"message": "Is there any way to expedite my shipment after dispatch?", "intent": "delivery_delay", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Post-dispatch speed request."},
    {"message": "Delivery attempted notice left but nobody rang the bell.", "intent": "delivery_delay", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Missed delivery attempt inquiry."},
    {"message": "Where is my stuff?! You promised Tuesday!", "intent": "delivery_delay", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Colloquial late delivery query."},
    {"message": "My meds were inside that package and temperature sensitive. Need this immediately or I will be hospitalized.", "intent": "delivery_delay", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Medical emergency / high-risk situation requiring human intervention."},
    {"message": "Driver literally threw my package onto the roof, how do I get it down?", "intent": "delivery_delay", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Bizarre delivery mishap needing specialized dispatch review."},
    {"message": "Tracking says delivered to mailroom but receptionist states no courier arrived.", "intent": "delivery_delay", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Commercial address misplacement."},
    {"message": "Can I change delivery address while package is out for delivery with courier?", "intent": "delivery_delay", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "In-transit rerouting request."},
    {"message": "Package stolen off my porch, I have security cam footage of the thief.", "intent": "delivery_delay", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Theft / police report claim needing manual claims handling."},
    {"message": "Delayed again? This is getting ridiculous.", "intent": "delivery_delay", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Short frustrated delivery complaint."},
    {"message": "Courier marked business closed at 2 PM on a weekday.", "intent": "delivery_delay", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Delivery attempt dispute."},
    {"message": "My parcel says delivered to neighbor in unit 4B but unit 4B is vacant.", "intent": "delivery_delay", "expected_action": "AUTO_HANDLE", "difficulty": "HARD", "evidence_available": "TRUE", "notes": "Complex delivery anomaly."},
    {"message": "Delivery driver backed into my driveway gate and broke it while delivering!", "intent": "delivery_delay", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Property damage claim; strictly escalate."},
    {"message": "Expected delivery was 3 days ago, no status update since.", "intent": "delivery_delay", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Late shipment inquiry."},
    {"message": "Can I pick up the package from the carrier hub directly instead of waiting?", "intent": "delivery_delay", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Hub pickup inquiry."},
    {"message": "Tracking shows returned to sender because address was illegible.", "intent": "delivery_delay", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Return to sender exception."},
    {"message": "Package was left in the rain and ruined.", "intent": "delivery_delay", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Weather damage on delivery."},
    {"message": "Still waiting... order 402-9981241-11234", "intent": "delivery_delay", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Short inquiry with order ID."},

    # --- 2. REFUND_REQUEST (25 examples) ---
    {"message": "I was charged twice for my Prime membership this month! Please refund.", "intent": "refund_request", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Classic duplicate subscription charge."},
    {"message": "My refund has not arrived in my bank account yet. It has been 10 business days.", "intent": "refund_request", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Standard refund turnaround inquiry."},
    {"message": "Returned my item over a week ago, when will I get my money back?", "intent": "refund_request", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Return refund timeline."},
    {"message": "You promised a promotional credit refund of $20 and it is not on my statement.", "intent": "refund_request", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Promotional balance dispute."},
    {"message": "I will file a chargeback with my bank and report you for fraud if my $1200 refund is not issued today!", "intent": "refund_request", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Chargeback / fraud threat; high risk policy escalation."},
    {"message": "Why was a restocking fee deducted from my refund? Item was defective!", "intent": "refund_request", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Restocking fee dispute."},
    {"message": "Can I get a refund to a different credit card since my original card was closed?", "intent": "refund_request", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Closed card refund destination."},
    {"message": "Refund issued as gift card balance instead of original payment method.", "intent": "refund_request", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Refund payment method correction."},
    {"message": "My attorney advised me to contact you one last time before initiating legal recovery for this withheld refund.", "intent": "refund_request", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Legal attorney mention; mandatory escalation."},
    {"message": "Order was cancelled by seller but charge is still pending on my debit card.", "intent": "refund_request", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Pending authorization drop explanation."},
    {"message": "Where is my money?? Order 114-8871923-9912.", "intent": "refund_request", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Aggressive but standard refund tracking."},
    {"message": "Item price dropped by $30 the day after I bought it. Can I get a price difference refund?", "intent": "refund_request", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Post-purchase price match policy."},
    {"message": "I returned 3 items in one box but only got refunded for 1.", "intent": "refund_request", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Multi-item consolidated return discrepancy needing manual investigation."},
    {"message": "Accidentally bought Kindle book, haven't read it, please refund.", "intent": "refund_request", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Digital Kindle return within window."},
    {"message": "Bank says Amazon never sent the refund transaction batch.", "intent": "refund_request", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "ARN/trace number request."},
    {"message": "Charged for annual plan when I only signed up for free trial.", "intent": "refund_request", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Trial conversion refund."},
    {"message": "Unauthorized recurring charges appearing on my credit card every month.", "intent": "refund_request", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Suspected fraudulent billing."},
    {"message": "How long does a refund to PayPal take compared to bank debit?", "intent": "refund_request", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Payment method timeframe comparison."},
    {"message": "My gift card refund didn't credit back to my account balance.", "intent": "refund_request", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Gift card balance reconciliation."},
    {"message": "Representative promised a full refund in chat yesterday but no email confirmation arrived.", "intent": "refund_request", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Agent commitment verification."},
    {"message": "Can I decline an exchange and request a straight refund instead?", "intent": "refund_request", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Exchange vs refund preference."},
    {"message": "Charged $99 for something I never purchased in my life.", "intent": "refund_request", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Unauthorized transaction claim."},
    {"message": "Received partial refund of $14 instead of the full $65.", "intent": "refund_request", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Partial refund inquiry."},
    {"message": "Refund status says complete in portal but zero credit on bank app.", "intent": "refund_request", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Portal vs bank settlement delay."},
    {"message": "Money back please.", "intent": "refund_request", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Terse refund demand."},

    # --- 3. CANCELLATION_REQUEST (20 examples) ---
    {"message": "Can you help me cancel my order 112-984712-441? It was an accidental purchase.", "intent": "cancellation_request", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Standard self-service order cancellation."},
    {"message": "I placed an order 10 minutes ago and need to cancel it right now.", "intent": "cancellation_request", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Immediate post-order cancellation."},
    {"message": "Tried cancelling through the app but the cancel button is greyed out.", "intent": "cancellation_request", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Cancellation button disabled (preparing for shipment)."},
    {"message": "How do I cancel my Subscribe & Save scheduled delivery for next week?", "intent": "cancellation_request", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Subscribe & Save cancellation guidance."},
    {"message": "I want to cancel only item 2 in my order, not the entire basket.", "intent": "cancellation_request", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Partial line-item cancellation."},
    {"message": "Please stop shipment immediately! Wrong address entered by mistake.", "intent": "cancellation_request", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Urgent address mistake cancellation."},
    {"message": "Cancel order #8841-2991.", "intent": "cancellation_request", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Terse cancellation command."},
    {"message": "Order says 'Preparing for dispatch', can customer support force a cancellation?", "intent": "cancellation_request", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Warehouse dispatch lock explanation."},
    {"message": "I requested cancellation 2 hours ago but just received a shipping confirmation notification.", "intent": "cancellation_request", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Race condition cancellation advice (refuse delivery)."},
    {"message": "How do I cancel a digital software preorder before release date?", "intent": "cancellation_request", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Digital preorder cancellation."},
    {"message": "Cancel my Amazon Music Unlimited subscription right now.", "intent": "cancellation_request", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Subscription cancellation."},
    {"message": "Accidentally bought 10 units instead of 1, please cancel 9 units.", "intent": "cancellation_request", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Quantity mistake cancellation."},
    {"message": "If this custom engraved order cannot be cancelled, I will contest the transaction.", "intent": "cancellation_request", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Custom goods non-cancellable dispute with threat."},
    {"message": "My toddler ordered a movie on Fire TV, cancel it immediately.", "intent": "cancellation_request", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Accidental VOD purchase cancellation."},
    {"message": "Third-party seller ignored my cancellation request and shipped anyway.", "intent": "cancellation_request", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Marketplace seller dispute policy."},
    {"message": "I need to cancel an order placed with a gift card before it redeems.", "intent": "cancellation_request", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Gift card tender reversal."},
    {"message": "Can I cancel an order while it is in transit?", "intent": "cancellation_request", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "In-transit cancellation clarification."},
    {"message": "Is there a penalty or fee for cancelling an order prior to delivery?", "intent": "cancellation_request", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Cancellation policy clarification."},
    {"message": "Can someone cancel my scheduled grocery delivery slot for tomorrow morning?", "intent": "cancellation_request", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Fresh slot cancellation."},
    {"message": "Please cancel.", "intent": "cancellation_request", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Very short inquiry."},

    # --- 4. DAMAGED_ITEM (20 examples) ---
    {"message": "My package arrived damaged. The box was crushed and the item inside is broken.", "intent": "damaged_item", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Standard damaged item complaint."},
    {"message": "The ceramic plates I ordered arrived shattered into hundreds of pieces.", "intent": "damaged_item", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Broken fragile goods."},
    {"message": "Opened the box and the perfume bottle leaked everywhere, completely empty.", "intent": "damaged_item", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Liquid spill / damaged goods."},
    {"message": "Brand new TV screen has a huge internal crack right down the middle when powered on.", "intent": "damaged_item", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "High value damaged electronics replacement."},
    {"message": "Received an opened box with missing parts and scratches. This was sold as brand new!", "intent": "damaged_item", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Used item sold as new / missing parts."},
    {"message": "The lithium battery inside the electronic device started smoking and melted the plastic casing!", "intent": "damaged_item", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Safety hazard / fire risk; mandatory high-priority escalation."},
    {"message": "Clothing item arrived stained with grease.", "intent": "damaged_item", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Soiled apparel replacement."},
    {"message": "Book cover is torn and several pages are ripped out.", "intent": "damaged_item", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Damaged print book."},
    {"message": "Box was soaking wet and disintegrating when delivered by the carrier.", "intent": "damaged_item", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Carrier water damage."},
    {"message": "Laptop turns on but half the keyboard keys do not register.", "intent": "damaged_item", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Defective hardware."},
    {"message": "Broken glass cut my child's hand when opening the crushed delivery box!", "intent": "damaged_item", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Physical injury caused by damaged shipment; mandatory escalation."},
    {"message": "Item arrived without the power adapter, cannot use it.", "intent": "damaged_item", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Missing accessory."},
    {"message": "Food package seal was broken and expiration date was last month.", "intent": "damaged_item", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Expired / tampered grocery item."},
    {"message": "Do I have to return the broken glass shards to get a refund?", "intent": "damaged_item", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Hazardous return exception policy."},
    {"message": "Watch glass is badly scratched out of the sealed box.", "intent": "damaged_item", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Cosmetic defect on arrival."},
    {"message": "Guitar neck snapped in half during transit.", "intent": "damaged_item", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Severe transit breakage."},
    {"message": "Second replacement item arrived damaged just like the first one!", "intent": "damaged_item", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Repeated fulfillment failure; customer frustration."},
    {"message": "Shoes sent were scuffed and clearly worn outside.", "intent": "damaged_item", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Used merchandise return."},
    {"message": "Monitor has 5 dead pixels in the center of the display.", "intent": "damaged_item", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Electronic defect tolerance inquiry."},
    {"message": "Crushed box, broken inside.", "intent": "damaged_item", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Short damaged goods note."},

    # --- 5. ACCOUNT_ACCESS (20 examples) ---
    {"message": "I cannot log into my account. It says password incorrect even after reset.", "intent": "account_access", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Password reset loop issue."},
    {"message": "Not receiving the 2-step verification OTP code on my mobile phone.", "intent": "account_access", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "2FA SMS OTP delivery delay."},
    {"message": "My account has been locked due to suspicious activity, please help unlock it.", "intent": "account_access", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Account security lock recovery steps."},
    {"message": "I believe my account was hacked, unauthorized orders were placed in another country!", "intent": "account_access", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Account takeover / fraud in progress; urgent escalation."},
    {"message": "How do I change the phone number linked to my two-factor authentication?", "intent": "account_access", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Phone number update guidance."},
    {"message": "No longer have access to my old email address, how can I regain access to my account?", "intent": "account_access", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Email loss account verification flow."},
    {"message": "Locked out of my Prime account on Apple TV, keeps asking for QR code verification.", "intent": "account_access", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Smart TV device registration."},
    {"message": "Someone changed my account password and registered phone without my permission.", "intent": "account_access", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Compromised account credential hijack."},
    {"message": "Why does the login page keep asking me to solve captchas over and over?", "intent": "account_access", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Captcha loop browser troubleshooting."},
    {"message": "Can I merge two separate Amazon accounts under one email?", "intent": "account_access", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Account merging policy."},
    {"message": "Authenticator app died with my broken phone, cannot generate backup 2FA code.", "intent": "account_access", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "2FA backup recovery procedure."},
    {"message": "How do I permanently delete my account and all associated personal data?", "intent": "account_access", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "GDPR / privacy account closure."},
    {"message": "Keep getting logged out every 5 minutes while browsing products.", "intent": "account_access", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Cookie / session timeout troubleshooting."},
    {"message": "Account suspended with active orders pending, what do I do?", "intent": "account_access", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Account suspension appeal."},
    {"message": "Is there a way to view active login sessions across devices?", "intent": "account_access", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Device management settings."},
    {"message": "Password reset link emailed to me says expired 10 seconds after clicking.", "intent": "account_access", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Expired token glitch."},
    {"message": "Can two family members share one Prime login legally?", "intent": "account_access", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Amazon Household feature advice."},
    {"message": "System says my phone number is already registered to another user.", "intent": "account_access", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Recycled phone number conflict."},
    {"message": "Can't log in at all.", "intent": "account_access", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Short login complaint."},
    {"message": "Locked out.", "intent": "account_access", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Terse login issue."},

    # --- 6. RETURN_POLICY_INQUIRY (20 examples) ---
    {"message": "How do I return a gift that I received without notifying the sender?", "intent": "return_policy_inquiry", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Gift return procedure."},
    {"message": "Where do I drop off my return package? Do I need to print a label?", "intent": "return_policy_inquiry", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Label-free UPS / Kohl's return instructions."},
    {"message": "What is the return window for electronics purchased during the holiday season?", "intent": "return_policy_inquiry", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Holiday extended return window query."},
    {"message": "Can I return an opened mattress if it's within the 30 day trial period?", "intent": "return_policy_inquiry", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Bulky item return policy."},
    {"message": "Is return shipping free for Prime members or do I have to pay postage?", "intent": "return_policy_inquiry", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Return postage fees inquiry."},
    {"message": "Return QR code expired before I could make it to the drop-off locker.", "intent": "return_policy_inquiry", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Return QR code regeneration."},
    {"message": "Can I return an item without the original manufacturer cardboard box?", "intent": "return_policy_inquiry", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Original packaging requirement."},
    {"message": "Can I return beauty products or cosmetics after breaking the seal?", "intent": "return_policy_inquiry", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Health and personal care return exceptions."},
    {"message": "Can I return an item purchased from Amazon Warehouse Deals?", "intent": "return_policy_inquiry", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Warehouse deals return eligibility."},
    {"message": "How do I schedule a carrier home pickup for a heavy return parcel?", "intent": "return_policy_inquiry", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "UPS pickup return option."},
    {"message": "I lost my return tracking receipt from the drop-off counter.", "intent": "return_policy_inquiry", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Drop-off receipt loss guidance."},
    {"message": "Can international orders be returned using local domestic post?", "intent": "return_policy_inquiry", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "International return shipping."},
    {"message": "Are downloadable software codes eligible for return or exchange?", "intent": "return_policy_inquiry", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Digital software non-returnable policy."},
    {"message": "What happens if I accidentally return the wrong item in the return box?", "intent": "return_policy_inquiry", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Wrong item returned process."},
    {"message": "How many days do I have to ship the return after printing the label?", "intent": "return_policy_inquiry", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Return label validity window."},
    {"message": "Can I return an item bought with an promotional gift code?", "intent": "return_policy_inquiry", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Promotional balance return rules."},
    {"message": "Can I return food items or groceries?", "intent": "return_policy_inquiry", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Grocery non-returnable refund policy."},
    {"message": "Does Amazon provide a printed return label if I don't own a printer?", "intent": "return_policy_inquiry", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Printerless QR code drop-off."},
    {"message": "How to return an item?", "intent": "return_policy_inquiry", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Short return inquiry."},
    {"message": "Return policy details.", "intent": "return_policy_inquiry", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Terse return keyword."},

    # --- 7. TECHNICAL_PROBLEM (20 examples) ---
    {"message": "Streaming quality on Prime Video is buffering constantly on my TV.", "intent": "technical_problem", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Video buffering troubleshooting."},
    {"message": "The mobile app crashes every time I click proceed to checkout on iOS 17.", "intent": "technical_problem", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "App crash at checkout."},
    {"message": "Getting error code 500 when trying to view my shopping cart on Google Chrome.", "intent": "technical_problem", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "HTTP 500 error clearing cache."},
    {"message": "Alexa won't connect to my home Wi-Fi network after router restart.", "intent": "technical_problem", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Smart speaker connectivity issue."},
    {"message": "Kindle Paperwhite screen is frozen on the screensaver and not responding to power button.", "intent": "technical_problem", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Kindle hard reboot steps."},
    {"message": "Credit card payment gateway keeps failing with 'Payment revision needed' even though card is valid.", "intent": "technical_problem", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Payment authorization error."},
    {"message": "Search bar on the website is returning zero results for every keyword today.", "intent": "technical_problem", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Website search outage."},
    {"message": "Prime Video subtitles are out of sync by 10 seconds on season 3.", "intent": "technical_problem", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Audio/subtitle sync issue."},
    {"message": "App says 'Unable to load orders' whenever I refresh the tab.", "intent": "technical_problem", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Network connection error in app."},
    {"message": "Purchased audiobook won't download in Audible app, stuck at 0%.", "intent": "technical_problem", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Audible download stall."},
    {"message": "Two-factor SMS arrives 45 minutes late, always invalid.", "intent": "technical_problem", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Carrier SMS gateway latency."},
    {"message": "One-click ordering button disappeared from product pages.", "intent": "technical_problem", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "1-Click ordering toggle."},
    {"message": "Fire TV stick is in an endless boot loop showing only logo.", "intent": "technical_problem", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Fire TV boot recovery."},
    {"message": "Can't play downloaded offline videos on iPad during flight.", "intent": "technical_problem", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "DRM license check for offline media."},
    {"message": "Website displays in Spanish even though my language setting is English.", "intent": "technical_problem", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Language preference reset."},
    {"message": "Item details page won't let me select sizes from the dropdown.", "intent": "technical_problem", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Browser JS glitch."},
    {"message": "Prime Video 4K HDR playback defaults to 480p on gigabit fiber internet.", "intent": "technical_problem", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "HDCP / streaming quality negotiation."},
    {"message": "Cannot apply promo code at checkout, error says invalid format.", "intent": "technical_problem", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Promo coupon formatting issue."},
    {"message": "App crashed.", "intent": "technical_problem", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Short app crash report."},
    {"message": "Your website is broken.", "intent": "technical_problem", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Terse glitch report."},

    # --- 8. PRODUCT_QUESTION (15 examples) ---
    {"message": "I need an official VAT invoice for my business purchase.", "intent": "product_question", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "VAT invoice download instructions."},
    {"message": "When will the 256GB space grey model be back in stock?", "intent": "product_question", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Stock restocking alert notification."},
    {"message": "Does this coffee maker work on 220V European voltage or only 110V?", "intent": "product_question", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Electrical spec verification."},
    {"message": "Is this refurbished product backed by the 1-year Amazon Renewed guarantee?", "intent": "product_question", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Renewed guarantee warranty terms."},
    {"message": "Can I buy replacement ear cushions for this headphone model?", "intent": "product_question", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Spare parts compatibility."},
    {"message": "How do I download the manufacturer user manual for order #112-998?", "intent": "product_question", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Product documentation link."},
    {"message": "Does the warranty cover accidental spills and drops on this tablet?", "intent": "product_question", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "SquareTrade / Asurion protection terms."},
    {"message": "Is this seller an authorized dealer for Sony cameras?", "intent": "product_question", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Authorized reseller inquiry."},
    {"message": "What is the expiration date on this pack of protein powder before I order?", "intent": "product_question", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Perishable expiration query."},
    {"message": "Can I trade in my old iPhone for gift card credit on this product page?", "intent": "product_question", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Amazon Trade-In program eligibility."},
    {"message": "Are the dimensions in the listing inclusive of the mounting brackets?", "intent": "product_question", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Product specification details."},
    {"message": "Need tax exemption certificate applied to my business account.", "intent": "product_question", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Tax exemption registration."},
    {"message": "Will this case fit the iPhone 15 Pro or only standard 15?", "intent": "product_question", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Phone accessory sizing."},
    {"message": "Can I get an invoice?", "intent": "product_question", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Short invoice query."},
    {"message": "Product warranty details?", "intent": "product_question", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Terse warranty inquiry."},

    # --- 9. COMPLAINT_ESCALATION (20 examples: heavily ESCALATE & High Risk) ---
    {"message": "Your customer service agent hung up on me earlier. Terrible experience.", "intent": "complaint_escalation", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Agent misconduct complaint; requires supervisor escalation."},
    {"message": "I have been transferred to 5 different representatives and no one solved my issue!", "intent": "complaint_escalation", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Repeated transfer failure."},
    {"message": "I demand to speak with an executive supervisor or manager immediately.", "intent": "complaint_escalation", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Direct supervisor demand."},
    {"message": "I am filing a formal complaint with the Federal Trade Commission (FTC) and the Better Business Bureau (BBB) regarding your deceptive practices!", "intent": "complaint_escalation", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Regulatory / FTC / BBB complaint threat."},
    {"message": "My lawyer is drafting a lawsuit for breach of contract and deceptive trade practices.", "intent": "complaint_escalation", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Lawsuit threat; legal risk trigger."},
    {"message": "Your delivery driver cursed at my wife and shoved our gate!", "intent": "complaint_escalation", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Physical altercation / driver hostility."},
    {"message": "I have been a loyal Prime member for 14 years and you treat me like garbage.", "intent": "complaint_escalation", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "General venting; can be empathetic support auto-handled with standard DM invite."},
    {"message": "Worst service I have ever experienced in my entire life.", "intent": "complaint_escalation", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Generic dissatisfaction expression."},
    {"message": "If this is not resolved within 1 hour I will make sure this goes viral on TikTok and Twitter.", "intent": "complaint_escalation", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "PR viral threat."},
    {"message": "An agent promised me a compensation credit in writing and now another agent says that agent lied.", "intent": "complaint_escalation", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Agent dispute / broken promise."},
    {"message": "This company is a complete scam and thieves!", "intent": "complaint_escalation", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Standard social media venting."},
    {"message": "I will contact local consumer protection bureau unless an account manager calls me.", "intent": "complaint_escalation", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Consumer protection escalation threat."},
    {"message": "I am utterly disgusted with the lack of help from your chat bots.", "intent": "complaint_escalation", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Bot dissatisfaction, auto-route to human DM."},
    {"message": "Your warehouse sent dangerous contaminated products and refused to apologize.", "intent": "complaint_escalation", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Contamination hazard complaint."},
    {"message": "Three weeks of waiting and 10 phone calls with zero resolution.", "intent": "complaint_escalation", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Chronic unresolved ticket."},
    {"message": "Rep was rude and unhelpful.", "intent": "complaint_escalation", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Brief agent complaint."},
    {"message": "Horrible experience today.", "intent": "complaint_escalation", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Short negative feedback."},
    {"message": "I need a senior escalation specialist, not Tier 1 support.", "intent": "complaint_escalation", "expected_action": "ESCALATE", "difficulty": "HARD", "evidence_available": "FALSE", "notes": "Senior tier demand."},
    {"message": "Cancel everything and delete my profile, I'm never shopping here again.", "intent": "complaint_escalation", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Angry churn threat."},
    {"message": "Disgraceful support.", "intent": "complaint_escalation", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Short complaint."},

    # --- 10. UNKNOWN_OTHER (15 examples) ---
    {"message": "Hello is anyone there today?", "intent": "unknown_other", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Greeting with no support context."},
    {"message": "Just wanted to say shoutout to whoever packed my order, great job!", "intent": "unknown_other", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Compliment / positive feedback."},
    {"message": "Can you guys sell pet tigers in the future? Asking for a friend haha.", "intent": "unknown_other", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "FALSE", "notes": "Joke / nonsensical inquiry."},
    {"message": "What is the meaning of life?", "intent": "unknown_other", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "FALSE", "notes": "Philosophical query."},
    {"message": "Testing 1 2 3.", "intent": "unknown_other", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "FALSE", "notes": "Test message."},
    {"message": "Good morning team!", "intent": "unknown_other", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Morning greeting."},
    {"message": "Why is the sky blue?", "intent": "unknown_other", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "FALSE", "notes": "Out of domain trivia."},
    {"message": "👍", "intent": "unknown_other", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "FALSE", "notes": "Single emoji."},
    {"message": "Anybody working right now?", "intent": "unknown_other", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Availability check."},
    {"message": "Check your DMs please.", "intent": "unknown_other", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "TRUE", "notes": "Meta DM notification without topic."},
    {"message": "Hey", "intent": "unknown_other", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "One word greeting."},
    {"message": "asdfghjk", "intent": "unknown_other", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "FALSE", "notes": "Gibberish string."},
    {"message": "Do you like pineapple on pizza?", "intent": "unknown_other", "expected_action": "AUTO_HANDLE", "difficulty": "MEDIUM", "evidence_available": "FALSE", "notes": "Casual banter."},
    {"message": "Can someone help me?", "intent": "unknown_other", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "TRUE", "notes": "Vague help plea with no topic."},
    {"message": "?", "intent": "unknown_other", "expected_action": "AUTO_HANDLE", "difficulty": "EASY", "evidence_available": "FALSE", "notes": "Single punctuation mark."}
]

def build_golden_set():
    root = get_project_root()
    golden_dir = root / "data" / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    for idx, ex in enumerate(DEFAULT_GOLDEN_EXAMPLES, start=1):
        records.append({
            "id": f"gold_{idx:03d}",
            "conversation_id": f"gold_conv_{idx:03d}",
            "message": ex["message"],
            "intent": ex["intent"],
            "expected_action": ex["expected_action"],
            "difficulty": ex["difficulty"],
            "evidence_available": ex["evidence_available"],
            "notes": ex["notes"]
        })
        
    df = pd.DataFrame(records)
    out_path = golden_dir / "golden_set.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Built golden evaluation dataset with {len(df)} curated examples at {out_path}")
    
    print("\n" + "="*60)
    print("GOLDEN SET SUMMARY STATISTICS")
    print("="*60)
    print(f"Total benchmark examples: {len(df)}")
    print("\nBreakdown by Intent:")
    print(df["intent"].value_counts().to_string())
    print("\nBreakdown by Expected Action:")
    print(df["expected_action"].value_counts().to_string())
    print("\nBreakdown by Difficulty Tier:")
    print(df["difficulty"].value_counts().to_string())
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Golden evaluation dataset builder and labeling CLI.")
    parser.add_argument("--build-default", action="store_true", default=True, help="Build the standardized 200-example golden set.")
    args = parser.parse_args()
    
    if args.build_default:
        build_golden_set()

if __name__ == "__main__":
    main()
