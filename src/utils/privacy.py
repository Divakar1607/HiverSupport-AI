import re

# Compiled regex patterns for PII detection
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_REGEX = re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
URL_REGEX = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*')
ORDER_ID_REGEX = re.compile(r'\b(?:\d{3}-\d{3,7}-\d{3,7}|[A-Z0-9]{8,14}|\#\d{4,10})\b')
TWITTER_USER_REGEX = re.compile(r'@\d{3,10}')  # Kaggle anonymized user handles like @115712

def anonymize_text(text: str) -> str:
    """
    Sanitizes and masks PII in text:
    - Email addresses -> [EMAIL]
    - Phone numbers -> [PHONE]
    - URLs -> [URL]
    - Order/Account IDs -> [ORDER_ID]
    - Anonymized Twitter handles -> @customer
    """
    if not isinstance(text, str):
        return ""
        
    cleaned = EMAIL_REGEX.sub("[EMAIL]", text)
    cleaned = PHONE_REGEX.sub("[PHONE]", cleaned)
    cleaned = URL_REGEX.sub("[URL]", cleaned)
    cleaned = ORDER_ID_REGEX.sub("[ORDER_ID]", cleaned)
    cleaned = TWITTER_USER_REGEX.sub("@customer", cleaned)
    
    # Strip excess whitespace
    cleaned = " ".join(cleaned.split())
    return cleaned
