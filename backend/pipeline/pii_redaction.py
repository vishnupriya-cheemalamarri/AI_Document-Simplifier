"""
Regex-based PII redaction, applied before any text reaches the LLM or gets
embedded/indexed (see docs/ARCHITECTURE.md > Guardrails).

This is intentionally simple pattern matching, not a full PII-detection
model — it catches emails, phone numbers, and account/SSN-like numeric
patterns, which is what ARCHITECTURE.md specifies.
"""

import re
from typing import Tuple

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
ACCOUNT_RE = re.compile(r"(?<!\d)\d{12,19}(?!\d)")  # card/account-like digit runs
PHONE_RE = re.compile(r"(?<!\d)(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)")

# Order matters: match the more specific patterns (SSN, long account numbers)
# before the looser phone-number pattern so they aren't partially consumed by it.
PATTERNS = (
    (EMAIL_RE, "[REDACTED_EMAIL]"),
    (SSN_RE, "[REDACTED_SSN]"),
    (ACCOUNT_RE, "[REDACTED_ACCOUNT]"),
    (PHONE_RE, "[REDACTED_PHONE]"),
)


def redact_pii(text: str) -> Tuple[str, bool]:
    """Return (redacted_text, was_redacted)."""
    redacted = text
    was_redacted = False
    for pattern, placeholder in PATTERNS:
        redacted, n = pattern.subn(placeholder, redacted)
        if n:
            was_redacted = True
    return redacted, was_redacted
