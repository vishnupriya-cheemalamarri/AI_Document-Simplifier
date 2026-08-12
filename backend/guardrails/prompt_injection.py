"""
Prompt injection detection via a DeBERTa-based text classifier
(see ARCHITECTURE.md > Guardrails).

IMPORTANT: config.PROMPT_INJECTION_MODEL_NAME
(protectai/deberta-v3-base-prompt-injection-v2) is flagged in
ARCHITECTURE.md as needing verification against HuggingFace before
being relied on for a demo — it has not been confirmed as of this writing.

If the model fails to load (wrong ID, no network, etc.) this module fails
OPEN: injection checks are skipped (logged as a warning) rather than
crashing the app, so a bad model ID degrades a guardrail instead of taking
down the whole demo. Fix the model ID and restart to re-enable the check.
"""

import logging
import threading
from typing import Optional

from .. import config

logger = logging.getLogger(__name__)

_classifier = None
_load_failed = False
_lock = threading.Lock()


def load_classifier():
    """Load the classifier once. Safe to call eagerly at app startup."""
    global _classifier, _load_failed
    if not config.ENABLE_PROMPT_INJECTION_CHECK:
        return None
    if _classifier is None and not _load_failed:
        with _lock:
            if _classifier is None and not _load_failed:
                try:
                    from transformers import pipeline

                    _classifier = pipeline(
                        "text-classification", model=config.PROMPT_INJECTION_MODEL_NAME
                    )
                except Exception:
                    logger.warning(
                        "Prompt injection classifier %r failed to load. Failing OPEN: "
                        "injection checks are disabled until this is fixed. Verify the "
                        "model ID against HuggingFace (see ARCHITECTURE.md).",
                        config.PROMPT_INJECTION_MODEL_NAME,
                        exc_info=True,
                    )
                    _load_failed = True
    return _classifier


def check_prompt_injection(text: str) -> dict:
    """Return {flagged, label, score, checked}. checked=False means the
    classifier is unavailable and the check was skipped (fail-open)."""
    classifier = load_classifier()
    if classifier is None:
        return {"flagged": False, "label": None, "score": None, "checked": False}

    result = classifier(text, truncation=True)[0]
    label = result["label"]
    score = float(result["score"])
    # Match loosely on label text rather than an exact enum, since the model
    # card's exact label set hasn't been verified (see module docstring).
    is_injection_label = "injection" in label.lower() or label.lower() in {"label_1", "malicious"}
    flagged = is_injection_label and score >= config.PROMPT_INJECTION_THRESHOLD
    return {"flagged": flagged, "label": label, "score": score, "checked": True}
