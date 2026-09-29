"""
Service S5: Time Sensitivity Detection Module (WP1).
Rule-based, deterministic detection of time-sensitive queries and claims.
"""
from __future__ import annotations

import re
from typing import Literal

TemporalScopeType = Literal["current", "past", "unspecified"]


def detect_time_sensitivity(text: str) -> tuple[bool, TemporalScopeType]:
    """
    Detects if text is time-sensitive and determines its temporal scope.

    Args:
        text: Question or claim text string.

    Returns:
        tuple of (is_time_sensitive: bool, temporal_scope: "current" | "past" | "unspecified")
    """
    clean_text = text.lower().strip()

    # Rule 1: Present-tense time-sensitive markers
    present_markers = [
        r"\bcurrent\b",
        r"\bcurrently\b",
        r"\bnow\b",
        r"\btoday\b",
        r"\bpresent\b",
        r"\blatest\b",
        r"\bincumbent\b",
        r"\bas of\b",
        r"\bat the moment\b",
    ]

    has_present_marker = any(re.search(p, clean_text) for p in present_markers)

    # Rule 2: Present-tense role-holder patterns
    role_pattern = r"\bis the (chief minister|cm|president|prime minister|ceo|governor|mayor|head|chairperson|minister)\b"
    has_present_role = bool(re.search(role_pattern, clean_text))

    # Rule 3: Past-tense markers
    past_markers = [
        r"\bformer\b",
        r"\bwas the\b",
        r"\bex-\b",
        r"\bin 19\d{2}\b",
        r"\bin 20[0-1]\d\b",  # Years up to 2019
    ]
    has_past_marker = any(re.search(p, clean_text) for p in past_markers)

    # Resolution logic
    if has_past_marker and not has_present_marker:
        return True, "past"
    elif has_present_marker or has_present_role:
        return True, "current"
    elif "in 202" in clean_text or "2026" in clean_text or "2025" in clean_text:
        return True, "current"
    else:
        return False, "unspecified"
