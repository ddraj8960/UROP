"""
Service S4a NLI Scorer Implementation.
Scores premise (passage) vs hypothesis (claim) entailment, neutral, and contradiction probabilities.
"""
from __future__ import annotations

import logging
import re
from typing import Any
from hmafact.schemas.graph import NLIScores

logger = logging.getLogger(__name__)

# Optional transformers NLI cross-encoder model
_NLI_PIPELINE = None


def _init_nli_pipeline() -> Any:
    global _NLI_PIPELINE
    if _NLI_PIPELINE is not None:
        return _NLI_PIPELINE

    try:
        from transformers import pipeline
        logger.info("Initializing DeBERTa NLI cross-encoder pipeline...")
        _NLI_PIPELINE = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
            device=-1,  # CPU
        )
        return _NLI_PIPELINE
    except Exception as e:
        logger.debug("Transformers NLI pipeline unavailable (%s); using heuristic fallback NLI scorer.", e)
        return None


def score_nli(premise: str, hypothesis: str) -> NLIScores:
    """
    Score NLI entailment between premise (evidence passage) and hypothesis (atomic claim).

    Args:
        premise: Passage text string.
        hypothesis: Claim text string.

    Returns:
        NLIScores object with entail, neutral, contradict probabilities.
    """
    p_clean = premise.lower().strip()
    h_clean = hypothesis.lower().strip()

    # Fast heuristic NLI scorer
    h_words = set(re.findall(r"\w+", h_clean)) - {"is", "was", "are", "were", "the", "a", "an", "in", "of", "to", "and"}
    p_words = set(re.findall(r"\w+", p_clean)) - {"is", "was", "are", "were", "the", "a", "an", "in", "of", "to", "and"}

    if not h_words:
        return NLIScores(entail=0.33, neutral=0.34, contradict=0.33)

    overlap = len(h_words.intersection(p_words)) / max(len(h_words), 1)

    # Check for direct negation or opposition markers
    negation_words = {"not", "no", "never", "didnt", "wasnt", "isnt", "cannot", "against", "refuses", "denies", "prohibits"}
    has_p_neg = bool(negation_words.intersection(p_words))
    has_h_neg = bool(negation_words.intersection(h_words))

    # Opposing term pair check
    opposing_pairs = [("against", "recommends"), ("against", "for"), ("false", "true"), ("deny", "confirm"), ("prohibit", "allow")]
    negation_conflict = has_p_neg != has_h_neg or any(
        (w1 in p_clean and w2 in h_clean) or (w2 in p_clean and w1 in h_clean)
        for w1, w2 in opposing_pairs
    )

    # Numeric / Year mismatch check
    p_years = set(re.findall(r"\b\d{4}\b", p_clean))
    h_years = set(re.findall(r"\b\d{4}\b", h_clean))
    numeric_mismatch = bool(h_years and p_years and not h_years.intersection(p_years))

    is_contradiction = negation_conflict or numeric_mismatch

    if overlap > 0.3 or is_contradiction:
        if is_contradiction:  # Contradiction flag
            entail = round(0.05 * overlap, 3)
            contradict = round(0.75 + 0.15 * max(overlap, 0.5), 3)
            neutral = round(1.0 - entail - contradict, 3)
        else:
            entail = round(0.60 + 0.35 * overlap, 3)
            contradict = round(0.05 * (1.0 - overlap), 3)
            neutral = round(1.0 - entail - contradict, 3)
    else:
        entail = round(0.15 * overlap, 3)
        contradict = round(0.15, 3)
        neutral = round(1.0 - entail - contradict, 3)

    return NLIScores(
        entail=max(0.0, min(1.0, entail)),
        neutral=max(0.0, min(1.0, neutral)),
        contradict=max(0.0, min(1.0, contradict)),
    )
