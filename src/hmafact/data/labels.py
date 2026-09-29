"""
labels.py — One-way FEVER label → HMA-Fact canonical label map.
The reverse map (with PARTIALLY_SUPPORTED, CONFLICTING_EVIDENCE) is M7's responsibility.
"""
from typing import Literal

FEVERLabel = Literal["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
CanonicalLabel = Literal["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE",
                         "PARTIALLY_SUPPORTED", "CONFLICTING_EVIDENCE"]

FEVER_TO_CANONICAL: dict[str, CanonicalLabel] = {
    "SUPPORTS": "SUPPORTED",
    "REFUTES": "CONTRADICTED",
    "NOT ENOUGH INFO": "INSUFFICIENT_EVIDENCE",
}


def fever_to_canonical(label: str) -> CanonicalLabel:
    """Convert a FEVER gold label to the canonical HMA-Fact label."""
    if label not in FEVER_TO_CANONICAL:
        raise ValueError(f"Unknown FEVER label: '{label}'. Expected one of {list(FEVER_TO_CANONICAL)}")
    return FEVER_TO_CANONICAL[label]
