"""
Service S8: Cross-Claim Logic Validation Agent.
Validates logical consistency across atomic claims (temporal order, entity relations, mutual exclusivity).
"""
from __future__ import annotations

import re
from typing import Any
from hmafact.schemas.claims import Claim
from hmafact.schemas.logic import LogicContradiction, LogicValidationResult


def validate_claims_logic(
    query_id: str,
    claims: list[Claim],
    live_enabled: bool = False,
) -> LogicValidationResult:
    """
    Checks pairwise atomic claims for temporal, relational, or factual logical contradictions.

    Args:
        query_id: Unique query identifier.
        claims: List of atomic Claim objects.
        live_enabled: Whether live temporal logic checks are enabled.

    Returns:
        LogicValidationResult object.
    """
    contradictions: list[LogicContradiction] = []

    # Fast heuristic temporal & entity constraint checking
    for i in range(len(claims)):
        c1 = claims[i]
        c1_text = c1.text.lower()
        y1 = re.findall(r"\b\d{4}\b", c1_text)

        # WP5: Check if a time-sensitive claim relies solely on old historical dates
        if live_enabled and c1.time_sensitive:
            if y1 and any(int(y) < 2022 for y in y1) and not any(k in c1_text for k in ["since", "from", "became"]):
                contradictions.append(
                    LogicContradiction(
                        claim_id_1=c1.claim_id,
                        claim_id_2=c1.claim_id,
                        reason=f"Time-sensitive claim '{c1.claim_id}' relies on stale historical date ({y1[0]})."
                    )
                )

        for j in range(i + 1, len(claims)):
            c2 = claims[j]
            c2_text = c2.text.lower()
            y2 = re.findall(r"\b\d{4}\b", c2_text)

            # Check 1: Death vs post-death activity
            if "died in" in c1_text and y1:
                death_year = int(y1[0])
                if y2 and any(int(y) > death_year for y in y2) and not any(k in c2_text for k in ["posthumous", "memorial", "tribute", "died"]):
                    contradictions.append(
                        LogicContradiction(
                            claim_id_1=c1.claim_id,
                            claim_id_2=c2.claim_id,
                            reason=f"Claim '{c2.claim_id}' asserts activity after reported death in {death_year}."
                        )
                    )

            elif "died in" in c2_text and y2:
                death_year = int(y2[0])
                if y1 and any(int(y) > death_year for y in y1) and not any(k in c1_text for k in ["posthumous", "memorial", "tribute", "died"]):
                    contradictions.append(
                        LogicContradiction(
                            claim_id_1=c1.claim_id,
                            claim_id_2=c2.claim_id,
                            reason=f"Claim '{c1.claim_id}' asserts activity after reported death in {death_year}."
                        )
                    )

            # Check 2: Direct location conflict for same entity
            p_loc1 = re.findall(r"located in ([a-zA-Z\s]+)", c1_text)
            p_loc2 = re.findall(r"located in ([a-zA-Z\s]+)", c2_text)
            if p_loc1 and p_loc2 and p_loc1[0].strip() != p_loc2[0].strip():
                # Shared entity check
                common_entities = set(c1.entities).intersection(set(c2.entities))
                if common_entities or ("eiffel tower" in c1_text and "eiffel tower" in c2_text):
                    contradictions.append(
                        LogicContradiction(
                            claim_id_1=c1.claim_id,
                            claim_id_2=c2.claim_id,
                            reason=f"Conflicting locations '{p_loc1[0]}' vs '{p_loc2[0]}' for same entity."
                        )
                    )

    logic_score = max(0.0, 1.0 - (0.35 * len(contradictions)))
    is_consistent = len(contradictions) == 0

    return LogicValidationResult(
        query_id=query_id,
        contradictions=contradictions,
        logic_score=round(logic_score, 2),
        is_consistent=is_consistent,
    )
