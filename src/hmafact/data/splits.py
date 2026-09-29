"""
splits.py — Pure allocation and stratified-split functions with no I/O.
All functions are deterministic given the same seed.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, TypeVar

T = TypeVar("T")


class AllocationError(Exception):
    pass


def allocate(strata_sizes: dict[str, int], total: int) -> dict[str, int]:
    """
    Distribute `total` items across strata proportionally using the
    largest-remainder method. Raises AllocationError if any quota
    would exceed its stratum size.

    Example:
        allocate({"bridge": 5921, "comparison": 1484}, 400)
        -> {"bridge": 320, "comparison": 80}
    """
    if total < 0:
        raise AllocationError(f"total must be >= 0, got {total}")

    grand_total = sum(strata_sizes.values())
    if total > grand_total:
        raise AllocationError(
            f"Requested total ({total}) exceeds sum of all strata ({grand_total}). "
            "Not enough items to sample."
        )
    if grand_total == 0:
        raise AllocationError("All strata are empty.")

    # Compute exact (float) quotas
    exact: dict[str, float] = {k: (v / grand_total) * total for k, v in strata_sizes.items()}

    # Floor allocation
    floors: dict[str, int] = {k: int(v) for k, v in exact.items()}
    remainders: dict[str, float] = {k: exact[k] - floors[k] for k in exact}

    # Distribute remainder seats by largest fractional part
    seats_left = total - sum(floors.values())
    sorted_keys = sorted(remainders, key=lambda k: remainders[k], reverse=True)
    result: dict[str, int] = dict(floors)
    for k in sorted_keys[:seats_left]:
        result[k] += 1

    # Validate quotas fit within strata
    for k, quota in result.items():
        if quota > strata_sizes[k]:
            raise AllocationError(
                f"Stratum '{k}' has {strata_sizes[k]} items but requires {quota}. "
                "Reduce total or add more data."
            )

    return result


def stratified_split(
    items: list[Any],
    key: str,
    n_dev: int,
    n_test: int,
    seed: int,
    *,
    dev_quotas: dict[str, int] | None = None,
    test_quotas: dict[str, int] | None = None,
) -> tuple[list[Any], list[Any]]:
    """
    Split `items` into (dev, test) with stratification by `item[key]`.

    - Sorts by `sample_id` first for determinism (items must have a `.sample_id` attribute
      or be dicts with a `sample_id` key).
    - Uses `allocate()` unless explicit `dev_quotas` / `test_quotas` are provided.
    - Dev and test are drawn from the same shuffled stratum → disjoint by construction.

    Args:
        items:       List of objects (BenchmarkSample or raw dicts) with a `key` attribute/key.
        key:         Attribute name used for stratification.
        n_dev:       Total dev size.
        n_test:      Total test size.
        seed:        Random seed for reproducibility.
        dev_quotas:  Optional explicit per-stratum dev counts (e.g. FEVER fixed quotas).
        test_quotas: Optional explicit per-stratum test counts.

    Returns:
        (dev_list, test_list)
    """

    def _get(item: Any, k: str) -> Any:
        return getattr(item, k) if hasattr(item, k) else item[k]

    # Sort by sample_id for determinism
    items = sorted(items, key=lambda x: _get(x, "sample_id"))

    # Group into strata
    groups: dict[str, list[Any]] = defaultdict(list)
    for item in items:
        groups[_get(item, key)].append(item)

    strata_sizes = {k: len(v) for k, v in groups.items()}

    # Resolve quotas
    resolved_test = test_quotas if test_quotas is not None else allocate(strata_sizes, n_test)
    resolved_dev = dev_quotas if dev_quotas is not None else allocate(strata_sizes, n_dev)

    dev_items: list[Any] = []
    test_items: list[Any] = []
    rng = random.Random(seed)

    for stratum in sorted(groups.keys()):
        pool = groups[stratum][:]
        rng.shuffle(pool)
        t_count = resolved_test.get(stratum, 0)
        d_count = resolved_dev.get(stratum, 0)
        if t_count + d_count > len(pool):
            raise AllocationError(
                f"Stratum '{stratum}' needs {t_count + d_count} items but only has {len(pool)}."
            )
        test_items.extend(pool[:t_count])
        dev_items.extend(pool[t_count: t_count + d_count])

    return dev_items, test_items
