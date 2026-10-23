"""How an attacker chooses which reference records to draw from.

One implementation, imported everywhere, for the reason ``vote.py`` gives: v01
grew six copies of the k-NN vote that disagreed on tie-breaking. ``hub_order``
was heading the same way -- it lived inside ``02b_05`` and the insertion arm had
no access to it at all, which is why ``insert x hub`` is unmeasured (SPEC.md
§5.4, ``build 2``) rather than unmeasurable.

Selection is ORTHOGONAL to geometry. ``relabel`` and ``duplicate`` are the same
construction -- copy a genuine vector -- at two selection settings, and the fact
that this was encoded in the attack NAME rather than in a column is the offence
SPEC.md §5.5 names.
"""
from __future__ import annotations

import numpy as np

from schemas import Selection

#: Share of the attacked celltype that ``concentrated`` draws its pool from.
#: v01's value, kept so ``duplicate`` reproduces exactly.
CONCENTRATED_POOL_FRACTION = 0.01


def hub_order(pool: np.ndarray, nbr_idx: np.ndarray, n_ref: int) -> np.ndarray:
    """Pool records ordered by how many neighbour slots they occupy, busiest first.

    Args:
        pool: (n_pool,) reference row indices carrying the attacked label.
        nbr_idx: (n_q, k) int64 retrieved reference rows, from the queries whose
            traffic the attacker is assumed to have observed.
        n_ref: reference size, for the bincount length.

    Returns:
        (n_pool,) positions into ``pool``, busiest first. Ties break on reference
        row index via a stable sort, so selection is deterministic and needs no
        seed -- the same guarantee v01 made.
    """
    hits = np.bincount(nbr_idx.ravel(), minlength=n_ref)
    return np.argsort(-hits[pool], kind="stable")


def choose(selection: str, src_idx: np.ndarray, n_p: int,
           rng: np.random.Generator,
           hub_nbr: np.ndarray | None = None,
           n_ref: int | None = None) -> np.ndarray:
    """Which reference rows the attacker draws, ``n_p`` of them.

    The RNG is consumed EXACTLY as the pre-axis code consumed it, so ``uniform``
    and ``concentrated`` reproduce ``relabel`` and ``duplicate`` bit for bit --
    asserted in ``test_selection.py``. ``hub`` consumes none: it is deterministic
    given the observed traffic, which is v01's guarantee and the reason it needs
    no seed.

    Args:
        selection: a :class:`schemas.Selection` value.
        src_idx: (n_src,) reference rows carrying the source term.
        n_p: how many records to return.
        rng: seeded upstream, never created here.
        hub_nbr: (n_q, k) observed neighbour rows. ``hub`` only, REQUIRED there.
        n_ref: reference size. ``hub`` only, REQUIRED there.

    Returns:
        (n_p,) reference row indices, with repeats where the budget exceeds the
        distinct records the rule makes available.

    Raises:
        ValueError: unknown selection, or ``hub`` without the observed traffic.
    """
    if selection == Selection.UNIFORM:
        return rng.choice(src_idx, size=n_p, replace=n_p > src_idx.size)
    if selection == Selection.CONCENTRATED:
        pool_n = max(1, int(round(CONCENTRATED_POOL_FRACTION * src_idx.size)))
        pool = rng.choice(src_idx, size=pool_n, replace=False)
        return rng.choice(pool, size=n_p, replace=True)
    if selection == Selection.HUB:
        if hub_nbr is None or n_ref is None:
            raise ValueError(
                "hub selection requires the observed neighbour table and the "
                "reference size. It ranks records by top-k occupancy, so "
                "without traffic there is nothing to rank and a silent "
                "fallback to uniform would report the wrong attack.")
        order = hub_order(src_idx, hub_nbr, n_ref)
        # Busiest first, cycling where the budget exceeds the celltype. Mirrors
        # 02b_05's pool[order[:k]] exactly at n_p <= n_src.
        return src_idx[order[np.arange(n_p) % len(order)]]
    raise ValueError(f"unknown selection {selection!r}")
