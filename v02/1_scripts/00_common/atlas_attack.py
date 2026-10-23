"""Choosing records and forging labels at ATLAS scope.

The atlas arm's budget is a fraction of the whole 7,913,892-record reference,
not of one celltype, and its poison has NO TARGET: each chosen record is given a
random WRONG term, so the poisoned records do not agree with each other and do
not vote as a bloc. That is the honest null the targeted arm is measured
against, and "poisoning must be targeted to work at all" is a finding that needs
a number rather than an assertion.

Shared because ``02a_01`` (insert) and ``02a_02`` (modify) must choose the same
records and forge the same labels from the same seed. If they did not, the one
axis between them -- whether the genuine record survives -- would be confounded
with which records were hit. ``vote.py`` records what happens when this rule is
broken.

``build_uniform`` and ``seed_of`` are VERBATIM from ``02a_01``, which reproduced
them verbatim from v01's ``run_global_attack``. Their RNG consumption order is
load-bearing and pinned by a golden test.
"""
from __future__ import annotations

import hashlib

import numpy as np

from selection import hub_order

#: Records whose shipped label has no ontology id. Poisoning one wastes budget
#: on a no-op, so v01 excluded them and so does this.
UNMAPPED = "UNMAPPED"

#: A celltype too small to be a sensible random label to forge.
MIN_CLASS = 2000


def seed_of(mode: str, budget: float | None, draw: int) -> int:
    """Stable seed for one draw. Verbatim from v01, so the poison matches.

    blake2b rather than Python's ``hash()``, which is salted per process and
    would make the same triple yield different poison in different runs.

    Args:
        mode: the arm's name. budget: atlas fraction. draw: replicate index.

    Returns:
        A 32-bit seed.
    """
    key = f"{mode}|{budget!r}|{draw}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")


def wrong_labels(lab: np.ndarray, src: np.ndarray, big: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
    """A random WRONG term for each chosen record.

    Redrawn ONCE where the draw happened to match the record's own label. A
    record relabelled to its own term is not poisoned at all, so the redraw
    matters: it cuts the collision rate from 1/n_terms to 1/n_terms^2.

    It is a single redraw, not a loop, so collisions are not eliminated -- and
    that is v01's behaviour, kept deliberately. Measured on the shipped
    reference: 180 celltypes carry at least MIN_CLASS records, so the rate falls
    from 1/180 = 5.6e-3 to 1/180^2 = 3.1e-5. At the largest budget, 10% of the
    atlas or 791,389 records, that is ~24 records keeping their own label. A
    no-op at that scale, not a bias.

    Looping until no collision would be marginally cleaner and would consume the
    RNG differently, re-poisoning the 30 committed conditions of ``02a_01`` and
    breaking the pairing with v01 that makes them worth having. The trade is not
    worth 24 records in 791,389.

    Args:
        lab: (n_ref,) reference labels. src: (n_p,) chosen rows.
        big: terms large enough to forge. rng: seeded upstream.

    Returns:
        (n_p,) forged terms.
    """
    pl = rng.choice(big, len(src))
    same = pl == lab[src]
    if same.any():
        pl[same] = rng.choice(big, int(same.sum()))
    return pl


def build_uniform(lab: np.ndarray, usable: np.ndarray, big: np.ndarray,
                  n_p: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """v01's uniform poison: scattered records, each given a random WRONG label.

    Args:
        lab: (n_ref,) reference labels. usable: rows with a CL id.
        big: terms with at least :data:`MIN_CLASS` records. n_p: budget in
            records. seed: passed in, never generated.

    Returns:
        ``(rows, forged_labels)``, aligned.
    """
    rng = np.random.default_rng(seed)
    src = rng.choice(usable, min(n_p, len(usable)), replace=n_p > len(usable))
    return src, wrong_labels(lab, src, big, rng)


def build_hub(lab: np.ndarray, usable: np.ndarray, big: np.ndarray, n_p: int,
              seed: int, nbr_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The same, but spending the budget on the BUSIEST records in the atlas.

    Hub targeting is worth 19x at fixed budget within a celltype (94.7% of
    retrieval traffic against 5.1%); whether it transfers to the whole atlas is
    unmeasured, and it is the obvious way a scattered attack could stop being a
    null result.

    Record selection is DETERMINISTIC given the observed traffic -- v01's
    guarantee for hub ranking -- so ``seed`` varies only the forged labels. A
    draw therefore still means something here, just less than under uniform.

    Args:
        lab/usable/big/n_p/seed: as :func:`build_uniform`.
        nbr_idx: (n_q, k) int64 rows the clean retrieval returned.

    Returns:
        ``(rows, forged_labels)``, busiest first.
    """
    order = hub_order(usable, nbr_idx, len(lab))
    src = usable[order[np.arange(min(n_p, len(usable))) % len(order)]]
    return src, wrong_labels(lab, src, big, np.random.default_rng(seed))
