"""The k-NN vote. One implementation, imported everywhere.

v01 grew SIX independent copies of this -- in ``sweep.py``, ``defense.py``,
``validate_shortcut.py`` and three analysis scripts -- and they did not agree.
Some broke ties on argmax order, which is the order labels happen to first appear
in the array; one followed ``config.vote.tie_break``. So a clean baseline and the
attacked results it was compared against could be scored by different rules, and
the difference was never quantified.

That is the same failure ``schemas.py`` exists to prevent, one level up: the vote
is the single most load-bearing function in the project -- every accuracy, every
attack success and every defence number is downstream of it -- so it belongs in
one place that all lanes import. Treat this module as orchestrator-owned, like
``schemas.py``.

The counting and tie-break here are taken from the implementation written for
``02b_01``, which was measurably better than the first draft of this module: a
``bincount`` over a flattened index rather than ``np.add.at``, and a SORTED
vocabulary so ``argmax`` yields the lowest CL id without a separate rank array.

TIES ARE REPORTED, NEVER SILENTLY RESOLVED. A tie has to be broken to return an
answer, and ``config.vote.tie_break`` declares how (``lowest_cl_id``: among the
labels sharing the top count, the smallest ontology id). But that rule is
deterministic, not defensible -- it is an arbitrary choice recorded so runs
reproduce, and SCimilarity's own rule differs again (``max`` over a dict filled
in neighbour order returns whichever tied label had the nearest single
neighbour). So every call also returns how many labels were tied and whether the
PIVOTAL label was among them, and callers are expected to publish bounds rather
than a point.

**Only a tie containing the pivotal label is ambiguous.** A three-way tie among
labels that are all wrong is already determined -- the answer is wrong either
way, and which wrong label is returned changes nothing. The pivotal label is the
query's true label when scoring accuracy, or the attacker's target when scoring
attack success. Counting every tie as ambiguous overstates the uncertainty;
counting none understates it.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

#: The declared rule, mirroring ``config.vote.tie_break``. Among labels sharing
#: the top count, return the smallest Cell Ontology id. CL ids are zero-padded to
#: seven digits, so lexicographic order is numeric order.
TIE_BREAK = "lowest_cl_id"


class VoteResult(NamedTuple):
    """One vote per query, plus what the tie-break had to decide.

    Attributes:
        pred: (n_q,) str, the winning label under :data:`TIE_BREAK`.
        n_tied: (n_q,) int8, how many labels shared the top count. 1 means no
            tie; the tie rate is ``(n_tied > 1).mean()``.
        pivotal_in_tie: (n_q,) bool, the AMBIGUOUS set: the pivotal label was
            among the top-count labels AND there was a tie, so the arbitrary
            tie-break is what decided this query. False when the pivotal label
            won outright -- that answer is determined, not decided. All False
            when no pivotal label was given.
    """

    pred: np.ndarray
    n_tied: np.ndarray
    pivotal_in_tie: np.ndarray


def vote(labels: np.ndarray, pivotal: np.ndarray | str | None = None,
         tie_break: str = TIE_BREAK) -> VoteResult:
    """Plurality vote over each query's retrieved neighbour labels.

    Args:
        labels: (n_q, k) str, one retrieved celltype label per neighbour.
        pivotal: the label whose presence in a tie makes that tie ambiguous --
            a per-query array (n_q,) such as the true labels, or one string such
            as the attacker's target. None means ties are counted but none are
            marked ambiguous.
        tie_break: only ``lowest_cl_id`` is implemented. Passing anything else
            raises rather than silently falling back to argmax order, which is
            how the v01 copies drifted apart.

    Returns:
        :class:`VoteResult`.

    Raises:
        ValueError: unknown ``tie_break``, or a ``pivotal`` array whose length
            does not match ``labels``.
    """
    if tie_break != TIE_BREAK:
        raise ValueError(
            f"tie_break {tie_break!r} is not implemented; only {TIE_BREAK!r} is. "
            "Add it here rather than breaking ties differently in a caller.")
    n_q, k = labels.shape

    # sort=True is what implements the tie-break: with a SORTED vocabulary,
    # argmax returns the lowest CL id among joint winners for free. CL ids are
    # zero-padded to seven digits, so lexicographic order is numeric order.
    codes, uniq = pd.factorize(labels.ravel(), sort=True)
    uniq = np.asarray(uniq)
    n_vocab = len(uniq)

    # bincount over a flattened (row, code) index rather than np.add.at, which
    # is an unbuffered scatter-add and ran ~4x slower across v01's whole sweep.
    flat = (np.arange(n_q)[:, None] * n_vocab + codes.reshape(n_q, k)).ravel()
    counts = np.bincount(flat, minlength=n_q * n_vocab).reshape(n_q, n_vocab)

    at_top = counts == counts.max(axis=1, keepdims=True)
    n_tied = at_top.sum(axis=1).astype(np.int8)
    pred = uniq[counts.argmax(axis=1)]

    if pivotal is None:
        pivotal_in_tie = np.zeros(n_q, dtype=bool)
    else:
        piv = np.asarray(pivotal)
        if piv.ndim == 0:
            piv = np.repeat(piv, n_q)
        if len(piv) != n_q:
            raise ValueError(f"pivotal has {len(piv)} entries, labels has {n_q}")
        col = np.searchsorted(uniq, piv)
        col[col >= n_vocab] = 0
        present = uniq[col] == piv          # absent from every neighbourhood
        pivotal_in_tie = np.zeros(n_q, dtype=bool)
        rows = np.flatnonzero(present)
        # BOTH conditions required: holding the top count is not enough, or an
        # outright win would be counted as decided-by-tie-break.
        pivotal_in_tie[rows] = at_top[rows, col[rows]] & (n_tied[rows] > 1)
    return VoteResult(pred, n_tied, pivotal_in_tie)


def bounds(res: VoteResult, target: np.ndarray | str) -> tuple[np.ndarray, np.ndarray]:
    """Best and worst case for ``target`` over the queries the tie-break decided.

    A tie containing ``target`` could go either way, so a rate computed from
    ``res.pred`` alone is a point inside an interval rather than the answer.

    Args:
        res: from :func:`vote`, built with ``pivotal=target``.
        target: the label being counted -- attacker's target, or true label.

    Returns:
        best: (n_q,) float, 1.0 where target wins or could win a tie.
        worst: (n_q,) float, 1.0 only where target wins outright.
    """
    tgt = np.asarray(target)
    if tgt.ndim == 0:
        tgt = np.repeat(tgt, len(res.pred))
    hit = (res.pred == tgt).astype(float)
    untied = res.n_tied == 1
    return np.maximum(hit, res.pivotal_in_tie.astype(float)), np.where(untied, hit, 0.0)
