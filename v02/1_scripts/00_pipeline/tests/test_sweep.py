"""Golden-value tests for stage 4, sweep.

The fixture is built so every expected number below is hand-checkable.
Queries are the basis vectors e0 and e1; poison vectors have exact cosines
against them by construction:

    p0 = [0.95,      0.31224990, 0, 0]   cos(e0)=0.95      cos(e1)=0.31224990
    p1 = [0.5,       0.86602540, 0, 0]   cos(e0)=0.50      cos(e1)=0.86602540
    p2 = [0.2,       0.97979590, 0, 0]   cos(e0)=0.20      cos(e1)=0.97979590

with k = 3 and clean neighbours supplied directly.
"""

from __future__ import annotations
# 1_scripts cannot be a package (leading digit), so the shared library in
# 00_common is put on the path and imported flat.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_common"))


import numpy as np
import pytest

from schemas import (
    MIN_POISON_SCORE,
    N_POISON_IN_TOPK,
    POISON_SURVIVES_QC,
    Y_PRED_CLEAN,
    Y_PRED_POISONED,
)
from sweep import run_sweep

K = 3
QUERY = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
POISON = np.array([[0.95, 0.31224990, 0.0, 0.0],
                   [0.50, 0.86602540, 0.0, 0.0],
                   [0.20, 0.97979590, 0.0, 0.0]], dtype=np.float32)
POISON_LABELS = np.array(["CL:B", "CL:B", "CL:B"])
CLEAN_COS = np.array([[0.90, 0.85, 0.80], [0.70, 0.60, 0.50]], dtype=np.float32)
CLEAN_LABELS = np.array([["CL:A"] * 3, ["CL:A"] * 3])
Y_TRUE = np.array(["CL:A", "CL:A"])
QUERY_IDS = np.array(["q0", "q1"])


def _sweep(poison_emb=POISON, poison_labels=POISON_LABELS, block_size=5000):
    return run_sweep(
        query_emb=QUERY, query_ids=QUERY_IDS, y_true=Y_TRUE,
        clean_cos=CLEAN_COS, clean_labels=CLEAN_LABELS,
        poison_emb=poison_emb, poison_labels=poison_labels,
        k=K, block_size=block_size,
    )


def test_clean_prediction_ignores_poison_entirely() -> None:
    # q0 and q1 both have three CL:A clean neighbours.
    assert list(_sweep()[Y_PRED_CLEAN]) == ["CL:A", "CL:A"]


def test_golden_poison_count_in_topk() -> None:
    # q0 merged top-3: 0.95(P) 0.90 0.85          -> 1 poison
    # q1 merged top-3: 0.97979(P) 0.86602(P) 0.70 -> 2 poison
    assert list(_sweep()[N_POISON_IN_TOPK]) == [1, 2]


def test_golden_vote_flips_only_when_poison_wins_the_majority() -> None:
    # q0: 2 x CL:A vs 1 x CL:B -> holds.  q1: 2 x CL:B vs 1 x CL:A -> flips.
    assert list(_sweep()[Y_PRED_POISONED]) == ["CL:A", "CL:B"]


def test_golden_min_poison_score() -> None:
    # nearest poison: q0 c=0.95 -> 1/0.05 = 20;  q1 c=0.9797959 -> 49.4949
    got = _sweep()[MIN_POISON_SCORE].to_numpy()
    assert np.allclose(got, [20.0, 49.4949], atol=1e-3)


def test_golden_qc_survival_straddles_the_threshold() -> None:
    # 20 < 33 <= 49.49
    assert list(_sweep()[POISON_SURVIVES_QC]) == [False, True]


def test_min_poison_score_uses_nearest_poison_even_when_it_misses_the_topk() -> None:
    # THE naming hazard, guarded. `min_poison_score` holds the score of the
    # NEAREST poison, which is the MAXIMUM score over the poison set. A single
    # poison too far to enter the top-k must still be reported.
    far = np.array([[0.30, 0.95393920, 0.0, 0.0]], dtype=np.float32)  # cos(e0)=0.3
    out = run_sweep(
        query_emb=QUERY[:1], query_ids=QUERY_IDS[:1], y_true=Y_TRUE[:1],
        clean_cos=CLEAN_COS[:1], clean_labels=CLEAN_LABELS[:1],
        poison_emb=far, poison_labels=np.array(["CL:B"]), k=K, block_size=5000,
    )
    assert out[N_POISON_IN_TOPK].iloc[0] == 0           # did not enter top-k
    assert np.isclose(out[MIN_POISON_SCORE].iloc[0], 1 / 0.7, atol=1e-4)


def test_exact_duplicate_scores_infinite_and_survives_qc() -> None:
    # cosine == 1 is reachable: the duplicate attack copies vectors exactly.
    # The divide-by-zero is the correct, documented outcome, not an accident:
    # asserting it is raised is stronger than letting it drift past unnoticed.
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        out = run_sweep(
            query_emb=QUERY[:1], query_ids=QUERY_IDS[:1], y_true=Y_TRUE[:1],
            clean_cos=CLEAN_COS[:1], clean_labels=CLEAN_LABELS[:1],
            poison_emb=QUERY[:1], poison_labels=np.array(["CL:B"]), k=K,
            block_size=5000,
        )
    assert np.isinf(out[MIN_POISON_SCORE].iloc[0])
    assert bool(out[POISON_SURVIVES_QC].iloc[0]) is True


def test_blocking_does_not_change_results() -> None:
    # Chunking exists to keep peak memory flat; it must be arithmetically inert.
    one = _sweep(block_size=1)
    many = _sweep(block_size=5000)
    assert one.equals(many)


def test_vote_ties_break_on_lowest_cl_id() -> None:
    # k=2 with one clean CL:A and one poison CL:B is a 1-1 tie -> CL:A wins.
    out = run_sweep(
        query_emb=QUERY[:1], query_ids=QUERY_IDS[:1], y_true=Y_TRUE[:1],
        clean_cos=CLEAN_COS[:1, :2], clean_labels=CLEAN_LABELS[:1, :2],
        poison_emb=POISON[:1], poison_labels=np.array(["CL:B"]), k=2,
        block_size=5000,
    )
    assert out[Y_PRED_POISONED].iloc[0] == "CL:A"


def test_empty_poison_set_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty poison set"):
        _sweep(poison_emb=np.zeros((0, 4), dtype=np.float32),
               poison_labels=np.array([], dtype=object))


def test_exact_ties_resolve_to_the_genuine_record() -> None:
    # THE TIE POLICY, pinned. `duplicate` copies reference vectors verbatim, so a
    # poisoned record and the record it copied have a mathematically exact
    # similarity tie. Which one ranks first is a benchmark decision, not a
    # measurement — in the protein arm a one-ULP float difference flipped it and
    # swung pooled attack success by a third.
    #
    # Policy here is INCUMBENT: the genuine record wins. It holds because the
    # merge concatenates clean neighbours BEFORE poison and sorts stably, so
    # equal similarities keep clean ahead. If either of those changes, this test
    # fails rather than the number silently moving.
    q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    twin = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)   # exact copy
    with pytest.warns(RuntimeWarning, match='divide by zero'):
        out = run_sweep(
            query_emb=q, query_ids=np.array(["q0"]), y_true=np.array(["CL:A"]),
            clean_cos=np.array([[1.0, 0.9]], dtype=np.float32),      # rank-1 is the twin
            clean_labels=np.array([["CL:A", "CL:A"]]),
            poison_emb=twin, poison_labels=np.array(["CL:B"]),
            k=2, block_size=10,
        )
    # both tie at cosine 1.0; the genuine record must hold rank 1, so the vote
    # stays CL:A rather than flipping on an arbitrary ordering
    assert out[Y_PRED_POISONED].iloc[0] == "CL:A"


def test_tie_policy_is_stable_across_repeated_runs() -> None:
    q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    twin = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    seen = set()
    for _ in range(50):
        with pytest.warns(RuntimeWarning, match='divide by zero'):
            o = run_sweep(
                query_emb=q, query_ids=np.array(["q0"]), y_true=np.array(["CL:A"]),
                clean_cos=np.array([[1.0, 0.9]], dtype=np.float32),
                clean_labels=np.array([["CL:A", "CL:A"]]),
                poison_emb=twin, poison_labels=np.array(["CL:B"]), k=2, block_size=10,
            )
        seen.add((str(o[Y_PRED_POISONED].iloc[0]), int(o[N_POISON_IN_TOPK].iloc[0])))
    assert len(seen) == 1, f"tie resolved inconsistently: {seen}"
