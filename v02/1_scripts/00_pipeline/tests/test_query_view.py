"""``02b_30``'s two tables must be joinable on ``query_id``, and complete.

The whole point of the step is that a figure can join a per-query prediction to
a per-query coordinate. That join is silent when it goes wrong: a missing
condition drops a panel, a duplicated query double-counts a cell, and a
merge on ``query_id`` reports neither. So the shapes are pinned here --

* the per-query table has EXACTLY one row per query per condition,
* the UMAP table has exactly one row per query, with unique ids,

-- together with the brute-force nearest-poison distance, which is the one
number in the step that is computed rather than read and is therefore the one
that can be silently wrong. It is checked against a full matmul on a fixture
small enough to hold both, and against itself at three block sizes.

The step's module name starts with a digit, so there is no dotted import path
(CLAUDE.md §3); it is loaded from its file like any other data file.
"""
import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

from celltype_sweep import Retrieved
from schemas import (MIN_POISON_SCORE, N_POISON_IN_TOPK, QUERY_ID, RATE,
                     THREAT_MODEL, Y_PRED_CLEAN, Y_PRED_POISONED, Y_TRUE)
from vote import VoteResult

STEP_PATH = (pathlib.Path(__file__).resolve().parents[3]
             / "1_scripts" / "02_b_attack_celltype"
             / "02b_30_no_access_query_view.py")

#: The approved ceiling for this step, from the conversation of 2026-09-28.
#: Each condition reloads the shipped 9.3 GB index, so a ninth is not a rounding
#: error -- it is another minute and a half of work nobody approved.
BUDGET = 8

N_Q = 7


def _load():
    """Import the step module from its path."""
    spec = importlib.util.spec_from_file_location("step_02b_30", STEP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def step():
    return _load()


@pytest.fixture
def meta() -> pd.DataFrame:
    """A tiny query set: ids, donors and ground truth, in the cache's shape."""
    return pd.DataFrame({
        QUERY_ID: [f"q{i}" for i in range(N_Q)],
        "donor_id": [f"d{i % 3}" for i in range(N_Q)],
        Y_TRUE: ["CL:0000057"] * 4 + ["CL:0000623"] * 3,
    })


def _condition(step, source: str, selection: str, rate: float) -> dict:
    """The scalar identity columns of one condition, as ``main`` builds them."""
    return {THREAT_MODEL: "no_access", "retrieval_path": step.RETRIEVAL_PATH,
            "source": source, "selection": selection, RATE: rate}


def _vote(pred: list[str], pivotal_in_tie: list[bool]) -> VoteResult:
    """A stand-in vote: the prediction, the tie width, and the ambiguity flag."""
    return VoteResult(pred=np.array(pred),
                      n_tied=np.where(pivotal_in_tie, 2, 1).astype(np.int8),
                      pivotal_in_tie=np.array(pivotal_in_tie))


def _rows(step, meta: pd.DataFrame, keys: dict) -> pd.DataFrame:
    """One condition's per-query frame over deterministic stand-in votes."""
    took = ["CL:0000097"] * N_Q
    tied = [i % 2 == 0 for i in range(N_Q)]
    return step.query_rows(
        meta,
        clean=_vote([keys["source"]] * N_Q, [False] * N_Q),
        poisoned=_vote(took, tied),
        hit=_vote(took, [not t for t in tied]),
        n_poison_in_topk=np.arange(N_Q),
        near_cos=np.linspace(0.90, 0.99, N_Q).astype(np.float32),
        keys=keys)


def test_one_row_per_query_per_condition(step, meta):
    """Every condition contributes every query exactly once, and no more.

    The failure this pins is a partial condition: a table that is 5 queries
    short on one panel still reads, still plots, and reports a rate over a
    denominator nobody notices has changed.
    """
    conds = [_condition(step, "CL:0000057", "hub", 0.25),
             _condition(step, "CL:0000057", "hub", 1.0),
             _condition(step, "CL:0000623", "uniform", 1.0)]
    out = pd.concat([_rows(step, meta, c) for c in conds], ignore_index=True)

    assert len(out) == len(conds) * N_Q
    per_condition = out.groupby(["source", "selection", RATE],
                                observed=True)[QUERY_ID]
    assert (per_condition.size() == N_Q).all()
    assert (per_condition.nunique() == N_Q).all()
    for _, grp in out.groupby(["source", "selection", RATE], observed=True):
        assert set(grp[QUERY_ID]) == set(meta[QUERY_ID])


def test_every_contract_column_is_present(step, meta):
    """The per-query table speaks ``raw_result``'s vocabulary, not its own."""
    df = _rows(step, meta, _condition(step, "CL:0000057", "hub", 1.0))
    for col in (QUERY_ID, Y_TRUE, Y_PRED_CLEAN, Y_PRED_POISONED,
                N_POISON_IN_TOPK, MIN_POISON_SCORE, "poison_survives_qc"):
        assert col in df.columns


def test_every_row_says_whether_the_tie_break_decided_it(step, meta):
    """The point verdict never travels without its ambiguity flag.

    CLAUDE.md §7: ties are reported, never silently resolved. At parity the
    tie-break decides more than half the attacked celltype's queries
    (HANDOFF §3.34), so a per-query table that carried only
    ``y_pred_poisoned`` would publish one side of a coin flip as a fact.

    The flags are METRIC-SPECIFIC and the count is not: only a tie containing
    the pivotal label is ambiguous, and the pivotal label differs between
    accuracy and attack success. Here the two stand-in votes are deliberately
    given opposite flags over one shared prediction, so a column that copied
    the wrong pivot's flag would show up.
    """
    df = _rows(step, meta, _condition(step, "CL:0000057", "hub", 1.0))
    for col in ("n_tied", "tie_decides_accuracy", "tie_decides_attack_success",
                "n_tied_clean", "tie_decides_accuracy_clean"):
        assert col in df.columns
    acc = df["tie_decides_accuracy"].to_numpy()
    hit = df["tie_decides_attack_success"].to_numpy()
    assert (acc != hit).all()
    # a flagged query had a tie to be decided; an unflagged one did not
    np.testing.assert_array_equal(df["n_tied"].to_numpy() > 1, acc)
    assert not df["tie_decides_accuracy_clean"].any()


def test_the_two_poisoned_votes_must_be_one_vote(step, meta):
    """Two pivots of one vote share a prediction. Differing ones are two votes.

    Scoring accuracy from one retrieval and attack success from another is the
    defect CLAUDE.md §7 names ("one retrieval regime per comparison"), and a
    caller that passed mismatched votes would produce a row whose two metrics
    describe different neighbourhoods.
    """
    keys = _condition(step, "CL:0000057", "hub", 1.0)
    with pytest.raises(ValueError, match="two pivots of ONE vote"):
        step.query_rows(
            meta,
            clean=_vote(["CL:0000057"] * N_Q, [False] * N_Q),
            poisoned=_vote(["CL:0000097"] * N_Q, [False] * N_Q),
            hit=_vote(["CL:0000084"] * N_Q, [False] * N_Q),
            n_poison_in_topk=np.zeros(N_Q),
            near_cos=np.zeros(N_Q, dtype=np.float32), keys=keys)


def test_rate_keeps_its_full_width(step, meta):
    """``rate`` is a KEY, not a measurement, so it is never narrowed.

    ``seed_of`` hashes ``repr(rate)``: float32 turns 0.7 into
    ``'0.699999988079071'`` and names a different poison set. A reader
    re-deriving the poison from the column would silently build the wrong one.
    """
    df = _rows(step, meta, _condition(step, "CL:0000057", "hub", 0.7))
    assert df[RATE].dtype == np.float64
    assert df[N_POISON_IN_TOPK].dtype == np.int16


def test_query_rows_rejects_a_short_array(step, meta):
    """A prediction array that does not cover every query is a hard error."""
    keys = _condition(step, "CL:0000057", "hub", 1.0)
    took = ["CL:0000097"] * N_Q
    with pytest.raises(ValueError, match="y_pred_clean"):
        step.query_rows(
            meta,
            clean=_vote(["CL:0000057"] * (N_Q - 1), [False] * (N_Q - 1)),
            poisoned=_vote(took, [False] * N_Q),
            hit=_vote(took, [False] * N_Q),
            n_poison_in_topk=np.zeros(N_Q),
            near_cos=np.zeros(N_Q, dtype=np.float32), keys=keys)


def test_umap_has_one_row_per_query(step, meta):
    """The layout table is a bijection with the query set, so the join is 1:1."""
    xy = np.arange(2 * N_Q, dtype=np.float32).reshape(N_Q, 2)
    df = step.umap_rows(meta, xy)

    assert len(df) == N_Q
    assert df[QUERY_ID].is_unique
    assert set(df[QUERY_ID]) == set(meta[QUERY_ID])
    assert THREAT_MODEL in df.columns
    np.testing.assert_array_equal(df[["x", "y"]].to_numpy(), xy)


def test_umap_rejects_a_coordinate_count_that_is_not_the_query_count(step, meta):
    with pytest.raises(ValueError, match="coordinates"):
        step.umap_rows(meta, np.zeros((N_Q - 1, 2), dtype=np.float32))


def _unit(rng, n: int, d: int = 8) -> np.ndarray:
    v = rng.normal(size=(n, d)).astype(np.float32)
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def test_nearest_poison_cosine_matches_the_full_matmul(step):
    """The blocked running max is the max. Golden against the whole matrix.

    Blocking exists only to bound memory, so it has to be arithmetically inert:
    three block shapes -- one block, many query blocks, many poison blocks --
    must return the same array, and that array must be the one a single
    ``(q @ pe.T).max(axis=1)`` produces.
    """
    rng = np.random.default_rng(0)
    q, pe = _unit(rng, 37), _unit(rng, 53)
    expected = (q @ pe.T).max(axis=1)

    for q_block, p_block in ((1000, 1000), (5, 1000), (1000, 7), (4, 6)):
        got = step.nearest_poison_cosine(q, pe, q_block=q_block,
                                         p_block=p_block)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)


def test_nearest_poison_cosine_sees_poison_the_graph_would_not_return(step):
    """It reports the nearest poison even when it is nowhere near the top-k.

    This is the reason the column is computed rather than read off the graph:
    a single planted near-duplicate among many far ones must set the distance,
    and an index asked for k neighbours may never mention it.
    """
    rng = np.random.default_rng(1)
    q = _unit(rng, 1)
    far = -np.repeat(q, 40, axis=0)                 # cosine -1 to the query
    near = q.copy()                                 # cosine 1
    pe = np.vstack([far, near]).astype(np.float32)
    got = step.nearest_poison_cosine(q, pe, q_block=1, p_block=8)
    np.testing.assert_allclose(got, 1.0, atol=1e-6)


def test_nearest_poison_cosine_rejects_an_empty_set(step):
    """A condition poisoning nothing would report the attack as costless."""
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError, match="empty poison set"):
        step.nearest_poison_cosine(_unit(rng, 3), np.zeros((0, 8), np.float32))


def test_topk_nearest_is_minus_inf_when_the_graph_returned_no_poison():
    """``Retrieved.nearest_topk_cos`` says "not returned", not "not near".

    The distinction is the step's whole reason for computing the brute-force
    column beside it, so the silent value is pinned rather than assumed.
    """
    is_p = np.array([[False, False], [False, True]])
    cos = np.array([[0.9, 0.8], [0.7, 0.6]], dtype=np.float32)
    r = Retrieved(is_poison=is_p, topk_cos=cos, acc=None, hit=None)
    got = r.nearest_topk_cos()
    assert got[0] == -np.inf
    np.testing.assert_allclose(got[1], 0.6, atol=1e-6)


def test_the_condition_list_stays_inside_the_approved_budget(step):
    """Eight conditions were approved. A ninth is a new approval, not an edit."""
    assert len(step.CONDITIONS) <= BUDGET
    assert len(set(step.CONDITIONS)) == len(step.CONDITIONS)
