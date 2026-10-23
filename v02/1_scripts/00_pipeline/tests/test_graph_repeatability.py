"""``02b_34`` exists to size a band, so its two comparisons must not lie.

The step's expensive half -- 24 insertions into the shipped 9.3 GB index -- is
untestable here. Its cheap half is where a wrong number would come from, and
it is entirely deterministic:

* :func:`row_overlap` says how much the GRAPH moved between two retrievals. Get
  it wrong and the whole point of the step inverts: an overlap that reads too
  high turns "the graph is unstable" into "the graph is fine".
* :func:`spread` turns repeats into the band that gets quoted. Get its ``n``
  wrong on the churn columns and every one of them is diluted by a trivial
  self-comparison worth exactly 1.0.

Both are checked on hand-worked inputs. The step's module name starts with a
digit, so there is no dotted import path (CLAUDE.md §3); it is loaded from its
file.
"""
import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

STEP_PATH = (pathlib.Path(__file__).resolve().parents[3]
             / "1_scripts" / "02_b_attack_celltype"
             / "02b_34_no_access_graph_repeatability.py")

#: The ceiling approved in conversation on 2026-09-28. Each retrieval reloads
#: the shipped index, so a grid that quietly grew past this is unapproved
#: compute and the test says so before the run does.
BUDGET = 24


@pytest.fixture(scope="module")
def step():
    spec = importlib.util.spec_from_file_location("step_02b_34", STEP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_grid_is_within_the_approved_budget(step):
    """Conditions x repeats is the 24 retrievals that were approved."""
    assert len(step.CONDITIONS) * step.N_REPEATS <= step.BUDGET == BUDGET


def test_one_below_parity_and_one_at_it(step):
    """Ties concentrate at dose 1.0, so a repeat study that skipped it would
    miss the case where one swapped neighbour flips an answer."""
    doses = {rate for _, rate in step.CONDITIONS}
    assert 1.0 in doses, "no condition at parity"
    assert any(d < 1.0 for d in doses), "no condition below parity"


def test_row_overlap_is_per_query_and_order_blind(step):
    """Hand-worked: identical rows, a reordered row, a half-swapped row.

    Row 2 is the one that matters. It holds the same four ids in a different
    sequence, and a set comparison must call that a complete overlap -- the
    plurality vote cannot tell the two apart either.
    """
    a = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]], dtype=np.int64)
    b = np.array([[1, 2, 3, 4],
                  [8, 7, 6, 5],
                  [9, 10, 90, 91]], dtype=np.int64)
    assert step.row_overlap(a, b).tolist() == pytest.approx([1.0, 1.0, 0.5])


def test_row_overlap_does_not_match_across_queries(step):
    """Query 0's neighbours are query 1's, and neither shares its own.

    The packing that lets one ``np.isin`` answer every row is exactly where this
    could go wrong: without the row offset both rows would read 1.0, and a graph
    that returned a completely different neighbourhood for every query would be
    reported as perfectly stable.
    """
    a = np.array([[1, 2], [3, 4]], dtype=np.int64)
    b = np.array([[3, 4], [1, 2]], dtype=np.int64)
    assert step.row_overlap(a, b).tolist() == [0.0, 0.0]


def test_row_overlap_is_symmetric(step):
    """Neither retrieval is privileged; the fraction is the same either way."""
    rng = np.random.default_rng(0)
    a = np.array([rng.choice(50, size=8, replace=False) for _ in range(20)],
                 dtype=np.int64)
    b = np.array([rng.choice(50, size=8, replace=False) for _ in range(20)],
                 dtype=np.int64)
    assert step.row_overlap(a, b).tolist() == step.row_overlap(b, a).tolist()


def test_row_overlap_refuses_a_repeated_id(step):
    """Membership counting would score a duplicated id twice and read high."""
    a = np.array([[1, 1, 2, 3]], dtype=np.int64)
    b = np.array([[1, 2, 3, 4]], dtype=np.int64)
    with pytest.raises(ValueError, match="repeats an id"):
        step.row_overlap(a, b)


def test_row_overlap_refuses_an_id_past_the_row_stride(step):
    """An id at the stride collides with the next query's key space."""
    a = np.array([[int(step.ROW_KEY_STRIDE), 1]], dtype=np.int64)
    b = np.array([[0, 1]], dtype=np.int64)
    with pytest.raises(ValueError, match="row stride"):
        step.row_overlap(a, b)


def test_row_overlap_refuses_mismatched_shapes(step):
    with pytest.raises(ValueError, match="not the same retrieval shape"):
        step.row_overlap(np.zeros((2, 3), dtype=np.int64),
                         np.zeros((2, 4), dtype=np.int64))


def test_agreement_separates_graph_churn_from_answer_churn(step):
    """The two levels are different sizes, and that is the step's finding.

    Every query here gets half its neighbourhood replaced, so the graph moved a
    great deal. Only one of four predictions changed, and only one of those four
    queries is the attacked celltype. A single number could not say that.
    """
    a = np.array([[1, 2, 3, 4]] * 4, dtype=np.int64)
    b = np.array([[1, 2, 30, 40]] * 4, dtype=np.int64)
    pred_a = np.array(["CL:0000057", "CL:0000057", "CL:0000084", "CL:0000084"])
    pred_b = np.array(["CL:0000057", "CL:0000097", "CL:0000084", "CL:0000084"])
    got = step.agreement(a, b, pred_a, pred_b, within=np.array([0, 1]))

    assert got["global_topk_overlap_mean"] == pytest.approx(0.5)
    assert got["global_topk_overlap_min"] == pytest.approx(0.5)
    assert got["global_topk_set_identical"] == 0.0
    assert got["global_topk_ordered_identical"] == 0.0
    assert got["global_nn1_identical"] == 1.0   # position 1 held, 3 and 4 did not
    assert got["within_pred_changed"] == pytest.approx(0.5)
    assert got["global_pred_changed"] == pytest.approx(0.25)


def test_agreement_does_not_dilute_the_attacked_celltype(step):
    """The churn is concentrated where the poison is, and the scopes say so.

    Query 0 is the attacked celltype and its whole neighbourhood was replaced;
    the other nine are untouched. Pooled, that is a 0.90 overlap and reads as a
    stable graph. On the scope that matters it is 0.0. The first run of this
    step reported only the pooled figure, which is what this test exists to stop
    happening again.
    """
    a = np.tile(np.array([1, 2, 3, 4], dtype=np.int64), (10, 1))
    b = a.copy()
    b[0] = [91, 92, 93, 94]
    pred = np.array(["CL:0000623"] * 10)
    got = step.agreement(a, b, pred, pred, within=np.array([0]))

    assert got["within_topk_overlap_mean"] == 0.0
    assert got["global_topk_overlap_mean"] == pytest.approx(0.9)
    assert got["within_nn1_identical"] == 0.0
    assert got["global_nn1_identical"] == pytest.approx(0.9)


def test_agreement_refuses_an_empty_within_scope(step):
    """A mean over no queries is NaN, not a stable celltype."""
    a = np.array([[1, 2]], dtype=np.int64)
    pred = np.array(["CL:0000623"])
    with pytest.raises(ValueError, match="within_ scope"):
        step.agreement(a, a, pred, pred, within=np.array([], dtype=int))


def test_agreement_with_itself_is_the_trivial_comparison(step):
    """Repeat 1 against repeat 1: a 1.0 that is construction, not measurement.

    :func:`spread` drops these, and this is the value it is dropping.
    """
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    pred = np.array(["CL:0000057", "CL:0000084"])
    got = step.agreement(a, a, pred, pred, within=np.array([0]))
    assert got["global_topk_overlap_mean"] == 1.0
    assert got["within_topk_overlap_mean"] == 1.0
    assert got["global_topk_ordered_identical"] == 1.0
    assert got["global_pred_changed"] == 0.0
    assert got["within_pred_changed"] == 0.0


def _repeats(step, values: list[float], churn: list[float]) -> pd.DataFrame:
    """One condition's repeat rows: a rate that moves and a churn column."""
    n = len(values)
    return pd.DataFrame({
        "source": ["CL:0000623"] * n, "rate": [1.0] * n,
        "repeat": list(range(1, n + 1)),
        "within_accuracy": values,
        "baseline_accuracy": [0.46368] * n,
        "within_topk_overlap_mean": churn,
        # identity columns, which carry no band and must not become metrics
        "elapsed_seconds": [40.0] * n, "condition_seed": [12345] * n,
    })


def test_spread_reports_the_observed_width_not_only_the_sd(step):
    """mean, sample sd, min, max and range, all recomputed by hand."""
    v = [0.10, 0.12, 0.11, 0.13, 0.09, 0.11]
    df = _repeats(step, v, [1.0, 0.8, 0.8, 0.8, 0.8, 0.8])
    sp = step.spread(df, ["source", "rate"]).set_index("metric")

    row = sp.loc["within_accuracy"]
    assert row["n_repeats"] == 6
    assert row["mean"] == pytest.approx(float(np.mean(v)))
    assert row["sd"] == pytest.approx(float(np.std(v, ddof=1)))
    assert row["min"] == pytest.approx(0.09)
    assert row["max"] == pytest.approx(0.13)
    assert row["range"] == pytest.approx(0.04)
    assert row["rel_range"] == pytest.approx(0.04 / float(np.mean(v)))
    assert row["first"] == pytest.approx(0.10)


def test_spread_drops_the_self_comparison_from_a_churn_column(step):
    """Repeat 1's 1.0 is itself against itself and would inflate the mean.

    Five repeats of 0.8 average 0.8. Including the trivial 1.0 would report
    0.833 and a spurious range of 0.2, which reads as instability in the
    measurement rather than in the graph.
    """
    df = _repeats(step, [0.1] * 6, [1.0, 0.8, 0.8, 0.8, 0.8, 0.8])
    sp = step.spread(df, ["source", "rate"]).set_index("metric")

    row = sp.loc["within_topk_overlap_mean"]
    assert row["n_repeats"] == 5
    assert row["mean"] == pytest.approx(0.8)
    assert row["range"] == pytest.approx(0.0)
    assert row["first"] == pytest.approx(0.8)   # repeat 2, not repeat 1


def test_spread_keeps_the_clean_side_as_a_zero_control(step):
    """The clean baseline inserts nothing, so its band must be exactly zero.

    This is what localises the variation to ``add_items``. A step that dropped
    constant columns would drop its own control.

    ``range`` is the assertion and ``sd`` is not, because ``np.std`` over six
    identical doubles returns ~6e-17 rather than 0 -- Welford-style arithmetic
    on a mean that is not exactly representable. ``max - min`` has no such
    residue, which is a second reason the step emits it.
    """
    sp = step.spread(_repeats(step, [0.1] * 6, [1.0] * 6),
                     ["source", "rate"]).set_index("metric")
    assert sp.loc["baseline_accuracy", "range"] == 0.0
    assert sp.loc["baseline_accuracy", "sd"] < 1e-12


def test_spread_excludes_identity_columns(step):
    """A seed and a stopwatch are not metrics and have no band."""
    sp = step.spread(_repeats(step, [0.1] * 6, [1.0] * 6), ["source", "rate"])
    assert not set(sp["metric"]) & {"repeat", "elapsed_seconds",
                                    "condition_seed"}


def test_spread_is_long_on_the_metric_axis(step):
    """One row per (condition, metric) -- CLAUDE.md §7, long format only."""
    df = pd.concat([_repeats(step, [0.1] * 6, [1.0] * 6),
                    _repeats(step, [0.2] * 6, [1.0] * 6).assign(rate=0.25)])
    sp = step.spread(df, ["source", "rate"])
    assert len(sp) == 2 * 3          # two conditions x three metric columns
    assert set(sp["rate"]) == {0.25, 1.0}
    assert sp.groupby(["source", "rate", "metric"]).size().max() == 1


def test_poison_digest_separates_two_different_poison_sets(step):
    """The control that proves the repeats vary the graph and not the attack."""
    rng = np.random.default_rng(0)
    pe = rng.standard_normal((16, 128)).astype(np.float32)
    assert step.poison_digest(pe) == step.poison_digest(pe.copy())

    other = pe.copy()
    other[3, 7] += np.float32(1e-3)
    assert step.poison_digest(other) != step.poison_digest(pe)
