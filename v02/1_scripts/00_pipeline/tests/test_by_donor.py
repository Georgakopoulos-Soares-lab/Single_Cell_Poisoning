"""``by_donor`` must re-pool to exactly what ``scope`` already publishes.

The threshold dose (SPEC.md 5.4c) is estimated from per-donor sums rather than
from the aggregated sweep table, because a threshold interval needs ONE donor
draw held fixed across every dose on the ladder and an already-pooled table
cannot supply that. That only works if the two reductions of the same indicator
dict agree: if ``sum(num)/sum(den)`` over all donors disagreed with the point
estimate in the published table, the threshold would be a threshold on a
different curve than the one every figure in the project draws.

So the central test here is an identity, not a tolerance -- with one deliberate
exception, documented at :func:`test_pooled_matches_scope`.
"""
import numpy as np
import pytest

from celltype_sweep import Bench, by_donor, scope
from schemas import DEN_SUFFIX, NUM_SUFFIX

N_Q = 400
N_DONORS = 9


@pytest.fixture
def world() -> tuple[np.ndarray, np.ndarray, dict]:
    """A small scored condition: donors, a within-scope selection, indicators.

    Returns:
        ``(donor, sel, ind)`` -- (n_q,) donor ids, (m,) selected query indices,
        and an indicator dict in both shapes ``scope`` accepts.
    """
    rng = np.random.default_rng(0)
    donor = rng.integers(0, N_DONORS, N_Q).astype(np.int64)
    sel = np.flatnonzero(rng.random(N_Q) < 0.4)
    reach = (rng.random(N_Q) < 0.85).astype(float)
    ind = {
        "retrieval": (rng.random(N_Q) < 0.6).astype(float),
        # a per-query DIFFERENCE, so its numerator goes negative
        "net_attack_success": rng.integers(-1, 2, N_Q).astype(float),
        # a mean over a subset: numerator and denominator differ
        "mean_error_dist": (rng.integers(0, 5, N_Q) * reach, reach),
    }
    return donor, sel, ind


def test_pooled_matches_scope(world) -> None:
    """Pooling the donor sums reproduces ``scope``'s point estimate exactly.

    Compared with ``==`` on the ratio rather than ``approx``: both sides sum the
    same float64 values with ``np.bincount`` and divide once, so any difference
    would be a difference in WHICH values were summed, not floating-point noise.
    A tolerance here would hide exactly the bug this test exists to catch.
    """
    donor, sel, ind = world
    vocab = np.unique(donor)
    got = by_donor("within", sel, donor, vocab, **ind)
    ref = scope("within", sel, donor, 0, **ind)

    for name in ind:
        pooled = (got[f"within_{name}{NUM_SUFFIX}"].sum()
                  / got[f"within_{name}{DEN_SUFFIX}"].sum())
        assert pooled == ref[f"within_{name}"], name


def test_every_donor_gets_a_row_even_with_no_queries(world) -> None:
    """Absent donors are structural zeros, not missing rows.

    The threshold bootstrap holds one donor draw fixed across all doses, which
    requires the donor roster to be identical in every condition. A donor who
    contributes no query to ``within_`` must still appear, with a denominator of
    0 -- dropping the row would silently change the roster per condition and
    make the fixed draw impossible.
    """
    donor, _, ind = world
    vocab = np.unique(donor)
    lonely = np.flatnonzero(donor == vocab[0])[:1]      # one donor, one query
    got = by_donor("within", lonely, donor, vocab, **ind)

    assert len(got["within_retrieval" + DEN_SUFFIX]) == len(vocab)
    assert got["within_retrieval" + DEN_SUFFIX][0] == 1.0
    assert (got["within_retrieval" + DEN_SUFFIX][1:] == 0.0).all()
    assert not np.isnan(got["within_retrieval" + NUM_SUFFIX]).any()


def test_n_queries_counts_queries_not_denominators(world) -> None:
    """``n_queries`` is the plain count, which ``mean_error_dist`` is not.

    ``mean_error_dist`` denominates by REACHABILITY, so its denominator is
    smaller than the query count wherever a prediction has no ontology path. A
    reader weighting donors must not reach for that denominator by mistake.
    """
    donor, sel, ind = world
    vocab = np.unique(donor)
    got = by_donor("within", sel, donor, vocab, **ind)

    assert got["within_n_queries"].sum() == len(sel)
    assert (got["within_mean_error_dist" + DEN_SUFFIX].sum()
            < got["within_n_queries"].sum())
    # a rate indicator's denominator IS the query count
    assert got["within_retrieval" + DEN_SUFFIX].sum() == len(sel)


def test_negative_numerators_survive(world) -> None:
    """A ``_net`` numerator may be negative and must not be clipped."""
    donor, sel, ind = world
    vocab = np.unique(donor)
    got = by_donor("within", sel, donor, vocab, **ind)
    assert (got["within_net_attack_success" + NUM_SUFFIX] < 0).any()


def test_golden_hand_checked() -> None:
    """A fixed tiny input with an expected output worked out by hand.

    Three donors, six queries. Donor 7 has queries 0 and 1 with retrieval 1 and
    0; donor 8 has queries 2, 3 with 1 and 1; donor 9 has queries 4, 5 with 0
    and 0. Selecting all six: numerators 1, 2, 0 and denominators 2, 2, 2, so
    the pooled rate is 3/6 = 0.5.
    """
    donor = np.array([7, 7, 8, 8, 9, 9], dtype=np.int64)
    ind = {"retrieval": np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])}
    got = by_donor("within", np.arange(6), donor, np.array([7, 8, 9]), **ind)

    assert got["within_retrieval" + NUM_SUFFIX].tolist() == [1.0, 2.0, 0.0]
    assert got["within_retrieval" + DEN_SUFFIX].tolist() == [2.0, 2.0, 2.0]
    assert got["within_n_queries"].tolist() == [2, 2, 2]
    assert (got["within_retrieval" + NUM_SUFFIX].sum()
            / got["within_retrieval" + DEN_SUFFIX].sum()) == 0.5


def test_per_donor_puts_both_scopes_on_one_row(world) -> None:
    """``within_`` and ``global_`` share a row, and both re-pool exactly.

    The SCOPE exception (CLAUDE.md §7) exists so a paired difference between
    scopes is a column subtraction rather than a self-join. That only holds if
    both scopes span the SAME donor roster in the same order -- which is why
    :meth:`Bench.per_donor` emits the full roster for both rather than each
    scope's own donors.
    """
    import types

    donor, sel, ind = world
    fake = types.SimpleNamespace(donor=donor, truth=np.zeros(N_Q))
    df = Bench.per_donor(fake, ind, sel)

    assert len(df) == len(np.unique(donor))
    assert df["donor_id"].is_monotonic_increasing
    for sc, s_sel in (("within", sel), ("global", np.arange(N_Q))):
        ref = scope(sc, s_sel, donor, 0, **ind)
        for name in ind:
            pooled = df[f"{sc}_{name}{NUM_SUFFIX}"].sum() / df[f"{sc}_{name}{DEN_SUFFIX}"].sum()
            assert pooled == ref[f"{sc}_{name}"], (sc, name)
    assert df["global_n_queries"].sum() == N_Q
    assert df["within_n_queries"].sum() == len(sel)


# ---------------------------------------------------------------------------
# donor_tables -- which by_donor table represents a step. A bare
# glob("*_by_donor.parquet") reads the EXACT table for 02b_11 and 02b_15,
# because their graph run writes <stem>_by_donor_graph.parquet and the _graph
# lands after the _by_donor. Found 2026-09-26, before either aggregator ran.
# ---------------------------------------------------------------------------

def _touch(d, *names):
    for n in names:
        (d / n).write_bytes(b"")
    return d


def test_graph_variant_wins_over_exact(tmp_path) -> None:
    """The whole point: never summarise the understating table when the faithful
    one is sitting beside it."""
    from flood_common import donor_tables
    _touch(tmp_path,
           "02b_15_with_access_alpha_sweep_by_donor.parquet",
           "02b_15_with_access_alpha_sweep_by_donor_graph.parquet")
    got = [p.name for p in donor_tables(tmp_path)]
    assert got == ["02b_15_with_access_alpha_sweep_by_donor_graph.parquet"]


def test_order_of_discovery_does_not_decide(tmp_path) -> None:
    """Graph must win whether it sorts before or after the exact name."""
    from flood_common import donor_tables
    _touch(tmp_path, "aa_by_donor_graph.parquet", "aa_by_donor.parquet")
    assert [p.name for p in donor_tables(tmp_path)] == ["aa_by_donor_graph.parquet"]


def test_exact_is_kept_when_no_graph_run_exists(tmp_path) -> None:
    from flood_common import donor_tables
    _touch(tmp_path, "02b_13_no_access_flood_geometry_by_donor.parquet")
    got = [p.name for p in donor_tables(tmp_path)]
    assert got == ["02b_13_no_access_flood_geometry_by_donor.parquet"]


def test_unmerged_shards_are_excluded(tmp_path) -> None:
    """merge_shards has no cleanup, so shard files persist beside the canonical
    table. Reading both would double-count every condition in them."""
    from flood_common import donor_tables
    _touch(tmp_path,
           "02b_02_no_access_flood_by_donor.parquet",
           "02b_02_no_access_flood_by_donor_shard0.parquet",
           "02b_02_no_access_flood_by_donor_shard1.parquet",
           "02b_02_no_access_flood_by_donor_shard2.parquet")
    got = [p.name for p in donor_tables(tmp_path)]
    assert got == ["02b_02_no_access_flood_by_donor.parquet"]


def test_graph_shards_are_excluded_too(tmp_path) -> None:
    from flood_common import donor_tables
    _touch(tmp_path,
           "02b_15_with_access_alpha_sweep_by_donor_graph_shard0.parquet",
           "02b_15_with_access_alpha_sweep_by_donor_graph_shard1.parquet")
    assert donor_tables(tmp_path) == []


def test_one_entry_per_step(tmp_path) -> None:
    from flood_common import donor_tables
    _touch(tmp_path,
           "02b_01_no_access_celltype_sweep_by_donor.parquet",
           "02b_02_no_access_flood_by_donor.parquet",
           "02b_11_with_access_inplace_by_donor.parquet",
           "02b_11_with_access_inplace_by_donor_graph.parquet",
           "02b_15_with_access_alpha_sweep_by_donor.parquet",
           "02b_15_with_access_alpha_sweep_by_donor_graph.parquet")
    got = [p.name for p in donor_tables(tmp_path)]
    assert len(got) == 4, got
    assert sum("_graph" in n for n in got) == 2
    steps = [n.partition("_by_donor")[0] for n in got]
    assert len(set(steps)) == len(steps), "a step appeared twice"


def test_empty_directory_returns_empty(tmp_path) -> None:
    """The callers turn this into their own sys.exit message, so it must not
    raise here."""
    from flood_common import donor_tables
    assert donor_tables(tmp_path) == []


def test_non_donor_tables_are_ignored(tmp_path) -> None:
    from flood_common import donor_tables
    _touch(tmp_path,
           "02b_13_no_access_flood_geometry.parquet",
           "02b_13_no_access_flood_geometry_by_donor.parquet",
           "summary.parquet")
    got = [p.name for p in donor_tables(tmp_path)]
    assert got == ["02b_13_no_access_flood_geometry_by_donor.parquet"]
