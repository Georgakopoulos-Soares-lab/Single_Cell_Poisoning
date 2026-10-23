"""Golden-value tests for stage 5, aggregate.

Fixture: one condition, two seeds, four query cells each.

    seed 0   n_poison_in_topk [1,0,0,0]   -> retrieval 0.25
             y_pred_poisoned wrong on 1   -> outcome   0.25
    seed 1   n_poison_in_topk [1,1,0,0]   -> retrieval 0.50
             y_pred_poisoned wrong on 2   -> outcome   0.50

so both pooled rates are 0.375.
"""

from __future__ import annotations
# 1_scripts cannot be a package (leading digit), so the shared library in
# 00_common is put on the path and imported flat.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_common"))


import numpy as np
import pandas as pd
import pytest

from schemas import (
    ATTACK,
    ATTACK_SUCCESS_RATE,
    CI_HI,
    CI_LO,
    CLEAN_ERROR_RATE,
    CLEAN_TARGET_RATE,
    DONOR_ID,
    FOCUS_SOURCE,
    NET_ATTACK_SUCCESS,
    MIN_POISON_SCORE,
    N_POISON_IN_TOPK,
    N_SEEDS,
    ONTOLOGY_DIST,
    OUTCOME_FAIL_RATE,
    PAIR,
    POISON_SURVIVES_QC,
    QC_EVASION_FOCUS,
    QC_EVASION_RATE,
    QUERY_ID,
    RATE,
    RETRIEVAL_FAIL_RATE,
    SEED,
    Y_PRED_CLEAN,
    Y_PRED_POISONED,
    Y_TRUE,
)
from aggregate import aggregate_raw

A, B = "CL:A", "CL:B"


def _seed_rows(seed: int, n_poison: list[int], wrong: int, clean_wrong: int = 0,
               qc: list[bool] | None = None) -> pd.DataFrame:
    n = len(n_poison)
    return pd.DataFrame({
        QUERY_ID: [f"q{i}" for i in range(n)],
        Y_TRUE: [A] * n,
        Y_PRED_CLEAN: [B] * clean_wrong + [A] * (n - clean_wrong),
        Y_PRED_POISONED: [B] * wrong + [A] * (n - wrong),
        N_POISON_IN_TOPK: np.array(n_poison, dtype=np.int16),
        MIN_POISON_SCORE: np.full(n, 40.0, dtype=np.float32),
        POISON_SURVIVES_QC: qc if qc is not None else [True] * n,
        ATTACK: "relabel", PAIR: "mac_to_endo", RATE: np.float32(1e-3),
        SEED: np.int16(seed), ONTOLOGY_DIST: np.int8(6),
    })


def _raw() -> pd.DataFrame:
    return pd.concat([_seed_rows(0, [1, 0, 0, 0], wrong=1),
                      _seed_rows(1, [1, 1, 0, 0], wrong=2)], ignore_index=True)


def test_one_row_per_condition_group() -> None:
    out = aggregate_raw(_raw(), bootstrap_seed=0)
    assert len(out) == 1


def test_golden_retrieval_fail_rate() -> None:
    # mean of per-seed rates: (0.25 + 0.50) / 2
    out = aggregate_raw(_raw(), bootstrap_seed=0)
    assert np.isclose(out[RETRIEVAL_FAIL_RATE].iloc[0], 0.375)


def test_golden_outcome_fail_rate() -> None:
    out = aggregate_raw(_raw(), bootstrap_seed=0)
    assert np.isclose(out[OUTCOME_FAIL_RATE].iloc[0], 0.375)


def test_clean_error_rate_is_reported_separately() -> None:
    # Without it, the classifier's own errors get attributed to the attack.
    raw = pd.concat([_seed_rows(0, [1, 0, 0, 0], wrong=1, clean_wrong=1),
                     _seed_rows(1, [1, 1, 0, 0], wrong=2, clean_wrong=1)],
                    ignore_index=True)
    out = aggregate_raw(raw, bootstrap_seed=0)
    assert np.isclose(out[CLEAN_ERROR_RATE].iloc[0], 0.25)


def test_golden_qc_evasion_rate() -> None:
    raw = pd.concat([_seed_rows(0, [1, 0, 0, 0], wrong=1, qc=[True, True, False, False]),
                     _seed_rows(1, [1, 1, 0, 0], wrong=2, qc=[True, False, False, False])],
                    ignore_index=True)
    out = aggregate_raw(raw, bootstrap_seed=0)
    assert np.isclose(out[QC_EVASION_RATE].iloc[0], 0.375)  # (0.5 + 0.25) / 2


def test_n_seeds_counts_replicates_actually_present() -> None:
    # Never assume the grid was complete.
    out = aggregate_raw(_raw(), bootstrap_seed=0)
    assert out[N_SEEDS].iloc[0] == 2


def test_ci_brackets_the_point_estimate() -> None:
    # Ported: the interval now brackets NET_ATTACK_SUCCESS, not outcome failure.
    rows = [_paired_rows(s, ["d0"] * 4 + ["d1"] * 4, [False] * 8,
                         [True] * 2 + [False] * 6) for s in range(4)]
    out = aggregate_raw(pd.concat(rows, ignore_index=True), bootstrap_seed=0,
                        pair_meta={"mac_to_endo": [{"source": A, "target": B}]},
                        donor_column=DONOR_ID)
    r = out.iloc[0]
    assert r[CI_LO] <= r[NET_ATTACK_SUCCESS] <= r[CI_HI]

def test_ci_lies_within_the_plausible_range_of_an_effect() -> None:
    # Ported: a percentile bootstrap on a difference of rates cannot leave
    # [-1, 1], and with a uniformly positive effect should stay non-negative.
    rows = [_paired_rows(s, ["d0"] * 4 + ["d1"] * 4, [False] * 8,
                         [True] * 2 + [False] * 6) for s in range(4)]
    out = aggregate_raw(pd.concat(rows, ignore_index=True), bootstrap_seed=0,
                        pair_meta={"mac_to_endo": [{"source": A, "target": B}]},
                        donor_column=DONOR_ID)
    r = out.iloc[0]
    assert -1.0 <= r[CI_LO] <= r[CI_HI] <= 1.0
    assert r[CI_LO] >= 0.0

def test_a_single_donor_and_seed_collapses_the_interval() -> None:
    # Ported from the seed-only version. With one donor and one seed there is
    # one independent observation on each axis, so every resample is identical
    # and the interval must collapse. A cell-level bootstrap would report a
    # narrow but non-zero interval here, which is the false precision this
    # design exists to avoid.
    df = _paired_rows(0, ["d0"] * 8, [False] * 8, [True] * 2 + [False] * 6)
    out = aggregate_raw(df, bootstrap_seed=0,
                        pair_meta={"mac_to_endo": [{"source": A, "target": B}]},
                        donor_column=DONOR_ID)
    r = out.iloc[0]
    assert np.isclose(r[CI_LO], r[CI_HI])

def test_same_bootstrap_seed_reproduces_the_interval() -> None:
    rows = [_paired_rows(s, ["d0"] * 4 + ["d1"] * 4, [False] * 8,
                         [True] * 2 + [False] * 6) for s in range(4)]
    df = pd.concat(rows, ignore_index=True)
    meta = {"mac_to_endo": [{"source": A, "target": B}]}
    a = aggregate_raw(df, bootstrap_seed=3, pair_meta=meta, donor_column=DONOR_ID).iloc[0]
    b = aggregate_raw(df, bootstrap_seed=3, pair_meta=meta, donor_column=DONOR_ID).iloc[0]
    assert (a[CI_LO], a[CI_HI]) == (b[CI_LO], b[CI_HI])

def test_ontology_distance_survives_aggregation() -> None:
    # It is an experimental axis, not a covariate to be averaged away.
    out = aggregate_raw(_raw(), bootstrap_seed=0)
    assert out[ONTOLOGY_DIST].iloc[0] == 6


def test_separate_conditions_stay_separate() -> None:
    other = _seed_rows(0, [0, 0, 0, 0], wrong=0)
    other[PAIR] = "mac_to_alvmac"
    out = aggregate_raw(pd.concat([_raw(), other], ignore_index=True), bootstrap_seed=0)
    assert len(out) == 2
    assert set(out[PAIR]) == {"mac_to_endo", "mac_to_alvmac"}


def test_label_columns_stored_as_categories_can_still_be_compared() -> None:
    # REGRESSION. raw_result stores y_true / y_pred_* as `category` per the
    # schema, to keep 1,260 x 49,456 rows small. Within one frame the columns
    # hold DIFFERENT category sets -- y_true sees only the true labels, while
    # y_pred_poisoned also sees the attacker's target -- and pandas refuses to
    # compare categoricals whose categories differ:
    #   "Categoricals can only be compared if 'categories' are the same."
    # This killed aggregation after all 1,260 sweeps had already succeeded.
    df = _seed_rows(0, [1, 1, 0, 0], wrong=2)
    df[Y_TRUE] = pd.Categorical(df[Y_TRUE], categories=[A])
    df[Y_PRED_CLEAN] = pd.Categorical(df[Y_PRED_CLEAN], categories=[A, B])
    df[Y_PRED_POISONED] = pd.Categorical(df[Y_PRED_POISONED], categories=[A, B])
    out = aggregate_raw(df, bootstrap_seed=0)
    assert np.isclose(out[OUTCOME_FAIL_RATE].iloc[0], 0.5)
    assert np.isclose(out[CLEAN_ERROR_RATE].iloc[0], 0.0)


def test_attack_success_counts_only_source_cells_landing_on_the_target() -> None:
    # 4 cells, all truly CL:A (the source). Two are voted CL:B (the target)
    # after poisoning -> 50%. Cells whose true label is not the source are not
    # what the attack aims at and must not dilute the denominator.
    df = _seed_rows(0, [1, 1, 0, 0], wrong=2)
    out = aggregate_raw(df, bootstrap_seed=0,
                        pair_meta={"mac_to_endo": [{"source": A, "target": B}]})
    assert np.isclose(out[ATTACK_SUCCESS_RATE].iloc[0], 0.5)


def test_attack_success_ignores_cells_of_other_types() -> None:
    df = _seed_rows(0, [1, 1, 0, 0], wrong=2)
    df[Y_TRUE] = [A, A, "CL:Z", "CL:Z"]      # last two are not the source
    # of the two source cells, both were voted B -> 100%
    out = aggregate_raw(df, bootstrap_seed=0,
                        pair_meta={"mac_to_endo": [{"source": A, "target": B}]})
    assert np.isclose(out[ATTACK_SUCCESS_RATE].iloc[0], 1.0)


def test_clean_target_rate_is_the_unpoisoned_baseline() -> None:
    # Without it, a target the classifier already confuses the source with
    # would read as attack success.
    df = _seed_rows(0, [1, 1, 0, 0], wrong=2, clean_wrong=1)
    out = aggregate_raw(df, bootstrap_seed=0,
                        pair_meta={"mac_to_endo": [{"source": A, "target": B}]})
    assert np.isclose(out[CLEAN_TARGET_RATE].iloc[0], 0.25)


def test_source_absent_from_the_queries_is_refused() -> None:
    df = _seed_rows(0, [1, 1, 0, 0], wrong=2)
    df[Y_TRUE] = ["CL:Z"] * 4
    with pytest.raises(ValueError, match="no query cells"):
        aggregate_raw(df, bootstrap_seed=0,
                      pair_meta={"mac_to_endo": [{"source": A, "target": B}]})


def test_a_combo_emits_one_row_per_member_source() -> None:
    # The comparison the experiment exists to make -- member-under-combo vs
    # member-alone -- needs the same denominator on both sides, so each member
    # gets its own row scoped to its own source cells.
    df = _seed_rows(0, [1, 1, 1, 1], wrong=2)
    df[PAIR] = "combo"
    df[Y_TRUE] = [A, A, "CL:Z", "CL:Z"]
    df[Y_PRED_POISONED] = [B, B, "CL:Y", A]
    out = aggregate_raw(
        df, bootstrap_seed=0,
        pair_meta={"combo": [{"source": A, "target": B},
                             {"source": "CL:Z", "target": "CL:Y"}]},
    )
    assert len(out) == 2
    assert set(out[FOCUS_SOURCE]) == {A, "CL:Z"}
    got = dict(zip(out[FOCUS_SOURCE], out[ATTACK_SUCCESS_RATE]))
    assert np.isclose(got[A], 1.0)          # both CL:A cells voted CL:B
    assert np.isclose(got["CL:Z"], 0.5)     # one of two CL:Z cells voted CL:Y


def test_a_single_attack_still_emits_exactly_one_row() -> None:
    df = _seed_rows(0, [1, 1, 0, 0], wrong=2)
    out = aggregate_raw(df, bootstrap_seed=0,
                        pair_meta={"mac_to_endo": [{"source": A, "target": B}]})
    assert len(out) == 1
    assert out[FOCUS_SOURCE].iloc[0] == A


def test_streaming_by_group_matches_aggregating_everything_at_once() -> None:
    # 2_results/GSE136831/00_raw/raw is 7,672 files x 49,456 rows = 379M rows. Concatenating them
    # all reached 18 GB resident with 27 GB of swap before being killed. Since
    # a group is (attack, pair, rate) across seeds, and no statistic crosses
    # group boundaries, groups can be aggregated one at a time. This pins that
    # the streaming path gives an identical answer.
    a = _seed_rows(0, [1, 0, 0, 0], wrong=1)
    b = _seed_rows(1, [1, 1, 0, 0], wrong=2)
    c = _seed_rows(0, [0, 0, 0, 0], wrong=0)
    c[PAIR] = "other_pair"
    meta = {"mac_to_endo": [{"source": A, "target": B}],
            "other_pair": [{"source": A, "target": B}]}

    at_once = aggregate_raw(pd.concat([a, b, c], ignore_index=True),
                            bootstrap_seed=0, pair_meta=meta)
    streamed = pd.concat(
        [aggregate_raw(pd.concat([a, b], ignore_index=True), bootstrap_seed=0,
                       pair_meta=meta),
         aggregate_raw(c, bootstrap_seed=0, pair_meta=meta)],
        ignore_index=True,
    )
    key = [ATTACK, PAIR, RATE]
    lhs = at_once.sort_values(key).reset_index(drop=True)
    rhs = streamed.sort_values(key).reset_index(drop=True)
    for col in (RETRIEVAL_FAIL_RATE, OUTCOME_FAIL_RATE, ATTACK_SUCCESS_RATE):
        assert np.allclose(lhs[col], rhs[col]), col


def test_focused_qc_evasion_counts_only_cells_of_the_attacked_type() -> None:
    # qc_evasion_rate is over ALL queries and is therefore dominated by how much
    # of the query set the attacked type represents: macrophages are 36% of
    # these queries, NK cells 3.8%. The focused version says how exposed the
    # attacked TYPE is, and orders the types completely differently.
    df = _seed_rows(0, [1, 1, 1, 1], wrong=2)
    df[Y_TRUE] = [A, A, "CL:Z", "CL:Z"]
    df[POISON_SURVIVES_QC] = [True, False, True, True]
    out = aggregate_raw(df, bootstrap_seed=0,
                        pair_meta={"mac_to_endo": [{"source": A, "target": B}]})
    assert np.isclose(out[QC_EVASION_RATE].iloc[0], 0.75)    # 3 of 4 overall
    assert np.isclose(out[QC_EVASION_FOCUS].iloc[0], 0.50)   # 1 of 2 CL:A cells


def _donor_rows(seed, n_poison, wrong, donors):
    df = _seed_rows(seed, n_poison, wrong=wrong)
    df[DONOR_ID] = donors
    return df


def test_donor_bootstrap_collapses_when_every_cell_shares_one_donor() -> None:
    # THE discriminating test. With one donor there is one independent
    # observation, so resampling donors can produce only that donor and the
    # interval must collapse. A cell-level bootstrap would report a narrow but
    # non-zero interval here — precisely the false precision this fixes.
    df = pd.concat([_donor_rows(s, [1, 1, 0, 0], 2, ["d0"] * 4) for s in (0, 1)],
                   ignore_index=True)
    out = aggregate_raw(df, bootstrap_seed=0,
                        pair_meta={"mac_to_endo": [{"source": A, "target": B}]},
                        donor_column=DONOR_ID)
    assert np.isclose(out[CI_LO].iloc[0], out[CI_HI].iloc[0])


def test_more_donors_tighten_the_interval() -> None:
    # Replaces a comparison between seed-only and donor bootstraps, which no
    # longer both exist. The property that matters now: more independent donors
    # means a tighter interval on the same average effect.
    #
    # Donors must DIFFER in effect for the interval to have anything to measure.
    # With every donor identical the between-donor variance is zero and both
    # intervals collapse to width zero, which says nothing.
    meta = {"mac_to_endo": [{"source": A, "target": B}]}

    def width(n_donors):
        rows = []
        for s in range(4):
            donors, pois = [], []
            for i in range(n_donors):
                donors += [f"d{i}"] * 4
                hits = i % 4                      # 0, 1, 2 or 3 of this donor's 4 cells
                pois += [True] * hits + [False] * (4 - hits)
            rows.append(_paired_rows(s, donors, [False] * len(donors), pois))
        out = aggregate_raw(pd.concat(rows, ignore_index=True), bootstrap_seed=0,
                            pair_meta=meta, donor_column=DONOR_ID)
        return float(out[CI_HI].iloc[0] - out[CI_LO].iloc[0])

    w_few, w_many = width(4), width(40)
    assert w_few > 0, "no between-donor variance to measure"
    assert w_many < w_few, f"{w_many:.3f} not tighter than {w_few:.3f}"


def _paired_rows(seed, donors, clean_hits, poison_hits):
    """One row per cell; each donor supplies its own clean and poisoned outcome."""
    n = len(donors)
    return pd.DataFrame({
        QUERY_ID: [f"q{seed}_{i}" for i in range(n)],
        Y_TRUE: [A] * n,
        Y_PRED_CLEAN: [B if h else A for h in clean_hits],
        Y_PRED_POISONED: [B if h else A for h in poison_hits],
        N_POISON_IN_TOPK: np.ones(n, dtype=np.int16),
        MIN_POISON_SCORE: np.full(n, 40.0, dtype=np.float32),
        POISON_SURVIVES_QC: [True] * n,
        DONOR_ID: donors,
        ATTACK: "relabel", PAIR: "mac_to_endo", RATE: np.float32(0.25),
        SEED: np.int16(seed), ONTOLOGY_DIST: np.int8(6),
    })


def test_pairing_cancels_donor_variation_in_the_baseline() -> None:
    # THE point of pairing. Two donors with wildly different baselines — d0 is
    # never confused with the target, d1 usually is — and the attack adds the
    # same small effect to each. Unpaired, the interval would be dominated by
    # the 80-point baseline gap. Paired, it sits near the effect.
    rows = []
    for s in range(6):
        donors = ["d0"] * 10 + ["d1"] * 10
        clean = [False] * 10 + [True] * 8 + [False] * 2
        pois = [True] * 1 + [False] * 9 + [True] * 9 + [False] * 1
        rows.append(_paired_rows(s, donors, clean, pois))
    out = aggregate_raw(pd.concat(rows, ignore_index=True), bootstrap_seed=0,
                        pair_meta={"mac_to_endo": [{"source": A, "target": B}]},
                        donor_column=DONOR_ID)
    r = out.iloc[0]
    assert np.isclose(r[NET_ATTACK_SUCCESS], 0.10, atol=0.02), r[NET_ATTACK_SUCCESS]
    assert float(r[CI_HI] - r[CI_LO]) < 0.30, "pairing did not cancel the baseline"


def test_net_attack_success_is_the_difference_of_the_two_rates() -> None:
    rows = [_paired_rows(s, ["d0"] * 4 + ["d1"] * 4,
                         [False] * 8, [True] * 2 + [False] * 6) for s in range(3)]
    out = aggregate_raw(pd.concat(rows, ignore_index=True), bootstrap_seed=0,
                        pair_meta={"mac_to_endo": [{"source": A, "target": B}]},
                        donor_column=DONOR_ID)
    r = out.iloc[0]
    assert np.isclose(r[NET_ATTACK_SUCCESS],
                      r[ATTACK_SUCCESS_RATE] - r[CLEAN_TARGET_RATE], atol=1e-6)


def test_point_estimates_are_unchanged_by_donor_resampling() -> None:
    # The bootstrap changes the INTERVAL, not the estimate. If the point moved,
    # every previously reported number would silently shift.
    rows = [_paired_rows(s, ["d0"] * 4 + ["d1"] * 4,
                         [False] * 8, [True] * 2 + [False] * 6) for s in range(3)]
    df = pd.concat(rows, ignore_index=True)
    meta = {"mac_to_endo": [{"source": A, "target": B}]}
    a = aggregate_raw(df, bootstrap_seed=0, pair_meta=meta)
    b = aggregate_raw(df, bootstrap_seed=0, pair_meta=meta, donor_column=DONOR_ID)
    for col in (OUTCOME_FAIL_RATE, RETRIEVAL_FAIL_RATE, ATTACK_SUCCESS_RATE):
        assert np.isclose(a[col].iloc[0], b[col].iloc[0]), col
