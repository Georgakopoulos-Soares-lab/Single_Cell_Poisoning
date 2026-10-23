"""Tests for the defended sweep — what produces the defence table.

A defence is only a result once its COSTS are measured beside its benefit. A
filter that rejects everything drives attack success to zero and is useless, so
the evaluation reports four numbers per configuration:

    attack_success   did the vote land on the attacker's label
    clean_accuracy   what the same defence costs on unpoisoned data
    false_positive   genuine records the defence rejected
    abstention       queries it declined rather than answered
"""

from __future__ import annotations
# 1_scripts cannot be a package (leading digit), so the shared library in
# 00_common is put on the path and imported flat.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_common"))


import numpy as np

from defense import DefenseConfig
from defended_sweep import evaluate_defense

A, B = "CL:A", "CL:B"


def _case():
    """Two queries. Both truly CL:A. Poison claims CL:B and holds 3 of 5 slots."""
    labels = np.array([[B, B, B, A, A], [B, B, B, A, A]])
    is_poison = np.array([[True, True, True, False, False]] * 2)
    donor = np.array([["atk", "atk", "atk", "d1", "d2"]] * 2)
    lc = np.array([[0.0, 0.0, 0.0, 0.9, 0.9]] * 2)
    ok = np.array([[True, True, True, True, True]] * 2)
    return dict(labels=labels, is_poison=is_poison, donor=donor,
                label_consistency=lc, recompute_ok=ok,
                y_true=np.array([A, A]), target=B)


def test_undefended_attack_succeeds() -> None:
    r = evaluate_defense(cfg=DefenseConfig(), **_case())
    assert np.isclose(r["attack_success"], 1.0)


def test_label_consistency_defeats_the_relabel_attack() -> None:
    r = evaluate_defense(cfg=DefenseConfig(label_consistency=True, lc_threshold=0.5),
                         **_case())
    assert np.isclose(r["attack_success"], 0.0)


def test_donor_vote_defeats_flooding_from_one_source() -> None:
    r = evaluate_defense(cfg=DefenseConfig(donor_vote=True), **_case())
    assert np.isclose(r["attack_success"], 0.0)


def test_false_positive_rate_counts_genuine_records_rejected() -> None:
    # A threshold high enough to reject honest records must show that cost.
    r = evaluate_defense(cfg=DefenseConfig(label_consistency=True, lc_threshold=0.95),
                         **_case())
    assert r["false_positive"] > 0


def test_a_defence_that_rejects_nothing_has_no_false_positives() -> None:
    r = evaluate_defense(cfg=DefenseConfig(label_consistency=True, lc_threshold=0.5),
                         **_case())
    assert np.isclose(r["false_positive"], 0.0)


def test_abstention_is_reported_separately_from_error() -> None:
    # Declining is a refusal, not a wrong answer. Conflating them would let a
    # defence that answers nothing look perfectly accurate.
    c = _case()
    r = evaluate_defense(cfg=DefenseConfig(abstain=True, abstain_min_margin=0.9), **c)
    assert np.isclose(r["abstention"], 1.0)
    assert np.isclose(r["attack_success"], 0.0)


def test_clean_accuracy_is_measured_without_poison() -> None:
    # The same filter runs on the unpoisoned neighbourhood, so its cost to
    # ordinary annotation is visible.
    c = _case()
    r = evaluate_defense(cfg=DefenseConfig(), **c)
    assert np.isclose(r["clean_accuracy"], 1.0)


def test_an_aggressive_filter_shows_its_cost_in_clean_accuracy() -> None:
    c = _case()
    c["label_consistency"] = np.array([[0.0, 0.0, 0.0, 0.1, 0.1]] * 2)  # honest ones look bad too
    r = evaluate_defense(cfg=DefenseConfig(label_consistency=True, lc_threshold=0.5), **c)
    assert r["clean_accuracy"] < 1.0


def test_accuracy_is_the_poisoned_vote_not_the_poison_removed_one() -> None:
    """Regression, 2026-10-04: callers stored ``correct`` (poison removed) as
    accuracy, so an undefended attack that flips every query read as fully
    accurate. ``accurate`` is the vote actually returned."""
    r = evaluate_defense(cfg=DefenseConfig(), per_query=True, **_case())
    assert not r["accurate"].any()          # every query voted CL:B, truth CL:A
    assert np.isclose(r["accuracy"], 0.0)
    assert r["correct"].all()               # with poison removed, A wins
    assert np.isclose(r["clean_accuracy"], 1.0)


def test_defended_accuracy_recovers_when_poison_is_filtered() -> None:
    r = evaluate_defense(cfg=DefenseConfig(label_consistency=True,
                                           lc_threshold=0.5),
                         per_query=True, **_case())
    assert r["accurate"].all()
    assert np.isclose(r["accuracy"], 1.0)
