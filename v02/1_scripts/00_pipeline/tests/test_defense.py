"""Tests for the defence layers.

Ported from the protein arm's design (Phase 9), adapted to single-cell. Each
layer targets a measured attack rather than a plausible-sounding threat:

  recompute          -> the `interpolate` ceiling. Its vectors are blends no
                        count matrix produces, so re-deriving embeddings from
                        raw data rejects them and nothing else.
  label_consistency  -> `relabel`. A relabelled cell sits among its true type
                        while carrying another type's label, so its own
                        neighbourhood contradicts it.
  donor_vote         -> `duplicate`. Copies of one cell collapse to one vote.
  abstain            -> declines rather than guessing on a thin margin.
"""

from __future__ import annotations
# 1_scripts cannot be a package (leading digit), so the shared library in
# 00_common is put on the path and imported flat.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_common"))


import numpy as np
import pytest

from defense import DefenseConfig, admit, defended_vote, label_agreement

A, B = "CL:A", "CL:B"


def test_no_defence_reproduces_the_plain_majority_vote() -> None:
    lab = np.array([A, A, B])
    assert defended_vote(lab, DefenseConfig()) == A


def test_recompute_drops_records_with_no_underlying_data() -> None:
    # Three poisoned blends outvote two genuine cells until recompute runs.
    lab = np.array([B, B, B, A, A])
    ok = np.array([False, False, False, True, True])
    assert defended_vote(lab, DefenseConfig()) == B
    assert defended_vote(lab, DefenseConfig(recompute=True), recompute_ok=ok) == A


def test_recompute_does_nothing_against_verbatim_reference_vectors() -> None:
    # relabel and duplicate emit real reference embeddings, so every record
    # passes recompute. This is why the ceiling gap is the measurement.
    lab = np.array([B, B, B, A, A])
    ok = np.ones(5, dtype=bool)
    assert defended_vote(lab, DefenseConfig(recompute=True), recompute_ok=ok) == B


def test_label_consistency_drops_a_record_its_neighbourhood_contradicts() -> None:
    lab = np.array([B, B, B, A, A])
    lc = np.array([0.0, 0.0, 0.0, 0.95, 0.95])   # the three B's are relabelled
    assert defended_vote(lab, DefenseConfig(label_consistency=True, lc_threshold=0.5),
                         label_consistency=lc) == A


def test_label_consistency_keeps_genuinely_rare_but_coherent_records() -> None:
    # A real minority type whose neighbours agree with it must survive.
    lab = np.array([B, A, A])
    lc = np.array([0.9, 0.9, 0.9])
    assert defended_vote(lab, DefenseConfig(label_consistency=True, lc_threshold=0.5),
                         label_consistency=lc) == A


def test_donor_vote_collapses_copies_of_one_cell() -> None:
    # duplicate's mechanism: four copies from one donor outvote two genuine
    # cells from two donors, until each donor gets one vote.
    lab = np.array([B, B, B, B, A, A])
    donor = np.array(["attacker", "attacker", "attacker", "attacker", "d1", "d2"])
    assert defended_vote(lab, DefenseConfig()) == B
    assert defended_vote(lab, DefenseConfig(donor_vote=True), donor=donor) == A


def test_donor_vote_leaves_genuinely_independent_agreement_alone() -> None:
    lab = np.array([B, B, B, A, A])
    donor = np.array(["d1", "d2", "d3", "d4", "d5"])
    assert defended_vote(lab, DefenseConfig(donor_vote=True), donor=donor) == B


def test_abstain_declines_on_a_thin_margin() -> None:
    lab = np.array([A, A, B, B])           # 2 v 2
    assert defended_vote(lab, DefenseConfig(abstain=True, abstain_min_margin=0.25)) is None


def test_abstain_answers_when_the_margin_is_clear() -> None:
    lab = np.array([A, A, A, B])           # 3 v 1, margin 0.5
    assert defended_vote(lab, DefenseConfig(abstain=True, abstain_min_margin=0.25)) == A


def test_layers_compose() -> None:
    lab = np.array([B, B, B, B, A, A])
    donor = np.array(["atk", "atk", "atk", "atk", "d1", "d2"])
    lc = np.array([0.0, 0.0, 0.0, 0.0, 0.9, 0.9])
    cfg = DefenseConfig(label_consistency=True, lc_threshold=0.5, donor_vote=True)
    assert defended_vote(lab, cfg, label_consistency=lc, donor=donor) == A


def test_config_names_itself_for_reporting() -> None:
    assert DefenseConfig().name == "none"
    assert DefenseConfig(recompute=True, donor_vote=True).name == "recompute+donor"


def test_filtering_everything_away_abstains_rather_than_guessing() -> None:
    lab = np.array([B, B])
    ok = np.array([False, False])
    assert defended_vote(lab, DefenseConfig(recompute=True), recompute_ok=ok) is None


def test_label_agreement_is_the_fraction_of_neighbours_that_concur() -> None:
    assert label_agreement(A, [A, A, A, B]) == 0.75
    assert label_agreement(A, [B, B, B, B]) == 0.0
    # no neighbours is no evidence, so do not flag on emptiness
    assert label_agreement(A, []) == 1.0


def test_admit_requires_every_enabled_filter_to_pass() -> None:
    cfg = DefenseConfig(recompute=True, label_consistency=True, lc_threshold=0.5)
    assert admit(cfg, recompute_ok=True, label_consistency=0.9)
    assert not admit(cfg, recompute_ok=False, label_consistency=0.9)
    assert not admit(cfg, recompute_ok=True, label_consistency=0.1)


def test_missing_evidence_for_an_enabled_layer_is_refused() -> None:
    # Silently treating absent scores as passing would report a defence that
    # never ran as a defence that found nothing.
    with pytest.raises(ValueError, match="recompute_ok"):
        defended_vote(np.array([A, B]), DefenseConfig(recompute=True))
