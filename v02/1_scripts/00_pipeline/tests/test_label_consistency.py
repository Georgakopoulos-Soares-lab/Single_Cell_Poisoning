"""Tests for label-consistency scoring — the input the defence needs.

A record's score is the fraction of ITS OWN nearest reference neighbours that
carry the same label. A relabelled cell sits among its true type while claiming
another, so its score collapses toward zero; a genuine record sits among its own
kind and scores high.
"""

from __future__ import annotations
# 1_scripts cannot be a package (leading digit), so the shared library in
# 00_common is put on the path and imported flat.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_common"))


import numpy as np

from label_consistency import score_records

# Two tight clusters: rows 0-4 are CL:A near e0, rows 5-9 are CL:B near e1.
REF = np.zeros((10, 4), dtype=np.float32)
REF[:5, 0] = 1.0
REF[5:, 1] = 1.0
REF += np.array([[0, 0, 0.01, 0]], dtype=np.float32) * np.arange(10).reshape(-1, 1)
REF /= np.linalg.norm(REF, axis=1, keepdims=True)
REF_LAB = np.array(["CL:A"] * 5 + ["CL:B"] * 5)


def test_a_genuine_record_agrees_with_its_own_neighbourhood() -> None:
    s = score_records(REF[:1], np.array(["CL:A"]), REF, REF_LAB, k=4)
    assert s[0] > 0.9


def test_a_relabelled_record_disagrees_with_its_own_neighbourhood() -> None:
    # Same vector as a genuine CL:A cell, but claiming CL:B — the relabel attack.
    s = score_records(REF[:1], np.array(["CL:B"]), REF, REF_LAB, k=4)
    assert s[0] < 0.1


def test_scoring_excludes_the_record_itself() -> None:
    # A record is always its own nearest neighbour and always agrees with itself.
    # Counting that inflates every score and would let a lone poisoned record
    # vouch for itself.
    s = score_records(REF[:1], np.array(["CL:B"]), REF, REF_LAB, k=1)
    assert s[0] == 0.0


def test_scores_are_bounded() -> None:
    s = score_records(REF, REF_LAB, REF, REF_LAB, k=3)
    assert ((s >= 0) & (s <= 1)).all()


def test_poison_is_scored_against_the_CLEAN_reference() -> None:
    # THE design decision. Twenty copies of one relabelled cell are scored
    # against the reference as it stood BEFORE the submission, so they cannot
    # corroborate each other. Scoring against the poisoned reference would let a
    # flood vouch for itself and make the defence look useless.
    flood = np.repeat(REF[:1], 20, axis=0)
    flood_lab = np.array(["CL:B"] * 20)
    s = score_records(flood, flood_lab, REF, REF_LAB, k=4)
    assert (s < 0.1).all(), "a flood corroborated itself — clean reference not used"


def test_batching_does_not_change_scores() -> None:
    a = score_records(REF, REF_LAB, REF, REF_LAB, k=3, block_size=1)
    b = score_records(REF, REF_LAB, REF, REF_LAB, k=3, block_size=100)
    assert np.allclose(a, b)
