"""Tests for stage 6, validate_shortcut.

Stage 6 answers one question: when retrieval uses an approximate graph index
instead of exhaustive search, does the merge-and-rerank shortcut still give the
same answer as honestly rebuilding the index with the poison inside it?

Under exhaustive search the shortcut is provably exact (SPEC.md §4.2), so there
is nothing to validate. Under HNSW an inserted poisoned node changes graph
traversal, and the proof no longer holds. These tests cover the comparison
logic; building the index itself is I/O and is not unit-tested.
"""

from __future__ import annotations

import numpy as np
import pytest

from validate_shortcut import agreement_report, topk_overlap


def test_identical_neighbourhoods_overlap_completely() -> None:
    a = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.allclose(topk_overlap(a, a.copy()), [1.0, 1.0])


def test_disjoint_neighbourhoods_overlap_not_at_all() -> None:
    a = np.array([[1, 2, 3]])
    b = np.array([[7, 8, 9]])
    assert np.allclose(topk_overlap(a, b), [0.0])


def test_partial_overlap_is_the_fraction_shared() -> None:
    # rows share {1,2} of 3 -> 2/3; second row shares all 3 -> 1.0
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[1, 2, 9], [6, 5, 4]])
    assert np.allclose(topk_overlap(a, b), [2 / 3, 1.0])


def test_overlap_ignores_ordering() -> None:
    # The vote is a majority over a set; rank within the top-k does not matter.
    a = np.array([[1, 2, 3]])
    b = np.array([[3, 1, 2]])
    assert np.allclose(topk_overlap(a, b), [1.0])


def test_mismatched_shapes_are_refused() -> None:
    with pytest.raises(ValueError, match="same shape"):
        topk_overlap(np.array([[1, 2, 3]]), np.array([[1, 2]]))


def test_report_counts_vote_agreement() -> None:
    # 3 of 4 queries get the same label from both paths.
    rep = agreement_report(
        shortcut_idx=np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
        rebuild_idx=np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
        shortcut_pred=np.array(["A", "A", "B", "B"]),
        rebuild_pred=np.array(["A", "A", "B", "A"]),
        shortcut_n_poison=np.array([0, 1, 2, 0]),
        rebuild_n_poison=np.array([0, 1, 2, 0]),
    )
    assert np.isclose(rep["vote_agreement"], 0.75)


def test_report_flags_perfect_agreement_as_exact() -> None:
    # This is what the smoke profile must produce: exhaustive search makes the
    # shortcut exact, so any disagreement there is a bug, not approximation.
    idx = np.array([[1, 2], [3, 4]])
    rep = agreement_report(
        shortcut_idx=idx, rebuild_idx=idx.copy(),
        shortcut_pred=np.array(["A", "B"]), rebuild_pred=np.array(["A", "B"]),
        shortcut_n_poison=np.array([1, 0]), rebuild_n_poison=np.array([1, 0]),
    )
    assert rep["exact"] is True
    assert np.isclose(rep["mean_overlap"], 1.0)
    assert np.isclose(rep["vote_agreement"], 1.0)


def test_report_is_not_exact_when_neighbourhoods_differ() -> None:
    rep = agreement_report(
        shortcut_idx=np.array([[1, 2]]), rebuild_idx=np.array([[1, 9]]),
        shortcut_pred=np.array(["A"]), rebuild_pred=np.array(["A"]),
        shortcut_n_poison=np.array([0]), rebuild_n_poison=np.array([0]),
    )
    # Votes agree, neighbourhoods do not. Reporting only vote agreement would
    # hide that the shortcut retrieved different cells and got lucky.
    assert rep["exact"] is False
    assert np.isclose(rep["vote_agreement"], 1.0)
    assert np.isclose(rep["mean_overlap"], 0.5)


def test_report_records_signed_poison_count_difference() -> None:
    # Signed, not absolute: systematically finding MORE poison under rebuild
    # means the shortcut understates retrieval failure, which is a different
    # problem from noise in both directions.
    rep = agreement_report(
        shortcut_idx=np.array([[1, 2], [3, 4]]),
        rebuild_idx=np.array([[1, 2], [3, 4]]),
        shortcut_pred=np.array(["A", "A"]), rebuild_pred=np.array(["A", "A"]),
        shortcut_n_poison=np.array([1, 1]), rebuild_n_poison=np.array([2, 3]),
    )
    assert np.isclose(rep["mean_poison_delta"], 1.5)
