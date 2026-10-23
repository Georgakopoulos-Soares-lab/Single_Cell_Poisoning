"""Golden-value tests for stage 2, retrieve_clean.

Fixture: four reference vectors in 3-d, two queries on the axes, so every
cosine is exact by construction.

    r0 = [1.0, 0.0, 0]  CL:A      cos(e0)=1.0   cos(e1)=0.0
    r1 = [0.8, 0.6, 0]  CL:A      cos(e0)=0.8   cos(e1)=0.6
    r2 = [0.0, 1.0, 0]  CL:B      cos(e0)=0.0   cos(e1)=1.0
    r3 = [0.6, 0.8, 0]  CL:B      cos(e0)=0.6   cos(e1)=0.8
"""

from __future__ import annotations

import numpy as np
import pytest

from retrieve_clean import retrieve_exhaustive, subsample_reference

REF = np.array([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0],
                [0.0, 1.0, 0.0], [0.6, 0.8, 0.0]], dtype=np.float32)
REF_LAB = np.array(["CL:A", "CL:A", "CL:B", "CL:B"])
Q = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def _run(k=3, block_size=1000):
    return retrieve_exhaustive(Q, REF, REF_LAB, k=k, block_size=block_size)


def test_returns_k_neighbours_per_query() -> None:
    cos, lab, idx = _run(k=3)
    assert cos.shape == lab.shape == idx.shape == (2, 3)


def test_golden_cosines_for_query_on_first_axis() -> None:
    cos, _, _ = _run(k=3)
    assert np.allclose(cos[0], [1.0, 0.8, 0.6], atol=1e-6)


def test_golden_cosines_for_query_on_second_axis() -> None:
    cos, _, _ = _run(k=3)
    assert np.allclose(cos[1], [1.0, 0.8, 0.6], atol=1e-6)


def test_neighbours_are_sorted_by_decreasing_similarity() -> None:
    # Stage 4 merges this list assuming it is already ordered.
    cos, _, _ = _run(k=4)
    assert (np.diff(cos, axis=1) <= 0).all()


def test_golden_labels_follow_their_vectors() -> None:
    # q0's top-3 are r0(A), r1(A), r3(B); q1's are r2(B), r3(B), r1(A).
    _, lab, _ = _run(k=3)
    assert list(lab[0]) == ["CL:A", "CL:A", "CL:B"]
    assert list(lab[1]) == ["CL:B", "CL:B", "CL:A"]


def test_indices_point_at_the_right_reference_rows() -> None:
    _, _, idx = _run(k=3)
    assert list(idx[0]) == [0, 1, 3]
    assert list(idx[1]) == [2, 3, 1]


def test_blocking_does_not_change_results() -> None:
    a_cos, a_lab, a_idx = _run(block_size=1)
    b_cos, b_lab, b_idx = _run(block_size=1000)
    assert np.array_equal(a_cos, b_cos) and np.array_equal(a_lab, b_lab)
    assert np.array_equal(a_idx, b_idx)


def test_cosines_are_float32() -> None:
    # float64 would double memory on the real 50,000 x 50 result for no gain.
    cos, _, _ = _run()
    assert cos.dtype == np.float32


def test_k_larger_than_the_reference_is_refused() -> None:
    # Silently returning fewer than k would make stage 4's merge wrong in a way
    # nothing downstream could detect.
    with pytest.raises(ValueError, match="k=9 exceeds"):
        _run(k=9)


def test_subsample_is_reproducible_from_its_seed() -> None:
    a = subsample_reference(n_total=1000, n_keep=100, seed=7)
    b = subsample_reference(n_total=1000, n_keep=100, seed=7)
    assert np.array_equal(a, b)


def test_subsample_differs_between_seeds() -> None:
    a = subsample_reference(n_total=1000, n_keep=100, seed=1)
    b = subsample_reference(n_total=1000, n_keep=100, seed=2)
    assert not np.array_equal(a, b)


def test_subsample_is_sorted_and_without_repeats() -> None:
    # Sorted so hnswlib get_items reads roughly in order; unique so the 500k
    # subsample really holds 500k distinct cells.
    s = subsample_reference(n_total=1000, n_keep=100, seed=0)
    assert len(s) == len(set(s.tolist())) == 100
    assert (np.diff(s) > 0).all()


def test_subsample_larger_than_the_reference_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot keep"):
        subsample_reference(n_total=100, n_keep=1000, seed=0)
