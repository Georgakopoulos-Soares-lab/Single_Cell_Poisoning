"""Distance-weighted voting: weights, and the vote they produce."""
# 1_scripts cannot be a package (leading digit), so the shared library in
# 00_common is put on the path and imported flat.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_common"))
import numpy as np
import pytest

from defense import (
    SCORE_EPS,
    DefenseConfig,
    defended_vote,
    vote_weights,
)


def test_uniform_reproduces_majority_vote() -> None:
    """The default must be the plain majority vote, whatever the cosines."""
    labels = np.array(["a", "b", "b"])
    cos = np.array([0.999, 0.5, 0.4])
    assert defended_vote(labels, DefenseConfig(), cos=cos) == "b"


def test_one_close_neighbour_can_outvote_two_distant_ones() -> None:
    """This is the whole point of weighting, and its whole risk."""
    labels = np.array(["a", "b", "b"])
    cos = np.array([0.9999, 0.90, 0.89])
    assert defended_vote(labels, DefenseConfig(weight="score"), cos=cos) == "a"


def test_score_weight_is_the_scimilarity_score() -> None:
    """Weight must be 1 / (1 - c), the score thresholds are defined on."""
    cos = np.array([0.5, 0.9])
    w = vote_weights(cos, DefenseConfig(weight="score"))
    assert np.allclose(w, [2.0, 10.0])


def test_score_weight_is_clipped_at_cosine_one() -> None:
    """A verbatim copy at cos = 1.0 must not become an infinite vote."""
    w = vote_weights(np.array([1.0]), DefenseConfig(weight="score"))
    assert np.isfinite(w).all() and w[0] == pytest.approx(1.0 / SCORE_EPS)


def test_rank_weight_ignores_cosine_values() -> None:
    """Rank weighting depends only on order, so it cannot be gamed by distance."""
    a = vote_weights(np.array([0.99, 0.98, 0.97]), DefenseConfig(weight="rank"))
    b = vote_weights(np.array([0.50, 0.10, 0.01]), DefenseConfig(weight="rank"))
    assert np.allclose(a, b) and np.allclose(a, [1.0, 0.5, 1 / 3])


def test_softmax_temperature_controls_concentration() -> None:
    """A colder temperature must concentrate weight on the nearest neighbour."""
    cos = np.array([0.99, 0.90])
    cold = vote_weights(cos, DefenseConfig(weight="softmax", weight_temp=0.001))
    warm = vote_weights(cos, DefenseConfig(weight="softmax", weight_temp=1.0))
    assert cold[1] / cold[0] < warm[1] / warm[0]


def test_weighting_requires_cosines() -> None:
    """Enabling a scheme without cosines must fail loudly, not vote uniformly."""
    with pytest.raises(ValueError, match="cosines"):
        defended_vote(np.array(["a", "b"]), DefenseConfig(weight="score"))


def test_unknown_scheme_is_refused() -> None:
    """An unrecognised name must raise, never silently fall back to uniform."""
    with pytest.raises(ValueError, match="unknown weight scheme"):
        vote_weights(np.array([0.9]), DefenseConfig(weight="inverse_distance"))


def test_weighting_composes_with_donor_cap() -> None:
    """Filtering happens first; the weighted tally sees only survivors."""
    labels = np.array(["a", "a", "b"])
    donor = np.array(["x", "x", "y"])
    cos = np.array([0.999, 0.998, 0.99])
    # both "a" records are donor x, so a cap of one drops the second; "a" still
    # wins on weight because its survivor is nearest
    assert defended_vote(labels, DefenseConfig(donor_cap=1, weight="score"),
                         donor=donor, cos=cos) == "a"
