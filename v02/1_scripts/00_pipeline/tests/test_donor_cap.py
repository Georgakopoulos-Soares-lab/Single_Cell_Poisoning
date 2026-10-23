"""A cap of c votes per donor, generalising one-vote-per-donor."""
# 1_scripts cannot be a package (leading digit), so the shared library in
# 00_common is put on the path and imported flat.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_common"))
import numpy as np

from defense import DefenseConfig, defended_vote


def test_cap_of_one_matches_donor_vote() -> None:
    """donor_cap=1 must reproduce donor_vote exactly."""
    labels = np.array(["a", "a", "a", "b", "b"])
    donor = np.array(["x", "x", "x", "y", "z"])
    one = defended_vote(labels, DefenseConfig(donor_vote=True), donor=donor)
    cap = defended_vote(labels, DefenseConfig(donor_cap=1), donor=donor)
    assert one == cap == "b"


def test_larger_cap_keeps_more_of_each_donor() -> None:
    """With a cap of three, one donor's three records outvote two singletons."""
    labels = np.array(["a", "a", "a", "b", "b"])
    donor = np.array(["x", "x", "x", "y", "z"])
    assert defended_vote(labels, DefenseConfig(donor_cap=3), donor=donor) == "a"


def test_cap_truncates_in_rank_order() -> None:
    """The votes a donor keeps are its highest-ranked records, not any two."""
    labels = np.array(["a", "a", "b", "b", "b"])
    donor = np.array(["x", "x", "x", "y", "z"])
    # cap 2 keeps x's first two ("a","a"), then y and z contribute "b","b" -> tie
    # broken to the lowest identifier, which is "a"
    assert defended_vote(labels, DefenseConfig(donor_cap=2), donor=donor) == "a"


def test_cap_requires_donors() -> None:
    """Enabling the cap without donor ids must fail loudly."""
    try:
        defended_vote(np.array(["a", "b"]), DefenseConfig(donor_cap=2))
    except ValueError as e:
        assert "donor" in str(e)
    else:
        raise AssertionError("expected ValueError when donors are absent")
