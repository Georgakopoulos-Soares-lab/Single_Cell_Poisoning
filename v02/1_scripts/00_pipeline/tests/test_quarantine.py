"""Quarantine holds recently added records out of the vote."""
# 1_scripts cannot be a package (leading digit), so the shared library in
# 00_common is put on the path and imported flat.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_common"))
import numpy as np

from defense import DefenseConfig, admit, defended_vote


def test_admit_rejects_unseasoned_only_when_enabled() -> None:
    """A new record is rejected only if quarantine is switched on."""
    assert admit(DefenseConfig(), seasoned=False) is True
    assert admit(DefenseConfig(quarantine=True), seasoned=False) is False
    assert admit(DefenseConfig(quarantine=True), seasoned=True) is True


def test_quarantine_removes_new_records_from_the_vote() -> None:
    """Three new poisoned records cannot outvote two seasoned genuine ones."""
    labels = np.array(["target", "target", "target", "truth", "truth"])
    seasoned = np.array([False, False, False, True, True])
    assert defended_vote(labels, DefenseConfig()) == "target"
    assert defended_vote(labels, DefenseConfig(quarantine=True),
                         seasoned=seasoned) == "truth"


def test_quarantine_requires_ages() -> None:
    """Enabling the layer without ages must fail loudly, not pass silently."""
    labels = np.array(["a", "b"])
    try:
        defended_vote(labels, DefenseConfig(quarantine=True))
    except ValueError as e:
        assert "quarantine" in str(e)
    else:
        raise AssertionError("expected ValueError when ages are absent")


def test_everything_quarantined_declines_rather_than_guessing() -> None:
    """With nothing seasoned there is no vote to take; None is the answer."""
    labels = np.array(["a", "b"])
    out = defended_vote(labels, DefenseConfig(quarantine=True),
                        seasoned=np.array([False, False]))
    assert out is None
