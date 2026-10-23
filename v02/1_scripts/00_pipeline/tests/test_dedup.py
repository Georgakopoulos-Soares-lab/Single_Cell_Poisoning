"""Near-duplicate collapse inside a retrieved neighbourhood."""
# 1_scripts cannot be a package (leading digit), so the shared library in
# 00_common is put on the path and imported flat.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_common"))
import numpy as np
import pytest

from defense import DefenseConfig, dedup_keep, defended_vote


def unit(v):
    """Row-normalise to the hypersphere the reference lives on."""
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_identical_records_collapse_to_one() -> None:
    """Verbatim copies are the case this layer exists for."""
    e = unit([[1, 0], [1, 0], [1, 0], [0, 1]])
    assert dedup_keep(e, 0.99).tolist() == [0, 3]


def test_the_survivor_is_the_highest_ranked_member() -> None:
    """Collapse must keep the nearest of a group, not an arbitrary one."""
    e = unit([[1, 0.01], [1, 0], [0, 1]])
    assert dedup_keep(e, 0.99).tolist() == [0, 2]


def test_distinct_records_all_survive() -> None:
    """Below the threshold nothing is collapsed, so the vote is unchanged."""
    e = unit([[1, 0], [0, 1], [-1, 0]])
    assert dedup_keep(e, 0.99).tolist() == [0, 1, 2]


def test_a_loose_threshold_collapses_genuine_diversity() -> None:
    """The cost side: tau too low removes records that are not duplicates."""
    e = unit([[1, 0], [1, 0.6], [1, 1.2]])
    assert len(dedup_keep(e, 0.5)) == 1


def test_first_record_is_always_kept() -> None:
    """An empty vote is not a legitimate outcome of collapse."""
    e = unit([[1, 0]])
    assert dedup_keep(e, 0.0).tolist() == [0]


def test_collapse_changes_the_vote_it_should_change() -> None:
    """Copies of one record must stop outvoting genuinely distinct ones.

    The two "b" records are far apart, so collapse leaves both; the three "a"
    records are identical, so it leaves one. Without collapse "a" wins 3-2.
    """
    labels = np.array(["a", "a", "a", "b", "b"])
    e = unit([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert defended_vote(labels, DefenseConfig()) == "a"
    assert defended_vote(labels, DefenseConfig(dedup_tau=0.99), emb=e) == "b"


def test_dedup_runs_before_the_donor_cap() -> None:
    """Geometry first, so fabricated donor ids cannot preserve a duplicate."""
    labels = np.array(["a", "a", "a", "b", "b"])
    # the attacker gives every copy its own donor id, so a donor cap sees three
    # independent votes where geometry sees one record
    donor = np.array(["d1", "d2", "d3", "d4", "d5"])
    e = unit([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert defended_vote(labels, DefenseConfig(donor_cap=1), donor=donor) == "a"
    assert defended_vote(labels, DefenseConfig(dedup_tau=0.99, donor_cap=1),
                         donor=donor, emb=e) == "b"


def test_dedup_requires_embeddings() -> None:
    """Enabling the layer without vectors must fail loudly, not pass everything."""
    with pytest.raises(ValueError, match="embeddings"):
        defended_vote(np.array(["a", "b"]), DefenseConfig(dedup_tau=0.99))
