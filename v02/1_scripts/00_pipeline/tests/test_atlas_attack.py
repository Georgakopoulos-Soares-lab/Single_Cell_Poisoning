"""Atlas-scope record choice and label forging, shared by both arms.

SPEC.md §5.4. The two arms of ``02a`` must choose the SAME records and forge the
SAME labels from a given seed, or the one axis between them -- whether the
genuine record survives -- is confounded with which records were hit. That is
why the construction is shared, and these tests are what hold it still.
"""
import numpy as np
import pytest

from atlas_attack import (MIN_CLASS, UNMAPPED, build_hub, build_uniform,
                          seed_of, wrong_labels)


@pytest.fixture
def atlas():
    """500 records over 7 celltypes, every 37th unmapped."""
    lab = np.array([f"CL:{i % 7:07d}" for i in range(500)])
    lab[::37] = UNMAPPED
    usable = np.flatnonzero(lab != UNMAPPED)
    return lab, usable, np.unique(lab[usable])[:5]


def test_golden_uniform_is_pinned(atlas) -> None:
    """The exact records and labels 02a_01 produced BEFORE the shared move.

    Captured from the pre-refactor script and asserted here, because 02a_01 has
    30 committed conditions and v01 rows they pair with. A change to the RNG
    consumption order would silently re-poison them.
    """
    lab, usable, big = atlas
    assert seed_of("uniform", 0.01, 0) == 2115350906
    src, pl = build_uniform(lab, usable, big, 5, 2115350906)
    assert src.tolist() == [266, 258, 453, 21, 400]
    assert pl.tolist() == ["CL:0000001", "CL:0000003", "CL:0000002",
                           "CL:0000002", "CL:0000000"]


def test_seed_is_stable_across_processes(atlas) -> None:
    """blake2b, not hash(), which is salted per process."""
    assert seed_of("uniform", 0.05, 3) == 280978141


def test_collision_is_reduced_by_one_redraw_not_eliminated(atlas) -> None:
    """The forged label CAN still equal the record's own term, and that is v01's
    behaviour kept on purpose.

    The redraw happens once, not in a loop, so the collision rate falls from
    1/n_terms to 1/n_terms^2 rather than to zero. With 5 forgeable terms here
    that is 1/25, which this test observes; on the shipped reference there are
    180, so it is 3.1e-5 -- about 24 records in the 791,389 of the largest
    budget. Looping would consume the RNG differently and re-poison 02a_01's 30
    committed conditions, breaking the v01 pairing that makes them worth having.
    """
    lab, usable, big = atlas
    rng = np.random.default_rng(0)
    n, hit = 0, 0
    for _ in range(300):
        src = rng.choice(usable, 30)
        pl = wrong_labels(lab, src, big, rng)
        n += len(src)
        hit += int((pl == lab[src]).sum())
    rate = hit / n
    assert 0 < rate < 3 / len(big) ** 2      # reduced ~5-fold, not to zero
    assert rate < 1 / len(big)               # strictly better than no redraw


def test_unmapped_records_are_never_chosen(atlas) -> None:
    """Poisoning a record with no ontology id spends budget on a no-op."""
    lab, usable, big = atlas
    src, _ = build_uniform(lab, usable, big, 60, seed_of("uniform", 0.1, 0))
    assert (lab[src] != UNMAPPED).all()


def test_both_arms_choose_identical_records(atlas) -> None:
    """The property the shared module exists for.

    02a_01 inserts and 02a_02 overwrites; if they drew different records the
    comparison between them would measure the draw, not the access level.
    """
    lab, usable, big = atlas
    s = seed_of("uniform", 0.02, 1)
    a = build_uniform(lab, usable, big, 10, s)
    b = build_uniform(lab, usable, big, 10, s)
    assert a[0].tolist() == b[0].tolist()
    assert a[1].tolist() == b[1].tolist()


# ── hub ─────────────────────────────────────────────────────────────────────

def test_hub_spends_the_budget_on_the_busiest_records(atlas) -> None:
    lab, usable, big = atlas
    busy = [int(usable[3]), int(usable[9])]
    nbr = np.array(busy * 30).reshape(-1, 2)
    src, _ = build_hub(lab, usable, big, 2, seed_of("hub", 0.01, 0), nbr)
    assert sorted(src.tolist()) == sorted(busy)


def test_hub_record_choice_ignores_the_seed(atlas) -> None:
    """Deterministic given the traffic; the seed varies only the labels."""
    lab, usable, big = atlas
    nbr = np.array([int(usable[i % 12]) for i in range(60)]).reshape(-1, 2)
    a_src, a_lab = build_hub(lab, usable, big, 8, 1, nbr)
    b_src, b_lab = build_hub(lab, usable, big, 8, 2, nbr)
    assert a_src.tolist() == b_src.tolist()
    assert a_lab.tolist() != b_lab.tolist()


def test_hub_and_uniform_differ(atlas) -> None:
    """If they agreed, the selection axis would be measuring nothing."""
    lab, usable, big = atlas
    nbr = np.array([int(usable[i % 6]) for i in range(60)]).reshape(-1, 2)
    h, _ = build_hub(lab, usable, big, 6, 5, nbr)
    u, _ = build_uniform(lab, usable, big, 6, 5)
    assert h.tolist() != u.tolist()


def test_hub_forges_labels_the_same_way_uniform_does(atlas) -> None:
    """Selection changes which records are hit, never how the label is forged."""
    lab, usable, big = atlas
    nbr = np.array([int(usable[i % 20]) for i in range(80)]).reshape(-1, 2)
    src, pl = build_hub(lab, usable, big, 20, 7, nbr)
    assert set(pl) <= set(big)
    # same construction, so the same residual collision behaviour applies
    assert (pl != lab[src]).mean() > 0.8


def test_min_class_is_the_documented_floor() -> None:
    assert MIN_CLASS == 2000
