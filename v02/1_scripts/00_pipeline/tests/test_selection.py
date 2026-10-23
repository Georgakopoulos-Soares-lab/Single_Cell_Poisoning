"""Selection and label as AXES, not as things baked into an attack name.

SPEC.md §5.5. ``relabel`` and ``duplicate`` are one construction at two
selection settings; making that a parameter is what lets ``insert x hub`` and
the three ``keep`` controls be built at all (SPEC.md §5.4, ``build 2`` and
``build 4``).

The load-bearing tests here are the two REGRESSION ones. Parameterising a
constructor that has already produced 624 committed conditions is only safe if
the default path draws exactly the records it drew before -- so those assert
byte equality against the pre-axis behaviour, not approximate agreement.
"""
import numpy as np
import pytest
from fixtures import SRC, TGT, tiny_reference
from make_poison import DUPLICATE_POOL_FRACTION, build_poison
from selection import CONCENTRATED_POOL_FRACTION, choose, hub_order

from schemas import (POISON_EMB, POISON_LABEL, POISON_PROVENANCE, Attack,
                     LabelChange, Selection)

RATE, SEED = 0.6, 7


def _poison(**kw):
    emb, lab, ids = tiny_reference()
    return build_poison(ref_emb=emb, ref_labels=lab, ref_ids=ids, source=SRC,
                        target=TGT, rate=RATE, seed=SEED, **kw)


# ── regression: the default path must not have moved ────────────────────────

def test_uniform_reproduces_relabel_byte_for_byte() -> None:
    """Explicit uniform draws exactly what the unparameterised path drew.

    Byte equality, not approx: both consume the same Generator in the same
    order, so any difference is a different draw and not rounding.
    """
    a = _poison(attack=Attack.RELABEL)
    b = _poison(attack=Attack.RELABEL, selection=Selection.UNIFORM)
    assert (a[POISON_EMB] == b[POISON_EMB]).all()
    assert (a[POISON_PROVENANCE] == b[POISON_PROVENANCE]).all()


def test_concentrated_reproduces_duplicate_byte_for_byte() -> None:
    a = _poison(attack=Attack.DUPLICATE)
    b = _poison(attack=Attack.DUPLICATE, selection=Selection.CONCENTRATED)
    assert (a[POISON_EMB] == b[POISON_EMB]).all()
    assert (a[POISON_PROVENANCE] == b[POISON_PROVENANCE]).all()


def test_the_two_pool_fractions_agree() -> None:
    """The shared constant and the name v01 quotes must not drift apart."""
    assert CONCENTRATED_POOL_FRACTION == DUPLICATE_POOL_FRACTION == 0.01


# ── the label axis ──────────────────────────────────────────────────────────

def test_keep_leaves_the_true_label_on_the_record() -> None:
    got = _poison(attack=Attack.RELABEL, label_change=LabelChange.KEEP)
    assert (got[POISON_LABEL] == SRC).all()


def test_relabel_forges_the_target() -> None:
    assert (_poison(attack=Attack.RELABEL)[POISON_LABEL] == TGT).all()


def test_keep_and_relabel_differ_ONLY_in_the_label() -> None:
    """The property that makes ``keep`` a control rather than another attack.

    Same seed, same selection, so the same records are drawn and the same
    vectors submitted. If the embeddings differed, a keep-vs-relabel comparison
    would be confounded by which cells were picked, and could not isolate the
    effect of the false annotation.
    """
    a = _poison(attack=Attack.RELABEL, label_change=LabelChange.RELABEL)
    b = _poison(attack=Attack.RELABEL, label_change=LabelChange.KEEP)
    assert (a[POISON_EMB] == b[POISON_EMB]).all()
    assert (a[POISON_PROVENANCE] == b[POISON_PROVENANCE]).all()
    assert (a[POISON_LABEL] != b[POISON_LABEL]).all()


def test_keep_applies_to_every_geometry() -> None:
    """Including interpolate, whose vectors move even though the label does not."""
    got = _poison(attack=Attack.INTERPOLATE, label_change=LabelChange.KEEP)
    assert (got[POISON_LABEL] == SRC).all()


# ── the hub axis ────────────────────────────────────────────────────────────

def _traffic(rows: list[int], n_q: int = 6, k: int = 3) -> np.ndarray:
    """A neighbour table whose slots are filled by ``rows``, cycled."""
    return np.array(rows * ((n_q * k) // len(rows) + 1))[:n_q * k].reshape(n_q, k)


def test_hub_order_puts_the_busiest_first() -> None:
    """Row 2 occupies most slots, then row 0, then row 1. Checked by hand."""
    pool = np.array([0, 1, 2])
    nbr = np.array([[2, 2, 0], [2, 0, 1], [2, 2, 2]])   # 2 x6, 0 x2, 1 x1
    assert hub_order(pool, nbr, 3).tolist() == [2, 0, 1]


def test_hub_selection_draws_the_busiest_records() -> None:
    emb, lab, ids = tiny_reference()
    src = np.flatnonzero(lab == SRC)                     # rows 0..4
    nbr = _traffic([int(src[3])] * 5 + [int(src[1])])    # row 3 busiest, then 1
    got = choose(Selection.HUB, src, 2, np.random.default_rng(0),
                 hub_nbr=nbr, n_ref=len(lab))
    assert got.tolist() == [int(src[3]), int(src[1])]


def test_hub_needs_no_seed() -> None:
    """Deterministic given the traffic -- v01's guarantee, and 02b_05's."""
    emb, lab, ids = tiny_reference()
    src = np.flatnonzero(lab == SRC)
    nbr = _traffic([int(src[2]), int(src[0])])
    a = choose(Selection.HUB, src, 3, np.random.default_rng(0),
               hub_nbr=nbr, n_ref=len(lab))
    b = choose(Selection.HUB, src, 3, np.random.default_rng(999),
               hub_nbr=nbr, n_ref=len(lab))
    assert (a == b).all()


def test_hub_cycles_when_the_budget_exceeds_the_celltype() -> None:
    emb, lab, ids = tiny_reference()
    src = np.flatnonzero(lab == SRC)                     # 5 records
    nbr = _traffic([int(i) for i in src])
    got = choose(Selection.HUB, src, 7, np.random.default_rng(0),
                 hub_nbr=nbr, n_ref=len(lab))
    assert len(got) == 7
    # the 6th and 7th records repeat the 1st and 2nd -- busiest first, cycled
    assert (got[5:] == got[:2]).all()
    assert len(np.unique(got)) == 5


def test_hub_without_traffic_fails_loudly() -> None:
    """A silent fallback to uniform would report the wrong attack entirely."""
    emb, lab, ids = tiny_reference()
    with pytest.raises(ValueError, match="hub selection requires"):
        choose(Selection.HUB, np.flatnonzero(lab == SRC), 2,
               np.random.default_rng(0))


def test_unknown_selection_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unknown selection"):
        choose("busiest-ish", np.arange(5), 2, np.random.default_rng(0))


# ── the combination the six missing types need ──────────────────────────────

def test_relabel_x_hub_is_expressible() -> None:
    """SPEC.md §5.4 ``build 2``: the selection symmetry the insert arm lacked."""
    emb, lab, ids = tiny_reference()
    src = np.flatnonzero(lab == SRC)
    nbr = _traffic([int(src[4])] * 5 + [int(src[0])])
    got = build_poison(ref_emb=emb, ref_labels=lab, ref_ids=ids,
                       attack=Attack.RELABEL, source=SRC, target=TGT,
                       rate=RATE, seed=SEED, selection=Selection.HUB,
                       hub_nbr=nbr)
    assert (got[POISON_LABEL] == TGT).all()
    assert got[POISON_PROVENANCE][0] == ids[src[4]]
