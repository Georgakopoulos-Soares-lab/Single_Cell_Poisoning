"""Tests for simultaneous multi-type attacks.

A combo poisons several source classes at once. The question it exists to
answer is whether concurrent attacks compound or interfere, which is only
answerable if a member's success under a combo can be compared against the same
member attacked alone — so both must be measured over the same denominator.
"""

from __future__ import annotations
# 1_scripts cannot be a package (leading digit), so the shared library in
# 00_common is put on the path and imported flat.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_common"))


import numpy as np
import pytest

from schemas import Attack
from make_poison import build_combo_poison
from fixtures import SRC, TGT, tiny_reference

THIRD = "CL:0000084"   # a third label, present in the padded fixture

#: A combo applies ONE attack across several source classes. ``flood`` is not
#: among them: ``build_combo_poison`` delegates to ``build_poison``, which does
#: not construct floods, and what a multi-member flood should mean -- one anchor
#: budget per member, or one shared across them -- has never been defined. The
#: refusal is asserted below rather than left to whichever error surfaces first.
COMBO_ATTACKS = (Attack.RELABEL, Attack.DUPLICATE, Attack.INTERPOLATE)


def _padded():
    """tiny_reference plus 5 cells of a third type, to act as a second source."""
    emb, lab, ids = tiny_reference()
    extra = emb[:5] * 0.5 + emb[5:] * 0.5
    extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
    return (
        np.vstack([emb, extra.astype(np.float32)]),
        np.concatenate([lab, np.array([THIRD] * 5)]),
        np.concatenate([ids, np.array([f"x{i}" for i in range(5)])]),
    )


def test_combo_poison_is_the_union_of_its_members() -> None:
    emb, lab, ids = _padded()
    members = [{"source": SRC, "target": TGT}, {"source": THIRD, "target": TGT}]
    out = build_combo_poison(ref_emb=emb, ref_labels=lab, ref_ids=ids,
                             attack=Attack.RELABEL, members=members, rate=0.6, seed=0)
    # 0.6 of 5 source cells = 3, twice over
    assert out["emb"].shape == (6, 4)


def test_each_member_contributes_its_own_rate_of_its_own_class() -> None:
    # Rate is per source class, so a member with more cells contributes more.
    emb, lab, ids = _padded()
    lab = lab.copy()
    lab[10:] = THIRD
    members = [{"source": SRC, "target": TGT}, {"source": THIRD, "target": TGT}]
    out = build_combo_poison(ref_emb=emb, ref_labels=lab, ref_ids=ids,
                             attack=Attack.RELABEL, members=members, rate=0.6, seed=0)
    assert out["emb"].shape[0] == 3 + 3


def test_combo_records_which_member_each_record_came_from() -> None:
    # Without this the poison set is unattributable and no per-member analysis
    # is possible, which is the whole point of the experiment.
    emb, lab, ids = _padded()
    members = [{"source": SRC, "target": TGT}, {"source": THIRD, "target": TGT}]
    out = build_combo_poison(ref_emb=emb, ref_labels=lab, ref_ids=ids,
                             attack=Attack.RELABEL, members=members, rate=0.6, seed=0)
    assert set(out["member_source"]) == {SRC, THIRD}
    assert len(out["member_source"]) == len(out["emb"])


def test_members_may_carry_different_targets() -> None:
    emb, lab, ids = _padded()
    members = [{"source": SRC, "target": TGT}, {"source": THIRD, "target": SRC}]
    out = build_combo_poison(ref_emb=emb, ref_labels=lab, ref_ids=ids,
                             attack=Attack.RELABEL, members=members, rate=0.6, seed=0)
    assert set(out["label"]) == {TGT, SRC}


def test_all_combo_vectors_are_unit_norm() -> None:
    emb, lab, ids = _padded()
    members = [{"source": SRC, "target": TGT}, {"source": THIRD, "target": TGT}]
    for attack in COMBO_ATTACKS:
        out = build_combo_poison(ref_emb=emb, ref_labels=lab, ref_ids=ids,
                                 attack=attack, members=members, rate=0.6, seed=0)
        assert np.allclose(np.linalg.norm(out["emb"], axis=1), 1.0, atol=1e-6)


def test_combo_covers_every_attack_it_does_not_explicitly_refuse() -> None:
    # COMBO_ATTACKS is narrowed by hand; a new Attack member must force a
    # decision here rather than quietly dropping out of the unit-norm check.
    assert set(COMBO_ATTACKS) | {Attack.FLOOD} == set(Attack)


def test_a_combo_flood_fails_loudly_rather_than_building_something_else() -> None:
    emb, lab, ids = _padded()
    members = [{"source": SRC, "target": TGT}, {"source": THIRD, "target": TGT}]
    with pytest.raises(ValueError, match="unknown attack"):
        build_combo_poison(ref_emb=emb, ref_labels=lab, ref_ids=ids,
                           attack=Attack.FLOOD, members=members, rate=0.6, seed=0)


def test_same_seed_reproduces_the_combo() -> None:
    emb, lab, ids = _padded()
    members = [{"source": SRC, "target": TGT}, {"source": THIRD, "target": TGT}]
    kw = dict(ref_emb=emb, ref_labels=lab, ref_ids=ids, attack=Attack.INTERPOLATE,
              members=members, rate=0.6)
    assert np.array_equal(build_combo_poison(seed=5, **kw)["emb"],
                          build_combo_poison(seed=5, **kw)["emb"])


def test_members_get_different_seeds_so_they_are_not_the_same_draw() -> None:
    # If every member drew with the same seed, a combo of two identical-sized
    # classes would poison the same row indices in each — an artefact, not an
    # attack.
    emb, lab, ids = _padded()
    members = [{"source": SRC, "target": TGT}, {"source": SRC, "target": THIRD}]
    out = build_combo_poison(ref_emb=emb, ref_labels=lab, ref_ids=ids,
                             attack=Attack.RELABEL, members=members, rate=0.6, seed=0)
    first, second = out["provenance"][:3], out["provenance"][3:]
    assert list(first) != list(second)


def test_a_combo_needs_at_least_two_members() -> None:
    emb, lab, ids = _padded()
    with pytest.raises(ValueError, match="at least two"):
        build_combo_poison(ref_emb=emb, ref_labels=lab, ref_ids=ids,
                           attack=Attack.RELABEL,
                           members=[{"source": SRC, "target": TGT}], rate=0.6, seed=0)
