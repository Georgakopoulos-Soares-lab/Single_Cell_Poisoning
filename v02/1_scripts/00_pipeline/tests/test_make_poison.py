"""Golden-value tests for stage 3, make_poison."""

from __future__ import annotations
# 1_scripts cannot be a package (leading digit), so the shared library in
# 00_common is put on the path and imported flat.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_common"))


import numpy as np
import pytest

from schemas import Attack
from make_poison import build_poison
from attacks import build_attack
from fixtures import SRC, TGT, tiny_reference

#: The attacks ``build_poison`` constructs. Each draws its vectors FROM the
#: reference, so ``rate`` counts source-class records and ``provenance`` names
#: the cell each one came from. ``flood`` satisfies none of the three: its
#: records are synthetic, its count falls out of the anchor split, and it needs
#: parameters this function does not take. It is built by
#: :func:`attacks.build_attack` and tested against that entry point below.
FROM_REFERENCE = (Attack.RELABEL, Attack.DUPLICATE, Attack.INTERPOLATE)

#: Flood's three construction parameters. Nothing anywhere supplies defaults for
#: these by design, so a test states them as deliberately as a run does.
FLOOD_KW = {"n_anchors": 2, "spread": 0.99, "jitter": 1.0}


def _flood(rate: float = 0.6, seed: int = 0, **overrides) -> dict:
    emb, labels, ids = tiny_reference()
    kw = {**FLOOD_KW, **overrides}
    return build_attack(Attack.FLOOD, ref_emb=emb, ref_labels=labels,
                        ref_ids=ids, source=SRC, target=TGT, rate=rate,
                        seed=seed, **kw)


def _build(attack: str, rate: float = 0.6, seed: int = 0) -> dict:
    emb, labels, ids = tiny_reference()
    return build_poison(
        ref_emb=emb, ref_labels=labels, ref_ids=ids,
        attack=attack, source=SRC, target=TGT, rate=rate, seed=seed,
    )


def test_poison_count_is_rate_times_the_SOURCE_CLASS() -> None:
    # rate is a fraction of the SOURCE CLASS, not of the whole reference.
    # "25% of the macrophages in this atlas are mislabelled" is a threat model;
    # "25% of the entire atlas is poisoned macrophages" is arithmetically
    # impossible -- the reference holds 86,983 macrophages and 25% of 7.9M
    # would need 1,978,473 of them.
    # Fixture: 5 source cells of 10 rows, rate 0.6 -> 3 poisoned records.
    assert _build(Attack.RELABEL, rate=0.6)["emb"].shape == (3, 4)


def test_poison_count_ignores_reference_size() -> None:
    # Doubling the reference while holding the source class fixed must not
    # change how much poison a given rate produces.
    import numpy as _np
    emb, labels, ids = tiny_reference()
    padded_emb = _np.vstack([emb, emb[5:]])          # +5 TARGET cells only
    padded_lab = _np.concatenate([labels, labels[5:]])
    padded_ids = _np.concatenate([ids, _np.array([f"extra{i}" for i in range(5)])])
    from make_poison import build_poison
    out = build_poison(ref_emb=padded_emb, ref_labels=padded_lab, ref_ids=padded_ids,
                       attack=Attack.RELABEL, source=SRC, target=TGT,
                       rate=0.6, seed=0)
    assert out["emb"].shape == (3, 4)


def test_every_poison_carries_the_attacker_chosen_target_label() -> None:
    p = _build(Attack.RELABEL)
    assert set(p["label"]) == {TGT}


def test_relabel_leaves_vectors_untouched() -> None:
    # The whole point of relabel: the geometry is genuine, only the label lies.
    emb, labels, _ = tiny_reference()
    p = _build(Attack.RELABEL)
    src_rows = emb[labels == SRC]
    for v in p["emb"]:
        assert np.isclose(np.abs(src_rows @ v).max(), 1.0, atol=1e-6)


def test_duplicate_reproduces_a_reference_vector_exactly() -> None:
    # cosine == 1 against its source, which is what makes score() infinite.
    emb, labels, _ = tiny_reference()
    p = _build(Attack.DUPLICATE)
    src_rows = emb[labels == SRC]
    assert np.isclose((src_rows @ p["emb"].T).max(axis=0), 1.0, atol=1e-6).all()


def test_interpolate_moves_each_vector_closer_to_the_target() -> None:
    # Relative to where it started, not in absolute terms: at alpha = 0.3 a
    # poisoned vector is only 30% of the way across and is still nearer its
    # source. What must hold is that it moved.
    emb, labels, ids = tiny_reference()
    p = _build(Attack.INTERPOLATE)
    tgt_centroid = emb[labels == TGT].mean(axis=0)
    tgt_centroid /= np.linalg.norm(tgt_centroid)
    for v, prov in zip(p["emb"], p["provenance"], strict=True):
        origin = emb[ids == prov][0]
        assert v @ tgt_centroid > origin @ tgt_centroid


def test_interpolate_moves_further_for_larger_alpha() -> None:
    emb, labels, ids = tiny_reference()
    p = _build(Attack.INTERPOLATE)
    tgt_centroid = emb[labels == TGT].mean(axis=0)
    tgt_centroid /= np.linalg.norm(tgt_centroid)
    gained = [v @ tgt_centroid - emb[ids == prov][0] @ tgt_centroid
              for v, prov in zip(p["emb"], p["provenance"], strict=True)]
    order = np.argsort(p["alpha"])
    assert np.all(np.diff(np.asarray(gained)[order]) > 0)


def test_alpha_is_recorded_for_interpolate_and_nan_otherwise() -> None:
    assert np.isnan(_build(Attack.RELABEL)["alpha"]).all()
    a = _build(Attack.INTERPOLATE)["alpha"]
    assert ((a >= 0.3) & (a <= 0.9)).all()


def test_all_poison_vectors_are_unit_norm() -> None:
    for attack in FROM_REFERENCE:
        p = _build(attack)
        assert np.allclose(np.linalg.norm(p["emb"], axis=1), 1.0, atol=1e-6)


def test_every_attack_is_built_by_one_constructor_or_the_other() -> None:
    # FROM_REFERENCE is narrowed by hand, so a fifth attack added to the enum
    # would otherwise slip past the unit-norm guarantee unnoticed -- which is
    # precisely how flood escaped it. Adding an Attack member must break this.
    assert set(FROM_REFERENCE) | {Attack.FLOOD} == set(Attack)


def test_flood_vectors_are_unit_norm() -> None:
    # Fixture: 5 source cells, rate 0.6 -> 3 records across 2 anchors. The count
    # is asserted because np.allclose is vacuously true on an empty array, and a
    # silent empty poison set reads as a robust system rather than a broken one.
    emb = _flood()["emb"]
    assert emb.shape == (3, 4)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-6)


def test_flood_records_name_no_source_cell() -> None:
    # Flood rotates its anchors toward the source centroid and jitters the
    # copies, so no record is a copy of a reference cell. Naming one would make
    # the poison look traceable to a genuine record it did not come from.
    _, _, ids = tiny_reference()
    prov = _flood()["provenance"]
    assert set(prov) == {"synthetic"}
    assert not set(prov) & set(ids)


def test_flood_refuses_to_run_without_its_three_parameters() -> None:
    # A default here is a silent choice of result: FLOOD_JITTER=0.25 was once
    # read off a rate, and since jitter is cosine RETAINED it would have put
    # poison at cosine 0.24 against neighbours at 0.98 -- measuring the
    # strongest attack in the project as inert.
    for dropped in FLOOD_KW:
        with pytest.raises(ValueError, match=dropped):
            _flood(**{dropped: None})


def test_provenance_names_the_source_cell() -> None:
    _, labels, ids = tiny_reference()
    p = _build(Attack.RELABEL)
    assert set(p["provenance"]) <= set(ids[labels == SRC])


def test_same_seed_reproduces_identical_poison() -> None:
    # Reproducibility of a poisoning benchmark is the whole point.
    assert np.array_equal(_build(Attack.INTERPOLATE, seed=7)["emb"],
                          _build(Attack.INTERPOLATE, seed=7)["emb"])


def test_different_seed_produces_different_poison() -> None:
    assert not np.array_equal(_build(Attack.INTERPOLATE, seed=1)["emb"],
                              _build(Attack.INTERPOLATE, seed=2)["emb"])


def test_absent_source_term_fails_loudly() -> None:
    # A silent empty poison set would look like a robust system.
    emb, labels, ids = tiny_reference()
    with pytest.raises(ValueError, match="no reference cells"):
        build_poison(ref_emb=emb, ref_labels=labels, ref_ids=ids,
                     attack=Attack.RELABEL, source="CL:9999999", target=TGT,
                     rate=0.3, seed=0)


def test_relabel_uses_distinct_source_cells() -> None:
    # relabel attaches a false label to genuine, distinct records.
    p = _build(Attack.RELABEL)
    assert len(set(p["provenance"])) == len(p["provenance"])


def test_duplicate_concentrates_copies_on_few_source_cells() -> None:
    # duplicate amplifies a small set of records rather than spreading across
    # many. Without this, duplicate is indistinguishable from relabel whenever
    # the source pool is large, and the attack axis collapses.
    p = _build(Attack.DUPLICATE)
    assert len(set(p["provenance"])) < len(p["provenance"])


def test_every_attack_declares_a_capability_tier() -> None:
    # A tier is a labelled axis, not a hidden assumption: a ceiling result
    # printed beside realistic ones without saying so reads as measured risk.
    from schemas import ATTACK_TIER, AttackTier
    assert set(ATTACK_TIER) == set(Attack)
    assert ATTACK_TIER[Attack.INTERPOLATE] == AttackTier.CEILING
    assert ATTACK_TIER[Attack.RELABEL] == AttackTier.REALISTIC
    assert ATTACK_TIER[Attack.DUPLICATE] == AttackTier.REALISTIC
    assert ATTACK_TIER[Attack.FLOOD] == AttackTier.CEILING


def test_only_relabel_and_duplicate_emit_verbatim_reference_vectors() -> None:
    # This is WHY interpolate and flood are ceilings: their vectors are blends
    # and rotations that no real count matrix produces, so re-deriving
    # embeddings from the raw data rejects them. relabel and duplicate emit
    # verbatim reference vectors, which no such check can tell from genuine
    # records.
    emb, labels, _ = tiny_reference()
    src_rows = emb[labels == SRC]
    for attack in (Attack.RELABEL, Attack.DUPLICATE):
        p = _build(attack)
        for v in p["emb"]:
            assert np.isclose(np.abs(src_rows @ v).max(), 1.0, atol=1e-6), attack
    for emb_out in (_build(Attack.INTERPOLATE)["emb"], _flood()["emb"]):
        assert not any(np.isclose(np.abs(src_rows @ v).max(), 1.0, atol=1e-6)
                       for v in emb_out)
