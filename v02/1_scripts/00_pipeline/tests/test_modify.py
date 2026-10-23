"""Overwriting records, and the exactness the deeper neighbour list buys.

SPEC.md §5.4 ``build 3``. The load-bearing test is
:func:`test_remerge_matches_brute_force`: the whole point of the removal-and-merge
shortcut is that it gives the SAME answer as rebuilding the neighbourhood from
scratch, and on a fixture small enough to do both, it must agree exactly.
"""
import numpy as np
import pytest
from fixtures import SRC, TGT, tiny_reference

from modify import (DEEP_K, build_modify, clean_graph_topk, graph_topk,
                    remerge)
from schemas import POISON_EMB, POISON_LABEL, Selection

FLOOD = {"n_anchors": 2, "spread": 0.99, "jitter": 1.0}


def _build(**kw):
    emb, lab, ids = tiny_reference()
    return build_modify(ref_emb=emb, ref_labels=lab, ref_ids=ids, source=SRC,
                        target=TGT, seed=3, **{**FLOOD, **kw})


# ── construction ────────────────────────────────────────────────────────────

def test_records_are_overwritten_at_most_once() -> None:
    """A record cannot be overwritten twice -- that is why concentrated is
    degenerate under modify and why the draw is without replacement."""
    got = _build(dose=1.0, selection=Selection.UNIFORM)
    assert len(np.unique(got["rows"])) == len(got["rows"]) == 5


def test_dose_above_parity_is_clipped_and_says_so() -> None:
    """There is no overwriting 125% of a celltype.

    Silently drawing with replacement would report a SMALLER attack than the
    dose claims, so the clip is recorded on the row instead.
    """
    got = _build(dose=1.25, selection=Selection.UNIFORM)
    assert got["n_overwritten"] == 5
    assert got["dose_clipped"] is True
    assert _build(dose=0.6, selection=Selection.UNIFORM)["dose_clipped"] is False


def test_replacements_align_with_the_rows_they_replace() -> None:
    got = _build(dose=0.6, selection=Selection.UNIFORM)
    assert len(got["rows"]) == len(got[POISON_EMB]) == got["n_overwritten"] == 3


def test_replacements_are_unit_norm() -> None:
    got = _build(dose=1.0, selection=Selection.UNIFORM)
    assert np.allclose(np.linalg.norm(got[POISON_EMB], axis=1), 1.0, atol=1e-5)


def test_axes_are_written_not_inferred() -> None:
    """SPEC.md §5.5: the table says what the attack was, not just its name."""
    got = _build(dose=0.6, selection=Selection.HUB,
                 hub_nbr=np.zeros((2, 3), int))
    assert got["access"] == "modify"
    assert got["geometry"] == "synthesised"
    assert got["selection"] == "hub"
    assert got["label_change"] == "relabel"


def test_hub_overwrites_the_busiest_records() -> None:
    emb, lab, ids = tiny_reference()
    src = np.flatnonzero(lab == SRC)
    nbr = np.full((4, 3), int(src[2]))          # row 2 occupies every slot
    got = _build(dose=0.2, selection=Selection.HUB, hub_nbr=nbr)
    assert got["rows"].tolist() == [int(src[2])]


def test_concentrated_is_refused_with_its_reason() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        _build(dose=0.6, selection=Selection.CONCENTRATED)


def test_hub_without_traffic_fails_loudly() -> None:
    with pytest.raises(ValueError, match="hub selection requires"):
        _build(dose=0.6, selection=Selection.HUB)


# ── the merge ───────────────────────────────────────────────────────────────

def _deep(q: np.ndarray, ref: np.ndarray, depth: int):
    """Exhaustive clean neighbours, nearest first."""
    cos = q @ ref.T
    order = np.argsort(-cos, axis=1, kind="stable")[:, :depth]
    return order, np.take_along_axis(cos, order, axis=1)


def test_remerge_matches_brute_force() -> None:
    """The shortcut equals rebuilding the neighbourhood from scratch.

    Exactly, not approximately: the same vectors are compared by the same
    metric, so any disagreement is a bookkeeping error in the removal step.
    """
    rng = np.random.default_rng(0)
    n_ref, n_q, d, k = 60, 25, 8, 5
    ref = rng.standard_normal((n_ref, d)).astype(np.float32)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    q = rng.standard_normal((n_q, d)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    lab = np.array([f"CL:{i % 4}" for i in range(n_ref)])

    rows = rng.choice(n_ref, 12, replace=False)
    new = rng.standard_normal((12, d)).astype(np.float32)
    new /= np.linalg.norm(new, axis=1, keepdims=True)
    new_lab = np.full(12, "CL:9")

    deep_idx, deep_cos = _deep(q, ref, n_ref)
    got, is_new, exact = remerge(deep_idx, deep_cos, rows, new, new_lab, lab,
                                 q, n_ref, k=k)
    assert exact.all()

    # the reference as it stands AFTER the overwrite, scored directly
    after = ref.copy()
    after[rows] = new
    after_lab = lab.copy()
    after_lab[rows] = "CL:9"
    want_idx = np.argsort(-(q @ after.T), axis=1, kind="stable")[:, :k]
    assert (got == after_lab[want_idx]).all()
    assert (is_new == np.isin(want_idx, rows)).all()


def test_remerge_flags_a_query_whose_deep_list_ran_out() -> None:
    """Not exact is REPORTED, never silently truncated.

    An attacker overwriting a large celltype can exhaust a query's list; a
    truncated neighbourhood would understate the attack, so the query is marked
    and must not be scored.
    """
    rng = np.random.default_rng(1)
    n_ref, d, k, depth = 40, 6, 5, 8
    ref = rng.standard_normal((n_ref, d)).astype(np.float32)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    q = ref[:3].copy()
    lab = np.array([f"CL:{i % 3}" for i in range(n_ref)])
    deep_idx, deep_cos = _deep(q, ref, depth)

    rows = np.unique(deep_idx[0][:depth - 2])        # leave only 2 survivors
    new = rng.standard_normal((len(rows), d)).astype(np.float32)
    new /= np.linalg.norm(new, axis=1, keepdims=True)
    _, _, exact = remerge(deep_idx, deep_cos, rows, new,
                          np.full(len(rows), "CL:9"), lab, q, n_ref, k=k)
    assert not exact[0]


def test_deep_k_is_deeper_than_k() -> None:
    """50 cannot work: if j of the top-50 are overwritten, 51..50+j must be
    promoted, and those may be overwritten too."""
    from schemas import K_NEIGHBORS
    assert DEEP_K > K_NEIGHBORS


# ── the shifted geometry: modify x shifted, the four v01 types ports ─────────

SHIFT = {"alpha_range": (0.5, 0.5), "geometry": "shifted"}


def test_shifted_moves_the_overwritten_records_themselves() -> None:
    """The difference from the insertion arm, and it is the whole attack.

    ``interpolate`` shifts a COPY and leaves the original in place, still
    voting correctly. Here the original is what gets displaced, so each record
    costs the defender a right vote as well as adding a wrong one. Concretely:
    every replacement must lie between its OWN row's old vector and the target
    centroid -- not between some other row's and the centroid.
    """
    emb, lab, ids = tiny_reference()
    got = _build(dose=1.0, selection=Selection.UNIFORM, **SHIFT)
    tgt_c = emb[lab == TGT].mean(axis=0)
    tgt_c /= np.linalg.norm(tgt_c)
    for row, new in zip(got["rows"], got[POISON_EMB], strict=True):
        want = 0.5 * emb[row] + 0.5 * tgt_c
        assert np.allclose(new, want / np.linalg.norm(want), atol=1e-6)


def test_alpha_zero_leaves_the_record_where_it_was() -> None:
    """The degenerate end of the axis, and a check that alpha means what it says."""
    emb, lab, ids = tiny_reference()
    got = _build(dose=1.0, selection=Selection.UNIFORM,
                 geometry="shifted", alpha_range=(0.0, 0.0))
    assert np.allclose(got[POISON_EMB], emb[got["rows"]], atol=1e-6)


def test_alpha_one_lands_every_record_on_the_target_centroid() -> None:
    emb, lab, ids = tiny_reference()
    got = _build(dose=1.0, selection=Selection.UNIFORM,
                 geometry="shifted", alpha_range=(1.0, 1.0))
    tgt_c = emb[lab == TGT].mean(axis=0)
    tgt_c /= np.linalg.norm(tgt_c)
    assert np.allclose(got[POISON_EMB], tgt_c, atol=1e-6)
    # every record now sits on ONE point -- a de-duplication filter's easiest case
    assert len(np.unique(got[POISON_EMB].round(5), axis=0)) == 1


def test_alpha_is_reported_per_record() -> None:
    got = _build(dose=1.0, selection=Selection.UNIFORM, **SHIFT)
    assert got["alpha"].shape == (5,)
    assert np.allclose(got["alpha"], 0.5)
    # NaN under synthesised, which has no alpha
    assert np.isnan(_build(dose=1.0, selection=Selection.UNIFORM,
                           **FLOOD)["alpha"]).all()


def test_shifted_keep_moves_vectors_without_lying_about_the_label() -> None:
    """One of the four ports: geometry attacks the retrieval, label stays true.

    The control that separates "the vectors moved" from "the annotation lied".
    """
    got = _build(dose=0.6, selection=Selection.UNIFORM,
                 label_change="keep", **SHIFT)
    assert (got[POISON_LABEL] == SRC).all()
    assert got["geometry"] == "shifted"


def test_all_four_shifted_types_are_constructible() -> None:
    """SPEC.md §5.4: modify x shifted x {relabel,keep} x {uniform,hub}."""
    emb, lab, ids = tiny_reference()
    nbr = np.full((4, 3), int(np.flatnonzero(lab == SRC)[1]))
    seen = set()
    for label_change in ("relabel", "keep"):
        for sel in (Selection.UNIFORM, Selection.HUB):
            got = _build(dose=0.6, selection=sel, label_change=label_change,
                         hub_nbr=nbr, **SHIFT)
            seen.add((got["geometry"], got["label_change"], got["selection"]))
    assert len(seen) == 4


def test_shifted_without_alpha_fails_loudly() -> None:
    """A default here would silently repeat v01's single alpha of 0.5."""
    with pytest.raises(ValueError, match="requires alpha_range"):
        _build(dose=0.6, selection=Selection.UNIFORM, geometry="shifted")


def test_synthesised_without_its_parameters_fails_loudly() -> None:
    emb, lab, ids = tiny_reference()
    with pytest.raises(ValueError, match="requires"):
        build_modify(ref_emb=emb, ref_labels=lab, ref_ids=ids, source=SRC,
                     target=TGT, dose=0.6, seed=3, selection=Selection.UNIFORM,
                     geometry="synthesised")


def test_unmoved_is_refused_with_its_reason() -> None:
    """unmoved under modify moves no embedding -- that is 02b_05's re-vote."""
    with pytest.raises(ValueError, match="02b_05"):
        _build(dose=0.6, selection=Selection.UNIFORM, geometry="unmoved")


def test_block_size_is_arithmetically_inert() -> None:
    """Chunking the matmul differently must not change a single returned value.

    The block exists only to bound the transient buffer -- len(q) x block x 4
    bytes, which is 3.96 GB at the default against 49,456 real queries and
    dominates remerge's ~26 GB peak. Lowering it is what lets this step share a
    machine with the index sweeps. Equality, not approx: the same products are
    summed in the same order within a row, so any difference would be a
    bookkeeping error in the chunk merge.
    """
    rng = np.random.default_rng(3)
    n_ref, n_q, d, k = 400, 60, 12, 6
    ref = rng.standard_normal((n_ref, d)).astype(np.float32)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    q = rng.standard_normal((n_q, d)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    lab = np.array([f"CL:{i % 5}" for i in range(n_ref)])
    rows = rng.choice(n_ref, 120, replace=False)
    new = rng.standard_normal((120, d)).astype(np.float32)
    new /= np.linalg.norm(new, axis=1, keepdims=True)
    new_lab = np.full(120, "CL:9")
    cos = q @ ref.T
    order = np.argsort(-cos, axis=1, kind="stable")
    deep_idx, deep_cos = order, np.take_along_axis(cos, order, axis=1)

    ref_out = remerge(deep_idx, deep_cos, rows, new, new_lab, lab, q, n_ref, k=k)
    for block in (7, 31, 120, 5000):
        got = remerge(deep_idx, deep_cos, rows, new, new_lab, lab, q, n_ref,
                      k=k, block=block)
        for a, b in zip(ref_out, got, strict=True):
            assert np.array_equal(a, b), f"block={block} changed the result"


# ---------------------------------------------------------------------------
# clean_graph_topk -- a clean baseline must come from the SAME retriever as the
# attacked side. CLAUDE.md 7, "One retrieval regime per comparison".
# ---------------------------------------------------------------------------

def _save_index(emb: np.ndarray, path) -> str:
    """Build and save an hnswlib index over ``emb``, ids 0..n-1."""
    import hnswlib
    n, d = emb.shape
    ix = hnswlib.Index(space="cosine", dim=d)
    ix.init_index(max_elements=n, ef_construction=200, M=16)
    ix.add_items(emb, np.arange(n))
    ix.save_index(str(path))
    return str(path)


def test_clean_graph_topk_is_a_noop_graph_topk(tmp_path) -> None:
    """The invariant the fix rests on: rewriting records with THEMSELVES and
    their own labels must give exactly what the clean path gives. If these two
    disagree, the clean baseline and the attacked side are not the same
    retriever, which is the defect found 2026-09-26 in 02b_11 / 02b_15."""
    rng = np.random.default_rng(7)
    n_ref, d, k, ef = 400, 8, 5, 64
    ref = rng.standard_normal((n_ref, d)).astype(np.float32)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    lab = np.array([f"CL:{i % 6}" for i in range(n_ref)])
    q = rng.standard_normal((30, d)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    idx = _save_index(ref, tmp_path / "k.bin")

    clean = clean_graph_topk(idx, lab, q, n_ref, ef, k=k)
    rows = np.arange(n_ref)                      # every record...
    same, _, _ = graph_topk(idx, rows, ref, lab, lab, q, n_ref, ef, k=k)
    assert np.array_equal(clean, same), "no-op rewrite changed the answer"
    assert clean.shape == (30, k)


def test_clean_graph_topk_does_not_see_the_poison(tmp_path) -> None:
    """It must return the UNPOISONED answer: the same labels whether or not a
    poisoned variant of the index was also queried."""
    rng = np.random.default_rng(11)
    n_ref, d, k, ef = 300, 8, 5, 64
    ref = rng.standard_normal((n_ref, d)).astype(np.float32)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    lab = np.array([f"CL:{i % 4}" for i in range(n_ref)])
    q = rng.standard_normal((20, d)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    idx = _save_index(ref, tmp_path / "k.bin")

    before = clean_graph_topk(idx, lab, q, n_ref, ef, k=k)
    rows = np.arange(50)
    moved = np.tile(q[0], (50, 1)).astype(np.float32)   # pile them on one query
    graph_topk(idx, rows, moved, np.full(50, "CL:BAD"), lab, q, n_ref, ef, k=k)
    after = clean_graph_topk(idx, lab, q, n_ref, ef, k=k)
    assert np.array_equal(before, after), "the clean path saw the poison"
    assert "CL:BAD" not in set(np.unique(before)), "attacker label in clean answer"


def test_clean_graph_topk_refuses_ef_below_k(tmp_path) -> None:
    rng = np.random.default_rng(3)
    ref = rng.standard_normal((50, 4)).astype(np.float32)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    idx = _save_index(ref, tmp_path / "k.bin")
    lab = np.array(["CL:0"] * 50)
    q = ref[:5]
    with pytest.raises(ValueError, match="ef"):
        clean_graph_topk(idx, lab, q, 50, ef=3, k=10)


def test_clean_graph_topk_refuses_a_size_mismatch(tmp_path) -> None:
    """A wrong n_ref means ids do not correspond to ref_label, which would
    mislabel every neighbour silently."""
    rng = np.random.default_rng(5)
    ref = rng.standard_normal((50, 4)).astype(np.float32)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    idx = _save_index(ref, tmp_path / "k.bin")
    lab = np.array(["CL:0"] * 60)
    with pytest.raises(ValueError, match="ids would not correspond"):
        clean_graph_topk(idx, lab, ref[:5], 60, ef=64, k=5)
