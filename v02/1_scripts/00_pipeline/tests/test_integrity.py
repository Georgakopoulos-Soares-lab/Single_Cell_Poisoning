"""``03b_04``'s in-place edit, the half of the step that decides what is hashed.

The step's claim is that an in-place relabel is INVISIBLE to a checksum of the
index and visible only to one of the label file. That claim is only as good as
the edited atlas it hashes, so the edit is checked here on a toy reference:

* ``unmoved`` must leave every embedding byte-identical and change exactly the
  drawn labels. Its first run (RUNLOG Run 44) died because this case was routed
  through ``build_modify``, which refuses it by design.
* ``shifted`` must move the vectors it relabels, or the index-hash column would
  read "missed" for an attack that did move the index.

Loaded from its file: the module name starts with a digit (CLAUDE.md §3).
"""
import importlib.util
import pathlib

import numpy as np
import pytest

STEP_PATH = (pathlib.Path(__file__).resolve().parents[2]
             / "03_b_defence_vs_modify" / "03b_04_both_integrity.py")

N_SRC, N_TGT, DIM, RATE, SEED = 100, 100, 128, 0.25, 7


@pytest.fixture(scope="module")
def step():
    spec = importlib.util.spec_from_file_location("step_03b_04", STEP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ref():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((N_SRC + N_TGT, DIM)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    lab = np.array(["CL:A"] * N_SRC + ["CL:B"] * N_TGT)
    ids = np.array([f"cell{i}" for i in range(len(lab))])
    hub_nbr = rng.integers(0, len(lab), size=(50, 10))
    return emb, lab, ids, hub_nbr


def _edit(step, ref, geom):
    emb, lab, ids, hub_nbr = ref
    return step.edit_in_place(geom, ref_emb=emb, ref_lab=lab, ref_ids=ids,
                              source="CL:A", target="CL:B", rate=RATE,
                              seed=SEED, hub_nbr=hub_nbr)


def test_unmoved_is_invisible_to_the_index_hash(step, ref):
    """The Run 44 crash, and the step's central claim, on one input."""
    emb, lab, _, _ = ref
    new_emb, new_lab, n = _edit(step, ref, "unmoved")
    assert n == round(RATE * N_SRC) == 25
    assert step.digest(new_emb) == step.digest(emb)
    assert step.digest(new_lab) != step.digest(lab)
    changed = np.flatnonzero(new_lab != lab)
    assert len(changed) == n
    assert (lab[changed] == "CL:A").all() and (new_lab[changed] == "CL:B").all()


def test_edit_does_not_mutate_the_reference(step, ref):
    emb, lab, _, _ = ref
    emb0, lab0 = emb.copy(), lab.copy()
    _edit(step, ref, "unmoved")
    _edit(step, ref, "shifted")
    assert np.array_equal(emb, emb0) and np.array_equal(lab, lab0)


def test_shifted_moves_exactly_the_records_it_relabels(step, ref):
    emb, lab, _, _ = ref
    new_emb, new_lab, n = _edit(step, ref, "shifted")
    moved = np.flatnonzero((new_emb != emb).any(axis=1))
    relabelled = np.flatnonzero(new_lab != lab)
    assert len(moved) == n == len(relabelled)
    assert np.array_equal(moved, relabelled)
    assert step.digest(new_emb) != step.digest(emb)
