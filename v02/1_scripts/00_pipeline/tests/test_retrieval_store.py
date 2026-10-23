"""Tests for retrieval_store -- keeping a graph retrieval so it can be re-scored.

The graph path is not reproducible, so a retrieval is evidence that cannot be
regenerated; these pin that what is written is what is read back.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1].parent / "00_common"))

import numpy as np
import pytest

import retrieval_store as rs

KEY = {"source": "CL:0000057", "attack": "relabel", "selection": "hub",
       "rate": 0.25, "n_anchors": "na"}


def _write(tmp_path, poison):
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 10_000, size=(7, 50))
    cos = rng.random((7, 50)).astype(np.float32)
    p = rs.save(tmp_path, "03a_01", "reluni", KEY, ids=ids, cos=cos,
                meta={"condition_seed": 123, "poison_digest": rs.digest(poison)},
                extra={"edited": np.arange(5)})
    return p, ids, cos


def test_round_trip(tmp_path) -> None:
    poison = np.ones((3, 128), np.float32)
    p, ids, cos = _write(tmp_path, poison)
    r = rs.load(p)
    assert np.array_equal(r["ids"], ids)
    assert np.allclose(r["cos"], cos, atol=1e-3)       # float16 storage
    assert r["meta"]["condition_seed"] == 123 and r["meta"]["key"] == KEY
    assert np.array_equal(r["edited"], np.arange(5))
    assert not list(p.parent.glob("*.tmp"))             # atomic: no temp left


def test_same_condition_same_path(tmp_path) -> None:
    a = rs.path_for(tmp_path, "03a_01", "x", KEY)
    b = rs.path_for(tmp_path, "03a_01", "x", dict(reversed(list(KEY.items()))))
    assert a == b


def test_verify_poison(tmp_path) -> None:
    poison = np.ones((3, 128), np.float32)
    r = rs.load(_write(tmp_path, poison)[0])
    rs.verify_poison(r, poison)
    with pytest.raises(ValueError):
        rs.verify_poison(r, poison * 0.5)
