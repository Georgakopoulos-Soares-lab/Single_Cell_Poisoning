"""Tests for stage 1, embed_queries.

The encoder itself belongs to scimilarity and is not retested here. What is
tested is our own contract around it: that what comes out is unit-normalised,
128-dimensional, aligned row-for-row with the metadata, and that a degenerate
embedding is refused rather than written to disk.

The encoder is injected, so these run without the 30 GB model.
"""

from __future__ import annotations
# 1_scripts cannot be a package (leading digit), so the shared library in
# 00_common is put on the path and imported flat.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_common"))


import numpy as np
import pandas as pd
import pytest

from schemas import EMB_DIM, QUERY_ID, Y_TRUE
from embed_queries import embed_and_validate

N = 5
OBS = pd.DataFrame(
    {"celltype_id": ["CL:0000235", "CL:0000583", "CL:0000235", "CL:0000084", "CL:0000236"]},
    index=[f"cell{i}" for i in range(N)],
)
X = np.arange(N * 12, dtype=np.float32).reshape(N, 12)


def _encoder(scale: float = 1.0, dim: int = EMB_DIM):
    """A stand-in encoder: deterministic, unnormalised, right shape."""
    def enc(x: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(0)
        return (rng.standard_normal((len(x), dim)) * scale).astype(np.float32)
    return enc


def test_embeddings_are_unit_norm() -> None:
    # The reference lives on the unit sphere; queries must too, or cosine
    # similarity silently stops being cosine similarity.
    emb, _ = embed_and_validate(X, OBS, _encoder(scale=7.3), "celltype_id")
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-6)


def test_embeddings_are_float32() -> None:
    emb, _ = embed_and_validate(X, OBS, _encoder(), "celltype_id")
    assert emb.dtype == np.float32


def test_one_row_out_per_cell_in() -> None:
    emb, meta = embed_and_validate(X, OBS, _encoder(), "celltype_id")
    assert len(emb) == len(meta) == N


def test_metadata_carries_barcode_and_ground_truth() -> None:
    _, meta = embed_and_validate(X, OBS, _encoder(), "celltype_id")
    assert list(meta[QUERY_ID]) == list(OBS.index)
    assert list(meta[Y_TRUE]) == list(OBS["celltype_id"])


def test_wrong_embedding_dimension_is_refused() -> None:
    # A 64-d encoder would produce numbers that merge happily with a 128-d
    # reference only after a silent broadcast or a crash deep in stage 4.
    with pytest.raises(ValueError, match="expected 128"):
        embed_and_validate(X, OBS, _encoder(dim=64), "celltype_id")


def test_zero_norm_embedding_is_refused() -> None:
    # A cell that embeds to the origin has no direction, so its cosine to
    # everything is undefined. Normalising it would divide by zero and quietly
    # produce nan neighbours.
    def dead(x: np.ndarray) -> np.ndarray:
        e = np.ones((len(x), EMB_DIM), dtype=np.float32)
        e[2] = 0.0
        return e
    with pytest.raises(ValueError, match="zero-norm"):
        embed_and_validate(X, OBS, dead, "celltype_id")


def test_missing_label_column_is_refused() -> None:
    with pytest.raises(KeyError, match="not_a_column"):
        embed_and_validate(X, OBS, _encoder(), "not_a_column")


def test_rows_are_not_reordered() -> None:
    # Row i of the embedding must be cell i of the metadata. A silent reorder
    # would attach every prediction to the wrong ground-truth label.
    def positional(x: np.ndarray) -> np.ndarray:
        e = np.zeros((len(x), EMB_DIM), dtype=np.float32)
        e[np.arange(len(x)), np.arange(len(x))] = 1.0   # cell i -> axis i
        return e
    emb, meta = embed_and_validate(X, OBS, positional, "celltype_id")
    assert [int(np.argmax(r)) for r in emb] == list(range(N))
    assert list(meta[QUERY_ID]) == list(OBS.index)


def test_unlabelled_cells_are_refused_by_default() -> None:
    # The real query set has 544 such cells (multiplets: two cells sequenced
    # as one, so no true cell type exists). Left alone they become the string
    # "nan", count as wrong in every condition, and inflate both the clean
    # error rate and the outcome failure rate by ~1% everywhere.
    obs = OBS.copy()
    obs.loc["cell2", "celltype_id"] = None
    with pytest.raises(ValueError, match="1 cell.*no .*label"):
        embed_and_validate(X, obs, _encoder(), "celltype_id")


def test_unlabelled_cells_can_be_dropped_when_asked_explicitly() -> None:
    obs = OBS.copy()
    obs.loc["cell2", "celltype_id"] = None
    emb, meta = embed_and_validate(X, obs, _encoder(), "celltype_id",
                                   drop_unlabeled=True)
    assert len(emb) == len(meta) == N - 1
    assert "cell2" not in set(meta[QUERY_ID])


def test_dropping_keeps_embeddings_aligned_with_metadata() -> None:
    # Dropping a row from the metadata but not the embedding would reintroduce
    # exactly the misalignment the row-order test guards against.
    obs = OBS.copy()
    obs.loc["cell1", "celltype_id"] = None

    # Keyed on row CONTENT, not row position, so the invariant holds whether
    # unlabelled cells are dropped before or after encoding. X row i begins at
    # 12*i, so the first value recovers which cell an embedding came from.
    def content_keyed(x: np.ndarray) -> np.ndarray:
        e = np.zeros((len(x), EMB_DIM), dtype=np.float32)
        for row, vals in enumerate(x):
            e[row, int(vals[0]) // 12] = 1.0
        return e

    emb, meta = embed_and_validate(X, obs, content_keyed, "celltype_id",
                                    drop_unlabeled=True)
    origin = [int(np.argmax(r)) for r in emb]
    assert origin == [0, 2, 3, 4]                       # cell1 gone
    assert list(meta[QUERY_ID]) == ["cell0", "cell2", "cell3", "cell4"]
