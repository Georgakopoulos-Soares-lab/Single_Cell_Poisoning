"""Hand-built tiny fixtures for golden-value tests.

Deliberately 4-dimensional and 10 rows: small enough that every expected value
below can be checked by hand, which is the point of a golden-value test.
"""

from __future__ import annotations

import numpy as np

SRC = "CL:0000235"  # macrophage, in the fixture
TGT = "CL:1001567"  # lung endothelial cell


def unit(a: np.ndarray) -> np.ndarray:
    """Row-normalise. a: (n, d) float; returns (n, d) float32, unit norm."""
    return (a / np.linalg.norm(a, axis=1, keepdims=True)).astype(np.float32)


def tiny_reference() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A 10 x 4 reference: 5 source cells near e0, 5 target cells near e1.

    Returns:
        emb: (10, 4) float32, unit norm
        labels: (10,) str
        ids: (10,) str
    """
    rng = np.random.default_rng(12345)
    src = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (5, 1)) + 0.05 * rng.standard_normal((5, 4))
    tgt = np.tile(np.array([0.0, 1.0, 0.0, 0.0]), (5, 1)) + 0.05 * rng.standard_normal((5, 4))
    emb = unit(np.vstack([src, tgt]))
    labels = np.array([SRC] * 5 + [TGT] * 5)
    ids = np.array([f"ref{i}" for i in range(10)])
    return emb, labels, ids
