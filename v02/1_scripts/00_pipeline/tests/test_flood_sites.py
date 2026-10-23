"""Where a flood puts its clusters — v01's ``mode``, isolated as one axis.

SPEC.md §5.4. The load-bearing test is the regression one: ``build_flood``
gained a parameter and every committed flood condition must still reproduce
byte for byte, because 520 of them are on disk and more pair with v01 rows.
"""
import numpy as np
import pytest

from flood_common import build_flood, kmeans_sites
from schemas import Sites


@pytest.fixture
def pool() -> np.ndarray:
    """A 2,000-cell celltype in 16 dimensions, in two loose clumps."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((1000, 16)) + 4 * np.eye(16)[0]
    b = rng.standard_normal((1000, 16)) + 4 * np.eye(16)[1]
    x = np.vstack([a, b]).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_the_default_path_is_byte_identical(pool) -> None:
    """Adding `sites` must not move any committed flood condition.

    `==` on the arrays, not approx: the anchor branch consumes the RNG exactly
    as it did before the parameter existed, so any difference is a different
    draw rather than rounding.
    """
    a = build_flood(pool, 64, 0.5, 0.99, 1.0, 7)
    b = build_flood(pool, 64, 0.5, 0.99, 1.0, 7, sites=Sites.ANCHOR)
    assert np.array_equal(a, b)


def test_kmeans_sites_differ_from_anchors(pool) -> None:
    """If they agreed the axis would be measuring nothing."""
    a = build_flood(pool, 64, 0.5, 0.99, 1.0, 7, sites=Sites.ANCHOR)
    k = build_flood(pool, 64, 0.5, 0.99, 1.0, 7, sites=Sites.KMEANS)
    assert a.shape == k.shape
    assert not np.array_equal(a, k)


def test_both_arms_produce_the_same_budget(pool) -> None:
    """The comparison is paired, so the record count must not vary with sites."""
    for n in (16, 64, 256):
        a = build_flood(pool, n, 0.5, 0.99, 1.0, 7, sites=Sites.ANCHOR)
        k = build_flood(pool, n, 0.5, 0.99, 1.0, 7, sites=Sites.KMEANS)
        assert len(a) == len(k) == 1000


def test_every_vector_is_unit_norm(pool) -> None:
    for s in (Sites.ANCHOR, Sites.KMEANS):
        e = build_flood(pool, 32, 0.5, 0.99, 1.0, 7, sites=s)
        assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-5)


def test_kmeans_centres_sit_closer_to_the_pool_centre_than_real_cells(pool) -> None:
    """The mechanism behind v01's result, reproduced on a fixture.

    A centroid is an average, and an average of a cloud is not in the cloud: it
    sits nearer the middle than any real member does. v01 measured a real
    centroid at cosine 0.9496 to its queries against a 0.9796 fiftieth genuine
    neighbour -- below the retrieval bar. That is why the construction captures
    neighbourhoods without dominating them.
    """
    mid = pool.mean(axis=0)
    mid /= np.linalg.norm(mid)
    anchors = build_flood(pool, 64, 0.1, 1.0, 1.0, 7, sites=Sites.ANCHOR)
    centres = build_flood(pool, 64, 0.1, 1.0, 1.0, 7, sites=Sites.KMEANS)
    assert (centres @ mid).mean() > (anchors @ mid).mean()


def test_kmeans_is_capped_by_the_pool(pool) -> None:
    """More clusters than cells is not meaningful; the cap is silent but real."""
    assert len(kmeans_sites(pool, 5000, 0)) == len(pool)


def test_kmeans_is_reproducible_from_the_seed(pool) -> None:
    assert np.array_equal(kmeans_sites(pool, 32, 3), kmeans_sites(pool, 32, 3))


def test_unknown_sites_fails_loudly(pool) -> None:
    with pytest.raises(ValueError, match="unknown sites"):
        build_flood(pool, 32, 0.5, 0.99, 1.0, 7, sites="somewhere_nice")
