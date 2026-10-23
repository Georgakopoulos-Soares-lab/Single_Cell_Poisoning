"""Sharding a sweep must not change a single number.

Sharding exists only to overlap ``hnswlib.load_index``, which is single-threaded
and 59% of a condition. It divides the celltype loop across processes. The whole
argument for it is that it CANNOT affect a result, and these tests are what make
that a fact rather than a hope.
"""
import pathlib

import numpy as np
import pandas as pd
import pytest
from merge_shards import merge

from celltype_sweep import seed_of

SOURCES = [f"CL:{i:07d}" for i in range(26)]


# ── the invariant that makes sharding safe ──────────────────────────────────

def test_the_condition_seed_cannot_see_the_shard() -> None:
    """seed_of(source, rate, seed) takes no shard argument, so no poison moves.

    This is the entire safety argument. If the seed depended on position in the
    loop, dividing the loop would rebuild every poison set differently and the
    sharded run would not be the unsharded one.
    """
    for src in SOURCES:
        for rate in (0.002, 0.25, 1.25):
            assert seed_of(src, rate, 0) == seed_of(src, rate, 0)
    # and it depends on exactly the three things it should
    assert seed_of("CL:0000001", 0.25, 0) != seed_of("CL:0000002", 0.25, 0)
    assert seed_of("CL:0000001", 0.25, 0) != seed_of("CL:0000001", 0.50, 0)
    assert seed_of("CL:0000001", 0.25, 0) != seed_of("CL:0000001", 0.25, 1)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7])
def test_strided_shards_partition_the_celltypes_exactly(n: int) -> None:
    """Every celltype in exactly one shard, none twice, none dropped."""
    parts = [SOURCES[i::n] for i in range(n)]
    flat = [s for p in parts for s in p]
    assert sorted(flat) == sorted(SOURCES)
    assert len(flat) == len(set(flat)) == len(SOURCES)


def test_shards_are_balanced_within_one_celltype(n: int = 3) -> None:
    """A strided split cannot leave one process with twice the work."""
    sizes = [len(SOURCES[i::n]) for i in range(n)]
    assert max(sizes) - min(sizes) <= 1


# ── the merge refuses rather than papering over ─────────────────────────────

def _shard(tmp: pathlib.Path, stem: str, i: int, sources: list[str],
           n_conditions: int = 26) -> None:
    pd.DataFrame({"source": sources, "attack": "flood", "rate": 0.25,
                  "n_anchors": 64, "value": np.arange(len(sources), dtype=float),
                  "condition_index": range(len(sources)),
                  "n_conditions": n_conditions}
                 ).to_parquet(tmp / f"{stem}_shard{i}.parquet", index=False)


def test_a_complete_shard_set_merges(tmp_path) -> None:
    for i in range(3):
        _shard(tmp_path, "t", i, SOURCES[i::3])
    df = merge(tmp_path / "t.parquet")
    assert len(df) == 26
    assert sorted(df.source) == sorted(SOURCES)
    assert df.condition_index.tolist() == list(range(26))   # renumbered


def test_a_missing_shard_is_refused(tmp_path) -> None:
    """Shard 1 absent: 17 rows that would look like a complete small table."""
    for i in (0, 2):
        _shard(tmp_path, "t", i, SOURCES[i::3])
    with pytest.raises(ValueError, match="not 0..1|missing"):
        merge(tmp_path / "t.parquet")


def test_an_overlapping_shard_is_refused(tmp_path) -> None:
    """A hand-launched shard with the wrong index double-counts a celltype."""
    _shard(tmp_path, "t", 0, SOURCES[0::3])
    _shard(tmp_path, "t", 1, SOURCES[0::3])          # same slice by mistake
    _shard(tmp_path, "t", 2, SOURCES[2::3])
    with pytest.raises(ValueError, match="appears in shard"):
        merge(tmp_path / "t.parquet")


def test_an_unfinished_shard_is_refused(tmp_path) -> None:
    """A shard that crashed halfway leaves too few rows; n_conditions catches it."""
    _shard(tmp_path, "t", 0, SOURCES[0::3])
    _shard(tmp_path, "t", 1, SOURCES[1::3][:2])      # truncated
    _shard(tmp_path, "t", 2, SOURCES[2::3])
    with pytest.raises(ValueError, match="did not finish"):
        merge(tmp_path / "t.parquet")


def test_disagreeing_columns_are_refused(tmp_path) -> None:
    _shard(tmp_path, "t", 0, SOURCES[0::3])
    _shard(tmp_path, "t", 1, SOURCES[1::3])
    _shard(tmp_path, "t", 2, SOURCES[2::3])
    p = tmp_path / "t_shard2.parquet"
    pd.read_parquet(p).drop(columns=["value"]).to_parquet(p, index=False)
    with pytest.raises(ValueError, match="disagree on columns"):
        merge(tmp_path / "t.parquet")


def test_no_shards_at_all_is_refused(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        merge(tmp_path / "t.parquet")


def test_merge_refuses_when_completeness_cannot_be_verified(tmp_path) -> None:
    """The check must FAIL CLOSED. Until 2026-09-26 it was conditional on the
    column existing, so 02b_15 -- which did not emit it -- had a 324-row table
    from a 936-condition grid written as though complete."""
    for i in range(3):
        pd.DataFrame({"source": [f"CL:{i}"], "value": [1.0]}).to_parquet(
            tmp_path / f"t_shard{i}.parquet", index=False)
    with pytest.raises(ValueError, match="completeness cannot"):
        merge(tmp_path / "t.parquet")
    assert not (tmp_path / "t.parquet").exists(), "wrote a table it could not verify"


def test_merge_still_refuses_a_short_set_when_it_can_verify(tmp_path) -> None:
    """With the column present, the original arithmetic check still fires."""
    for i in range(3):
        pd.DataFrame({"source": [f"CL:{i}"], "n_conditions": [9]}).to_parquet(
            tmp_path / f"s_shard{i}.parquet", index=False)
    with pytest.raises(ValueError, match="n_conditions says 9"):
        merge(tmp_path / "s.parquet")
    assert not (tmp_path / "s.parquet").exists()


def test_merge_accepts_a_complete_verifiable_set(tmp_path) -> None:
    for i in range(3):
        pd.DataFrame({"source": [f"CL:{i}"] * 2,
                      "n_conditions": [6, 6]}).to_parquet(
            tmp_path / f"g_shard{i}.parquet", index=False)
    out = merge(tmp_path / "g.parquet")
    assert len(out) == 6
    assert (tmp_path / "g.parquet").exists()


def test_expect_allows_a_by_donor_merge_to_be_verified(tmp_path) -> None:
    """A by_donor table holds one row per condition PER DONOR, so it cannot carry
    n_conditions -- its row count is conditions x donors. Making the check fail
    closed on 2026-09-26 therefore blocked every by_donor merge until --expect
    gave it something to verify against."""
    for i in range(3):
        pd.DataFrame({"source": [f"CL:{i}"] * 4,
                      "donor_id": ["d0", "d1", "d0", "d1"]}).to_parquet(
            tmp_path / f"b_shard{i}.parquet", index=False)
    out = merge(tmp_path / "b.parquet", expect=12)   # 6 conditions x 2 donors
    assert len(out) == 12
    assert (tmp_path / "b.parquet").exists()


def test_expect_still_catches_a_short_by_donor_set(tmp_path) -> None:
    for i in range(3):
        pd.DataFrame({"source": [f"CL:{i}"] * 4,
                      "donor_id": ["d0", "d1", "d0", "d1"]}).to_parquet(
            tmp_path / f"c_shard{i}.parquet", index=False)
    with pytest.raises(ValueError, match="n_conditions says 20"):
        merge(tmp_path / "c.parquet", expect=20)
    assert not (tmp_path / "c.parquet").exists()


def test_expect_overrides_n_conditions_when_both_present(tmp_path) -> None:
    """--expect wins, so a caller who knows the true total is never overruled by
    a column that means something else."""
    for i in range(3):
        pd.DataFrame({"source": [f"CL:{i}"] * 2,
                      "n_conditions": [99, 99]}).to_parquet(
            tmp_path / f"d_shard{i}.parquet", index=False)
    out = merge(tmp_path / "d.parquet", expect=6)
    assert len(out) == 6


def test_successful_merge_removes_its_shards(tmp_path) -> None:
    """Shards are intermediates once the canonical table holds every row. Leaving
    them failed check_readme's bijection and check_schema, and those gate a
    commit."""
    for i in range(3):
        pd.DataFrame({"source": [f"CL:{i}"] * 2,
                      "n_conditions": [6, 6]}).to_parquet(
            tmp_path / f"k_shard{i}.parquet", index=False)
    merge(tmp_path / "k.parquet")
    assert (tmp_path / "k.parquet").exists()
    assert list(tmp_path.glob("k_shard*.parquet")) == [], "shards survived"


def test_a_REFUSED_merge_keeps_every_shard(tmp_path) -> None:
    """The sharp edge. Deleting on failure would destroy hours of compute that
    the merge itself judged unfit -- and the judgement can be wrong."""
    for i in range(3):
        pd.DataFrame({"source": [f"CL:{i}"], "n_conditions": [99]}).to_parquet(
            tmp_path / f"r_shard{i}.parquet", index=False)
    with pytest.raises(ValueError, match="n_conditions says 99"):
        merge(tmp_path / "r.parquet")
    assert len(list(tmp_path.glob("r_shard*.parquet"))) == 3, "shards were deleted"
    assert not (tmp_path / "r.parquet").exists()


def test_unverifiable_merge_keeps_every_shard(tmp_path) -> None:
    for i in range(3):
        pd.DataFrame({"source": [f"CL:{i}"]}).to_parquet(
            tmp_path / f"u_shard{i}.parquet", index=False)
    with pytest.raises(ValueError, match="completeness cannot"):
        merge(tmp_path / "u.parquet")
    assert len(list(tmp_path.glob("u_shard*.parquet"))) == 3
