"""Concatenate a sharded step's tables back into its canonical one.

A celltype-sharded sweep (see ``BIORAG_SHARD`` in ``02b_02``) writes
``<stem>_shard0.parquet``, ``_shard1``, ``_shard2``. This joins them. Sharding
exists only to overlap ``hnswlib.load_index``, which is single-threaded and 59%
of a condition, so the merged table must be indistinguishable from an unsharded
run -- and this file's job is to REFUSE rather than paper over it if it is not.

Four checks, each guarding a way a shard set can be wrong:

1. **No gaps.** Shards 0..N-1 must all be present. A missing shard silently
   drops whole celltypes, and the result would look like a complete table with
   fewer rows.
2. **No overlap.** The celltypes must partition: no source may appear twice. A
   strided slice cannot overlap, but a hand-launched shard with the wrong index
   can, and a duplicated celltype would be double-counted by every aggregation.
3. **Complete.** The merged row count must equal ``n_conditions``, the GLOBAL
   total each shard carries. This is the check that catches a shard that
   crashed halfway. A step that does not emit the column is REFUSED, not
   waved through -- see the comment at the check itself.
4. **One schema.** Columns must match across shards, or the concatenation would
   introduce nulls that no reader expects.

Shards are DELETED once the merge succeeds -- see the comment at the unlink.

``condition_index`` is renumbered on merge. It is shard-local while running --
each shard counts its own rows from zero -- and only means anything once the
pieces are in one table.

Run as ``python merge_shards.py <stem.parquet>``; it finds the shards beside
it. A by_donor table holds one row per condition PER DONOR, so its row count is
``conditions x donors`` and it cannot carry ``n_conditions``; pass
``--expect <conditions * donors>`` for those. Without either, the merge is
refused rather than written unverified.
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd


def merge(canonical: pathlib.Path,
          expect: int | None = None) -> pd.DataFrame:
    """Join ``<stem>_shard*.parquet`` beside ``canonical`` and write it.

    Args:
        canonical: the unsharded path the merged table should take.

    Returns:
        The merged frame, as written.

    Raises:
        FileNotFoundError: no shards found.
        ValueError: shards missing, overlapping, incomplete, or disagreeing on
            columns.
    """
    shards = sorted(canonical.parent.glob(f"{canonical.stem}_shard*{canonical.suffix}"))
    if not shards:
        raise FileNotFoundError(f"no shards beside {canonical}")

    idx = sorted(int(p.stem.rsplit("_shard", 1)[1]) for p in shards)
    if idx != list(range(len(idx))):
        raise ValueError(
            f"shard indices {idx} are not 0..{len(idx)-1}; a missing shard "
            f"drops whole celltypes and the merged table would look complete")

    frames = [pd.read_parquet(p) for p in shards]
    cols = [tuple(f.columns) for f in frames]
    if len(set(cols)) != 1:
        raise ValueError("shards disagree on columns; concatenating would "
                         "introduce nulls no reader expects")

    seen: dict[str, int] = {}
    for i, f in enumerate(frames):
        for src in f["source"].unique():
            if src in seen:
                raise ValueError(
                    f"celltype {src} appears in shard {seen[src]} AND shard "
                    f"{i}; every aggregation would double-count it")
            seen[src] = i

    df = pd.concat(frames, ignore_index=True).sort_values(
        ["source", *[c for c in ("attack", "rate", "n_anchors") if c in frames[0]]],
        kind="stable").reset_index(drop=True)
    if "condition_index" in df.columns:
        df["condition_index"] = range(len(df))

    # FAIL CLOSED. This was `if "n_conditions" in df.columns:` until 2026-09-26,
    # so on a step that did not emit the column the check SILENTLY SKIPPED and a
    # partial merge was written as though complete. That is exactly what happened:
    # 02b_15's graph run was killed at 35%, and this file wrote a 324-row table
    # from a 936-condition grid, 9 celltypes of 26, stamped retrieval_path=graph,
    # with nothing in it to say so. A guard that disappears when its input is
    # missing is worse than no guard, because the docstring above promises it.
    if expect is not None:
        want = expect
    elif "n_conditions" in df.columns:
        want = int(df["n_conditions"].iloc[0])
    else:
        raise ValueError(
            f"{canonical.name}: no 'n_conditions' column and no --expect, so "
            f"completeness cannot be verified -- and an unverifiable merge is "
            f"refused rather than written, because a partial table here is "
            f"indistinguishable from a complete one downstream. A CONDITION table "
            f"should emit n_conditions (the GLOBAL count, not the shard's). A "
            f"by_donor table cannot: its row count is conditions x donors, not "
            f"conditions, so pass --expect <conditions * donors>. Shard row "
            f"counts: {[len(f) for f in frames]}")
    if len(df) != want:
        raise ValueError(
            f"merged {len(df)} rows but n_conditions says {want}; a shard "
            f"did not finish. Shard row counts: "
            f"{[len(f) for f in frames]}")

    df.to_parquet(canonical, index=False)
    print(f"{len(shards)} shards -> {len(df):,} rows, {df['source'].nunique()} "
          f"celltypes -> {canonical.name}")

    # REMOVE THE SHARDS, but only here -- after all four checks passed and the
    # canonical table is on disk. Every row is now in it, verified by the count
    # check above, so a shard is a redundant intermediate and not a result.
    #
    # It matters that the PRODUCER does this. 2_results is written only by
    # pipeline execution (CLAUDE.md), so an agent deleting these by hand would
    # break the rule that makes the directory trustworthy; merge_shards clearing
    # its own intermediates as the last act of a successful merge does not.
    #
    # Leaving them was not harmless: 18 shard files failed check_readme's
    # bijection and 13 failed check_schema (a by_donor shard carries no
    # threat_model), and those four checks gate a commit. A step that cannot be
    # committed without --no-verify is a step whose provenance is weaker.
    for sh in shards:
        sh.unlink()
    print(f"  removed {len(shards)} merged shard files")
    return df


def main(args: list[str]) -> int:
    """Merge each canonical path named on the command line."""
    if not args:
        print(__doc__, file=sys.stderr)
        return 2
    expect: int | None = None
    paths: list[str] = []
    it = iter(args)
    for a in it:
        if a == "--expect":
            expect = int(next(it))
        elif a.startswith("--expect="):
            expect = int(a.split("=", 1)[1])
        else:
            paths.append(a)
    if not paths:
        print(__doc__, file=sys.stderr)
        return 2
    for a in paths:
        merge(pathlib.Path(a), expect=expect)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
