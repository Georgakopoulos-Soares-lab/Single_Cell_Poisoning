"""The condition seed, and the float width a stored rate has to keep.

``seed_of`` hashes ``repr(rate)``, so the seed depends on the float's WIDTH, not
just its value. Narrowing ``rate`` to float32 on the way into a results table
therefore breaks reproduction silently: the stored rate re-derives a different
seed, builds a different poison set, and the run looks irreproducible when it was
only the table that lost precision. These tests pin both halves of the fix --
``rate`` keeps its width, and the derived seed is written down so nobody has to
re-derive it at all.
"""

from __future__ import annotations
# 1_scripts cannot be a package (leading digit), so the shared library in
# 00_common is put on the path and imported flat.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_common"))


import numpy as np
import pandas as pd

from celltype_sweep import KEY_COLUMNS, OPTIONAL_DTYPES, seed_of

RATES = (0.25, 0.5, 0.7, 0.8, 0.85, 0.9, 1.0, 1.25)
SRC = "CL:0000235"


def test_seed_is_stable_across_processes() -> None:
    # blake2b, not hash(): hash() is salted per process, so a seed derived in one
    # run would not reproduce in the next. Pinned to the literal value, because
    # "it equals itself" would pass under a salted hash too.
    assert seed_of(SRC, 0.7, 0) == 0xF9CD7E4B


def test_every_condition_gets_its_own_seed() -> None:
    seeds = {seed_of(SRC, r, 0) for r in RATES}
    assert len(seeds) == len(RATES)


def test_narrowing_the_rate_changes_the_seed() -> None:
    # The failure this whole module exists to prevent. 0.7 as float32 re-reads
    # as 0.699999988079071, whose repr is a different string and therefore a
    # different key -- a DIFFERENT poison set, built in silence.
    narrowed = float(np.float32(0.7))
    assert narrowed != 0.7
    assert seed_of(SRC, narrowed, 0) != seed_of(SRC, 0.7, 0)


def test_write_keeps_the_rate_at_full_width() -> None:
    from celltype_sweep import write
    import tempfile

    rows = [{"rate": r, "condition_seed": seed_of(SRC, r, 0),
             "within_accuracy": 0.5} for r in RATES]
    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d) / "t.parquet"
        write(rows, out)
        df = pd.read_parquet(out)

    assert df["rate"].dtype == np.float64, "a key column must not be narrowed"
    # ... while an ordinary measurement column still is.
    assert df["within_accuracy"].dtype == np.float32


def test_a_stored_row_reproduces_its_own_seed() -> None:
    # The round trip a reader actually performs: read the table, re-derive the
    # seed from the stored rate, and get the seed the run used.
    from celltype_sweep import write
    import tempfile

    rows = [{"rate": r, "condition_seed": seed_of(SRC, r, 0), "source": SRC}
            for r in RATES]
    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d) / "t.parquet"
        write(rows, out)
        df = pd.read_parquet(out)

    for _, row in df.iterrows():
        assert seed_of(row["source"], float(row["rate"]), 0) == row["condition_seed"]


def test_rate_is_declared_a_key_and_the_seed_is_declared_an_integer() -> None:
    # Both halves are declarations other code reads, so pin them.
    assert "rate" in KEY_COLUMNS
    assert OPTIONAL_DTYPES["condition_seed"] == "Int64"
