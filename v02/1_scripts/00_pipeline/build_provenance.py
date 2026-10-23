"""Recover per-record provenance for the annotation index. THE MISSING PRODUCER.

``cache/reference_provenance.npz`` carries ``study`` and ``sample`` for all
7,913,892 records of the shipped annotation index. Everything that reasons about
who contributed a record depends on it — donor voting, the submission-volume cap,
the genuine-study analysis. Until now it had **no producer script and appeared
nowhere in the DAG**, so an outsider could not regenerate it and the numbers that
rest on it could not be checked. This closes that.

**Why a join is needed at all.** The annotation index ships only vectors and a
label column; it carries no study, sample, donor or timestamp. The larger
cellsearch corpus (23,381,150 cells) ships a TileDB metadata array with 25
attributes including ``study`` and ``sample``. The annotation index is a subset
of that corpus, so the metadata can be recovered — but only by finding, for each
annotation record, the cellsearch record it came from.

**The join key is the embedding itself**, established empirically rather than
assumed: annotation record 0 matches cellsearch record 19,402,081 with a maximum
elementwise difference of 3.0e-08, which is float32 round-off. Row order does not
align (label sequences agree at 1.46%, i.e. chance), and there is no shared id
column, so the vectors are the only available key.

**The match is within tolerance, not bit-identical, and that distinction cost a
rewrite.** The two copies of a cell differ by around 3e-08 per element — float32
round-off, most likely from re-normalisation when the annotation index was built.
A byte-exact hash join therefore recovers only 38.3% of records. The join must
allow that tolerance.

It is done by querying the shipped ``labelled_kNN.bin``, which is already an
HNSW index over exactly these 7,913,892 vectors, with each cellsearch embedding,
and accepting a hit only when the cosine distance is below ``TOL``. That
threshold is the safeguard a nearest-neighbour join otherwise lacks: two
genuinely distinct cells can sit arbitrarily close together, and accepting the
nearest one unconditionally would attach the wrong donor to a record. Records
with no hit under the threshold get an empty string, and coverage is reported
rather than assumed.

Writes cache/reference_provenance.npz.
"""

import pathlib
import sys
import time

import numpy as np

BASE = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "1_scripts" / "00_common"))

from dataset_config import cache_root, load  # noqa: E402

#: Reference-derived, so it stays at the SHARED cache root: the join is
#: annotation-record -> cellsearch-row and never looks at a query.
CACHE = cache_root(BASE, load(BASE))

CS = BASE / "0_data/reference/model_v1.1/cellsearch"
ANN = BASE / "0_data/reference/model_v1.1/annotation/labelled_kNN.bin"
BLOCK = 500_000
#: cosine distance below which two vectors are taken to be the same cell. The
#: observed round-off between the two copies is ~3e-08; 1e-6 clears that by two
#: orders of magnitude while staying far below the distance to any genuinely
#: different cell (the median nearest genuine neighbour sits at cosine 0.9965,
#: i.e. a distance of 3.5e-03 -- three thousand times larger).
TOL = 1e-6


def main() -> None:
    """Match every annotation record to its cellsearch row and save provenance."""
    import hnswlib
    import tiledb

    t0 = time.time()
    n = len(np.load(CACHE / "reference.npz")["label"])
    print(f"[{time.time()-t0:.0f}s] annotation records: {n:,}", flush=True)

    with tiledb.open(str(CS / "cell_metadata")) as A:
        q = A.query(attrs=["study", "sample"])[:]
    cs_study = np.asarray(q["study"]).astype(str)
    cs_sample = np.asarray(q["sample"]).astype(str)
    n_cs = len(cs_study)
    print(f"[{time.time()-t0:.0f}s] cellsearch records: {n_cs:,}", flush=True)

    index = hnswlib.Index(space="cosine", dim=128)
    index.load_index(str(ANN))
    index.set_ef(16)                      # k = 1 only; a wide beam buys nothing
    print(f"[{time.time()-t0:.0f}s] loaded the annotation index "
          f"({index.get_current_count():,} vectors)", flush=True)

    study = np.full(n, "", dtype=object)
    sample = np.full(n, "", dtype=object)
    found = 0
    with tiledb.open(str(CS / "cell_embedding")) as E:
        for lo in range(0, n_cs, BLOCK):
            hi = min(lo + BLOCK, n_cs)
            blk = np.asarray(E[lo:hi]["vals"]).reshape(hi - lo, -1)
            ids, dist = index.knn_query(blk, k=1, num_threads=6)
            ids = ids[:, 0]
            near = dist[:, 0] < TOL
            for k in np.flatnonzero(near):
                i = int(ids[k])
                if study[i] == "":
                    study[i] = cs_study[lo + k]
                    sample[i] = cs_sample[lo + k]
                    found += 1
            print(f"[{time.time()-t0:.0f}s]   scanned {hi:,}/{n_cs:,} "
                  f"— matched {found:,}/{n:,} ({found/n*100:.1f}%)", flush=True)

    out = CACHE / "reference_provenance.npz"
    np.savez_compressed(out, study=study, sample=sample)
    print(f"\n[{time.time()-t0:.0f}s] wrote {out}")
    print(f"  coverage: {found/n*100:.2f}%   "
          f"{len(set(study.tolist())-{''}):,} studies, "
          f"{len(set(sample.tolist())-{''}):,} samples")


if __name__ == "__main__":
    main()
