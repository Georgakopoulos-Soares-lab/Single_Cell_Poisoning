"""Step 01a_05 — how often does a GENUINE reference record disagree with its own neighbourhood?

The label-consistency defence rejects a record whose label disagrees with the
labels around it. Measured against poison alone that looks free: poison is
mislabelled by construction, so it scores badly by construction. The cost only
appears on the other side of the ledger, and this step measures it.

A genuine record can score badly for reasons that have nothing to do with an
attacker -- it sits on a boundary between two subtypes, its study annotated at a
coarser granularity than its neighbours' studies, or one study contributes most
of a celltype and its conventions are the local majority. In a reference built
from many studies that population is not small, and every record in it is one a
label-consistency filter would throw away. A defence whose false-positive rate is
unmeasured is not a defence, it is a claim.

Scope is the records that a query **actually retrieves**, not the whole
reference. A record no query ever reaches costs nothing to reject, so including
it would dilute the false-positive rate toward zero and flatter the defence.
The retrieval used is ``01a_01_no_access_hnsw_neighbors.npz`` -- SCimilarity's own
graph. v01 scored the records that EXACT search retrieved, which is a different
and slightly smaller set, and it hardcoded the four source terms; here they come
from ``config.pairs``, so adding a pair to the grid extends this step
automatically instead of silently leaving its records unscored.

Each record is scored against the same shipped graph, **loaded, never rebuilt**,
excluding its own self-match. The self-match is dropped because a record is
always its own nearest neighbour in the index it lives in, and keeping it would
add one guaranteed agreement to every score -- which is exactly the advantage
poison does not get, since poison is scored before it is inserted.

``n_retrievals`` is carried alongside, because rejecting a record that fills
thousands of query-neighbour slots and rejecting one that fills a single slot
are not the same cost. The retrieval-weighted rejection rate in the summary is
the one that says what a threshold would do to answers; the unweighted rate says
what it would do to the atlas. Both are reported.

**No confidence intervals.** Every other interval in this project resamples
donors, because the query cells are the unit and donors are their unit of
independence. These are reference RECORDS, not queries: the shipped annotation
bundle carries no donor or sample id for them at all, and the provenance join
that recovers one is a ceiling-tier artefact (``schemas.DEFENCE_TIER``). Quoting
a cell-resampled interval here would claim independence that has not been
established, so rates are reported as point estimates over a stated denominator.

**THREAT MODEL: none.** No attacker exists in this step -- these are the genuine
records, scored before any poison is constructed. ``threat_model`` is written
because this is the false-positive side of the defence that the ``no_access``
arm is evaluated with.

**Loads the 8.8 GB shipped index and the 4 GB reference embeddings — requires
explicit human approval (CLAUDE.md §1).**

Writes, under ``2_results/<dataset>/01_a_baseline/``:
  ``01a_05_no_access_genuine_lc.npz``              idx/lc, the array the defence layer loads
  ``01a_05_no_access_genuine_lc_scores.parquet``   one row per scored record
  ``01a_05_no_access_genuine_lc_summary.parquet``  rejection cost per threshold
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_common"))
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dataset_config import cache_root, load, query_cache  # noqa: E402
from provenance import emit  # noqa: E402
from schemas import EMB_DIM, EMB_DTYPE, K_NEIGHBORS, THREAT_MODEL, ThreatModel  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[2]
CFG = load(BASE)
DS = CFG["dataset"]
CACHE = cache_root(BASE, CFG)      # shared: reference.npz, poison/
QCACHE = query_cache(BASE, CFG)    # this dataset only -- see dataset_config
OUT = BASE / "2_results" / DS / "01_a_baseline"

#: SCimilarity's own default, and the value 01a_01 retrieves with. The two must
#: match: a genuine record scored at a wider ``ef`` than the queries were
#: retrieved at would be judged by a neighbourhood the deployed system never
#: sees. v01 used 120 here and 200 for retrieval, so neither matched the vendor.
DEFAULT_EF = 100

#: Records per ``knn_query`` call. Bounds the transient (block, k+1) id array
#: and keeps progress reporting meaningful on a multi-minute scan.
BLOCK = 5000

#: Thresholds a label-consistency filter might be set at. Reported as a ladder
#: rather than a chosen value: the choice belongs to the defence step, and this
#: table is what that choice should be made from.
LC_THRESHOLDS = (0.5, 0.7, 0.9)


def attacked_sources(cfg: dict) -> list[str]:
    """Source CL ids of every single-celltype pair in the grid, sorted.

    Combos are skipped: a combo's members are already present as singles, so
    including it would only duplicate ids.

    Args:
        cfg: parsed ``config.yaml``.

    Returns:
        Sorted unique source terms -- the celltypes whose queries define which
        reference records are worth scoring.
    """
    return sorted({p["source"] for p in cfg["pairs"] if "members" not in p})


def load_shipped_index(path: pathlib.Path, n_expected: int, ef: int):
    """Open the shipped graph read-only and configure it like SCimilarity does.

    Args:
        path: ``labelled_kNN.bin`` as shipped. n_expected: reference rows, so a
            mismatched index fails here rather than scoring records against ids
            that do not line up with ``reference.npz``. ef: query-time candidate
            list size; must admit ``k + 1``, since the self-match is retrieved
            and then dropped.

    Returns:
        A loaded ``hnswlib.Index``. Never built: ``init_index`` is not called.
    """
    import hnswlib

    if ef < K_NEIGHBORS + 1:
        raise ValueError(f"ef {ef} < k+1 {K_NEIGHBORS + 1}: the self-match is "
                         "retrieved and dropped, so one extra slot is needed")
    index = hnswlib.Index(space="cosine", dim=EMB_DIM)
    index.load_index(str(path))
    n = index.get_current_count()
    if n != n_expected:
        raise ValueError(
            f"index holds {n:,} vectors but reference.npz has {n_expected:,}; "
            "ids would not correspond and every score would be wrong"
        )
    index.set_ef(ef)
    return index


def label_consistency(index, emb: np.ndarray, lab: np.ndarray,
                      want: np.ndarray, threads: int) -> np.ndarray:
    """Share of each record's own k nearest neighbours that carry its label.

    Args:
        index: the loaded shipped graph. emb: (n_ref, 128) float32 unit-norm
            reference embeddings. lab: (n_ref,) str reference celltypes.
        want: (n,) int64 reference row ids to score. threads: hnswlib worker
            count.

    Returns:
        (n,) float32 agreement in [0, 1], aligned with ``want``. A record whose
        neighbourhood is entirely its own celltype scores 1.0; one that agrees
        with none of its neighbours scores 0.0.
    """
    out = np.empty(len(want), dtype=EMB_DTYPE)
    for lo in range(0, len(want), BLOCK):
        sel = want[lo:lo + BLOCK]
        ids, _ = index.knn_query(emb[sel], k=K_NEIGHBORS + 1, num_threads=threads)
        # Drop the self-match, then keep the first k of what remains. A record
        # missing from its own result -- possible under an approximate graph --
        # simply keeps its k nearest others, which is the honest reading.
        keep = np.where(ids == sel[:, None], -1, ids)
        for r in range(len(sel)):
            nbr = keep[r][keep[r] >= 0][:K_NEIGHBORS]
            out[lo + r] = (lab[nbr] == lab[sel[r]]).mean() if len(nbr) else 1.0
    return out


def main() -> None:
    t0 = time.time()
    emit(__file__)          # which code ran, HEAD or dirty tree
    ef = int(CFG.get("hnsw", {}).get("ef_search", DEFAULT_EF))
    threads = int(CFG.get("hnsw", {}).get("num_threads", 6))
    sources = attacked_sources(CFG)

    hn = np.load(OUT / "01a_01_no_access_hnsw_neighbors.npz", allow_pickle=True)
    meta = pd.read_parquet(QCACHE / "query_meta.parquet")
    idx = hn["idx"]
    if idx.shape != (len(meta), K_NEIGHBORS):
        raise ValueError(f"neighbours are {idx.shape}, expected "
                         f"{(len(meta), K_NEIGHBORS)}")

    y = meta["y_true"].to_numpy().astype(str)
    rows = np.flatnonzero(np.isin(y, sources))
    retrieved = idx[rows].ravel()
    want, n_retrievals = np.unique(retrieved, return_counts=True)
    print(f"[{time.time() - t0:.0f}s] {len(rows):,} queries of "
          f"{len(sources)} attacked celltypes reach {len(want):,} distinct "
          f"reference records in {len(retrieved):,} neighbour slots",
          file=sys.stderr)

    ref = np.load(CACHE / "reference.npz", allow_pickle=True)
    lab = ref["label"].astype(str)
    emb = ref["emb"].astype(EMB_DTYPE)

    idx_path = BASE.parent / CFG["paths"]["reference_dir"] / "annotation" / "labelled_kNN.bin"
    print(f"loading {idx_path} ({idx_path.stat().st_size / 1e9:.2f} GB) …",
          file=sys.stderr)
    index = load_shipped_index(idx_path, len(lab), ef)
    print(f"[{time.time() - t0:.0f}s] loaded; scoring at ef={ef}, "
          f"k={K_NEIGHBORS} excluding self", file=sys.stderr)

    lc = label_consistency(index, emb, lab, want, threads)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "01a_05_no_access_genuine_lc.npz",
                        idx=want, lc=lc)

    scores = pd.DataFrame({
        THREAT_MODEL: ThreatModel.NO_ACCESS,
        "ref_idx": want.astype(np.int64),
        "ref_celltype": lab[want],
        "label_consistency": lc,
        "n_retrievals": n_retrievals.astype(np.int64),
    })
    scores.to_parquet(OUT / "01a_05_no_access_genuine_lc_scores.parquet", index=False)

    summary = []
    for ct in sorted(pd.unique(scores.ref_celltype)) + ["__all__"]:
        sub = scores if ct == "__all__" else scores[scores.ref_celltype == ct]
        w = sub["n_retrievals"].to_numpy(dtype=float)
        v = sub["label_consistency"].to_numpy()
        row = {
            THREAT_MODEL: ThreatModel.NO_ACCESS,
            "ref_celltype": ct,
            "n_records": int(len(sub)),
            "n_retrieval_slots": int(w.sum()),
            "lc_median": float(np.median(v)),
            "lc_mean": float(v.mean()),
        }
        for t in LC_THRESHOLDS:
            rej = v < t
            # Unweighted: the share of the atlas this threshold deletes.
            row[f"reject_at_{t}"] = float(rej.mean())
            # Weighted: the share of query-neighbour slots it empties, which is
            # what a reader actually experiences as degraded annotation.
            row[f"reject_weighted_at_{t}"] = float(w[rej].sum() / w.sum())
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_parquet(OUT / "01a_05_no_access_genuine_lc_summary.parquet", index=False)

    a = summary[summary.ref_celltype == "__all__"].iloc[0]
    print(f"\n[{time.time() - t0:.0f}s] genuine label agreement over "
          f"{int(a.n_records):,} records: median {a.lc_median:.4f}, "
          f"mean {a.lc_mean:.4f}", file=sys.stderr)
    for t in LC_THRESHOLDS:
        print(f"  threshold {t}: rejects {a[f'reject_at_{t}'] * 100:5.1f}% of "
              f"genuine records, {a[f'reject_weighted_at_{t}'] * 100:5.1f}% of "
              f"retrieval slots", file=sys.stderr)
    print(f"[{time.time() - t0:.0f}s] wrote "
          f"{OUT / '01a_05_no_access_genuine_lc.npz'} and two tables",
          file=sys.stderr)


if __name__ == "__main__":
    main()
