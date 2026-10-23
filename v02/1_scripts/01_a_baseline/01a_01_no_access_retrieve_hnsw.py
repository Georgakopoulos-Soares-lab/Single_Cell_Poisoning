"""Step 01a_01 — what SCimilarity's OWN graph returns for each query, unpoisoned.

v01 touched the shipped index exactly once, in ``prepare_reference``, to pull the
vectors out of it. Every number after that came from exact brute force. That is
defensible -- exact search is ground truth, and it is what makes the merge
shortcut provable -- but it means v01 never measured what the deployed system
actually returns.

Stage 6 does query a real graph, but its rates start at 0.01: **there is no
unpoisoned condition in it**. So the gap between exact retrieval and SCimilarity's
approximate retrieval has been measured under attack and never on clean data,
even though every accuracy number in the project rests on the clean case.

This runs SCimilarity's retrieval exactly as ``CellAnnotation.get_predictions_knn``
does it:

* the shipped ``labelled_kNN.bin``, **loaded, never rebuilt**. Building a
  7.9M-node graph at M=80 takes hours and would produce a DIFFERENT graph from
  the one SCimilarity ships, which would void the claim that this is their
  reference.
* ``ef`` from config, defaulting to 100, which is that method's own default. No
  floor is applied on top of it -- ``cell_search_knn.py`` is a bare
  ``self.knn.set_ef(ef)``, and v01's ``max(ef, 4 * k)`` silently forced 200.
* ``k = K_NEIGHBORS``, and hnswlib's cosine space returns DISTANCE, converted
  here to similarity so it is comparable with the exact path.

**THREAT MODEL: none.** This is clean retrieval; no attacker exists. The
``threat_model`` column is still written, because a table without it cannot be
attributed later and v01 lost that for 35 of 39 tables.

**Loads an 8.8 GB index — requires explicit human approval (CLAUDE.md §1).**

Writes, under ``2_results/<dataset>/01_a_baseline/``:
  ``01a_01_no_access_hnsw_neighbors.npz``  idx/cos/label, (n_q, k) each
  ``01a_01_no_access_hnsw_summary.parquet``  one row, run provenance and shape
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

#: SCimilarity's own default for the annotation path. Overridable from the full
#: profile, which documents why it must not be raised.
DEFAULT_EF = 100


def load_shipped_index(path: pathlib.Path, n_expected: int, ef: int):
    """Open the shipped graph read-only and configure it like SCimilarity does.

    Args:
        path: ``labelled_kNN.bin`` as shipped. n_expected: reference rows, so a
            mismatched index fails here rather than producing ids that do not
            line up with ``reference.npz``. ef: query-time candidate list size.

    Returns:
        A loaded ``hnswlib.Index``. Never built: ``init_index`` is not called.
    """
    import hnswlib

    if ef < K_NEIGHBORS:
        raise ValueError(f"ef {ef} < k {K_NEIGHBORS}: hnswlib requires ef >= k")
    index = hnswlib.Index(space="cosine", dim=EMB_DIM)
    index.load_index(str(path))
    n = index.get_current_count()
    if n != n_expected:
        raise ValueError(
            f"index holds {n:,} vectors but reference.npz has {n_expected:,}; "
            "ids would not correspond and every label would be wrong"
        )
    index.set_ef(ef)
    return index


def main() -> None:
    t0 = time.time()
    emit(__file__)          # which code ran, HEAD or dirty tree
    ef = int(CFG.get("hnsw", {}).get("ef_search", DEFAULT_EF))

    ref_labels = np.load(CACHE / "reference.npz", allow_pickle=True)["label"]
    q = np.load(QCACHE / "query_emb.npy").astype(EMB_DTYPE)
    meta = pd.read_parquet(QCACHE / "query_meta.parquet")
    if len(q) != len(meta):
        raise ValueError(f"{len(q):,} embeddings but {len(meta):,} metadata rows")

    idx_path = BASE.parent / CFG["paths"]["reference_dir"] / "annotation" / "labelled_kNN.bin"
    print(f"loading {idx_path} ({idx_path.stat().st_size / 1e9:.2f} GB) …", file=sys.stderr)
    index = load_shipped_index(idx_path, len(ref_labels), ef)
    print(f"[{time.time() - t0:.0f}s] loaded; querying {len(q):,} cells at "
          f"ef={ef}, k={K_NEIGHBORS}", file=sys.stderr)

    ids, dist = index.knn_query(q, k=K_NEIGHBORS)
    ids = ids.astype(np.int64)
    # hnswlib's "cosine" space returns 1 - cos; the exact path stores similarity.
    cos = (1.0 - dist).astype(EMB_DTYPE)
    lab = ref_labels[ids]

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "01a_01_no_access_hnsw_neighbors.npz",
                        idx=ids, cos=cos, label=lab)

    pd.DataFrame([{
        THREAT_MODEL: ThreatModel.NO_ACCESS,   # clean; the arm this feeds
        "retrieval": "hnsw",
        "ef_search": ef,
        "k": K_NEIGHBORS,
        "n_queries": len(q),
        "n_reference": len(ref_labels),
        "index_file": idx_path.name,
        "mean_cos_top1": float(cos[:, 0].mean()),
        "mean_cos_top50": float(cos[:, -1].mean()),
        "seconds": round(time.time() - t0, 1),
    }]).to_parquet(OUT / "01a_01_no_access_hnsw_summary.parquet", index=False)

    print(f"[{time.time() - t0:.0f}s] wrote {OUT / '01a_01_no_access_hnsw_neighbors.npz'} "
          f"{ids.shape}", file=sys.stderr)


if __name__ == "__main__":
    main()
