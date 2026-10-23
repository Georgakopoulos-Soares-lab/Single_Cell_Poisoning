"""Fit UMAP on the whole annotation index. Produces the layout every map reads.

Written into the repository after being run from a scratchpad, which was a
mistake: it produces cache/umap_full_xy.npy and cache/umap_full_reducer.pkl,
and fig17 depends on both. A producer of a committed figure's input cannot live
in a temp directory.

MEASURED SCALING on this machine, so nobody re-derives it:

    200,000     84.8 s     6.9 GB
    500,000    411.9 s    13.7 GB
  1,000,000    977   s    19.7 GB
  7,913,892  5,635   s    25 GB steady    <- this run

Time exponent 1.25, memory exponent 0.52 -- memory grows SUBLINEARLY. An earlier
note in this repo assumed linear and declared the full atlas infeasible at
215 GB; it is not, and the actual run settled around 25 GB. The 94-minute
runtime also beat a 3.6 h projection, so the exponent fitted from smaller runs
overestimates at this scale.

Two things to know before re-running:

* ``random_state`` forces single-threaded execution. Reproducibility costs the
  parallelism, and that is the right trade here -- the layout is a published
  figure's coordinate system.
* Spectral initialisation FAILS on this graph and UMAP falls back to random
  init. The layout is still usable but is not comparable to a smaller fit that
  initialised spectrally, so do not overlay one on the other.

The reducer pickles its whole k-NN graph and is 17 GB. Load it only to
``transform`` new points; a cached projection is cheaper for anything else.
"""

import pathlib
import pickle
import sys
import time

BASE = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "1_scripts" / "00_common"))

import numpy as np  # noqa: E402
import umap  # noqa: E402

from dataset_config import cache_root, load  # noqa: E402

#: The fit is on REFERENCE embeddings alone, so the layout is shared by
#: every query dataset and stays at the cache root.
CACHE = cache_root(BASE, load(BASE))
OUT_XY = CACHE / "umap_full_xy.npy"
OUT_RED = CACHE / "umap_full_reducer.pkl"
UMAP_KW = dict(n_neighbors=15, min_dist=0.30, metric="cosine",
               random_state=0, verbose=True)


def main() -> None:
    """Fit on all reference embeddings and persist layout plus reducer."""
    t0 = time.time()
    X = np.load(CACHE / "reference.npz")["emb"].astype(np.float32)
    print(f"[{time.time()-t0:.0f}s] fitting on all {len(X):,} reference cells "
          f"({X.nbytes/1e9:.1f} GB input)", flush=True)
    r = umap.UMAP(**UMAP_KW).fit(X)
    np.save(OUT_XY, r.embedding_)
    with open(OUT_RED, "wb") as fh:
        pickle.dump(r, fh)
    print(f"[{time.time()-t0:.0f}s] wrote {OUT_XY.name} and {OUT_RED.name}",
          flush=True)


if __name__ == "__main__":
    main()
