"""A DEEPER clean neighbour list, for the attacks that remove records.

``clean_neighbors.npz`` stores exactly ``K_NEIGHBORS`` per query, and stage 2's
docstring explains why that is sufficient: inserting poison can only push clean
records DOWN the ranking, so the true merged top-k is a subset of the clean
top-k. Overwriting an embedding breaks that. The old vector is gone, and the
record that ranked (k+1)th must be PROMOTED -- from a list that has no (k+1)th
entry.

So ``modify x synthesised`` (SPEC.md §5.4, ``build 3``) needs depth
:data:`modify.DEEP_K`. This produces it, exhaustively, once. Every condition of
both attacks reuses it, exactly as every insertion condition reuses stage 2.

**Loads the full 4.4 GB reference and runs an exhaustive search.** Above
CLAUDE.md §1's 2 GB bar, so it needs explicit approval before it runs.

Writes ``cache/<dataset>/clean_neighbors_deep.npz``: ``idx`` (n_q, DEEP_K)
int64 and ``cos`` (n_q, DEEP_K) float32, nearest first. Labels are NOT stored -- at depth
500 they would add ~100 MB of strings that ``ref_label[idx]`` reproduces for
free.
"""
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "00_common"))
import time  # noqa: E402

import numpy as np  # noqa: E402

from dataset_config import cache_root, load, query_cache  # noqa: E402
from modify import DEEP_K  # noqa: E402
from retrieve_clean import retrieve_exhaustive  # noqa: E402

BASE = _HERE.parents[2]
CFG = load(BASE)
CACHE = cache_root(BASE, CFG)      # shared: reference.npz, poison/
QCACHE = query_cache(BASE, CFG)    # this dataset only -- see dataset_config
OUT = QCACHE / "clean_neighbors_deep.npz"

#: Peak bytes the (block, n_ref) cosine matrix may occupy.
#:
#: Stage 2's default of 1000 queries per block assumes a subsampled reference.
#: Against the full 7,913,892 rows one block would be 31.6 GB -- half the
#: machine -- so the block is sized from the reference instead of fixed.
#: Arithmetically inert; it changes only how the same matmul is chunked.
BLOCK_BYTES = 4_000_000_000


def main() -> None:
    t0 = time.time()
    ref = np.load(CACHE / "reference.npz", allow_pickle=True)
    emb, lab = ref["emb"], ref["label"]
    q = np.load(QCACHE / "query_emb.npy")
    block = max(1, BLOCK_BYTES // (len(emb) * 4))
    print(f"[{time.time()-t0:.0f}s] {len(q):,} queries x {len(emb):,} reference "
          f"at depth {DEEP_K}; block {block} queries "
          f"({block * len(emb) * 4 / 1e9:.1f} GB per block)", file=sys.stderr)

    cos, _, idx = retrieve_exhaustive(q, emb, lab, DEEP_K, block_size=block)
    np.savez(OUT, idx=idx, cos=cos)
    print(f"[{time.time()-t0:.0f}s] wrote {idx.shape} to {OUT} "
          f"({OUT.stat().st_size / 1e6:.0f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
