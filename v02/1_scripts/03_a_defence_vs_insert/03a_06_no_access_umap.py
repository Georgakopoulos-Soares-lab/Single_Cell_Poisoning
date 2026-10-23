"""Step 03a_06 — embedding-space coordinates for the defence triptych.

# No `from __future__ import annotations`: snakemake's script directive
# prepends a preamble, which would push it off line 1 and make it a SyntaxError.

An ANALYSIS step, not a figure, and the split is deliberate. A UMAP fit is a
computed quantity, so its coordinates are a RESULT and belong under
``2_results/`` where they can be inspected and re-read (CLAUDE.md 3 names
"coordinates" among what that tree holds). v01 wrote its atlas fit to
gitignored ``cache/`` instead -- ``cache/umap_full_reducer.pkl`` is 17 GB and
still sitting there -- and left two figures unreproducible from a clean clone.
That is the case CLAUDE.md 7's "a figure script must produce a figure" rule
exists to prevent, and this is the other half of it: a script that computes but
draws nothing is an analysis step and its output is a table.

TWO CONDITIONS, CHOSEN AS A CONTRAST, both read off ``03a_01``'s committed
matrix as the largest rescues in their attack family:

  CL:0002062 relabel dose 1.0 -> CL:0000160.  Attack success 85.28% undefended,
  0.00% under the stack. The attacker COPIES a genuine reference vector and
  writes a different label on it. The poison therefore lands exactly on top of
  the cells it imitates, and no amount of looking at the geometry separates
  them -- which is why ``recompute`` is worth precisely zero here and why the
  panel is the point: the attack is INVISIBLE to position.

  CL:0000583 flood dose 1.0 -> CL:0000097.  Attack success 90.46% undefended,
  0.00% under the stack. The attacker SYNTHESISES vectors around anchors. They
  are new points in space, so they are visible as a cloud, and ``recompute``
  removes all of them.

The pair is the argument. Same access level, same dose, same defence stack,
opposite mechanism -- and a defence that is total against one is worthless
against the other. Drawing only the flood would suggest poison is something you
can see.

PER-RECORD VERDICTS, not per-query rates. The triptych needs to know whether
each individual poisoned record survives a layer, which the rate tables cannot
say. Each is decided by the same rule the layer uses in ``03a_01``:
``recompute`` admits a record iff the attack left the vector verbatim;
``label_consistency`` admits it iff its score clears the budget threshold
derived from the GENUINE distribution. ``stack`` is the conjunction.

WHAT IS FITTED. The attacked celltype's own queries, the genuine reference
records those queries actually retrieved, and the retrieved poison -- subsampled
with a fixed seed to keep the fit tractable and the figure readable. Fitting
only what was RETRIEVED is the honest frame: a record no query reaches cannot
move a vote, and including the rest of a 7.9M-record atlas would bury the
neighbourhood the attack actually happens in.
"""
import pathlib
import sys
import time

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "00_common"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import umap  # noqa: E402

from attacks import build_attack  # noqa: E402
from celltype_sweep import Bench, seed_of  # noqa: E402
from dataset_config import cache_root, load  # noqa: E402
from modify import knn_of_vectors  # noqa: E402
from provenance import emit  # noqa: E402
from schemas import K_NEIGHBORS, POISON_EMB  # noqa: E402

BASE = _HERE.parents[2]
CFG = load(BASE)
DS = CFG["dataset"]
CACHE = cache_root(BASE, CFG)
TAB = BASE / "2_results" / DS / "03_a_defence_vs_insert"
OUT = TAB / "03a_06_no_access_umap.parquet"
IDX = BASE.parent / CFG["paths"]["reference_dir"] / "annotation" / "labelled_kNN.bin"

SEED = 0
#: Flood geometry, identical to 03a_01's. build_attack refuses to default these.
FLOOD_ANCHORS, FLOOD_SPREAD, FLOOD_JITTER = 16_384, 0.999, 1.0
#: Which attacks leave the vector verbatim -- the whole of what recompute tests.
VERBATIM = {"relabel": True, "duplicate": True, "interpolate": False,
            "flood": False}
#: The most generous budget 03a_01 measures. Chosen so the null cannot be
#: dismissed as an under-tuned defence.
LC_BUDGET = 0.10
#: Subsample caps. Large enough that the structure is not an artefact of
#: thinning, small enough that the fit is minutes rather than hours.
MAX_GENUINE, MAX_POISON, MAX_QUERY = 6000, 3000, 3000
#: UMAP, fixed seed. n_jobs=1 because umap-learn's parallel path is not
#: reproducible, which is the same defect CLAUDE.md 1 records for add_items.
UMAP_KW = dict(n_neighbors=30, min_dist=0.25, metric="cosine",
               random_state=SEED, n_jobs=1, verbose=False)
CONDITIONS = (("CL:0002062", "relabel", 1.0), ("CL:0000583", "flood", 1.0))


def take(rng: np.random.Generator, a: np.ndarray, n: int) -> np.ndarray:
    """A fixed-seed subsample of ``a``, or ``a`` itself when already small."""
    return a if len(a) <= n else rng.choice(a, size=n, replace=False)


def main() -> None:
    t0 = time.time()
    emit(__file__)
    b = Bench.load(CFG, BASE, t0)

    z = np.load(CACHE / "reference_knn_retrieved.npz")
    ref_knn = {int(r): nb for r, nb in zip(z["idx"], z["nbr"])}
    clean_rows = np.unique(b.clean_idx)
    miss = [r for r in clean_rows if int(r) not in ref_knn]
    if miss:
        sys.exit(f"{len(miss):,} retrieved rows have no cached kNN; run "
                 f"03a_01 with BIORAG_PREPARE=1 first.")
    g_lc = np.fromiter(
        (float((b.ref_label[ref_knn[int(r)]] == b.ref_label[int(r)]).mean())
         for r in clean_rows), float, len(clean_rows))
    thr = float(np.quantile(g_lc, LC_BUDGET))
    print(f"[{time.time()-t0:.0f}s] genuine LC over {len(g_lc):,} rows; "
          f"budget {LC_BUDGET:.0%} -> threshold {thr:.4f}", file=sys.stderr)

    frames: list[pd.DataFrame] = []
    for source, attack, rate in CONDITIONS:
        t1 = time.time()
        target, odist = b.target_of[source]
        b.clean_target(target)
        cond_seed = seed_of(source, rate, SEED)
        poison = build_attack(
            attack, ref_emb=b.ref_emb, ref_labels=b.ref_label, ref_ids=b.ref_id,
            source=source, target=target, rate=rate, seed=cond_seed,
            selection="hub", hub_nbr=b.clean_idx,
            **({"n_anchors": FLOOD_ANCHORS, "spread": FLOOD_SPREAD,
                "jitter": FLOOD_JITTER} if attack == "flood" else {}))
        pe = poison[POISON_EMB]
        ret = b.retrieve(pe, target)
        ids, is_p = ret.ids, ret.is_poison

        # Restrict to the ATTACKED celltype's own queries: the triptych is a
        # within-celltype story and the global neighbourhood would swamp it.
        qi = b.qi_of[source]
        ids_s, isp_s = ids[qi], is_p[qi]
        p_rows = np.unique(ids_s[isp_s] - b.n_ref) if isp_s.any() else np.empty(0, np.int64)
        g_rows = np.unique(ids_s[~isp_s])

        # Poison label consistency against the CLEAN reference -- the defender
        # screening a submission against the database as it stands.
        p_lc = np.ones(len(p_rows))
        if len(p_rows):
            nb = knn_of_vectors(str(IDX), pe[p_rows].astype(np.float32), b.n_ref,
                                b.ef, k=K_NEIGHBORS,
                                self_rows=np.full(len(p_rows), -1, np.int64))
            p_lc = (b.ref_label[nb] == target).mean(axis=1)

        rng = np.random.default_rng(cond_seed)
        q_take = take(rng, qi, MAX_QUERY)
        g_take = take(rng, g_rows, MAX_GENUINE)
        p_take_i = take(rng, np.arange(len(p_rows)), MAX_POISON)
        p_take = p_rows[p_take_i]

        X = np.concatenate([b.q[q_take], b.ref_emb[g_take], pe[p_take]]).astype(np.float32)
        print(f"[{time.time()-t1:.0f}s] {source} {attack}: {len(q_take):,} queries, "
              f"{len(g_take):,} genuine, {len(p_take):,} poison of "
              f"{len(p_rows):,} retrieved -> UMAP on {len(X):,}", file=sys.stderr)
        xy = umap.UMAP(**UMAP_KW).fit_transform(X)

        verbatim = VERBATIM[attack]
        lc_ok = p_lc[p_take_i] >= thr
        f = pd.DataFrame({
            "condition": f"{source} {attack}",
            "source": source, "target": target, "attack": attack, "rate": rate,
            "ontology_dist": odist, "condition_seed": cond_seed,
            "lc_budget": LC_BUDGET, "lc_threshold": thr,
            "role": (["query"] * len(q_take) + ["genuine"] * len(g_take)
                     + ["poison"] * len(p_take)),
            "label": np.concatenate([b.truth[q_take], b.ref_label[g_take],
                                     np.full(len(p_take), target)]),
            "x": xy[:, 0].astype(np.float32), "y": xy[:, 1].astype(np.float32),
            # Per-record verdicts. True = the layer ADMITS this record.
            "survives_recompute": np.concatenate([
                np.ones(len(q_take) + len(g_take), bool),
                np.full(len(p_take), verbatim)]),
            "survives_lc": np.concatenate([
                np.ones(len(q_take) + len(g_take), bool), lc_ok]),
            "poison_lc": np.concatenate([
                np.full(len(q_take) + len(g_take), np.nan), p_lc[p_take_i]]),
            "n_poison": len(pe), "n_poison_retrieved": len(p_rows),
        })
        f["survives_stack"] = f.survives_recompute & f.survives_lc
        frames.append(f)

    out = pd.concat(frames, ignore_index=True)
    out["threat_model"] = "no_access"
    out["access"] = "insert"
    out["retrieval_path"] = "hnsw"
    out["k"] = K_NEIGHBORS
    out["seed"] = SEED
    TAB.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"[{time.time()-t0:.0f}s] wrote {len(out):,} points to {OUT.name}",
          file=sys.stderr)
    for c, g in out.groupby("condition", observed=True):
        p = g[g.role == "poison"]
        print(f"  {c:28s} poison {len(p):,}  survives recompute "
              f"{int(p.survives_recompute.sum()):,}  lc "
              f"{int(p.survives_lc.sum()):,}  stack "
              f"{int(p.survives_stack.sum()):,}", file=sys.stderr)


if __name__ == "__main__":
    main()
