"""Step 02b_08 — one map of the whole atlas, with every attack projected into it.

``02b_06`` fits a separate layout per attacked celltype, which answers "where is
the poison relative to the two celltypes involved". It cannot answer "how does
the reference CHANGE", because twenty-six local layouts share no frame: a point
at (3, 8) in one means nothing in another. This step fits ONE layout, on the
atlas itself, and projects every attack at every dose into it, so the panels form
a grid -- attack down, degree across -- in which the only thing that varies is
what the attacker did.

THE DOSE AXIS IS THE POINT
--------------------------
A single snapshot at one budget cannot show an attack "landing"; it shows where
it ended up. Four budgets of the same attack in one frame show the reference
filling up, and show it filling up in different SHAPES: relabel spreads over the
whole celltype, duplicate stacks onto a fraction of it, flood concentrates on its
anchors, interpolate drifts off the manifold. Those are four different answers to
"what does this budget buy", and none of them is visible at a single dose.

WHAT IS FITTED, AND WHY IT IS A SUBSAMPLE
-----------------------------------------
:data:`BACKDROP_N` records drawn PROPORTIONALLY across all 202 labels, with a
floor of one record per label so no celltype vanishes. Proportional rather than
equal-per-celltype because the figure should show the atlas as it is -- equalising
would invent a density the attacker does not face.

**The subsample is a choice, not a limit -- corrected 2026-09-24 after
measuring.** This docstring previously claimed the full atlas "is not reachable
on this machine, and that is measured rather than assumed", citing v01's
extrapolation of ~13 hours and ~215 GB from its 200k and 500k timings. It was
not measured here; it was inherited from other hardware, and both halves are
wrong on this one.

Probes at 2M and 4M, each in its own process so peak RSS is not conflated:

  2M   10.5 min,       15.2 GB
  4M   ~25 min CPU,    24.4 GB

Time exponent 1.25, memory 4.6 GB per million on a 6 GB base -- roughly **59 min
and 42 GB** for all 7,913,892, comfortable in 64 GB. A full-atlas fit IS
available and this step could take it.

**And measuring ``transform`` is why the subsample STAYS.** The fit happens
once; projections do not -- there is one per condition, thousands across a
campaign. Measured against three index sizes, warm (the first call in a process
pays numba compilation and was discarded):

  500k   transform of 4,000 points   2.37 s cpu
  1M     transform of 4,000 points   4.73 s cpu

Exponent **1.00** -- transform is LINEAR in index size, where the fit is 1.09.
At 7.9M that is ~37 cpu-seconds per projection:

  104 projections (what this step runs today)      1.1 h cpu
  5,000 (19 attack types x 11 doses x 26 celltypes)  51.7 h cpu

So a full-atlas fit costs 46 min once and then taxes every regeneration for the
life of the project, 8x over the 1M layout. The fit was never the expensive
part. Two caveats that cut against this conclusion and are stated rather than
buried: these are cpu-seconds summed across threads, so wall time per projection
is lower -- transform is not forced single-threaded the way the fit is -- and the
exponent rests on two warm points, not three.

**And the reading does not change, so the choice is tested.** The same 20 poison
sets -- 5 celltypes x 4 attacks at parity -- projected into a 200k fit and a 1M
fit, scored on the quantity the figures rest on: how far poison lands from the
nearest real cell, in units of each layout's OWN median nearest-neighbour
spacing, so two fits at different absolute scales are comparable.

  attack        200k    1M
  relabel       1.15   1.07
  duplicate     1.11   1.08
  flood         0.88   0.86
  interpolate   2.17   1.91

Spearman rank correlation across all 20 values **0.925**; 18 of 20 ratios fall
between 0.84 and 1.33 with no systematic drift. The claim the figures actually
make -- interpolate lands about twice as far from the manifold as the other
three, which sit at ~1 spacing -- holds identically in both.

Stated as the inference it is: a fivefold increase in backdrop moving nothing is
evidence the reading has converged, not proof that 7.9M would agree. The case
for extrapolating is that 200k -> 1M is the larger jump in neighbourhood-graph
density, and nothing moved.

So the subsample is a COST decision (see the transform numbers above), and one
that has been checked rather than assumed.

Matching v01's ``fig15a`` on all three knobs -- sample size, stratification, and
``n_neighbors``/``min_dist`` -- so the two versions' maps are comparable rather
than merely similar.

THE POISON IS BUILT THE WAY THE SWEEPS BUILT IT
-----------------------------------------------
Every set comes from ``build_attack`` against the FULL reference, seeded with the
same ``seed_of(source, rate, seed)`` the sweeps used, so at a dose the sweeps
also ran the poison is byte-identical to the poison that was scored. Flood takes
the anchor count that won for that celltype in ``02b_02``.

Only the stored COORDINATES are thinned, to :data:`POISON_N` per condition, and
thinning happens after construction -- so ``n_poison`` and ``atlas_share`` are
the real cost of the attack, not the size of the draw.

Reading the output: ``kind`` is ``atlas`` for a backdrop record, which carries its
``celltype``, or ``poison``, which carries ``source``, ``attack``, ``rate``,
``n_poison`` and ``atlas_share``. One frame, two populations.

Writes, under ``2_results/<dataset>/02_b_attack_celltype/``:
  ``02b_08_no_access_umap_global_xy.parquet``

Cost: one fit on BACKDROP_N records, then one projection per
(source, attack, dose). The shipped HNSW index is NOT loaded -- no retrieval
happens here and nothing is scored; ``reference.npz`` peaks near 5 GB.
"""
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "00_common"))
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import umap  # noqa: E402

from attacks import build_attack  # noqa: E402
from celltype_sweep import FLOOD_JITTER, FLOOD_SPREAD, seed_of  # noqa: E402
from dataset_config import cache_root, load  # noqa: E402
from provenance import emit  # noqa: E402
from schemas import (ATTACK_TIER, POISON_EMB, THREAT_MODEL,  # noqa: E402
                     Attack, ThreatModel)

BASE = _HERE.parents[2]
CFG = load(BASE)
DS = CFG["dataset"]
CACHE = cache_root(BASE, CFG)       # shared: reference.npz, poison/
OUT = BASE / "2_results" / DS / "02_b_attack_celltype"
SWEEP = OUT / "02b_01_no_access_celltype_sweep_graph.parquet"
FLOOD_TABLE = OUT / "02b_02_no_access_flood_anchors.parquet"

#: Records drawn from the reference to fit the layout on. v01's fig15a used
#: 1,000,000 and the figures are read side by side, so this matches it. Expect
#: ~23 min and ~27 GB by the scaling above.
BACKDROP_N = 1_000_000

#: Coordinates stored for the LARGEST dose; every smaller dose is thinned at the
#: SAME FRACTION, so the stored point count scales with the budget.
#:
#: A constant cap per condition would be wrong here and wrong in a way that
#: destroys the figure: at a fixed 4,000 points, 18,951 records and 94,756
#: records draw identically, and the dose axis -- the entire point of this step --
#: becomes invisible. One fraction across the row keeps the picture proportional
#: to what was actually added.
#:
#: Counts are never taken from this table's row counts; ``n_poison`` and
#: ``atlas_share`` carry the real sizes on every row.
POISON_N = 5000

DRAW_SEED = 0
ATTACKS: tuple[str, ...] = (Attack.RELABEL, Attack.DUPLICATE,
                            Attack.INTERPOLATE, Attack.FLOOD)

#: The DOSE axis, which is the point of this step: the same attack at four
#: budgets, so the figure shows the reference filling up rather than a single
#: snapshot. Chosen to span 02b_01's cliff -- 0.25 is well below it, 1.0 is
#: parity where relabel jumps from 0.02% to 34%, 1.25 is past it.
#:
#: These are CONSTRUCTION rates, not measured conditions. 02b_02 swept flood at
#: 0.5/0.8/0.9/1.0 only, so flood at 0.25 and 1.25 here is built and drawn but
#: was never scored; nothing in this table or its figure reports success, so the
#: grid stays square without implying a measurement that does not exist.
RATES: tuple[float, ...] = (0.25, 0.5, 1.0, 1.25)

#: Which celltypes get the dose grid. () means every source in 02b_01 -- all 26
#: attackable celltypes, each with its own 4 attacks x 4 doses grid, all
#: projected into the SAME reference layout so the grids are comparable across
#: celltypes as well as within one.
SOURCES: tuple[str, ...] = ()

#: The dose at which 02b_02's winning anchor count is read. Flood's anchor count
#: is held FIXED across the dose column on purpose: varying two knobs at once
#: would make the row unreadable as a dose series, since a change between panels
#: could be either one. Parity is where the anchor sweep's optimum was measured.
ANCHOR_RATE = 1.0

SEED = 0

#: v01's fig15a settings. A smaller min_dist packs clusters into thin strings
#: and makes the atlas look wispy; 0.30 gives the solid continents the v01
#: figures show, which is what a reader compares against.
UMAP_SEED = 0
N_NEIGHBORS = 15
MIN_DIST = 0.30


def stratified_sample(lab: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Indices of about ``n`` records, allocated across labels by size.

    Proportional rather than equal-per-celltype: the backdrop should show the
    atlas as it is, and equalising would invent a density the attacker does not
    face. The floor of one keeps rare celltypes on the map at all.

    Args:
        lab: (n_ref,) str reference labels. n: target count. seed: RNG seed.

    Returns:
        (m,) int64 indices, m approximately n.
    """
    rng = np.random.default_rng(seed)
    terms, counts = np.unique(lab, return_counts=True)
    take = np.maximum(1, (counts / counts.sum() * n).astype(int))
    return np.concatenate([rng.choice(np.flatnonzero(lab == t),
                                      min(k, int(c)), replace=False)
                           for t, k, c in zip(terms, take, counts, strict=True)])


def fit_layout(emb: np.ndarray) -> umap.UMAP:
    """Fit the one global layout.

    ``init="pca"`` because the spectral initialiser fails on this data -- the
    celltypes are separated enough in cosine space that the neighbour graph comes
    apart and the eigensolver reports too small an eigengap, after which UMAP
    falls back to a random start that loses global structure. See ``02b_06``.

    Args:
        emb: (n, 128) float32, unit norm.

    Returns:
        A fitted reducer; ``transform`` projects new vectors into the same frame.
    """
    return umap.UMAP(n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, n_components=2,
                     metric="cosine", init="pca",
                     random_state=UMAP_SEED).fit(emb)


def best_anchor_count(source: str, flood: pd.DataFrame) -> int:
    """The anchor count that won for this celltype in ``02b_02``, at ANCHOR_RATE.

    Args:
        source: the attacked term. flood: ``02b_02``'s table.

    Returns:
        Anchor count, as swept.
    """
    f = flood[(flood.source == source)
              & (np.abs(flood.rate - ANCHOR_RATE) < 1e-3)]
    return int(f.loc[f.within_net_attack_success.idxmax(), "n_anchors"])


def main() -> None:
    t0 = time.time()
    emit(__file__)          # which code ran, HEAD or dirty tree
    pairs = (pd.read_parquet(SWEEP, columns=["source", "target"])
             .drop_duplicates().set_index("source")["target"].to_dict())
    flood = pd.read_parquet(FLOOD_TABLE)
    sources = SOURCES or tuple(sorted(pairs))
    missing = [s for s in sources if s not in pairs]
    if missing:
        raise ValueError(f"not swept in 02b_01, nothing to map: {missing}")

    z = np.load(CACHE / "reference.npz", allow_pickle=True)
    ref_emb = z["emb"]
    ref_lab = z["label"]
    ref_id = z["cell_id"]
    rng = np.random.default_rng(DRAW_SEED)
    back = np.sort(stratified_sample(ref_lab, BACKDROP_N, DRAW_SEED))
    print(f"[{time.time()-t0:.0f}s] {len(ref_lab):,} records, "
          f"{len(np.unique(ref_lab))} labels; backdrop {BACKDROP_N:,}",
          file=sys.stderr)

    reducer = fit_layout(ref_emb[back])
    xy = reducer.embedding_
    print(f"[{time.time()-t0:.0f}s] global layout fitted", file=sys.stderr)

    rows = [{THREAT_MODEL: str(ThreatModel.NO_ACCESS), "kind": "atlas",
             "celltype": str(ref_lab[j]), "source": None, "attack": None,
             "attack_tier": None, "rate": None, "n_poison": None,
             "atlas_share": None, "n_anchors": None,
             "cell_id": str(ref_id[j]),
             "x": float(xy[i, 0]), "y": float(xy[i, 1])}
            for i, j in enumerate(back)]

    n_ref = len(ref_lab)
    for source in sources:
        target = pairs[source]
        n_src = int((ref_lab == source).sum())
        n_anchors = best_anchor_count(source, flood)
        for attack in ATTACKS:
            # One fraction for the whole row, set by the largest dose.
            frac = min(1.0, POISON_N / (max(RATES) * n_src))
            for rate in RATES:
                cond_seed = seed_of(source, rate, SEED)
                kw = ({"n_anchors": n_anchors, "spread": FLOOD_SPREAD,
                       "jitter": FLOOD_JITTER} if attack == Attack.FLOOD else {})
                poison = build_attack(attack, ref_emb=ref_emb,
                                      ref_labels=ref_lab, ref_ids=ref_id,
                                      source=source, target=target, rate=rate,
                                      seed=cond_seed, **kw)
                p_emb = poison[POISON_EMB]
                n_p = len(p_emb)
                n_keep = max(1, int(round(frac * n_p)))
                keep = (np.arange(n_p) if n_keep >= n_p
                        else np.sort(rng.choice(n_p, n_keep, replace=False)))
                p_xy = reducer.transform(p_emb[keep])
                # n_poison and atlas_share describe the FULL set, never the
                # thinned draw: they are what the attack costs, and a figure
                # title quoting the draw size would understate it by up to 90x.
                rows += [{THREAT_MODEL: str(ThreatModel.NO_ACCESS),
                          "kind": "poison", "celltype": None, "source": source,
                          "attack": str(attack),
                          "attack_tier": str(ATTACK_TIER[attack]),
                          "rate": float(rate), "n_poison": n_p,
                          "atlas_share": n_p / n_ref,
                          "n_anchors": n_anchors if attack == Attack.FLOOD else None,
                          "cell_id": None,
                          "x": float(p_xy[i, 0]), "y": float(p_xy[i, 1])}
                         for i in range(len(p_xy))]
                del poison, p_emb
            print(f"[{time.time()-t0:.0f}s] {source} {attack}: "
                  f"{len(RATES)} doses projected at 1:{1/frac:.0f}",
                  file=sys.stderr)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(
        OUT / "02b_08_no_access_umap_global_xy.parquet", index=False)
    print(f"[{time.time()-t0:.0f}s] {len(rows):,} points written",
          file=sys.stderr)


if __name__ == "__main__":
    main()
