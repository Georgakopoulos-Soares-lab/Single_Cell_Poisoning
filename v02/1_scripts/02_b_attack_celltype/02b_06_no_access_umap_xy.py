"""Step 02b_06 — where the poison actually lands, in two dimensions.

Every number in ``02b_01`` and ``02b_02`` is a rate. A rate says how often the
attack won, never WHY, and the four no-access attacks fail and succeed for
different geometric reasons: ``interpolate`` aims away from the queries it must
fool, ``duplicate`` piles onto a handful of points, ``flood`` spreads across the
source manifold. Those are claims about position, and a position is what a map
shows.

**This is an analysis step, not a figure step.** A UMAP fit is coordinates, and
coordinates are a quantitative result -- so it writes a table and draws nothing.
v01 made the other choice: ``fig15a`` computed a fit, drew no image, and wrote
its output to gitignored ``cache/``, which left two figures unreproducible from a
clean clone. The drawing is ``02b_07``, which reads this table.

ONE LAYOUT PER CELLTYPE, SHARED BY EVERY PANEL
----------------------------------------------
The fit is computed ONCE per source celltype, on the genuine records of the
source and its target, and every poison set is then projected into that fixed
layout with ``transform``. Fitting each attack separately would give four
incomparable pictures -- UMAP's axes carry no meaning across fits, so a poison
cloud that moved between panels might only mean the layout rotated.

**The target's genuine cells are in the frame on purpose.** ``interpolate``
moves its vectors toward the target centroid; with only the source plotted, that
movement leaves the frame and reads as "the poison vanished" rather than "the
poison went somewhere useless". The two clouds together are what make the
comparison legible.

**Cosine, not Euclidean.** The reference is unit-normalised onto a hypersphere
and every distance in this project is cosine (``CLAUDE.md`` 1), so the embedding
uses ``metric="cosine"``. Fitting under Euclidean would draw a neighbourhood
structure the retrieval never sees.

THE POISON IS THE MEASURED POISON
---------------------------------
Nothing here constructs a fresh attack. Each poison set is rebuilt through
``build_attack`` at the same rate, with the same ``seed_of`` condition seed, that
``02b_01`` and ``02b_02`` scored -- so the picture is of the run, not of a
re-enactment of it. ``flood`` is drawn at the anchor count that was BEST for that
celltype in ``02b_02``, because a flood at an arbitrary anchor count is a
different attack (``02b_02`` found the optimum spread across the whole range).

The (source, target) pairs are read from ``02b_01``'s committed table rather than
re-derived from the ontology. Re-deriving would reproduce them, but reading them
makes it impossible for the map and the measurement to drift apart.

RATE 1.0, THE ATTACKER'S OPTIMUM UNDER INSERTION
------------------------------------------------
One poison record per genuine record. It is also where ``02b_01``'s cliff has
already been climbed, so the panels show the attacks at their strongest rather
than at a dose where three of the four do nothing.

Writes, under ``2_results/<dataset>/02_b_attack_celltype/``:
  ``02b_06_no_access_umap_xy.parquet``     one row per plotted point
  ``02b_06_no_access_umap_counts.parquet`` one row per (source, attack): how many
                                           records the attack adds to the atlas

**Initialised from PCA, not spectrally, and the reason is a finding.** UMAP's
default spectral initialiser needs a connected k-nearest-neighbour graph. Here
it fails -- the source and its target are separated enough in cosine space that
the graph comes apart and the eigensolver reports too small an eigengap -- and
UMAP falls back to a RANDOM start, which recovers local neighbourhoods but not
global structure, so the two celltypes render as speckle rather than as two
clouds. ``init="pca"`` sidesteps the failing solver and is deterministic. The
layout was already reproducible under the fallback, because ``random_state`` is
set; what changes is that the picture now shows the separation instead of
scattering it.

That the spectral solver fails at all is weak evidence FOR the separation the
map is drawn to show, so it is recorded here rather than suppressed.

Cost: one UMAP fit per celltype over the source and target pools. ~1 GB for the
fit; loading ``reference.npz`` peaks near 5 GB, and the shipped HNSW index is
NOT loaded -- no retrieval happens here.
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

#: Every celltype 02b_01 swept. Reading them from the table rather than listing
#: them keeps this step in step with the measurement: a celltype that enters or
#: leaves the sweep enters or leaves the map, with no list to forget to update.
SOURCES: tuple[str, ...] = ()   # () means "every source in 02b_01"

#: Points per cloud used to FIT the layout, and points per cloud STORED.
#:
#: The pools run from 10,438 to 368,537 records, and a deterministic UMAP is
#: single-threaded, so fitting the largest celltype against its target would take
#: longer than the other twenty-five together and write a table nothing can open.
#: Both caps are DRAWING decisions on a seeded draw, and neither touches a
#: reported quantity: n_poison, n_distinct and n_after in the counts table are
#: always computed on the FULL poison set before any subsampling, and the
#: per-celltype fit sizes are recorded beside them so a reader can see what the
#: layout was built from.
FIT_MAX = 20000
STORE_MAX = 20000
DRAW_SEED = 0

ATTACKS: tuple[str, ...] = (Attack.RELABEL, Attack.DUPLICATE,
                            Attack.INTERPOLATE, Attack.FLOOD)

RATE = 1.0
SEED = 0

#: UMAP's own reproducibility: with a random_state set, the implementation drops
#: to a single thread and the layout is deterministic. Slower, and the whole
#: point -- an undeterministic figure cannot be checked against itself.
UMAP_SEED = 0
N_NEIGHBORS = 30
MIN_DIST = 0.1


def load_pools(labels: np.ndarray, terms: set[str]) -> dict[str, np.ndarray]:
    """Row indices of every reference record carrying each term.

    Args:
        labels: (n_ref,) str, one Cell Ontology term per reference record.
        terms: the terms to locate.

    Returns:
        ``{term: (n_term,) int64 indices}``.
    """
    return {t: np.flatnonzero(labels == t) for t in sorted(terms)}


def fit_layout(emb: np.ndarray) -> umap.UMAP:
    """Fit one UMAP layout on unit-norm embeddings.

    Args:
        emb: (n, 128) float32, unit norm.

    Returns:
        A fitted reducer whose ``transform`` projects new vectors into the same
        two-dimensional frame.
    """
    return umap.UMAP(n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, n_components=2,
                     metric="cosine", init="pca",
                     random_state=UMAP_SEED).fit(emb)


def best_anchor_count(source: str) -> int:
    """The anchor count that won for this celltype in ``02b_02``, at RATE.

    A flood at an arbitrary anchor count is a different attack, so the map shows
    the configuration the measurement found best rather than a round number.

    Args:
        source: the attacked Cell Ontology term.

    Returns:
        Anchor count, as swept.
    """
    f = pd.read_parquet(FLOOD_TABLE)
    f = f[(f.source == source) & (np.abs(f.rate - RATE) < 1e-3)]
    return int(f.loc[f.within_net_attack_success.idxmax(), "n_anchors"])


def main() -> None:
    t0 = time.time()
    emit(__file__)          # which code ran, HEAD or dirty tree
    pairs = (pd.read_parquet(SWEEP, columns=["source", "target"])
             .drop_duplicates().set_index("source")["target"].to_dict())
    # Resolved BEFORE the pools are built, because the pools are built FROM it.
    sources = SOURCES or tuple(sorted(pairs))
    missing = [s for s in sources if s not in pairs]
    if missing:
        raise ValueError(f"not swept in 02b_01, so there is nothing to map: {missing}")

    # The FULL reference is held, not a two-celltype slice, and every poison set
    # is built against it. build_attack draws with rng.choice over the source
    # pool, so a slice would reproduce the sweep's selection only while the
    # pool's internal ORDER survived slicing -- true today, silently false the
    # first time anything reorders. The map would then show a different poison
    # set from the one that was measured, and nothing would say so. ~4.7 GB.
    z = np.load(CACHE / "reference.npz", allow_pickle=True)
    ref_emb = z["emb"]
    ref_lab = z["label"]
    ref_id = z["cell_id"]
    wanted = set(sources) | {pairs[s] for s in sources}
    pools = load_pools(ref_lab, wanted)
    print(f"[{time.time()-t0:.0f}s] reference held: {len(ref_lab):,} records; "
          f"pools " + ", ".join(f"{t} {len(v):,}" for t, v in pools.items()),
          file=sys.stderr)

    rng = np.random.default_rng(DRAW_SEED)

    def thin(idx: np.ndarray, cap: int) -> np.ndarray:
        """At most ``cap`` of ``idx``, drawn once from the seeded generator.

        Args:
            idx: (n,) int indices into the reference.
            cap: maximum to keep.

        Returns:
            (min(n, cap),) int indices, sorted so row order stays stable.
        """
        if len(idx) <= cap:
            return idx
        return np.sort(rng.choice(idx, size=cap, replace=False))

    rows: list[dict] = []
    counts: list[dict] = []
    for source in sources:
        target = pairs[source]
        fit_src = thin(pools[source], FIT_MAX)
        fit_tgt = thin(pools[target], FIT_MAX)
        sel = np.concatenate([fit_src, fit_tgt])
        fit_emb = ref_emb[sel]
        reducer = fit_layout(fit_emb)
        xy = reducer.embedding_
        print(f"[{time.time()-t0:.0f}s] {source}: layout fitted on {len(sel):,}",
              file=sys.stderr)

        for i, j in enumerate(sel):
            rows.append({
                THREAT_MODEL: str(ThreatModel.NO_ACCESS),
                "source": source, "target": target, "attack": "none",
                "attack_tier": None, "rate": RATE, "seed": SEED,
                "condition_seed": None, "n_anchors": None,
                "kind": "genuine_source" if ref_lab[j] == source
                        else "genuine_target",
                "cell_id": str(ref_id[j]),
                "x": float(xy[i, 0]), "y": float(xy[i, 1]),
            })
        # The FULL pool size, not the thinned one: this is the denominator every
        # rate in 02b_01 is taken over, and it must not follow a drawing cap.
        n_src = int(len(pools[source]))
        counts.append({THREAT_MODEL: str(ThreatModel.NO_ACCESS),
                       "source": source, "target": target, "attack": "none",
                       "attack_tier": None, "rate": RATE,
                       "n_genuine": n_src, "n_poison": 0, "n_distinct": 0,
                       "n_after": n_src, "n_anchors": None,
                       "n_fit_source": int(len(fit_src)),
                       "n_fit_target": int(len(fit_tgt))})

        for attack in ATTACKS:
            cond_seed = seed_of(source, RATE, SEED)
            n_anchors = best_anchor_count(source) if attack == Attack.FLOOD else None
            kw = ({"n_anchors": n_anchors, "spread": FLOOD_SPREAD,
                   "jitter": FLOOD_JITTER} if attack == Attack.FLOOD else {})
            poison = build_attack(attack, ref_emb=ref_emb, ref_labels=ref_lab,
                                  ref_ids=ref_id, source=source, target=target,
                                  rate=RATE, seed=cond_seed, **kw)
            p_emb = poison[POISON_EMB]
            n_distinct = len(np.unique(p_emb, axis=0))
            n_poison = len(p_emb)
            keep = (np.arange(n_poison) if n_poison <= STORE_MAX
                    else np.sort(rng.choice(n_poison, STORE_MAX, replace=False)))
            p_emb = p_emb[keep]
            p_xy = reducer.transform(p_emb)
            for i in range(len(p_emb)):
                rows.append({
                    THREAT_MODEL: str(ThreatModel.NO_ACCESS),
                    "source": source, "target": target, "attack": str(attack),
                    "attack_tier": str(ATTACK_TIER[attack]), "rate": RATE,
                    "seed": SEED, "condition_seed": cond_seed,
                    "n_anchors": n_anchors, "kind": "poison",
                    "cell_id": None,
                    "x": float(p_xy[i, 0]), "y": float(p_xy[i, 1]),
                })
            # DISTINCT vectors, not records. relabel, duplicate and flood all
            # sit on the source cloud, so a scatter cannot tell them apart; what
            # separates them is how many different points the budget buys.
            # duplicate draws from 1% of the celltype, and flood at jitter 1.0
            # applies no per-copy noise, so its distinct count is its anchor
            # count however large the budget. Counted here rather than in the
            # figure, because it is a statistic and figures do not compute.
            counts.append({THREAT_MODEL: str(ThreatModel.NO_ACCESS),
                           "source": source, "target": target,
                           "attack": str(attack),
                           "attack_tier": str(ATTACK_TIER[attack]), "rate": RATE,
                           "n_genuine": n_src, "n_poison": n_poison,
                           "n_distinct": n_distinct,
                           "n_after": n_src + n_poison, "n_anchors": n_anchors,
                           "n_fit_source": int(len(fit_src)),
                           "n_fit_target": int(len(fit_tgt))})
            print(f"[{time.time()-t0:.0f}s] {source} {attack}: {n_poison:,} "
                  f"records, {n_distinct:,} distinct, {len(p_emb):,} drawn",
                  file=sys.stderr)
            del poison, p_emb

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUT / "02b_06_no_access_umap_xy.parquet",
                                  index=False)
    pd.DataFrame(counts).to_parquet(
        OUT / "02b_06_no_access_umap_counts.parquet", index=False)
    print(f"[{time.time()-t0:.0f}s] {len(rows):,} points, {len(counts)} count rows",
          file=sys.stderr)


if __name__ == "__main__":
    main()
