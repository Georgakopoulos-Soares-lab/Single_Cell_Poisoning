"""Step 02b_15 — how far should a shifted record actually move?

``02b_11`` measured ``modify x shifted`` at alpha 0.5 as exactly inert: net
attack success 0.0000 at all thirteen doses, all four variants. Its retrieval
column explains why -- 0.0006. The poison is never retrieved, so it cannot win
a vote it never enters.

And the reason is arithmetic, not a property of the attack. The genuine top-50
neighbourhood spans ``1 - cos = 0.0195`` (measured in ``02b_13``); at alpha 0.5
a shifted record moves ``1 - cos = 0.1514``, **776% of that radius**. v01 ran
only 0.5, so the question "is shifted a weak attack?" has never been asked.
What both versions measured is that a record thrown eight radii away is not
retrieved.

This asks it. Alpha is the axis; SPEC.md §5.4d holds the ladder and the
displacement each rung produces. Six of the nine points sit inside or at the
neighbourhood edge -- the region neither version has sampled -- and the three
above 0.5 document the collapse rather than assuming it.

**Dose is fixed at 0.25** so alpha is the only thing varying, and because the
remerge cost is linear in records overwritten: at parity this grid would cost
four times as much to answer a question about geometry, not budget. 0.25 is
also the dose at which the insert and modify arms historically overlap.

**What the answer will look like.** Alpha trades reach against displacement the
way jitter trades distinctness against retrieval. A record that has barely moved
is retrieved, but it is then a genuine vector carrying a false label -- which is
``unmoved`` geometry, already measured by ``02b_05``. If low alpha simply
reproduces ``unmoved``, ``shifted`` has no regime of its own, and that is the
finding. If a middle alpha beats both, that is a more interesting one. The curve
is the result; no single rung is.

Reads ``cache/<dataset>/clean_neighbors_deep.npz`` and scores by removal-and-merge, exactly
as ``02b_11`` does. No index load.

Writes ``02b_15_with_access_alpha_sweep.parquet`` and its ``_by_donor``
companion, sharing ``02b_11``'s schema so the two concatenate.
"""
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "00_common"))
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from celltype_sweep import (by_donor, eligible_sources,  # noqa: E402
                            pick_targets, seed_of, write, write_by_donor)
from dataset_config import cache_root, load, query_cache  # noqa: E402
from modify import (DEEP_K, build_modify, clean_graph_topk,  # noqa: E402
                    graph_topk, remerge)
from ontology import Ontology  # noqa: E402
from provenance import emit  # noqa: E402
from schemas import (ACCESS, GEOMETRY, K_NEIGHBORS, LABEL_CHANGE,  # noqa: E402
                     POISON_EMB, POISON_LABEL, SELECTION, THREAT_MODEL,
                     Geometry, LabelChange, Selection, ThreatModel)
from vote import TIE_BREAK, vote  # noqa: E402

BASE = _HERE.parents[2]
CFG = load(BASE)
DS = CFG["dataset"]
CACHE = cache_root(BASE, CFG)      # shared: reference.npz, poison/
QCACHE = query_cache(BASE, CFG)    # this dataset only -- see dataset_config
TAB = BASE / "2_results" / DS / "02_b_attack_celltype"
OUT = TAB / "02b_15_with_access_alpha_sweep.parquet"
OUT_BY_DONOR = TAB / "02b_15_with_access_alpha_sweep_by_donor.parquet"

#: Which retrieval path to score on. ``exact`` (default) uses
#: ``modify.remerge`` over the depth-500 exhaustive list; ``graph`` writes the
#: replacements into the shipped ``labelled_kNN.bin`` with ``add_items`` on
#: existing ids, which updates each point AND rewires its links.
#:
#: They are not interchangeable. Four validated conditions: ``CL:0000235``
#: agreed to 0.0001 at alpha 0.02 and 0.2, ``CL:0000583`` differed by +0.1113
#: and +0.1240 -- celltype-dependent, and the graph was never weaker. ``exact``
#: therefore reports a LOWER BOUND.
#:
#: ``graph`` costs ~10x: an 18.3 s index load per condition plus 0.77 ms per
#: record updated. Default stays ``exact`` so the committed tables reproduce.
RETRIEVAL = os.environ.get("BIORAG_RETRIEVAL", "exact")
if RETRIEVAL not in ("exact", "graph"):
    raise ValueError(f"BIORAG_RETRIEVAL={RETRIEVAL!r}; expected exact or graph")
if RETRIEVAL == "graph":
    OUT = OUT.with_name(OUT.stem + "_graph" + OUT.suffix)
    OUT_BY_DONOR = OUT_BY_DONOR.with_name(
        OUT_BY_DONOR.stem + "_graph" + OUT_BY_DONOR.suffix)
    IDX = BASE.parent / CFG["paths"]["reference_dir"] / "annotation" / "labelled_kNN.bin"
    EF = int(CFG.get("hnsw", {}).get("ef_search", 100))

#: Split the CELLTYPE loop across processes. Same mechanism as ``02b_02``, same
#: guarantee: ``seed_of(source, dose, SEED)`` does not see the shard, so every
#: poison set is identical however the celltypes are divided, and the split is a
#: strided slice of an already-sorted derived list. ``test_shard.py`` asserts
#: both. Default is one shard, reproducing the unsharded filename exactly.
#:
#: Worth less here than in ``02b_02``: under ``BIORAG_RETRIEVAL=graph`` only the
#: index load parallelises (18.3 s, single-threaded), while the point updates
#: already run at 15.4x threading and saturate all 18 cores. Amdahl caps the
#: gain at 2.0x; 3 shards give ~1.5x.
SHARD = int(os.environ.get("BIORAG_SHARD", "0"))
N_SHARDS = int(os.environ.get("BIORAG_N_SHARDS", "1"))
if not 0 <= SHARD < N_SHARDS:
    raise ValueError(f"BIORAG_SHARD={SHARD} out of range for "
                     f"BIORAG_N_SHARDS={N_SHARDS}")
if N_SHARDS > 1:
    OUT = OUT.with_name(OUT.stem + f"_shard{SHARD}" + OUT.suffix)
    OUT_BY_DONOR = OUT_BY_DONOR.with_name(
        OUT_BY_DONOR.stem + f"_shard{SHARD}" + OUT_BY_DONOR.suffix)


#: Fixed. This step varies alpha, not budget -- see the module docstring.
DOSE = 0.25

#: The alpha ladder, SPEC.md §5.4d. Displacement as a share of the measured
#: genuine neighbourhood radius: 0%, 0%, 1%, 5%, 23%, 102%, 776%, 1781%, 2870%.
ALPHAS: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0)

#: The four ``shifted`` variants. ``synthesised`` is absent because it has no
#: alpha -- its geometry is set by spread and jitter, which ``02b_13`` owns.
#: ``concentrated`` is absent because it is degenerate under modify: a record
#: cannot be overwritten twice.
TYPES: tuple[tuple[str, str, str], ...] = tuple(
    (Geometry.SHIFTED, lc, sel)
    for lc in (LabelChange.RELABEL, LabelChange.KEEP)
    for sel in (Selection.UNIFORM, Selection.HUB)
)

#: Never used -- no ``synthesised`` condition runs here -- but passed so the
#: shared constructor's required-parameter check is satisfied by the same values
#: 02b_11 used, keeping the two tables' columns aligned.
N_ANCHORS, SPREAD, JITTER = 1024, 0.99, 1.0

#: How far a ``shifted`` replacement moves toward the target centroid.
#:
#: FIXED here, and 0.5 is v01's value, so these rows reproduce v01's
#: ``embed_shift`` before anything new is claimed. Alpha is a real axis -- v01
#: never varied it, which is why ``shifted x keep`` reads as inert -- but it
#: belongs to its own step for the reason 02b_02 owns the anchor axis: sweeping
#: it here would multiply this grid by eleven for a question this step does not
#: ask. Passed as a degenerate range so every record moves by the same amount.
#: Poison rows per matmul inside ``remerge``. Arithmetically inert -- asserted
#: by equality in ``test_modify`` -- and lowered from the default 20,000 purely
#: to share the machine: the transient buffer is ``len(q) x block x 4`` bytes,
#: so 20,000 costs 3.96 GB against 49,456 queries and drives this step's ~26 GB
#: peak. At 5,000 it is 0.99 GB, which leaves room for the index sweeps running
#: alongside.
TOP_POISON_BLOCK = 5_000

SEED = 0


def main() -> None:
    t0 = time.time()
    emit(__file__)          # which code ran, HEAD or dirty tree
    deep_path = QCACHE / "clean_neighbors_deep.npz"
    if not deep_path.exists():
        sys.exit(f"{deep_path} is missing. Run 00_pipeline/retrieve_clean_deep.py "
                 f"first -- stage 2's depth-{K_NEIGHBORS} list cannot serve this "
                 f"attack, because overwriting a record promotes the next one and "
                 f"there is no next one at that depth.")

    ref = np.load(CACHE / "reference.npz", allow_pickle=True)
    ref_emb, ref_lab, ref_id = ref["emb"], ref["label"], ref["cell_id"]
    n_ref = len(ref_lab)
    deep = np.load(deep_path)
    deep_idx, deep_cos = deep["idx"], deep["cos"]
    q = np.load(QCACHE / "query_emb.npy")
    meta = pd.read_parquet(QCACHE / "query_meta.parquet")
    truth = meta["y_true"].to_numpy().astype(str)
    donor = meta["donor_id"].to_numpy()
    donors = np.array(sorted(pd.unique(donor)))

    sources = eligible_sources(ref_lab, truth)
    onto = Ontology.from_obo(BASE.parent / CFG["ontology"]["obo"])
    target_of = pick_targets(sources, onto)
    sources = [s for s in sources if s in target_of]
    mine = sources[SHARD::N_SHARDS]

    # BASELINE-REGIME-FIXED 2026-09-26. The clean side, once, FROM THE SAME
    # RETRIEVER as the attacked side -- CLAUDE.md 7, "One retrieval regime per
    # comparison". Under ``exact`` the deep list's own top-k IS the clean answer
    # and no retrieval need be repeated. Under ``graph`` it is NOT: the attacked
    # side comes from the shipped HNSW graph, so subtracting an exhaustive
    # baseline from it makes accuracy_delta and net_attack_success straddle two
    # regimes. 01a_02 bounds that at median |delta| 0.000000 over 34 celltypes
    # but 0.110429 on CL:0002062 -- the same order as the alpha effect this step
    # measures. One index load per RUN, not per condition, which is what 02a_01
    # already pays at its lines 94-96.
    if RETRIEVAL == "graph":
        clean_lab = clean_graph_topk(str(IDX), ref_lab, q, n_ref, EF,
                                     k=K_NEIGHBORS)
    else:
        clean_lab = ref_lab[deep_idx[:, :K_NEIGHBORS]]
    clean_ok = (vote(clean_lab, pivotal=truth).pred == truth).astype(float)
    clean_target: dict[str, np.ndarray] = {}
    # What the attacker is assumed to have watched: the same top-k traffic
    # 02b_05's hub_order ranks on. Deliberately the EXHAUSTIVE list in BOTH
    # modes. It is the attacker's model of reference traffic, not a measurement
    # of this run, and hub selection derives the poison set from it -- so
    # switching it under ``graph`` would change WHICH records get attacked, and
    # the exact and graph tables would no longer share a poison set and could
    # not be compared. The comparison is the point.
    hub_nbr = deep_idx[:, :K_NEIGHBORS]

    n_cond = len(sources) * len(TYPES) * len(ALPHAS)
    print(f"[{time.time()-t0:.0f}s] {n_cond} conditions "
          f"({len(sources)} celltypes x {len(TYPES)} attack types x "
          f"{len(ALPHAS)} alphas at dose {DOSE}), deep list "
          f"{deep_idx.shape}\n  types: "
          + ", ".join(f"{g}/{lc}/{sel}" for g, lc, sel in TYPES),
          file=sys.stderr)

    rows: list[dict] = []
    donor_frames: list[pd.DataFrame] = []
    for source in mine:
        target, odist = target_of[source]
        if target not in clean_target:
            clean_target[target] = (
                vote(clean_lab, pivotal=target).pred == target).astype(float)
        ct = clean_target[target]
        for geometry, label_change, selection in TYPES:
            for alpha in ALPHAS:
                t1 = time.time()
                cond_seed = seed_of(source, DOSE, SEED)
                p = build_modify(
                    ref_emb=ref_emb, ref_labels=ref_lab, ref_ids=ref_id,
                    source=source, target=target, dose=DOSE, seed=cond_seed,
                    selection=selection, geometry=geometry,
                    label_change=label_change,
                    n_anchors=N_ANCHORS, spread=SPREAD, jitter=JITTER,
                    alpha_range=(alpha, alpha), hub_nbr=hub_nbr)
                if RETRIEVAL == "graph":
                    lab_k, is_new, exact = graph_topk(
                        str(IDX), p["rows"], p[POISON_EMB], p[POISON_LABEL],
                        ref_lab, q, n_ref, EF, k=K_NEIGHBORS)
                else:
                    lab_k, is_new, exact = remerge(
                        deep_idx, deep_cos, p["rows"], p[POISON_EMB],
                        p[POISON_LABEL], ref_lab, q, n_ref, k=K_NEIGHBORS,
                    block=TOP_POISON_BLOCK)

                r_acc = vote(lab_k, pivotal=truth)
                r_hit = vote(lab_k, pivotal=target)
                ok = (r_acc.pred == truth).astype(float)
                got = (r_hit.pred == target).astype(float)
                ind = {
                    "accuracy": ok, "baseline_accuracy": clean_ok,
                    "accuracy_delta": ok - clean_ok,
                    "retrieval": is_new.any(axis=1).astype(float),
                    "poison_in_topk": is_new.sum(axis=1).astype(float),
                    "attack_success": got, "clean_target_rate": ct,
                    "net_attack_success": got - ct,
                }
                # Inexact queries are DROPPED, not truncated. A query that lost
                # more of its neighbourhood than the deep list can replace would
                # be scored on a short list, which understates the attack.
                g_sel = np.flatnonzero(exact)
                w_sel = np.flatnonzero(exact & (truth == source))
                donor_frames.append(pd.DataFrame({
                    "donor_id": donors,
                    **by_donor("within", w_sel, donor, donors, **ind),
                    **by_donor("global", g_sel, donor, donors, **ind),
                }).assign(attack=f"{geometry}_inplace",
                          geometry=str(geometry),
                          label_change=str(label_change),
                          selection=str(selection),
                          source=source, target=target, rate=DOSE, seed=SEED,
                          condition_seed=cond_seed, alpha=alpha))

                rows.append({
                    THREAT_MODEL: str(ThreatModel.WITH_ACCESS),
                    ACCESS: p["access"], GEOMETRY: p["geometry"],
                    LABEL_CHANGE: p["label_change"], SELECTION: p["selection"],
                    "attack": f"{geometry}_inplace",
                    "retrieval_path": RETRIEVAL,
                    "k": K_NEIGHBORS, "tie_break": TIE_BREAK,
                    "source": source, "target": target, "ontology_dist": odist,
                    "rate": DOSE, "dose_basis": "celltype",
                    "dose_clipped": p["dose_clipped"], "seed": SEED,
                    "condition_seed": cond_seed,
                    # null on the geometry that does not carry them, so the
                    # two geometries share one schema and concatenate
                    "n_anchors": N_ANCHORS if geometry == Geometry.SYNTHESISED else None,
                    "flood_spread": SPREAD if geometry == Geometry.SYNTHESISED else None,
                    "flood_jitter": JITTER if geometry == Geometry.SYNTHESISED else None,
                    "alpha_lo": alpha, "alpha_hi": alpha,
                    "n_overwritten": p["n_overwritten"],
                    "n_celltype": int((ref_lab == source).sum()),
                    "n_reference": n_ref,
                    "atlas_share": p["n_overwritten"] / n_ref,
                    "n_exact": int(exact.sum()),
                    "n_inexact": int((~exact).sum()),
                    "n_queries_within": int(len(w_sel)),
                    "n_queries_global": int(len(g_sel)),
                    "within_accuracy": float(ok[w_sel].mean()) if len(w_sel) else np.nan,
                    "within_net_attack_success":
                        float((got - ct)[w_sel].mean()) if len(w_sel) else np.nan,
                    "within_retrieval":
                        float(is_new.any(axis=1)[w_sel].mean()) if len(w_sel) else np.nan,
                    # merge_shards verifies completeness against
                    # n_conditions, the GLOBAL total. Without it the check
                    # silently skips and a partial merge looks complete --
                    # which is how a 324-row table from this 936-condition
                    # grid reached disk on 2026-09-26.
                    "condition_index": len(rows), "n_conditions": n_cond,
                    "seconds": round(time.time() - t1, 1),
                })
                r = rows[-1]
                print(f"[{time.time()-t0:6.0f}s] {len(rows):4d}/{n_cond} {source} "
                      f"-> {target} {geometry[:5]}/{label_change[:3]}/"
                      f"{selection:<7} alpha {alpha:<6} "
                      f"n_ow {p['n_overwritten']:>7,}  "
                      f"acc {r['within_accuracy']:.4f}  "
                      f"net {r['within_net_attack_success']:+.4f}  "
                      f"inexact {r['n_inexact']:>5,}  ({r['seconds']:.0f}s)",
                      file=sys.stderr)
        write(rows, OUT)
        write_by_donor(donor_frames, OUT_BY_DONOR)

    tot_inexact = sum(r["n_inexact"] for r in rows)
    print(f"\n[{time.time()-t0:.0f}s] wrote {len(rows)}/{n_cond} rows to {OUT.name}\n"
          f"  inexact query-conditions dropped: {tot_inexact:,} "
          f"(deep list depth {DEEP_K})", file=sys.stderr)


if __name__ == "__main__":
    main()
