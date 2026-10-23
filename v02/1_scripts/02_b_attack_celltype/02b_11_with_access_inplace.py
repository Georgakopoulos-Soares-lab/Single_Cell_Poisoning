"""Step 02b_11 — the attacker overwrites records, embeddings and all.

The six write-arm types whose replacements MOVE (SPEC.md §5.4): ``modify`` x
{``shifted``, ``synthesised``} x {``relabel``, ``keep``} x {``uniform``,
``hub``}, as the attack table enumerates them. Together with ``02b_05``'s
``modify x unmoved`` they complete the geometry axis on the write arm, which was
the only axis present in one arm and absent from the other.

**What makes these different from their insertion twins.** ``02b_01``'s
``interpolate`` shifts a COPY and ``02b_02``'s ``flood`` ADDS records; either
way the genuine record stays in place, still voting for the right answer. Here
the genuine record is DISPLACED, so each poisoned record costs the defender a
correct vote as well as adding a wrong one. Same construction, same budget, same
ladder -- one axis apart. That is the comparison the symmetric grid exists to
make, and it has never been measured.

**Four of the six exist only in v01** (``modify x shifted``), which is frozen.
They are rebuilt here rather than copied forward, against the same shared
constructors the insertion arm uses, so the two arms cannot drift.

**No index rebuild, and the reason is COST, not capability. Corrected
2026-09-25.** This docstring previously said hnswlib cannot update a vector.
It can: ``add_items`` with an existing id updates the point AND rewires its
links, demonstrated by moving a 5,000-node index's furthest record (rank 4,999)
onto the query and seeing it returned at rank 0. The poison could therefore be
written into the shipped graph, as the insertion arm does -- at 18.3 s of index
load per condition, or ~39 h across this step and ``02b_15``.

Instead this scores by removal-and-merge over a DEEPER clean list: drop the
overwritten records from each query's neighbours, promote what survives,
brute-force the replacements, merge. ``modify.remerge`` does it and is tested
against a from-scratch rebuild. That is exact for EXHAUSTIVE retrieval and does
not reproduce GRAPH retrieval: relocating a record would relocate its node, and
that effect is absent here. Rows carry ``retrieval_path = "exact"``; ``01a_02``
bounds the two paths at 99.0-99.6% vote agreement on clean data, and whether
poisoning widens that is unmeasured. The one condition removal-and-merge cannot
satisfy -- a query losing more than ``DEEP_K - k`` of its neighbours -- is
REPORTED per condition and those queries are dropped rather than truncated.

**Requires ``cache/<dataset>/clean_neighbors_deep.npz``**, from
``00_pipeline/retrieve_clean_deep.py``. Stage 2's depth-k list cannot serve:
removing a record promotes the next one, and there is no next one at depth k.

Writes ``02b_11_with_access_inplace.parquet`` and its ``_by_donor`` companion,
sharing ``02b_05``'s indicator names so the two write-arm steps concatenate.
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
OUT = TAB / "02b_11_with_access_inplace.parquet"
OUT_BY_DONOR = TAB / "02b_11_with_access_inplace_by_donor.parquet"

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


#: The shared ladder, SPEC.md §5.4b, MINUS its top rung.
#:
#: 1.25 is absent because it cannot mean anything here: a record cannot be
#: overwritten twice, so the dose clips to 1.0 and the condition is
#: byte-identical to the one below it. Keeping it would cost 1.6 measured hours
#: to recompute a duplicate, and -- worse -- would put two rows with different
#: dose labels and identical numbers in a table where someone will eventually
#: read a difference between them. The insertion arm keeps 1.25 because there it
#: is a real condition: the draw goes with-replacement and covers LESS of the
#: celltype (HANDOFF §3.32).
DOSES = (0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.7, 0.8, 0.85,
         0.9, 1.0)

#: The (geometry, label_change, selection) triples this step runs, enumerated
#: explicitly rather than as a cross product because the cross product is not
#: the design. ``synthesised x keep`` is absent on purpose: the attack table
#: holds it pending the alpha sweep, and inventing it here would report a
#: condition nobody argued for. ``concentrated`` is absent throughout --
#: degenerate under modify, since a record cannot be overwritten twice.
TYPES: tuple[tuple[str, str, str], ...] = tuple(
    (g, lc, sel)
    for g in (Geometry.SHIFTED, Geometry.SYNTHESISED)
    for lc in (LabelChange.RELABEL, LabelChange.KEEP)
    for sel in (Selection.UNIFORM, Selection.HUB)
    if not (g == Geometry.SYNTHESISED and lc == LabelChange.KEEP)
)

#: Flood's geometry parameters, matching 02b_02 so the insert/modify contrast is
#: one axis wide. n_anchors is fixed here rather than swept: 02b_02 owns that
#: axis, and sweeping it in both steps would multiply the grid for a comparison
#: neither needs.
N_ANCHORS, SPREAD, JITTER = 1024, 0.99, 1.0

#: How far a ``shifted`` replacement moves toward the target centroid.
#:
#: FIXED here, and 0.5 is v01's value, so these rows reproduce v01's
#: ``embed_shift`` before anything new is claimed. Alpha is a real axis -- v01
#: never varied it, which is why ``shifted x keep`` reads as inert -- but it
#: belongs to its own step for the reason 02b_02 owns the anchor axis: sweeping
#: it here would multiply this grid by eleven for a question this step does not
#: ask. Passed as a degenerate range so every record moves by the same amount.
ALPHA = 0.5
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
    # but 0.110429 on CL:0002062 -- the same order as the graph-vs-exact
    # divergence this run exists to measure. One index load per RUN, not per
    # condition, which is what 02a_01 already pays at its lines 94-96.
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

    n_cond = len(sources) * len(TYPES) * len(DOSES)
    print(f"[{time.time()-t0:.0f}s] {n_cond} conditions "
          f"({len(sources)} celltypes x {len(TYPES)} attack types x "
          f"{len(DOSES)} doses), deep list {deep_idx.shape}\n  types: "
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
            for dose in DOSES:
                t1 = time.time()
                cond_seed = seed_of(source, dose, SEED)
                p = build_modify(
                    ref_emb=ref_emb, ref_labels=ref_lab, ref_ids=ref_id,
                    source=source, target=target, dose=dose, seed=cond_seed,
                    selection=selection, geometry=geometry,
                    label_change=label_change,
                    n_anchors=N_ANCHORS, spread=SPREAD, jitter=JITTER,
                    alpha_range=(ALPHA, ALPHA), hub_nbr=hub_nbr)
                if RETRIEVAL == "graph":
                    lab_k, is_new, exact = graph_topk(
                        str(IDX), p["rows"], p[POISON_EMB], p[POISON_LABEL],
                        ref_lab, q, n_ref, EF, k=K_NEIGHBORS)
                else:
                    lab_k, is_new, exact = remerge(
                        deep_idx, deep_cos, p["rows"], p[POISON_EMB],
                        p[POISON_LABEL], ref_lab, q, n_ref, k=K_NEIGHBORS)

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
                          source=source, target=target, rate=dose, seed=SEED,
                          condition_seed=cond_seed))

                rows.append({
                    THREAT_MODEL: str(ThreatModel.WITH_ACCESS),
                    ACCESS: p["access"], GEOMETRY: p["geometry"],
                    LABEL_CHANGE: p["label_change"], SELECTION: p["selection"],
                    "attack": f"{geometry}_inplace",
                    "retrieval_path": RETRIEVAL,
                    "k": K_NEIGHBORS, "tie_break": TIE_BREAK,
                    "source": source, "target": target, "ontology_dist": odist,
                    "rate": dose, "dose_basis": "celltype",
                    "dose_clipped": p["dose_clipped"], "seed": SEED,
                    "condition_seed": cond_seed,
                    # null on the geometry that does not carry them, so the
                    # two geometries share one schema and concatenate
                    "n_anchors": N_ANCHORS if geometry == Geometry.SYNTHESISED else None,
                    "flood_spread": SPREAD if geometry == Geometry.SYNTHESISED else None,
                    "flood_jitter": JITTER if geometry == Geometry.SYNTHESISED else None,
                    "alpha_lo": ALPHA if geometry == Geometry.SHIFTED else None,
                    "alpha_hi": ALPHA if geometry == Geometry.SHIFTED else None,
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
                      f"{selection:<7} dose {dose:<6} "
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
