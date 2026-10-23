"""Step 02a_01 — scattered poisoning, measured on SCimilarity's actual graph.

The honest null. Spend the budget on records scattered at random across the whole
atlas, each relabelled to a random OTHER Cell Ontology term. There is no target,
so the poisoned records do not agree with each other and do not vote as a bloc.
"Poisoning must be targeted to work at all" is a real security finding and it
needs a measurement rather than an assertion.

v01 measured this with exact brute force. This measures the SAME poison on the
deployed retrieval path: inserted into the shipped ``labelled_kNN.bin`` and
queried at ef=100, exactly as ``CellAnnotation.get_predictions_knn`` does.

**The poison is byte-identical to v01's.** The construction now lives in
``00_common/atlas_attack.py`` so that ``02a_02``, the write-access mirror of this
step, chooses the SAME records and forges the SAME labels from the same seed --
otherwise the one axis between the two arms, whether the genuine record
survives, would be confounded with which records were hit. The code moved
unchanged and a golden test pins its output. ``seed_of`` is blake2b of
``mode|budget|draw`` -- not Python's ``hash()``, which is salted per process --
so the same triple yields the same RNG everywhere. Both the record selection and
the random wrong labels are reproduced verbatim from v01's ``run_global_attack``.
The ONLY difference between the two runs is retrieval, which is what makes this a
paired comparison rather than two similar experiments.

**Five draws, not v01's twenty, and the pairing survives** because draws 0-4 are
the same draws. v01's spread across 20 draws is sd 0.00028 in accuracy at the 10%
budget against an effect of 0.00158; five draws give a standard error of 0.00013,
so the effect is 12x it. Note this is a DIFFERENT argument from the one that made
20 draws inadequate for the TARGETED arm -- there a draw sampled which celltype
to attack, 20 of 650 possible pairs, and the percentile interval was unstable.
Here a draw only varies which records are poisoned. No victim is sampled.

**THREAT MODEL: no_access.** Records are appended; the genuine ones remain and
keep voting. Against a graph index the poison must actually ENTER the graph, or
hnswlib never returns it and the attack is inert -- hence a fresh load per
condition, since hnswlib has no true removal.

**Loads the 8.8 GB index once per condition. Approved in conversation
2026-09-18.** Peak memory ~10 GB at the 10% budget.

Writes ``02a_01_no_access_uniform_graph.parquet``: one row per (budget, draw),
and its ``_by_donor`` companion.

**BY-DONOR-ADDED 2026-09-26.** The point estimates and intervals here were always
donor-resampled -- ``donor_ratio_ci`` bins per-query values by donor and
resamples the 78 donors, so nothing reported was ever a cell-level interval. What
was missing is the UNREDUCED per-donor sums, and without them the one comparison
this module exists to make cannot carry an interval. ``02a_03`` is this step's
mirror: same records, same seed, same forged labels, the only difference being
whether the genuine record survives. A paired interval on THAT difference needs
one donor draw held across both conditions, which needs each side's numerators
and denominators separately. ``02a_03`` emitted them and this step did not, so
the pair could be compared as points and not as an interval.

Also adds the four taxonomy axes (``access``, ``geometry``, ``label_change``,
``selection``), so this table can be sliced by threat model like every other one,
and ``condition_seed`` so a single condition can be re-derived. ``mode`` is kept
alongside ``selection`` rather than renamed, because the committed table has it.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_common"))
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from atlas_attack import (MIN_CLASS, UNMAPPED,  # noqa: E402
                          build_uniform, seed_of)
from celltype_sweep import by_donor, write_by_donor  # noqa: E402
from dataset_config import cache_root, load, query_cache  # noqa: E402
from flood_common import donor_ratio_ci  # noqa: E402
from provenance import emit  # noqa: E402
from vote import vote  # noqa: E402
from schemas import (ACCESS, DOSE_BASIS, EMB_DIM, EMB_DTYPE,  # noqa: E402
                     GEOMETRY, K_NEIGHBORS, LABEL_CHANGE, SCORE_QC, SELECTION,
                     THREAT_MODEL, Access, Geometry, LabelChange, Selection,
                     ThreatModel, scimilarity_score)

BASE = pathlib.Path(__file__).resolve().parents[2]
CFG = load(BASE)
DS = CFG["dataset"]
CACHE = cache_root(BASE, CFG)      # shared: reference.npz, poison/
QCACHE = query_cache(BASE, CFG)    # this dataset only -- see dataset_config
OUT = BASE / "2_results" / DS / "02_a_attack_atlas"
OUT_BY_DONOR = OUT / "02a_01_no_access_uniform_graph_by_donor.parquet"
IDX = BASE.parent / CFG["paths"]["reference_dir"] / "annotation" / "labelled_kNN.bin"

#: v01's grid, as fractions of the ATLAS. Kept identical so the comparison pairs.
BUDGETS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1)
DRAWS = 5
EF = int(CFG.get("hnsw", {}).get("ef_search", 100))
SEED = 0


def main() -> None:
    t0 = time.time()
    emit(__file__)          # which code ran, HEAD or dirty tree
    import hnswlib

    ref = np.load(CACHE / "reference.npz", allow_pickle=True)
    lab, emb = ref["label"], ref["emb"]
    n_ref = len(lab)
    q = np.load(QCACHE / "query_emb.npy").astype(EMB_DTYPE)
    meta = pd.read_parquet(QCACHE / "query_meta.parquet")
    truth = meta["y_true"].to_numpy().astype(str)
    donor = meta["donor_id"].to_numpy()
    donors = np.array(sorted(pd.unique(donor)))
    ones = np.ones(len(truth))

    usable = np.flatnonzero(lab != UNMAPPED)
    terms, counts = np.unique(lab[usable], return_counts=True)
    big = terms[counts >= MIN_CLASS]
    print(f"{n_ref:,} reference rows, {len(usable):,} with a CL id, "
          f"{len(big)} classes >= {MIN_CLASS:,}", file=sys.stderr)

    base_ix = hnswlib.Index(space="cosine", dim=EMB_DIM)
    base_ix.load_index(str(IDX))
    base_ix.set_ef(EF)
    b_idx, _ = base_ix.knn_query(q, k=K_NEIGHBORS)
    del base_ix
    # Kept per query, not collapsed. The by_donor sums need an indicator array,
    # and accuracy_delta needs a per-query clean side to subtract. This baseline
    # comes from the SAME HNSW index the attacked side is queried on, one clean
    # query before any insertion -- CLAUDE.md 7, "One retrieval regime per
    # comparison". It is the pattern 02b_11 and 02b_15 were fixed to match.
    clean_ok = (vote(lab[b_idx.astype(np.int64)],
                     pivotal=truth).pred == truth).astype(float)
    base_acc = float(clean_ok.mean())
    print(f"[{time.time()-t0:.0f}s] clean graph baseline accuracy {base_acc:.5f}",
          file=sys.stderr)

    rows = []
    donor_frames: list[pd.DataFrame] = []
    for budget in BUDGETS:
        n_p = int(round(budget * n_ref))
        for draw in range(DRAWS):
            t1 = time.time()
            cond_seed = seed_of("uniform", budget, draw)
            src, pl = build_uniform(lab, usable, big, n_p, cond_seed)
            ix = hnswlib.Index(space="cosine", dim=EMB_DIM)
            ix.load_index(str(IDX), max_elements=n_ref + len(src))
            ix.add_items(emb[src].astype(EMB_DTYPE), np.arange(n_ref, n_ref + len(src)))
            ix.set_ef(EF)
            ids, dist = ix.knn_query(q, k=K_NEIGHBORS)
            del ix
            ids = ids.astype(np.int64)

            is_p = ids >= n_ref
            _r = vote(np.concatenate([lab, pl])[ids], pivotal=truth)
            y_pois = _r.pred
            cos = (1.0 - dist).astype(EMB_DTYPE)
            near = np.where(is_p, cos, -np.inf).max(axis=1)
            has = np.isfinite(near)
            qc = scimilarity_score(near[has]) >= SCORE_QC if has.any() else np.array([])

            ok = (y_pois == truth).astype(float)
            got = is_p.any(axis=1).astype(float)
            a, alo, ahi = donor_ratio_ci(ok, ones, donor, SEED)
            r, rlo, rhi = donor_ratio_ci(got, ones, donor, SEED)

            # The same indicators, kept UNREDUCED per donor. qc_pass goes in as a
            # (numerator, denominator) pair because its denominator is not every
            # query: it is the queries that retrieved any poison at all, so a
            # donor contributing none must contribute 0/0 and not 0/1.
            qc_num = np.zeros(len(truth))
            qc_num[has] = qc.astype(float) if len(qc) else 0.0
            ind = {
                "accuracy": ok,
                "baseline_accuracy": clean_ok,
                "accuracy_delta": ok - clean_ok,
                "retrieval": got,
                "poison_in_topk": is_p.sum(axis=1).astype(float),
                "qc_pass": (qc_num, has.astype(float)),
            }
            g_sel = np.arange(len(truth))
            donor_frames.append(pd.DataFrame({
                "donor_id": donors,
                **by_donor("global", g_sel, donor, donors, **ind),
            }).assign(attack="relabel_insert_atlas",
                      geometry=str(Geometry.UNMOVED),
                      label_change=str(LabelChange.RELABEL),
                      selection=str(Selection.UNIFORM), source="atlas",
                      target="random", rate=budget, seed=draw))

            rows.append({
                THREAT_MODEL: str(ThreatModel.NO_ACCESS),
                ACCESS: str(Access.INSERT), GEOMETRY: str(Geometry.UNMOVED),
                LABEL_CHANGE: str(LabelChange.RELABEL),
                SELECTION: str(Selection.UNIFORM),
                "attack": "relabel_insert_atlas",
                "retrieval_path": "hnsw", "ef_search": EF, "mode": "uniform",
                "budget": budget, DOSE_BASIS: "atlas", "rate": budget,
                "draw": draw, "condition_seed": int(cond_seed),
                "n_poison": len(src), "n_distinct_inserted": int(len(np.unique(src))),
                "n_reference": n_ref, "atlas_share": len(src) / n_ref,
                "n_queries_global": int(len(truth)),
                "baseline_accuracy": base_acc,
                "accuracy": a, "acc_lo": alo, "acc_hi": ahi,
                "accuracy_delta": float((ok - clean_ok).mean()),
                "retrieval": r, "ret_lo": rlo, "ret_hi": rhi,
                "poison_in_topk": float(is_p.sum(axis=1).mean()),
                "qc_pass": float(qc.mean()) if len(qc) else float("nan"),
                "tie_rate": float((_r.n_tied > 1).mean()),
                "ambiguous_rate": float(_r.pivotal_in_tie.mean()),
                "n_queries": len(truth), "k": K_NEIGHBORS,
                "seconds": round(time.time() - t1, 1),
            })
            print(f"[{time.time()-t0:5.0f}s] budget {budget*100:5.1f}% draw {draw}  "
                  f"n_p {len(src):>7,}  acc {a:.5f}  retrieval {r:.4f}  "
                  f"({rows[-1]['seconds']:.0f}s)", file=sys.stderr)

    OUT.mkdir(parents=True, exist_ok=True)
    main_table = OUT / "02a_01_no_access_uniform_graph.parquet"
    pd.DataFrame(rows).to_parquet(main_table, index=False)
    # the donor table reports itself, from write_by_donor
    write_by_donor(donor_frames, OUT_BY_DONOR)
    print(f"[{time.time()-t0:.0f}s] wrote {len(rows)} rows to {main_table.name}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
