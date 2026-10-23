"""Step 02a_03 — scattered poisoning by an attacker who can OVERWRITE.

The write-access mirror of ``02a_01``, and the last two types in the designed
grid (SPEC.md §5.4): ``atlas x modify x unmoved x relabel`` at ``uniform`` and
``hub`` selection.

Numbered 03 rather than 02 because ``02a_02`` is already ``02a_01``'s FIGURE.
A step id is shared by its script, its tables and its figures precisely so that
one grep finds everything that step produced (CLAUDE.md §3); reusing 02 would
have made ``grep 02a_02`` return two unrelated experiments.

**One axis apart from 02a_01, deliberately.** The same records are chosen from
the same seed and given the same forged labels -- ``00_common/atlas_attack.py``
is shared for exactly that reason. The only difference is what happens to the
genuine record. ``02a_01`` APPENDS a copy carrying the wrong term, so the
original stays and keeps voting correctly; here the original's label is
OVERWRITTEN, so the right vote is destroyed rather than outnumbered. Any
difference between the two tables is that and nothing else.

**No target, and that is the experiment.** Each poisoned record gets a random
WRONG term rather than a shared one, so the poison does not vote as a bloc.
``02a_01`` measures that as a near-null under insertion. Whether it stays null
when the attacker can delete right answers instead of merely adding wrong ones
is untested, and it is the difference between "scattered poisoning does not
work" and "scattered poisoning does not work IF you can only submit".

**No index load, and no approximation.** Under ``with_access x unmoved`` the
attacker rewrites labels and moves no embedding, so ``labelled_kNN.bin`` is
byte-identical and the retrieved neighbours are exactly the clean ones. Every
condition is a re-vote over the cached clean neighbours with a permuted label
array -- seconds, against the ~9 GB load ``02a_01`` pays per condition. That is
a property of the threat model, not a shortcut.

**Hub selection is the open question.** ``uniform`` spends the budget at random;
``hub`` spends it on the busiest records in the whole atlas, by top-k occupancy.
Within a celltype that is worth 19x at fixed budget (``02b_05``). Whether it
transfers to a scattered, untargeted attack is unmeasured, and it is the obvious
way this arm could stop being a null result. Hub's record choice is
deterministic given the traffic, so a draw varies only the forged labels.

Writes ``02a_03_with_access_relabel_inplace.parquet`` and its ``_by_donor``
companion. Only ``global_`` scope exists here: there is no attacked celltype, so
there is no ``within_`` population to score.
"""
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "00_common"))
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from atlas_attack import (MIN_CLASS, UNMAPPED, build_hub,  # noqa: E402
                          build_uniform, seed_of)
from celltype_sweep import by_donor, write, write_by_donor  # noqa: E402
from dataset_config import cache_root, load, query_cache  # noqa: E402
from provenance import emit  # noqa: E402
from schemas import (ACCESS, DOSE_BASIS, GEOMETRY, K_NEIGHBORS,  # noqa: E402
                     LABEL_CHANGE, SELECTION, THREAT_MODEL, Access, Geometry,
                     LabelChange, Selection, ThreatModel)
from vote import TIE_BREAK, vote  # noqa: E402

BASE = _HERE.parents[2]
CFG = load(BASE)
DS = CFG["dataset"]
CACHE = cache_root(BASE, CFG)      # shared: reference.npz, poison/
QCACHE = query_cache(BASE, CFG)    # this dataset only -- see dataset_config
TAB = BASE / "2_results" / DS / "02_a_attack_atlas"
OUT = TAB / "02a_03_with_access_relabel_inplace.parquet"
OUT_BY_DONOR = TAB / "02a_03_with_access_relabel_inplace_by_donor.parquet"

#: The clean graph retrieval this step re-votes over, and whose top-k occupancy
#: ``hub`` ranks on. The same file ``01a_01`` scored the clean baseline from, so
#: the ranking and the scoring cannot describe different retrievals.
CLEAN = BASE / "2_results" / DS / "01_a_baseline" / "01a_01_no_access_hnsw_neighbors.npz"

#: v01's atlas ladder, identical to ``02a_01``'s so the two arms pair rung for
#: rung. NOT the celltype ladder of SPEC.md §5.4b -- the denominator here is the
#: 7,913,892-record atlas, not a celltype, and the two are different quantities
#: (SPEC.md §5.5). ``n_poison`` is on every row of both, which is what lets a
#: reader put them on one axis anyway.
BUDGETS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1)
DRAWS = 5
SELECTIONS = (Selection.UNIFORM, Selection.HUB)


def main() -> None:
    t0 = time.time()
    emit(__file__)          # which code ran, HEAD or dirty tree
    ref = np.load(CACHE / "reference.npz", allow_pickle=True)
    lab = ref["label"]
    n_ref = len(lab)
    clean = np.load(CLEAN, allow_pickle=True)
    nbr = clean["idx"].astype(np.int64)
    meta = pd.read_parquet(QCACHE / "query_meta.parquet")
    truth = meta["y_true"].to_numpy().astype(str)
    donor = meta["donor_id"].to_numpy()
    donors = np.array(sorted(pd.unique(donor)))

    usable = np.flatnonzero(lab != UNMAPPED)
    terms, counts = np.unique(lab[usable], return_counts=True)
    big = terms[counts >= MIN_CLASS]

    # The clean side, once. No embedding moves, so the neighbours are the
    # shipped ones and the clean vote does not depend on the condition.
    clean_ok = (vote(lab[nbr], pivotal=truth).pred == truth).astype(float)
    print(f"[{time.time()-t0:.0f}s] {n_ref:,} reference rows, {len(usable):,} "
          f"with a CL id, {len(big)} celltypes >= {MIN_CLASS:,}; clean "
          f"accuracy {clean_ok.mean():.5f}", file=sys.stderr)

    n_cond = len(BUDGETS) * len(SELECTIONS) * DRAWS
    rows: list[dict] = []
    donor_frames: list[pd.DataFrame] = []
    for selection in SELECTIONS:
        for budget in BUDGETS:
            n_p = int(round(budget * n_ref))
            for draw in range(DRAWS):
                t1 = time.time()
                seed = seed_of(f"{selection}_inplace", budget, draw)
                if selection == Selection.UNIFORM:
                    src, forged = build_uniform(lab, usable, big, n_p, seed)
                else:
                    src, forged = build_hub(lab, usable, big, n_p, seed, nbr)

                lab2 = lab.copy()
                lab2[src] = forged          # in place: no embedding moves
                edited = np.zeros(n_ref, bool)
                edited[src] = True
                in_topk = edited[nbr]

                r_acc = vote(lab2[nbr], pivotal=truth)
                ok = (r_acc.pred == truth).astype(float)
                ind = {
                    "accuracy": ok,
                    "baseline_accuracy": clean_ok,
                    "accuracy_delta": ok - clean_ok,
                    "retrieval": in_topk.any(axis=1).astype(float),
                    "poison_in_topk": in_topk.sum(axis=1).astype(float),
                }
                g_sel = np.arange(len(truth))
                donor_frames.append(pd.DataFrame({
                    "donor_id": donors,
                    **by_donor("global", g_sel, donor, donors, **ind),
                }).assign(attack="relabel_inplace_atlas",
                          geometry=str(Geometry.UNMOVED),
                          label_change=str(LabelChange.RELABEL),
                          selection=str(selection), source="atlas",
                          target="random", rate=budget, seed=draw))

                rows.append({
                    THREAT_MODEL: str(ThreatModel.WITH_ACCESS),
                    ACCESS: str(Access.MODIFY), GEOMETRY: str(Geometry.UNMOVED),
                    LABEL_CHANGE: str(LabelChange.RELABEL),
                    SELECTION: str(selection),
                    "attack": "relabel_inplace_atlas",
                    "retrieval_path": "cached_clean", "k": K_NEIGHBORS,
                    "tie_break": TIE_BREAK,
                    "rate": budget, DOSE_BASIS: "atlas", "draw": draw,
                    "condition_seed": int(seed),
                    "n_poison": int(len(src)),
                    "n_distinct_edited": int(len(np.unique(src))),
                    "n_reference": n_ref, "atlas_share": len(src) / n_ref,
                    "n_queries_global": int(len(truth)),
                    "baseline_accuracy": float(clean_ok.mean()),
                    "accuracy": float(ok.mean()),
                    "accuracy_delta": float((ok - clean_ok).mean()),
                    "retrieval": float(in_topk.any(axis=1).mean()),
                    "seconds": round(time.time() - t1, 1),
                })
                r = rows[-1]
                print(f"[{time.time()-t0:5.0f}s] {len(rows):3d}/{n_cond} "
                      f"{selection:<8} budget {budget:<6} draw {draw} "
                      f"n_p {len(src):>8,}  acc {r['accuracy']:.5f} "
                      f"({r['accuracy_delta']:+.5f})  "
                      f"ret {r['retrieval']:.4f}  ({r['seconds']:.0f}s)",
                      file=sys.stderr)
        write(rows, OUT)
        write_by_donor(donor_frames, OUT_BY_DONOR)

    print(f"\n[{time.time()-t0:.0f}s] wrote {len(rows)} rows to {OUT.name}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
