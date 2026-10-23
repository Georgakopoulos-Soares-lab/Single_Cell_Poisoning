"""Step 03a_03 — does reweighting the vote add anything the stack has not already taken?

# No `from __future__ import annotations` here: snakemake's script directive
# prepends a preamble, which would push it off line 1 and make it a SyntaxError.

Every other layer decides WHICH neighbours vote. ``weight`` decides how much
each one counts: nothing is rejected, the arithmetic of the tally changes. It is
the only layer in the second stage of ``defended_vote`` -- ``survivors`` filters,
then ``vote_weights`` weights what survived.

**IT IS MEASURED ONLY IN COMPOSITION, AND THAT IS THE POINT OF THE STEP.** v01
measured weighting ALONE (``weighted_vote.parquet``, B cell, no other layer on)
and found ``rank`` taking cluster-flood 0.5306 -> 0.3676 and duplicate 0.1475 ->
0.0285 for 0.57 pp of clean accuracy. But standalone is the configuration where
poison that a record-property layer would have rejected is still present and
still voting. In the stack the record layers have already removed it, so the
quantity a deployment actually faces is ``stack + w`` against ``stack +
uniform`` -- a marginal contribution, not a solo number. Reporting the solo
figure as if it were the defence's value is the same error as quoting a ratio
whose two sides differ in something other than the thing being compared.

**THE CONTROL IS NOT OPTIONAL, and v01 said so first.**
``run_weighted_vote.py`` insists that ``rank`` = 1/(1+i) decays fiftyfold across
a 50-neighbour list, so it behaves like a plain majority vote at a much smaller
k. Without a k sweep beside it, "weighting defends" and "fewer neighbours is
better" are indistinguishable. Both are measured here, off the same retrieval:
the weighting schemes at k = 50, and the unweighted vote truncated to k in
{5, 10, 20, 50}. Truncation is free -- the neighbour list is already ranked --
so the control costs nothing but a column.

**WHY PROXIMITY SCHEMES ARE EXPECTED TO FAIL.** Three of the four insert
attacks place poison NEARER the query than genuine neighbours sit; that is what
makes them work. So ``score`` = 1/(1-c) and ``cosine`` weight by exactly the
quantity the attacker is optimising, and v01 measured them moving flood by about
one point. ``rank`` uses order only and cannot be moved by how close the
attacker gets -- crowding to cosine 0.99999 still only makes you rank 1.

**AND WHY RANK IS NOT SAFE EITHER.** Rank weights concentrate: with H(50) =
4.499, rank 1 alone carries 22% of the total vote weight and the top five carry
51%, against 2% and 10% under a plain majority. So reweighting does not remove
the attacker's leverage, it concentrates it and hands it to whoever reaches the
front -- which is what hub selection and flood are both built to do. In the
stack that mostly does not bite, because the record layers reject the poison
before it is weighted. Standalone it would. The ``poison_rank1`` column records
how often poison actually holds the nearest slot, so the reader can see which
regime a row is in rather than inferring it.

Six celltypes, one dose at parity. The question is whether a marginal
contribution exists at all; if it does, it earns a wider grid.
"""
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "00_common"))
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from attacks import build_attack  # noqa: E402
from celltype_sweep import Bench, seed_of, write, write_by_donor  # noqa: E402
from dataset_config import cache_root, load  # noqa: E402
from defended_sweep import evaluate_defense  # noqa: E402
from defense import DefenseConfig  # noqa: E402
from modify import knn_of_vectors  # noqa: E402
from provenance import emit  # noqa: E402
from schemas import (ACCESS, K_NEIGHBORS, SELECTION, THREAT_MODEL,  # noqa: E402
                     Access, POISON_EMB, ThreatModel)
from vote import TIE_BREAK  # noqa: E402

BASE = _HERE.parents[2]
CFG = load(BASE)
DS = CFG["dataset"]
CACHE = cache_root(BASE, CFG)
TAB = BASE / "2_results" / DS / "03_a_defence_vs_insert"
OUT = TAB / "03a_03_no_access_vote_weighting.parquet"
OUT_BY_DONOR = TAB / "03a_03_no_access_vote_weighting_by_donor.parquet"
IDX = BASE.parent / CFG["paths"]["reference_dir"] / "annotation" / "labelled_kNN.bin"

CELLTYPES = ("CL:0000623", "CL:0000236", "CL:0000235", "CL:0000057",
             "CL:0000084", "CL:0001065")
ATTACKS = ("relabel", "duplicate", "interpolate", "flood")
VERBATIM = {"relabel": True, "duplicate": True,
            "interpolate": False, "flood": False}
#: Selection per attack. Until 2026-10-09 every attack here ran at "hub". The
#: attack grid builds duplicate at "concentrated", and duplicate at hub is a
#: different, much stronger attack (diagnosis/2026-10-04_attack_vs_defence_none,
#: H2a), so duplicate now uses the grid's selection. The others are grid arms
#: at hub already.
SELECTION_OF = {"relabel": "hub", "duplicate": "concentrated",
                "interpolate": "hub", "flood": "hub"}
#: Opt-in suffix for a re-run, so it never overwrites a committed table.
RUN_TAG = os.environ.get("BIORAG_TAG", "")

#: Parity. 02b_12 puts relabel at 39.3% and flood at 41.7% here, so every
#: attack is doing something and a marginal contribution has room to show.
#: NOT a titration: one dose and six celltypes. The full 14-rate ladder
#: has never been run here (recorded 2026-10-09 at the user's request).
DOSE = 1.0

#: The budget 03a_01 found sets a threshold of 0.0200, above every measured
#: pre-ingestion poison score. Fixed here: this step varies the WEIGHT axis and
#: holds everything else at the stack's operating point.
LC_BUDGET = 0.01

FLOOD_ANCHORS, FLOOD_SPREAD, FLOOD_JITTER = 16_384, 0.999, 1.0
SCHEMES = ("uniform", "rank", "score", "cosine", "softmax")
SOFTMAX_T = 0.001
#: The control. 1/(1+i) behaves like a majority vote at smaller k, so the
#: unweighted vote is truncated to these and reported beside the schemes.
K_SWEEP = (5, 10, 20, 50)

SEED = 0
SHARD = int(os.environ.get("BIORAG_SHARD", 0))
N_SHARDS = int(os.environ.get("BIORAG_N_SHARDS", 1))
if not 0 <= SHARD < N_SHARDS:
    raise ValueError(f"BIORAG_SHARD={SHARD} out of range for {N_SHARDS} shards")
if RUN_TAG or N_SHARDS > 1:
    suffix = ((f"_{RUN_TAG}" if RUN_TAG else "")
              + (f"_shard{SHARD}" if N_SHARDS > 1 else ""))
    OUT = OUT.with_name(OUT.stem + suffix + OUT.suffix)
    OUT_BY_DONOR = OUT_BY_DONOR.with_name(
        OUT_BY_DONOR.stem + suffix + OUT_BY_DONOR.suffix)


def main() -> None:
    t0 = time.time()
    emit(__file__)
    b = Bench.load(CFG, BASE, t0)
    prov = np.load(CACHE / "reference_provenance.npz", allow_pickle=True)
    ref_donor = pd.factorize(prov["sample"])[0].astype(np.int32)
    del prov

    z = np.load(CACHE / "reference_knn_retrieved.npz")
    ref_knn = {int(r): nb for r, nb in zip(z["idx"], z["nbr"])}
    clean_rows = np.unique(b.clean_idx)
    if len(np.setdiff1d(clean_rows,
                        np.fromiter(ref_knn, np.int64, len(ref_knn)))):
        sys.exit("reference kNN cache incomplete; run 03a_01 with "
                 "BIORAG_PREPARE=1 first.")
    genuine_lc = {int(r): float((b.ref_label[ref_knn[int(r)]]
                                 == b.ref_label[int(r)]).mean())
                  for r in clean_rows}
    thr = float(np.quantile(np.fromiter(genuine_lc.values(), float,
                                        len(genuine_lc)), LC_BUDGET))
    print(f"[{time.time()-t0:.0f}s] lc_budget {LC_BUDGET:.0%} -> threshold "
          f"{thr:.4f}", file=sys.stderr)

    sources = [s for s in CELLTYPES if s in set(b.sources)][SHARD::N_SHARDS]
    n_cond = len(sources) * len(ATTACKS)
    print(f"[{time.time()-t0:.0f}s] {n_cond} conditions x "
          f"{len(SCHEMES)} schemes + {len(K_SWEEP)} k-sweep controls",
          file=sys.stderr)

    rows: list[dict] = []
    donor_frames: list[pd.DataFrame] = []
    done = 0
    for source in sources:
        target, odist = b.target_of[source]
        b.clean_target(target)
        qi = b.qi_of[source]
        for attack in ATTACKS:
            t1 = time.time()
            cond_seed = seed_of(source, DOSE, SEED)
            poison = build_attack(
                attack, ref_emb=b.ref_emb, ref_labels=b.ref_label,
                ref_ids=b.ref_id, source=source, target=target, rate=DOSE,
                seed=cond_seed, selection=SELECTION_OF[attack], hub_nbr=b.clean_idx,
                **({"n_anchors": FLOOD_ANCHORS, "spread": FLOOD_SPREAD,
                    "jitter": FLOOD_JITTER} if attack == "flood" else {}))
            pe = poison[POISON_EMB]
            ret = b.retrieve(pe, target)
            ids, is_p = ret.ids, ret.is_poison
            labels = b.ref_label[np.where(is_p, 0, ids)]
            labels[is_p] = target
            emb_k = np.where(is_p[:, :, None],
                             pe[np.where(is_p, ids - b.n_ref, 0)],
                             b.ref_emb[np.where(is_p, 0, ids)])
            donor_k = ref_donor[np.where(is_p, 0, ids)]
            recompute_ok = (np.ones(labels.shape, bool) if VERBATIM[attack]
                            else ~is_p)
            # How often poison holds the NEAREST slot -- the quantity that
            # decides whether rank weighting is protective or amplifying.
            rank1 = float(is_p[:, 0].mean())

            p_rows = (np.unique(ids[is_p] - b.n_ref) if is_p.any()
                      else np.empty(0, np.int64))
            g_miss = np.array([r for r in np.unique(ids[~is_p])
                               if int(r) not in genuine_lc], dtype=np.int64)
            if len(p_rows) or len(g_miss):
                vecs = np.concatenate([pe[p_rows],
                                       b.ref_emb[g_miss]]).astype(np.float32)
                selfr = np.concatenate([np.full(len(p_rows), -1, np.int64),
                                        g_miss])
                nb = knn_of_vectors(str(IDX), vecs, b.n_ref, b.ef,
                                    k=K_NEIGHBORS, self_rows=selfr)
                if len(p_rows):
                    p_lc = (b.ref_label[nb[:len(p_rows)]] == target).mean(axis=1)
                for i, gr in enumerate(g_miss):
                    genuine_lc[int(gr)] = float(
                        (b.ref_label[nb[len(p_rows)+i]]
                         == b.ref_label[int(gr)]).mean())
            lc_k = np.ones(labels.shape)
            if len(p_rows):
                pos = np.searchsorted(p_rows, ids - b.n_ref)
                lc_k = np.where(is_p, p_lc[np.clip(pos, 0, len(p_rows)-1)], 1.0)
            gi = ~is_p
            lc_k[gi] = np.fromiter((genuine_lc[int(v)] for v in ids[gi]),
                                   float, int(gi.sum()))
            t_prep = time.time() - t1

            plan = [(w, K_NEIGHBORS, f"stack+{w}") for w in SCHEMES]
            # The control: NO defence, unweighted, truncated to k' <= 50. Free,
            # because the list is already ranked nearest-first.
            plan += [("uniform", kk, f"none@k{kk}") for kk in K_SWEEP]
            for scheme, kk, name in plan:
                stacked = name.startswith("stack")
                cfg = (DefenseConfig(recompute=True, label_consistency=True,
                                     lc_threshold=thr, dedup_tau=0.9999,
                                     weight=scheme, weight_temp=SOFTMAX_T)
                       if stacked else DefenseConfig())
                sl = slice(None, kk)
                r = evaluate_defense(
                    labels=labels[:, sl], is_poison=is_p[:, sl],
                    y_true=b.truth, target=target, cfg=cfg,
                    recompute_ok=recompute_ok[:, sl] if stacked else None,
                    label_consistency=lc_k[:, sl] if stacked else None,
                    donor=donor_k[:, sl], cos=ret.topk_cos[:, sl],
                    emb=emb_k[:, sl] if stacked else None, per_query=True)
                got = np.asarray(r["hit"], float)
                ok = np.asarray(r["accurate"], float)  # accuracy under attack, not "correct" (poison removed)
                ind = {"accuracy": ok, "baseline_accuracy": b.clean_ok,
                       "accuracy_delta": ok - b.clean_ok,
                       "attack_success": got,
                       "retrieval": is_p[:, sl].any(axis=1).astype(float)}
                donor_frames.append(b.per_donor(ind, qi).assign(
                    config=name, weight=scheme, k_used=kk, source=source,
                    attack=attack, rate=DOSE, seed=SEED,
                    condition_seed=cond_seed))
                rows.append({
                    THREAT_MODEL: str(ThreatModel.NO_ACCESS),
                    ACCESS: str(Access.INSERT), SELECTION: SELECTION_OF[attack],
                    "attack": attack, "retrieval_path": "hnsw",
                    "ef_search": b.ef, "k": K_NEIGHBORS, "k_used": kk,
                    "tie_break": TIE_BREAK, "config": name, "weight": scheme,
                    "stacked": stacked, "lc_budget": LC_BUDGET,
                    "flood_spread": (FLOOD_SPREAD if attack == "flood" else None),
                    "flood_jitter": (FLOOD_JITTER if attack == "flood" else None),
                    "lc_threshold": thr, "softmax_temp": SOFTMAX_T,
                    "source": source, "target": target,
                    "ontology_dist": odist, "rate": DOSE,
                    "dose_basis": "celltype", "seed": SEED,
                    "condition_seed": cond_seed,
                    "n_conditions": (len([s for s in CELLTYPES
                                          if s in set(b.sources)])
                                     * len(ATTACKS) * len(plan)),
                    "n_poison": int(len(pe)), "poison_rank1": rank1,
                    "within_accuracy": float(ok[qi].mean()),
                    "within_attack_success": float(got[qi].mean()),
                    "within_baseline_accuracy": float(b.clean_ok[qi].mean()),
                    "global_accuracy": float(ok.mean()),
                    "global_attack_success": float(got.mean()),
                    "false_positive": float(r["false_positive"]),
                    "no_survivors": float(r["abstention"]),
                    "clean_accuracy_defended": float(r["clean_accuracy"]),
                })
            done += 1
            print(f"[{time.time()-t0:7.0f}s] {done}/{n_cond} {source} {attack} "
                  f"n_p {len(pe):>8,} poison@rank1 {rank1:.3f} "
                  f"(prep {t_prep:.0f}s, {len(plan)} configs in "
                  f"{time.time()-t1-t_prep:.0f}s)", file=sys.stderr)
            del poison, pe, emb_k, labels

    TAB.mkdir(parents=True, exist_ok=True)
    write(rows, OUT)
    write_by_donor(donor_frames, OUT_BY_DONOR)


if __name__ == "__main__":
    main()
