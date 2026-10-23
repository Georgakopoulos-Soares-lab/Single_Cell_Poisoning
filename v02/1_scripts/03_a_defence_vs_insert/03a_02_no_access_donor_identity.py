"""Step 03a_02 — where a per-donor vote cap stops helping and starts helping the attacker.

# No `from __future__ import annotations` here: snakemake's script directive
# prepends a preamble, which would push it off line 1 and make it a SyntaxError.

``donor_cap`` limits how many records from one donor may vote. It is the only
layer whose SIGN flips, and the axis it flips along is how many identities the
attacker fabricates. This step measures the crossover.

**Why it is a separate step rather than a column of 03a_01.** A headline
configuration whose sign depends on an unobservable attacker parameter is not a
configuration a defender can adopt. 03a_01 therefore fixes ``splits = 1`` so its
matrix has one donor regime, and the sweep lives here with the axis explicit.

**WHAT v01 GOT WRONG, and this exists to retire it.** ``run_defence_table.py``
and ``run_defence_matrix.py`` both assign every poisoned record the donor string
``"attacker"`` -- ``run_defence_matrix.py`` line 216. The whole poison set
therefore collapses to ONE vote by construction, which is ``splits = 1`` without
anyone saying so. That is the defect SPEC.md 12 retracted for the "~24x at zero
false positives" claim, and the ``donor vote`` column of BOTH v01 matrices
inherits it. Those columns are unmeasured, not merely optimistic.

**The mechanism, which is arithmetic and not tuning.** A cap of ``c`` leaves a
neighbourhood roughly ``c x (distinct donors)`` votes. The genuine side holds
**12-17 distinct donors** in 50 slots -- a measured property of the reference,
not a knob. The attacker's side is however many ids they invent, at no cost. So
the cap turns the vote into a contest of donor COUNTS, which the attacker wins
by registering more addresses. v01's ``volume_defence.parquet`` puts the flip
between 4 and 8 identities; the theory predicts it near parity of donor counts,
12-17, and the grid below is dense enough there to tell those apart.

**The cap axis is bounded and its limit is known.** ``k = 50``, so a single
donor cannot contribute more than 50 records to a 50-slot neighbourhood: any cap
>= 50 is a no-op identical to switching the layer off. The sweep carries 20 and
50 to confirm that convergence empirically rather than asserting it.

**Everything here is free of retrieval cost except the retrieval itself.**
``splits`` rewrites the poison's donor METADATA -- same vectors, same
neighbourhoods -- so the entire ``cap x splits`` grid scores off ONE retrieval
per condition. And no configuration here reads label consistency, so this step
skips the per-condition index load that 03a_01 pays to score it.

Six celltypes, not 26. The question is the SHAPE of a crossover, and a shape
does not need the full atlas; 03a_01 carries the per-celltype cost picture.
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
from provenance import emit  # noqa: E402
from schemas import (ACCESS, K_NEIGHBORS, SELECTION, THREAT_MODEL,  # noqa: E402
                     Access, POISON_EMB, ThreatModel)
from vote import TIE_BREAK  # noqa: E402

BASE = _HERE.parents[2]
CFG = load(BASE)
DS = CFG["dataset"]
CACHE = cache_root(BASE, CFG)
TAB = BASE / "2_results" / DS / "03_a_defence_vs_insert"
OUT = TAB / "03a_02_no_access_donor_identity.parquet"
OUT_BY_DONOR = TAB / "03a_02_no_access_donor_identity_by_donor.parquet"

#: Six celltypes spanning the reference-size range, so the crossover is not
#: read off one favourable case. Resolved against Bench.sources at run time;
#: any that this dataset does not attack are dropped with a printed note.
CELLTYPES = ("CL:0000623", "CL:0000236", "CL:0000235", "CL:0000057",
             "CL:0000084", "CL:0001065")

#: The two attacks a donor cap can possibly touch. ``duplicate`` works BY
#: concentrating copies, so collapsing per donor removes its mechanism; that is
#: the case for the layer. ``relabel`` is the control -- the cap has no
#: mechanism against it, and v01 measured the cap turning a 0.00% relabel attack
#: into 37.82% at scale. ``flood`` and ``interpolate`` are omitted: they are
#: caught outright by ``recompute``, so their donor behaviour decides nothing.
ATTACKS = ("relabel", "duplicate")
#: Selection per attack. Until 2026-10-09 every attack here ran at "hub". The
#: attack grid builds duplicate at "concentrated", and duplicate at hub is a
#: different, much stronger attack (diagnosis/2026-10-04_attack_vs_defence_none,
#: H2a), so duplicate now uses the grid's selection. The others are grid arms
#: at hub already.
SELECTION_OF = {"relabel": "hub", "duplicate": "concentrated",
                "interpolate": "hub", "flood": "hub"}
#: Opt-in suffix for a re-run, so it never overwrites a committed table.
RUN_TAG = os.environ.get("BIORAG_TAG", "")
#: NOT a titration: two doses only. This step answers where the cap flips,
#: not how the flip moves with dose; the full 14-rate ladder has never been
#: run here (recorded 2026-10-09 at the user's request).
DOSES = (0.25, 1.0)

#: 0 is OFF -- ``defense.py`` gates on truthiness, so the filter is skipped
#: entirely rather than applied with a cap of zero. 1 is v01's ``donor_vote``,
#: carried as a rung of a continuum rather than a named layer so the retracted
#: case sits beside its neighbours. 20 and 50 confirm convergence to OFF.
CAPS = (0, 1, 2, 3, 5, 10, 20, 50)

#: Fabricated attacker identities. Dense between 8 and 17 because that is where
#: the genuine neighbourhood's own donor count sits and where the theory puts
#: the flip; v01's coarser grid could only bracket it between 4 and 8.
SPLITS = (1, 2, 4, 8, 12, 17, 24, 40)

#: Run the FULL attackable set instead of the six above. The six were chosen
#: to answer a shape question cheaply -- does the cap invert, and where -- and
#: they answered it: the crossover runs 35.86% at one attacker identity to
#: 75.46% at forty. Once that is a headline rather than a probe it should not
#: rest on six of twenty-six, so BIORAG_ALL_CELLTYPES=1 runs all of them and
#: tags the output rather than overwriting the committed six-celltype table.
#: Added 2026-09-30.
ALL_CELLTYPES = os.environ.get("BIORAG_ALL_CELLTYPES", "") == "1"
CT_TAG = "_all26" if ALL_CELLTYPES else ""

SEED = 0
SHARD = int(os.environ.get("BIORAG_SHARD", 0))
N_SHARDS = int(os.environ.get("BIORAG_N_SHARDS", 1))
if not 0 <= SHARD < N_SHARDS:
    raise ValueError(f"BIORAG_SHARD={SHARD} out of range for {N_SHARDS} shards")
if CT_TAG or RUN_TAG or N_SHARDS > 1:
    suffix = (CT_TAG + (f"_{RUN_TAG}" if RUN_TAG else "")
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
    n_real = int(ref_donor.max()) + 1
    del prov
    # The quantity the whole crossover is read against: how many distinct
    # donors a GENUINE neighbourhood holds. 12-17 in v01; measured here so the
    # comparison is against this reference rather than a remembered number.
    gd = np.array([len(np.unique(ref_donor[row])) for row in b.clean_idx])
    print(f"[{time.time()-t0:.0f}s] genuine donors per top-{K_NEIGHBORS}: "
          f"median {np.median(gd):.0f}, IQR {np.percentile(gd,25):.0f}-"
          f"{np.percentile(gd,75):.0f}", file=sys.stderr)

    pool = list(b.sources) if ALL_CELLTYPES else list(CELLTYPES)
    sources = [s for s in pool if s in set(b.sources)]
    missing = [s for s in pool if s not in set(b.sources)]
    if missing:
        print(f"  not attackable in {DS}, dropped: {', '.join(missing)}",
              file=sys.stderr)
    sources = sources[SHARD::N_SHARDS]

    n_cond = len(sources) * len(ATTACKS) * len(DOSES)
    n_rows_global = (len([s for s in pool if s in set(b.sources)])
                     * len(ATTACKS) * len(DOSES) * len(CAPS) * len(SPLITS))
    print(f"[{time.time()-t0:.0f}s] {n_cond} conditions x {len(CAPS)} caps x "
          f"{len(SPLITS)} splits", file=sys.stderr)

    rows: list[dict] = []
    donor_frames: list[pd.DataFrame] = []
    done = 0
    for source in sources:
        target, odist = b.target_of[source]
        b.clean_target(target)
        qi = b.qi_of[source]
        for attack in ATTACKS:
            for rate in DOSES:
                t1 = time.time()
                cond_seed = seed_of(source, rate, SEED)
                poison = build_attack(
                    attack, ref_emb=b.ref_emb, ref_labels=b.ref_label,
                    ref_ids=b.ref_id, source=source, target=target, rate=rate,
                    seed=cond_seed, selection=SELECTION_OF[attack], hub_nbr=b.clean_idx)
                pe = poison[POISON_EMB]
                # ONE retrieval for the whole cap x splits grid. splits only
                # rewrites donor metadata, so no vector and no neighbourhood
                # changes with it -- exactly the property that makes this step
                # affordable at all.
                ret = b.retrieve(pe, target)
                ids, is_p = ret.ids, ret.is_poison
                labels = b.ref_label[np.where(is_p, 0, ids)]
                labels[is_p] = target
                t_ret = time.time() - t1

                for splits in SPLITS:
                    rng = np.random.default_rng(cond_seed + splits)
                    p_donor = n_real + rng.integers(
                        0, splits, size=len(pe)).astype(np.int32)
                    donor_k = np.where(
                        is_p, p_donor[np.where(is_p, ids - b.n_ref, 0)],
                        ref_donor[np.where(is_p, 0, ids)])
                    for cap in CAPS:
                        cfg = DefenseConfig(donor_cap=cap) if cap else DefenseConfig()
                        r = evaluate_defense(
                            labels=labels, is_poison=is_p, y_true=b.truth,
                            target=target, cfg=cfg, donor=donor_k,
                            per_query=True)
                        got = np.asarray(r["hit"], float)
                        ok = np.asarray(r["accurate"], float)  # accuracy under attack, not "correct" (poison removed)
                        ind = {"accuracy": ok, "baseline_accuracy": b.clean_ok,
                               "accuracy_delta": ok - b.clean_ok,
                               "attack_success": got,
                               "retrieval": is_p.any(axis=1).astype(float)}
                        donor_frames.append(
                            b.per_donor(ind, qi).assign(
                                donor_cap=cap, splits=splits, source=source,
                                attack=attack, rate=rate, seed=SEED,
                                condition_seed=cond_seed))
                        rows.append({
                            THREAT_MODEL: str(ThreatModel.NO_ACCESS),
                            ACCESS: str(Access.INSERT), SELECTION: SELECTION_OF[attack],
                            "attack": attack, "retrieval_path": "hnsw",
                            "ef_search": b.ef, "k": K_NEIGHBORS,
                            "tie_break": TIE_BREAK,
                            "donor_cap": cap, "splits": splits,
                            "source": source, "target": target,
                            "ontology_dist": odist, "rate": rate,
                            "dose_basis": "celltype", "seed": SEED,
                            "condition_seed": cond_seed,
                            "n_conditions": n_rows_global,
                            "n_poison": int(len(pe)),
                            "genuine_donors_median": float(np.median(gd)),
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
                print(f"[{time.time()-t0:7.0f}s] {done}/{n_cond} {source} "
                      f"{attack} dose {rate:<5g} n_p {len(pe):>8,} "
                      f"(ret {t_ret:.0f}s, {len(CAPS)*len(SPLITS)} cells in "
                      f"{time.time()-t1-t_ret:.0f}s)", file=sys.stderr)
                del poison, pe, labels

    TAB.mkdir(parents=True, exist_ok=True)
    write(rows, OUT)
    write_by_donor(donor_frames, OUT_BY_DONOR)


if __name__ == "__main__":
    main()
