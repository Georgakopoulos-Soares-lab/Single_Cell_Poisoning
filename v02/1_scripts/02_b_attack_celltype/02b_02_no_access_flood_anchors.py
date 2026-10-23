"""Step 02b_02 — how many places should a flood spread its poison across?

At a fixed budget, does the attacker do better piling records into a few dense
clusters or spreading them thinly over many? ``02b_01`` holds the attack fixed
and varies the dose; this holds the dose band fixed and varies ``n_anchors``,
which is the axis that decides whether a flood works at all.

**COVERAGE BEATS DENSITY, and that is the hypothesis under test.** A query only
ever sees its own 50 nearest neighbours, so poison two celltypes away is poison
wasted. One dense pile captures the handful of queries beside it completely and
reaches nobody else; a thin layer puts a few records into everyone's
neighbourhood. v01 measured the anchor axis on three celltypes and found net
attack success rising monotonically with it. Net attack success, read off
``v01/2_results/GSE136831/02_a_attack_global/cluster_flood.parquet``
(``anchor directed``, rate 1.0) rather than quoted from memory:

    anchors               64      256     1024    4096    16384
    B cell                0.017   0.108   0.295   0.382   0.400
    macrophage            0.024   0.118   0.231   0.353   0.361
    NK cell               0.002   0.015   0.037   0.265   0.290
    classical monocyte    0.002   0.011   0.027   0.043   0.183

23x for B cell across the swept range, and its 16,384 row is the highest net
attack success in that entire table -- the strongest attack result the project
has. **But the curve had not flattened**: 4096 to 16384 still gained 17 points of
capture for B cell, so 16,384 is the last point measured and not a ceiling anyone
has found.

**And the four sources do not agree.** Monotone in every one, but the level at
16,384 runs 0.183 to 0.400 and the shape differs -- classical monocyte is still
nearly flat at 1024 where B cell has already reached 0.295, and NK cell does
almost all of its moving between 1024 and 4096. Four celltypes cannot say
whether that is a property of the celltype, of its size, or of its target. This
step asks the same question of all 26, which is the only way to tell an anchor
count that is good in general from one that was good for B cells.

**THREAT MODEL: no_access**, and the same deployed retrieval path as ``02b_01``
-- poison inserted into the shipped ``labelled_kNN.bin``, queried at ef=100, a
fresh ~9.3 GB load per condition because hnswlib has no true removal.

**Flood is the only attack here, and it is a CEILING construction.** Anchors are
real source cells, but they are rotated toward the pool's own centroid and the
result is renormalised, so the vectors correspond to no count matrix any
sequencer produced. A defender who re-derives embeddings from raw data rejects
them outright. The number this step produces is an upper bound on what the
construction buys, not a measurement of a threat, and it must never be reported
beside ``relabel`` or ``duplicate`` without that said.

**Where the anchor axis stops being an axis.** ``build_flood`` takes
``min(n_anchors, len(pool))`` anchors, so a celltype with fewer reference records
than the setting uses every record as an anchor and the higher settings all
collapse to the same construction -- flood degenerates toward jittered relabel
and the curve goes flat for a reason that has nothing to do with the attack.
Seven of these 26 celltypes are smaller than 16,384. A second ceiling bites
earlier: when the BUDGET is below the anchor count, each anchor receives one copy
and there are no clusters at all. ``n_anchors_effective`` and ``n_anchors_used``
record both, and a flat region in a curve drawn without them is an artefact
being read as a finding.

**Rates span the cliff rather than resolving it.** ``02b_01`` establishes where
the dose response lives; four points are enough to see whether the anchor
ordering is stable across it, and the budget goes to anchors instead.

**Everything between the poison set and the scored row lives in
``00_common/celltype_sweep.py``**, shared with ``02b_01``: the vote and its tie
bounds, the dual within/global scoring, the four accuracy scores, the
donor-paired intervals. This file owns the grid and the identity columns and
computes no statistic of its own, which is what makes the two tables directly
comparable rather than merely similar.

**520 conditions at ~31 s each, ~4.5 h. Runs concurrently with ``02b_01``
against the same shipped index.** Each process holds ~15 GB. No internal
parallelism: hnswlib already threads ``knn_query`` across every core, and two
processes contend for them as it is.

Writes ``02b_02_no_access_flood_anchors.parquet``: one row per
(source, n_anchors, rate, seed), rewritten after each source.
"""
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "00_common"))
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from attacks import build_attack, params_of  # noqa: E402
from celltype_sweep import (FLOOD_JITTER, FLOOD_SPREAD, Bench,  # noqa: E402
                            require_tiers, seed_of, write, write_by_donor)
from dataset_config import load  # noqa: E402
from provenance import emit  # noqa: E402
from schemas import (ACCESS, ATTACK_TIER, GEOMETRY, K_NEIGHBORS,  # noqa: E402
                     LABEL_CHANGE, POISON_EMB, SELECTION, THREAT_MODEL,
                     Access, Attack, Geometry, LabelChange, Selection,
                     ThreatModel)
from vote import TIE_BREAK  # noqa: E402

BASE = _HERE.parents[2]
CFG = load(BASE)
OUT = (BASE / "2_results" / CFG["dataset"] / "02_b_attack_celltype"
       / "02b_02_no_access_flood_anchors.parquet")

#: Flood's conditions kept at the DONOR, exactly as 02b_01 does it, for SPEC.md
#: 5.4c's threshold dose. Flood is five of the twelve insert sweeps, so omitting
#: it here would leave the threshold table covering the attack the project
#: measures as its strongest.
OUT_BY_DONOR = OUT.with_name("02b_02_no_access_flood_by_donor.parquet")

#: Split the CELLTYPE loop across processes, so several index loads overlap.
#:
#: 59% of a condition is ``hnswlib.load_index``, measured single-threaded at
#: 18.3 s wall against 18.3 s CPU. One thread cannot use more cores, so the only
#: way to spend the idle ones is to run more loads at once. Three shards turn
#: 15.7 h into roughly 5-6 h at ~11-14 GB each.
#:
#: **Cannot change a number.** The condition seed is ``seed_of(source, rate,
#: SEED)`` -- it does not see the shard, so every poison set is identical
#: however the celltypes are divided. The split is a strided slice of
#: ``b.sources``, which is itself sorted and derived, so the partition is exact
#: and reproducible. ``test_shard.py`` asserts both.
#:
#: Read from the environment rather than argparse because CLAUDE.md §7 forbids
#: arguments in ``1_scripts``. The default is a single shard, which reproduces
#: the unsharded run byte for byte, including the output filename.
SHARD = int(os.environ.get("BIORAG_SHARD", "0"))
N_SHARDS = int(os.environ.get("BIORAG_N_SHARDS", "1"))
if not 0 <= SHARD < N_SHARDS:
    raise ValueError(f"BIORAG_SHARD={SHARD} out of range for "
                     f"BIORAG_N_SHARDS={N_SHARDS}")
if N_SHARDS > 1:
    OUT = OUT.with_name(OUT.stem + f"_shard{SHARD}" + OUT.suffix)
    OUT_BY_DONOR = OUT_BY_DONOR.with_name(
        OUT_BY_DONOR.stem + f"_shard{SHARD}" + OUT_BY_DONOR.suffix)

ATTACK = Attack.FLOOD

#: The axis under test: how many clusters the budget is split across. v01's set,
#: kept identical so the per-celltype curves here can be laid against the
#: three-celltype curves in ``cluster_flood.parquet``. Powers of four, so a
#: plateau is visible rather than being stepped over.
ANCHORS: tuple[int, ...] = (64, 256, 1024, 4096, 16384)

#: Four dose points spanning the cliff ``02b_01`` resolves, not a dose ladder of
#: its own: enough to see whether the anchor ordering is stable across dose, and
#: no more, because the budget belongs to the anchor axis. 1.0 is the attacker's
#: The shared ladder, SPEC.md §5.4b. A strict superset of the four doses this
#: step used to run, so every committed flood condition still pairs with its row.
RATES: tuple[float, ...] = (0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
                            0.7, 0.8, 0.85, 0.9, 1.0, 1.25)

#: Flood's label axis. ``relabel`` is the attack and sweeps every anchor count;
#: ``keep`` is its CONTROL and runs at one.
#:
#: The control asks whether flood's harm comes from the false annotation or from
#: the geometry, and that question does not need the anchor axis -- it needs the
#: two labels compared at the SAME construction. Sweeping it would add 1,820
#: conditions and 22.9 measured hours to answer a question nobody asked.
#: :data:`KEEP_ANCHORS` is 1024, matching ``02b_11`` so the insert and modify
#: arms' controls are built alike.
KEEP_ANCHORS = 1024

#: ``(label_change, n_anchors)`` pairs, built from the two constants above.
TYPES: tuple[tuple[str, int], ...] = tuple(
    [(LabelChange.RELABEL, a) for a in ANCHORS]
    + [(LabelChange.KEEP, KEEP_ANCHORS)]
)


SEED = 0


def main() -> None:
    t0 = time.time()
    emit(__file__)          # which code ran, HEAD or dirty tree
    require_tiers((ATTACK,))
    b = Bench.load(CFG, BASE, t0)

    # GLOBAL count, not this shard's -- a reader compares len(df) against it to
    # tell a complete table from a partial one, and that must hold after merge.
    n_cond = len(b.sources) * len(TYPES) * len(RATES)
    mine = b.sources[SHARD::N_SHARDS]
    if N_SHARDS > 1:
        print(f"[{time.time()-t0:.0f}s] shard {SHARD}/{N_SHARDS}: "
              f"{len(mine)} of {len(b.sources)} celltypes -> "
              f"{len(mine)*len(TYPES)*len(RATES)} of {n_cond} conditions",
              file=sys.stderr)
    print(f"[{time.time()-t0:.0f}s] {n_cond} conditions ({len(b.sources)} celltypes "
          f"x {len(ANCHORS)} anchor counts x {len(RATES)} rates), "
          f"spread {FLOOD_SPREAD} jitter {FLOOD_JITTER}", file=sys.stderr)
    small = [s for s in b.sources if len(b.pool_of[s]) < max(ANCHORS)]
    print(f"[{time.time()-t0:.0f}s] {len(small)} celltypes hold fewer than "
          f"{max(ANCHORS):,} records, so their top anchor settings collapse to "
          f"one construction: {', '.join(small)}\n", file=sys.stderr)

    rows: list[dict] = []
    donor_frames: list[pd.DataFrame] = []
    for source in mine:
        target, attack_dist = b.target_of[source]
        qi, pool = b.qi_of[source], b.pool_of[source]
        ct = b.clean_target(target)

        for label_change, n_anchors in TYPES:
            for rate in RATES:
                t1 = time.time()
                # Derived once and WRITTEN to the table, so a reader
                # rebuilds this exact poison set from the column rather than
                # re-deriving it from a stored rate (celltype_sweep.KEY_COLUMNS).
                cond_seed = seed_of(source, rate, SEED)
                poison = build_attack(
                    ATTACK, ref_emb=b.ref_emb, ref_labels=b.ref_label,
                    ref_ids=b.ref_id, source=source, target=target, rate=rate,
                    seed=cond_seed, n_anchors=n_anchors,
                    spread=FLOOD_SPREAD, jitter=FLOOD_JITTER,
                    label_change=label_change,
                )
                n_p = len(poison[POISON_EMB])
                ind, reach = b.measure(poison[POISON_EMB], target, ct)
                del poison

                # Two separate ceilings on the anchor count, and they are not the
                # same one. _effective: the celltype is smaller than the request,
                # so every record is an anchor and flood degenerates toward
                # jittered relabel. _used: the budget is smaller than the anchor
                # count, so each anchor gets one copy and there are no clusters.
                eff = min(n_anchors, len(pool))
                row: dict = {
                    THREAT_MODEL: str(ThreatModel.NO_ACCESS),
                    "retrieval_path": "hnsw", "ef_search": b.ef,
                    "k": K_NEIGHBORS, "tie_break": TIE_BREAK,
                    "attack": str(ATTACK),
                    # the axes (SPEC.md 5.5). flood's anchors ARE its selection
                    # rule -- a fixed number of places the budget piles onto --
                    # so it reports as concentrated, like duplicate.
                    ACCESS: str(Access.INSERT),
                    GEOMETRY: str(Geometry.SYNTHESISED),
                    LABEL_CHANGE: str(label_change),
                    SELECTION: str(Selection.CONCENTRATED),
                    "dose_basis": "celltype",
                    "attack_tier": str(ATTACK_TIER[ATTACK]),
                    "source": source, "target": target,
                    "attack_ontology_dist": attack_dist,
                    "target_in_lineage": bool(
                        b.lin[np.searchsorted(b.vocab, source),
                              np.searchsorted(b.vocab, target)]),
                    "rate": rate, "seed": SEED,
                    "condition_seed": cond_seed,
                    **params_of(ATTACK, n_anchors=n_anchors,
                                spread=FLOOD_SPREAD, jitter=FLOOD_JITTER),
                    "n_anchors_effective": eff,
                    "n_anchors_used": min(eff, n_p),
                    "n_poison": n_p, "n_celltype": int(len(pool)),
                    "n_reference": b.n_ref, "atlas_share": n_p / b.n_ref,
                    "n_queries_within": int(len(qi)),
                    "n_queries_global": len(b.truth),
                    "condition_index": len(rows), "n_conditions": n_cond,
                }
                row.update(b.summarise(ind, reach, qi))
                row["seconds"] = round(time.time() - t1, 1)
                rows.append(row)

                # n_anchors rides along: it is an axis here and null in 02b_01,
                # so the two donor tables concatenate the same way the graph
                # tables do.
                donor_frames.append(b.per_donor(ind, qi).assign(
                    attack=str(Attack.FLOOD),
                    geometry=str(Geometry.SYNTHESISED),
                    label_change=str(label_change),
                    selection=str(Selection.CONCENTRATED),
                    source=source, target=target,
                    rate=rate, seed=SEED, condition_seed=cond_seed,
                    n_anchors=n_anchors,
                ))

                print(f"[{time.time()-t0:6.0f}s] {len(rows):3d}/{n_cond} {source} -> "
                      f"{target} {label_change[:3]}/anchors {n_anchors:>6,} (used "
                      f"{row['n_anchors_used']:>6,}) rate {rate:<5} "
                      f"n_p {n_p:>7,}  acc {row['within_accuracy']:.4f}  "
                      f"net {row['within_net_attack_success']:+.4f}  "
                      f"ret {row['within_retrieval']:.4f}  "
                      f"({row['seconds']:.0f}s)", file=sys.stderr)

        write(rows, OUT)
        write_by_donor(donor_frames, OUT_BY_DONOR)

    print(f"\n[{time.time()-t0:.0f}s] wrote {len(rows)}/{n_cond} rows to {OUT}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
