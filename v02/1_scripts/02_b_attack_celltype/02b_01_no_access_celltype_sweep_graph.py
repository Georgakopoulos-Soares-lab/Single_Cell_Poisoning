"""Step 02b_01 — every attackable celltype, dosed, on SCimilarity's own graph.

The central experiment. v01 measured the per-celltype dose curve with exact
brute force and the merge shortcut (``run_celltype_sweep.py``); this measures the
SAME attack on the deployed retrieval path -- poison inserted into the shipped
``labelled_kNN.bin`` and queried at ef=100, exactly as
``CellAnnotation.get_predictions_knn`` does it. Against a graph index the poison
must actually ENTER the graph or hnswlib never returns it and the attack is
inert, which is why each condition pays a fresh ~9.3 GB load: hnswlib has no true
removal, so a poisoned index cannot be un-poisoned and reused.

**THREAT MODEL: no_access.** Records are appended; the genuine ones remain and
keep voting. The attacker adds a wrong vote, they never remove a right one.

**Three attacks, not four.** ``flood`` is an ``Attack`` member and builds through
the same dispatcher, but it carries an axis the others do not -- ``n_anchors``,
across which v01 measured net attack success moving 0.017 to 0.400. Folding it in
at one arbitrary anchor count would report a point as if it were the attack, so
it runs as its own step, ``02b_02``, which sweeps that axis. The two tables share
this module's schema and concatenate.

**The poison is byte-identical to v01's.** No construction is reimplemented
here: ``00_common/attacks.py`` dispatches to ``make_poison.build_poison``,
untouched, and v01's ``seed_of`` is reproduced verbatim (blake2b of
``source|rate|seed``, not Python's ``hash()``, which is salted per process). All
eight rates below are on v01's ladder, so every ``relabel`` condition here pairs
one-to-one with a v01 row and the ONLY difference between them is retrieval.
That is what makes the comparison a paired one rather than two similar
experiments.

Note the seed key deliberately omits ``attack``: ``relabel`` and ``interpolate``
therefore draw the SAME source records, so interpolation's effect is the
vectors' movement toward the target and nothing else. ``duplicate`` diverges by
construction -- its first RNG draw is the 1%-of-celltype pool it copies from.

**Dose is a share of the SOURCE CELLTYPE, not of the atlas**, and the two
denominators must never be confused. An atlas share drawn from one celltype's
pool runs out above ~0.1%: sampling then falls back to with-replacement and
silently turns ``relabel`` into ``duplicate``, so the attack would change
identity along the dose axis instead of getting stronger. ``atlas_share``
carries the other denominator as a column so both are available without either
being the axis by accident.

**Everything between the poison set and the scored row lives in
``00_common/celltype_sweep.py``**, shared with ``02b_02``: the vote and its tie
bounds, the dual scoring, the four accuracy scores, the donor-paired intervals.
This file owns the grid and the identity columns and computes no statistic of
its own. See that module for what each measurement means -- in particular that
``within_`` is efficacy and ``global_`` is impact, that attack success is
reported raw, clean and net, and that ``mean_error_dist`` moves opposite to
every other accuracy column.

**Loads the shipped 9.3 GB index once per condition, 624 times. Approved in
conversation 2026-09-18.** Peak memory ~15 GB. Single process: hnswlib threads
the query internally, and ``02b_02`` runs concurrently against the same index.

Writes ``02b_01_no_access_celltype_sweep_graph.parquet``: one row per
(attack, source, rate, seed), rewritten after each source so a truncated run is
still readable -- compare ``len(df)`` against the ``n_conditions`` column.
"""
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "00_common"))
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from attacks import build_attack, params_of  # noqa: E402
from celltype_sweep import (Bench, require_tiers, seed_of,  # noqa: E402
                            write, write_by_donor)
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
       / "02b_01_no_access_celltype_sweep_graph.parquet")

#: The same conditions kept at the DONOR instead of pooled, for SPEC.md 5.4c's
#: threshold dose. It cannot be derived from OUT: that table is already
#: aggregated past the donor, and a threshold interval needs one donor draw held
#: fixed across every dose on the ladder. Written from indicators this loop
#: already holds, so it costs one groupby per condition and no extra retrieval.
OUT_BY_DONOR = OUT.with_name("02b_01_no_access_celltype_sweep_by_donor.parquet")

#: The insertion types this step runs: ``(geometry, label_change, selection)``
#: with the ``attack`` name each dispatches through.
#:
#: The name is kept for CONTINUITY, not for meaning. ``duplicate`` is
#: ``unmoved x concentrated`` and ``relabel`` is ``unmoved x uniform`` -- one
#: construction at two selection settings (SPEC.md §5.5) -- and now that
#: selection is a column, the name's only remaining job is to let a v02 row join
#: the 624 conditions already committed under it. The axes are what a reader
#: should predicate on.
#:
#: ``Attack.FLOOD`` is excluded and measured by ``02b_02``, which owns the anchor
#: axis. ``synthesised x keep`` likewise.
TYPES: tuple[tuple[str, str, str, str], ...] = (
    (Geometry.UNMOVED, LabelChange.RELABEL, Selection.UNIFORM, Attack.RELABEL),
    (Geometry.UNMOVED, LabelChange.RELABEL, Selection.CONCENTRATED, Attack.DUPLICATE),
    (Geometry.UNMOVED, LabelChange.RELABEL, Selection.HUB, Attack.RELABEL),
    (Geometry.UNMOVED, LabelChange.KEEP, Selection.UNIFORM, Attack.RELABEL),
    (Geometry.UNMOVED, LabelChange.KEEP, Selection.CONCENTRATED, Attack.DUPLICATE),
    (Geometry.SHIFTED, LabelChange.RELABEL, Selection.UNIFORM, Attack.INTERPOLATE),
)

#: The shared ladder, SPEC.md §5.4b. Fourteen doses as a fraction of the SOURCE
#: CELLTYPE, identical for every celltype-scope type in both arms -- which is
#: what makes insert against modify a comparison rather than two adjacent
#: curves.
#:
#: A strict superset of the eight v01 doses this step used to run, so every
#: committed condition still pairs one-to-one with its v01 row and the six new
#: rungs extend the curve rather than replacing it. The three lowest exist for
#: the write arm, which is already saturated at the lowest dose anyone had run;
#: the six from 0.7 up resolve the insertion cliff.
#:
#: Above parity the draw goes with-replacement, so more records cover LESS of
#: the celltype (HANDOFF §3.32). Those rows are a different attack, not more of
#: the same one, and the selection column now says so.
RATES: tuple[float, ...] = (0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
                            0.7, 0.8, 0.85, 0.9, 1.0, 1.25)

#: One seed. v01 used five and found the seed-to-seed spread negligible beside
#: the donor-to-donor spread the interval is built on; five here would cost
#: another 15 hours of index loading for a component that is already dominated.
SEED = 0


def main() -> None:
    t0 = time.time()
    emit(__file__)          # which code ran, HEAD or dirty tree
    require_tiers(tuple({a for *_, a in TYPES}))
    alpha_range = tuple(CFG["poison"]["alpha_range"])
    b = Bench.load(CFG, BASE, t0)

    n_cond = len(b.sources) * len(TYPES) * len(RATES)
    print(f"[{time.time()-t0:.0f}s] {n_cond} conditions ({len(TYPES)} types x "
          f"{len(b.sources)} celltypes x {len(RATES)} doses)\n  "
          + "\n  ".join(f"{g}/{lc}/{sel} (dispatches as {a})"
                        for g, lc, sel, a in TYPES) + "\n", file=sys.stderr)

    rows: list[dict] = []
    donor_frames: list[pd.DataFrame] = []
    for source in b.sources:
        target, attack_dist = b.target_of[source]
        qi, pool = b.qi_of[source], b.pool_of[source]
        ct = b.clean_target(target)

        for geometry, label_change, selection, attack in TYPES:
            for rate in RATES:
                t1 = time.time()
                # Derived once and WRITTEN to the table, so a reader
                # rebuilds this exact poison set from the column rather than
                # re-deriving it from a stored rate (celltype_sweep.KEY_COLUMNS).
                cond_seed = seed_of(source, rate, SEED)
                poison = build_attack(
                    attack, ref_emb=b.ref_emb, ref_labels=b.ref_label,
                    ref_ids=b.ref_id, source=source, target=target, rate=rate,
                    seed=cond_seed, alpha_range=alpha_range,
                    label_change=label_change, selection=selection,
                    hub_nbr=b.clean_idx,
                )
                n_p = len(poison[POISON_EMB])
                ind, reach = b.measure(poison[POISON_EMB], target, ct)
                del poison

                row: dict = {
                    THREAT_MODEL: str(ThreatModel.NO_ACCESS),
                    "retrieval_path": "hnsw", "ef_search": b.ef,
                    "k": K_NEIGHBORS, "tie_break": TIE_BREAK,
                    "attack": str(attack),
                    "attack_tier": str(ATTACK_TIER[attack]),
                    # the axes, so a reader predicates on the design rather than
                    # on a name that encodes two of them (SPEC.md 5.5)
                    ACCESS: str(Access.INSERT), GEOMETRY: str(geometry),
                    LABEL_CHANGE: str(label_change), SELECTION: str(selection),
                    "dose_basis": "celltype",
                    # the ATTACK's distance: source -> the attacker's target,
                    # fixed before any query is scored. The OUTCOME's distance is
                    # *_mean_error_dist.
                    "source": source, "target": target,
                    "attack_ontology_dist": attack_dist,
                    "target_in_lineage": bool(
                        b.lin[np.searchsorted(b.vocab, source),
                              np.searchsorted(b.vocab, target)]),
                    "rate": rate, "seed": SEED,
                    "condition_seed": cond_seed,
                    # the flood knobs come back null for all three of these, so
                    # no spread/jitter is passed; 02b_02 owns that geometry
                    **params_of(attack, alpha_range=alpha_range),
                    # null here and populated in 02b_02, so the two tables share
                    # one schema and concatenate without a reindex
                    "n_anchors_effective": None, "n_anchors_used": None,
                    "n_poison": n_p, "n_celltype": int(len(pool)),
                    "n_reference": b.n_ref, "atlas_share": n_p / b.n_ref,
                    "n_queries_within": int(len(qi)),
                    "n_queries_global": len(b.truth),
                    "condition_index": len(rows), "n_conditions": n_cond,
                }
                row.update(b.summarise(ind, reach, qi))
                row["seconds"] = round(time.time() - t1, 1)
                rows.append(row)

                # keys first, so the frame is self-describing without a join
                # back to OUT on a float rate that may not round-trip
                donor_frames.append(b.per_donor(ind, qi).assign(
                    attack=str(attack), geometry=str(geometry),
                    label_change=str(label_change), selection=str(selection),
                    source=source, target=target,
                    rate=rate, seed=SEED, condition_seed=cond_seed,
                ))

                print(f"[{time.time()-t0:6.0f}s] {len(rows):3d}/{n_cond} {source} -> "
                      f"{target} {geometry[:5]}/{label_change[:3]}/"
                      f"{selection[:4]:<4} dose {rate:<5} n_p {n_p:>7,}  "
                      f"acc {row['within_accuracy']:.4f}  "
                      f"net {row['within_net_attack_success']:+.4f}  "
                      f"ret {row['within_retrieval']:.4f}  "
                      f"err {row['within_mean_error_dist']:.2f}h  "
                      f"amb {row['within_ambiguous_rate_success']:.3f}  "
                      f"({row['seconds']:.0f}s)", file=sys.stderr)

        write(rows, OUT)
        write_by_donor(donor_frames, OUT_BY_DONOR)

    print(f"\n[{time.time()-t0:.0f}s] wrote {len(rows)}/{n_cond} rows to {OUT}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
