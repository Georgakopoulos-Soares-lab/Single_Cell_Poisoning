"""Step 02b_13 — flood's two geometry knobs, titrated at last.

``02b_02`` sweeps flood's anchor count and holds ``spread`` and ``jitter`` fixed
at 0.99 and 1.0. Those two were never varied in v02, and the attack table has
said so since it was written -- its verdict for this attack type reads
"extend ladder + spread/jitter", and only the ladder half was done.

**It matters more than a loose end.** v01's ``flood_push`` varied spread at
fixed everything else and found (B cell, 65,536 anchors, rate 4.0, jitter
0.999):

    spread 0.99   net 0.531      <- the value v02 uses everywhere
    spread 0.995  net 0.725
    spread 0.999  net 0.886
    spread 0.9999 net 0.888

NK cell runs 0.387, 0.678, 0.933, 0.953 across the same four. So v02's fixed
0.99 is the WEAKEST of the four settings v01 measured, and every flood number in
this project -- including the 0.400 that makes it the strongest attack here --
is quoted at roughly half strength. That is a reason to titrate, not a detail.

**What each knob does**, from ``flood_common.build_flood``:

``spread``  the cosine RETAINED toward the real anchor cell when the vector is
            rotated toward the source centroid. 1.0 leaves the anchor untouched;
            lower values pull it off the manifold toward an average cell that no
            real cell resembles. Higher is evidently better, which is the same
            lesson ``interpolate`` teaches by failing -- aiming at a centroid is
            aiming at empty space.
``jitter``  the cosine retained after per-copy noise. 1.0 means every copy of an
            anchor is IDENTICAL, which is why ``02b_07`` measures 368,537 flood
            records collapsing to 64 distinct vectors. Lower values buy
            distinctness -- de-duplication resistance -- and cost a little
            retrieval. That trade is the quantity this step exists to produce,
            so ``n_distinct`` is recorded on every row beside the attack rates.

**One dose, one anchor count, on purpose.** The grid is spread x jitter, 42
combinations over 26 celltypes. Adding the dose ladder would multiply it by 14
for a question this step does not ask: whether the geometry helps is not
expected to reverse with budget, and if it does, that is ``02b_02``'s axis to
find. Dose is fixed at parity, where flood is strongest and where v01's numbers
above were taken.

Writes ``02b_13_no_access_flood_geometry.parquet`` and its ``_by_donor``
companion, sharing this module's schema so it concatenates with ``02b_02``.
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
       / "02b_13_no_access_flood_geometry.parquet")
OUT_BY_DONOR = OUT.with_name("02b_13_no_access_flood_geometry_by_donor.parquet")

ATTACK = Attack.FLOOD

#: The radius of a genuine top-50 neighbourhood, as ``1 - cosine``.
#:
#: MEASURED, not assumed: the median cosine to the 50th neighbour in
#: ``cache/<dataset>/clean_neighbors.npz`` is 0.98048, so the whole neighbourhood spans
#: 0.0195, about 11.3 degrees. The 1st neighbour sits at 0.98695 and the 25th at
#: 0.98185, so the shell is thin -- almost all of a query's 50 neighbours lie
#: between 9.3 and 11.3 degrees away.
#:
#: This is the scale both knobs have to be read against. A poison vector
#: displaced by less than this stays inside the neighbourhood it is trying to
#: join; one displaced by much more cannot be retrieved at all. The grids below
#: are chosen to BRACKET it rather than to look tidy.
NEIGHBOURHOOD_1MCOS = 0.0195

#: Cosine retained toward the real anchor, log-spaced on ``1 - spread``.
#:
#: As a fraction of NEIGHBOURHOOD_1MCOS these are 0%, 0.5%, 5%, 51%, 102%, 256%
#: and 512%. An earlier draft of this step ran 0.99 to 1.0 only -- a span of
#: 0% to 51% -- which never leaves the neighbourhood and therefore can only
#: measure the plateau. v01 already shows that plateau is flat (0.999 gives net
#: 0.886 and 0.9999 gives 0.888), so three of those five points measured the
#: same thing and none reached the cliff. The cliff is where the poison stops
#: being retrieved, and that is necessarily OUTSIDE the radius.
#:
#: 0.99 is kept because v02's 520 committed flood conditions were run at it.
#: 1.0 is the limit: if no rotation wins, rotating toward the centroid is not
#: merely weak but harmful, and flood reduces to duplicate with jitter.
SPREADS: tuple[float, ...] = (1.0, 0.9999, 0.999, 0.99, 0.98, 0.95, 0.9)

#: Cosine retained after per-copy noise; 0%, 5%, 51%, 102%, 256%, 512% of the
#: radius.
#:
#: 0.98 is the value that makes the construction's own claim true.
#: ``flood_common.build_flood`` says jitter "is what defeats near-duplicate
#: filtering: the copies are no more similar to each other than genuine cells
#: from one study are" -- but v02 runs jitter 1.0, where the copies are
#: BYTE-IDENTICAL, which is exactly why ``02b_07`` finds 368,537 records
#: collapsing to 64 distinct vectors. At 0.98 the copies sit one neighbourhood
#: radius apart, which is the docstring's property actually delivered. Whether
#: that costs retrieval is what this axis measures.
#:
#: The far end is anchored by a known result rather than a guess: jitter 0.25
#: puts poison at cosine 0.244 against a 50th neighbour at 0.980, never
#: retrieved. 0.9 at 512% of the radius should already show the collapse
#: beginning; if it does not, that is itself worth knowing.
JITTERS: tuple[float, ...] = (1.0, 0.999, 0.99, 0.98, 0.95, 0.9)

#: Fixed while the geometry varies. 16,384 is the top of 02b_02's anchor sweep
#: and the lower of the two counts v01's flood_push used, so the numbers above
#: are the ones this step is comparable to.
N_ANCHORS = 16_384

#: Parity, where flood is strongest and where v01's spread series was measured.
DOSE = 1.0
SEED = 0


def distinct(emb: np.ndarray) -> int:
    """How many of the poisoned vectors are actually different.

    The quantity a near-duplicate defence acts on, and the reason ``jitter``
    is worth titrating: at jitter 1.0 every copy of an anchor is identical, so
    a budget of 368,537 records can be 64 points (``02b_07``).

    Args:
        emb: (n_p, 128) float32 poison vectors.

    Returns:
        Distinct row count.
    """
    return int(len(np.unique(emb, axis=0)))


def main() -> None:
    t0 = time.time()
    emit(__file__)          # which code ran, HEAD or dirty tree
    require_tiers((ATTACK,))
    b = Bench.load(CFG, BASE, t0)
    n_cond = len(b.sources) * len(SPREADS) * len(JITTERS)
    print(f"[{time.time()-t0:.0f}s] {n_cond} conditions ({len(SPREADS)} spreads "
          f"x {len(JITTERS)} jitters x {len(b.sources)} celltypes) at "
          f"dose {DOSE}, {N_ANCHORS:,} anchors\n", file=sys.stderr)

    rows: list[dict] = []
    donor_frames: list[pd.DataFrame] = []
    for source in b.sources:
        target, attack_dist = b.target_of[source]
        qi, pool = b.qi_of[source], b.pool_of[source]
        ct = b.clean_target(target)

        for spread in SPREADS:
            for jitter in JITTERS:
                t1 = time.time()
                # The seed key omits spread and jitter deliberately, as it omits
                # attack: every cell of this grid then draws the SAME anchors,
                # so a difference between two cells is the geometry and not the
                # anchor draw.
                cond_seed = seed_of(source, DOSE, SEED)
                poison = build_attack(
                    ATTACK, ref_emb=b.ref_emb, ref_labels=b.ref_label,
                    ref_ids=b.ref_id, source=source, target=target, rate=DOSE,
                    seed=cond_seed, n_anchors=N_ANCHORS, spread=spread,
                    jitter=jitter,
                )
                pe = poison[POISON_EMB]
                n_p, n_d = len(pe), distinct(pe)
                ind, reach = b.measure(pe, target, ct)
                del poison, pe

                row: dict = {
                    THREAT_MODEL: str(ThreatModel.NO_ACCESS),
                    ACCESS: str(Access.INSERT),
                    GEOMETRY: str(Geometry.SYNTHESISED),
                    LABEL_CHANGE: str(LabelChange.RELABEL),
                    SELECTION: str(Selection.CONCENTRATED),
                    "retrieval_path": "hnsw", "ef_search": b.ef,
                    "k": K_NEIGHBORS, "tie_break": TIE_BREAK,
                    "attack": str(ATTACK),
                    "attack_tier": str(ATTACK_TIER[ATTACK]),
                    "source": source, "target": target,
                    "attack_ontology_dist": attack_dist,
                    "target_in_lineage": bool(
                        b.lin[np.searchsorted(b.vocab, source),
                              np.searchsorted(b.vocab, target)]),
                    "rate": DOSE, "dose_basis": "celltype", "seed": SEED,
                    "condition_seed": cond_seed,
                    **params_of(ATTACK, n_anchors=N_ANCHORS, spread=spread,
                                jitter=jitter),
                    # the knobs expressed against the MEASURED genuine
                    # neighbourhood radius, so "how far off-manifold is this"
                    # is on the row rather than left to the reader
                    "spread_radius_frac": (1.0 - spread) / NEIGHBOURHOOD_1MCOS,
                    "jitter_radius_frac": (1.0 - jitter) / NEIGHBOURHOOD_1MCOS,
                    "neighbourhood_1mcos": NEIGHBOURHOOD_1MCOS,
                    "n_anchors_effective": min(N_ANCHORS, len(pool)),
                    "n_anchors_used": None,
                    "n_poison": n_p, "n_distinct": n_d,
                    "distinct_share": n_d / n_p,
                    "n_celltype": int(len(pool)),
                    "n_reference": b.n_ref, "atlas_share": n_p / b.n_ref,
                    "n_queries_within": int(len(qi)),
                    "n_queries_global": len(b.truth),
                    "condition_index": len(rows), "n_conditions": n_cond,
                }
                row.update(b.summarise(ind, reach, qi))
                row["seconds"] = round(time.time() - t1, 1)
                rows.append(row)

                donor_frames.append(b.per_donor(ind, qi).assign(
                    attack=str(ATTACK), geometry=str(Geometry.SYNTHESISED),
                    label_change=str(LabelChange.RELABEL),
                    selection=str(Selection.CONCENTRATED),
                    source=source, target=target, rate=DOSE, seed=SEED,
                    condition_seed=cond_seed, flood_spread=spread,
                    flood_jitter=jitter, n_anchors=N_ANCHORS,
                ))

                print(f"[{time.time()-t0:6.0f}s] {len(rows):3d}/{n_cond} {source} "
                      f"spread {spread:<7}({row['spread_radius_frac']:>5.0%}) "
                      f"jitter {jitter:<7}({row['jitter_radius_frac']:>5.0%}) "
                      f"n_p {n_p:>7,} distinct {n_d:>7,} "
                      f"({row['distinct_share']:.3%})  "
                      f"net {row['within_net_attack_success']:+.4f}  "
                      f"ret {row['within_retrieval']:.4f}  "
                      f"({row['seconds']:.0f}s)", file=sys.stderr)

        write(rows, OUT)
        write_by_donor(donor_frames, OUT_BY_DONOR)

    print(f"\n[{time.time()-t0:.0f}s] wrote {len(rows)}/{n_cond} rows to "
          f"{OUT.name}", file=sys.stderr)


if __name__ == "__main__":
    main()
