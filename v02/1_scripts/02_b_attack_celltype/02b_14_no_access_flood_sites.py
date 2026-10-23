"""Step 02b_14 — does it matter WHERE the flood puts its clusters?

Flood's last unswept axis. ``02b_02`` varies how MANY clusters there are and
``02b_13`` varies how far each record is displaced from its cluster; neither
asks what the cluster sits ON. v02 has only ever placed them on real reference
cells.

**v01 measured this and it matters.** Its ``cluster_flood`` reports three
"constructions", and on their only balanced slice -- rate 0.25, 64/256/1024
sites, four celltypes -- the best beats the worst by a median 13.1x and up to
47.4x on net attack success. That is not a knob anyone can leave at a default
without saying so.

**But v01 confounded two axes and this step separates them.** Its three modes
were ``(sites, displacement)`` pairs: ``anchor exact`` is real cells with NO
displacement, ``anchor directed`` is real cells with a 0.99 directed rotation,
and ``centroid`` is k-means centres with a 0.995 random scatter. Two of the
three differ only in displacement, which is ``spread`` and belongs to
``02b_13``. The genuine third axis is the SITES, and it was never measured
against a fixed displacement. Here spread and jitter are held at 0.99 and 1.0
for both arms, so a difference between them is the site choice alone.

**What v01's numbers predict, and why it is worth measuring anyway.** Pooled
over that slice, ``centroid`` captured the MOST neighbourhoods (0.066 median)
and converted the FEWEST (0.075 win-given-capture, against 0.68 for anchors).
A centroid is an average, and in 128 dimensions the average of a cloud is not
in the cloud -- v01 measured one at cosine 0.9496 to its queries against a
0.9796 fiftieth neighbour, i.e. below the retrieval bar. It squeaks into
neighbourhoods without ever dominating one.

That is a finding about DEFENCES, not just about attacks: a retrieval-based
filter would flag the centroid construction hardest while it is the least
dangerous of the three. A benchmark that reports only outcome failure cannot
say that, which is the distinction CLAUDE.md §1 exists to protect.

Writes ``02b_14_no_access_flood_sites.parquet`` and its ``_by_donor``
companion, sharing this module's schema so it concatenates with ``02b_02`` and
``02b_13``.
"""
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
                     Access, Attack, Geometry, LabelChange, Selection, Sites,
                     ThreatModel)
from vote import TIE_BREAK  # noqa: E402

BASE = _HERE.parents[2]
CFG = load(BASE)
OUT = (BASE / "2_results" / CFG["dataset"] / "02_b_attack_celltype"
       / "02b_14_no_access_flood_sites.parquet")
OUT_BY_DONOR = OUT.with_name("02b_14_no_access_flood_sites_by_donor.parquet")

ATTACK = Attack.FLOOD

#: The axis. Real reference cells against cluster centres of the celltype.
SITES: tuple[str, ...] = (Sites.ANCHOR, Sites.KMEANS)

#: How many clusters, shared by both arms so the comparison is paired rung for
#: rung. v01 could not do this -- it ran centroids at 1-1024 and anchors at
#: 64-16384, overlapping on only three counts. Capped at the pool size inside
#: ``build_flood``, which matters for the smaller celltypes: 4,096 centres
#: cannot be fitted to a 10,438-record celltype in any meaningful way, and
#: ``n_anchors_effective`` records what was actually used.
SITE_COUNTS: tuple[int, ...] = (64, 256, 1024, 4096)

#: v01's ``cluster_flood`` dose, so its numbers are what this is comparable to.
DOSE = 0.25

#: Held FIXED across both arms. This is the whole point: v01's site choice was
#: confounded with displacement, and holding these constant is what isolates
#: the axis. 02b_13 owns their titration.
SPREAD, JITTER = FLOOD_SPREAD, FLOOD_JITTER
SEED = 0


def main() -> None:
    t0 = time.time()
    emit(__file__)          # which code ran, HEAD or dirty tree
    require_tiers((ATTACK,))
    b = Bench.load(CFG, BASE, t0)
    n_cond = len(b.sources) * len(SITES) * len(SITE_COUNTS)
    print(f"[{time.time()-t0:.0f}s] {n_cond} conditions ({len(SITES)} site "
          f"kinds x {len(SITE_COUNTS)} counts x {len(b.sources)} celltypes) "
          f"at dose {DOSE}, spread {SPREAD}, jitter {JITTER}\n", file=sys.stderr)

    rows: list[dict] = []
    donor_frames: list[pd.DataFrame] = []
    for source in b.sources:
        target, attack_dist = b.target_of[source]
        qi, pool = b.qi_of[source], b.pool_of[source]
        ct = b.clean_target(target)

        for sites in SITES:
            for n_sites in SITE_COUNTS:
                t1 = time.time()
                # The seed key omits sites and the count, as it omits attack, so
                # both arms build from the same stream and a difference between
                # them is the construction rather than the draw.
                cond_seed = seed_of(source, DOSE, SEED)
                poison = build_attack(
                    ATTACK, ref_emb=b.ref_emb, ref_labels=b.ref_label,
                    ref_ids=b.ref_id, source=source, target=target, rate=DOSE,
                    seed=cond_seed, n_anchors=n_sites, spread=SPREAD,
                    jitter=JITTER, sites=sites,
                )
                pe = poison[POISON_EMB]
                n_p = len(pe)
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
                    **params_of(ATTACK, n_anchors=n_sites, spread=SPREAD,
                                jitter=JITTER, sites=sites),
                    "n_anchors_effective": min(n_sites, len(pool)),
                    "n_anchors_used": None,
                    "n_poison": n_p, "n_celltype": int(len(pool)),
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
                    condition_seed=cond_seed, flood_sites=str(sites),
                    n_anchors=n_sites, flood_spread=SPREAD,
                    flood_jitter=JITTER,
                ))

                print(f"[{time.time()-t0:6.0f}s] {len(rows):3d}/{n_cond} {source} "
                      f"{sites:<14} sites {n_sites:>6,} n_p {n_p:>7,}  "
                      f"cap {row['within_retrieval']:.4f}  "
                      f"net {row['within_net_attack_success']:+.4f}  "
                      f"({row['seconds']:.0f}s)", file=sys.stderr)

        write(rows, OUT)
        write_by_donor(donor_frames, OUT_BY_DONOR)

    print(f"\n[{time.time()-t0:.0f}s] wrote {len(rows)}/{n_cond} rows to "
          f"{OUT.name}", file=sys.stderr)


if __name__ == "__main__":
    main()
