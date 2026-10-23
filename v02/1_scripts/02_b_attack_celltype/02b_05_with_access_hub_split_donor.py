"""Step 02b_05 — does hub targeting still work if the attacker never saw your queries?

The write-access arm's headline is that editing the 2,042 busiest records of a
celltype flips 94% of its predictions while editing 2,042 random ones flips 0.5%.
That 190x gap is the project's strongest claim, and as v01 measured it, it rests
on a circularity: ``hub_rank`` counts how often each record appears in the top-50
OF THE VERY QUERIES THE ATTACK IS SCORED ON. v01's own docstring calls this
"ORACLE knowledge" and records that a split-donor check "retains most of the
effect" -- but that number was never written to a table, so under this project's
rules it does not exist.

This produces it. Three selections, identical budget, all scored on the SAME
held-out donors:

``oracle``   hubs ranked using every donor, including the ones scored on. v01's
             assumption, and an attacker who has already seen your traffic.
``split``    hubs ranked on donors A ONLY, scored on donors B. An attacker who
             has watched SOMEONE's queries, not yours -- anyone running the same
             public tool on similar tissue.
``random``   records drawn at random from the celltype. The floor.

Read the gaps, not the levels. ``split`` near ``oracle`` means hub structure
generalises across donors and the threat is realistic. ``split`` near ``random``
means the attack needs your queries specifically, the 94% is a ceiling rather
than a threat, and it must never be quoted unqualified.

**No graph work.** Under ``with_access`` the attacker rewrites labels and moves
no embedding, so ``labelled_kNN.bin`` is byte-identical and the retrieved
neighbours are exactly the clean ones. Every condition is a re-vote over cached
neighbours with a permuted label array -- seconds, not the 20 s index load the
insertion arm pays per condition. That is a property of the threat model, not an
approximation.

**Donor is the unit throughout**, both for the split and for the bootstrap: cells
from one subject share genotype, batch and dissociation, so splitting on cells
would leak the same donor into both halves and flatter the attacker.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_common"))
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from celltype_sweep import (by_donor, eligible_sources,  # noqa: E402
                            pick_targets, write, write_by_donor)
from dataset_config import cache_root, load, query_cache  # noqa: E402
from flood_common import donor_ratio_ci  # noqa: E402
from ontology import Ontology  # noqa: E402
from provenance import emit  # noqa: E402
from schemas import THREAT_MODEL, ThreatModel  # noqa: E402
from vote import TIE_BREAK, bounds, vote  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[2]
CFG = load(BASE)
DS = CFG["dataset"]
CACHE = cache_root(BASE, CFG)      # shared: reference.npz, poison/
QCACHE = query_cache(BASE, CFG)    # this dataset only -- see dataset_config
OUT = (BASE / "2_results" / DS / "02_b_attack_celltype"
       / "02b_05_with_access_hub_split_donor.parquet")

#: The same conditions kept at the DONOR, for SPEC.md 5.4c's threshold dose.
#:
#: This table carries indicators OUT does not, and the difference is what makes
#: the two arms comparable at all. OUT reports ``attack_success`` over every
#: scored query -- a GLOBAL denominator of 26,055 -- while the insertion arm's
#: headline is ``within_``, the attacked celltype's own 120-17,571 queries. OUT
#: has no ``retrieval`` and no ``net_attack_success`` whatever. A threshold read
#: off the two tables as they stand would compare a within-celltype outcome rate
#: against a whole-dataset one and call the difference an arm effect.
#:
#: So both scopes and the full indicator set are emitted here, matching
#: ``Bench.measure`` name for name. OUT's own columns are untouched; nothing
#: already published changes meaning.
OUT_BY_DONOR = OUT.with_name("02b_05_with_access_hub_split_by_donor.parquet")

#: Share of the attacked celltype's records the attacker overwrites -- the
#: shared ladder, SPEC.md §5.4b, minus its top rung.
#:
#: A strict superset of the four doses this step used to run, so 0.05 stays the
#: headline condition and every committed row still pairs. 1.25 is absent for
#: the reason ``02b_11`` gives: a record cannot be overwritten twice, so it
#: clips to 1.0 and would duplicate the row below it under a different label.
#: The three doses below 0.01 are the point of the extension -- this arm is
#: already at 0.694 traffic captured at the LOWEST dose anyone has run, so its
#: entire unmeasured region is underneath.
RATES = (0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.7, 0.8, 0.85,
         0.9, 1.0)
#: How the attacker chooses which records to edit.
SELECTIONS = ("oracle", "split", "random")
#: Donor partitions. Several, because one split could be lucky.
SPLIT_SEEDS = (0, 1, 2)
BOOT_SEED = 0


def hub_order(pool: np.ndarray, nbr_idx: np.ndarray, n_ref: int) -> np.ndarray:
    """Pool records ordered by how many neighbour slots they occupy, busiest first.

    Args:
        pool: (n_pool,) reference row indices carrying the attacked label.
        nbr_idx: (n_q, k) int64 retrieved reference rows, from the donors whose
            traffic the attacker is assumed to have observed.
        n_ref: reference size, for the bincount length.

    Returns:
        (n_pool,) positions into ``pool``, busiest first. Ties break on reference
        row index via a stable sort, so selection is deterministic and needs no
        seed -- the same guarantee v01 made.
    """
    hits = np.bincount(nbr_idx.ravel(), minlength=n_ref)
    return np.argsort(-hits[pool], kind="stable")


def main() -> None:
    t0 = time.time()
    emit(__file__)          # which code ran, HEAD or dirty tree
    ref = np.load(CACHE / "reference.npz", allow_pickle=True)
    lab = ref["label"]
    n_ref = len(lab)
    nbr = np.load(QCACHE / "clean_neighbors.npz", allow_pickle=True)["idx"].astype(np.int64)
    meta = pd.read_parquet(QCACHE / "query_meta.parquet")
    truth = meta["y_true"].to_numpy().astype(str)
    donor = meta["donor_id"].to_numpy()

    sources = eligible_sources(lab, truth)
    onto = Ontology.from_obo(BASE.parent / CFG["ontology"]["obo"])
    target_of = pick_targets(sources, onto)
    sources = [s for s in sources if s in target_of]
    donors = np.array(sorted(pd.unique(donor)))
    print(f"[{time.time()-t0:.0f}s] {len(sources)} celltypes, {len(donors)} donors, "
          f"{len(RATES)}x{len(SELECTIONS)}x{len(SPLIT_SEEDS)} conditions each "
          f"= {len(sources)*len(RATES)*len(SELECTIONS)*len(SPLIT_SEEDS)} rows",
          file=sys.stderr)

    # The clean side, computed ONCE. Under with_access no embedding moves, so
    # the retrieved neighbours are the shipped ones and the clean vote does not
    # depend on the condition -- only on which label is pivotal. Recomputing it
    # per condition would cost 936 redundant votes and risk the two sides being
    # scored by different rules, which is exactly the v01 failure vote.py exists
    # to prevent.
    clean_ok = (vote(lab[nbr], pivotal=truth).pred == truth).astype(float)
    clean_target: dict[str, np.ndarray] = {}

    rows, donor_frames, n = [], [], 0
    for split_seed in SPLIT_SEEDS:
        rng = np.random.default_rng(split_seed)
        perm = rng.permutation(len(donors))
        rank_donors = set(donors[perm[:len(donors) // 2]])       # A: attacker sees
        score_mask = ~np.isin(donor, list(rank_donors))          # B: scored on
        # The attacker's view of traffic. oracle sees everything; split sees A only.
        view = {"oracle": np.ones(len(truth), bool),
                "split": np.isin(donor, list(rank_donors))}
        for source in sources:
            target, odist = target_of[source]
            pool = np.flatnonzero(lab == source)
            if target not in clean_target:
                clean_target[target] = (
                    vote(lab[nbr], pivotal=target).pred == target).astype(float)
            ct = clean_target[target]
            for sel in SELECTIONS:
                order = (hub_order(pool, nbr[view[sel]], n_ref) if sel in view
                         else np.random.default_rng(split_seed).permutation(len(pool)))
                for rate in RATES:
                    k = max(1, int(round(rate * len(pool))))
                    edited = pool[order[:k]]
                    lab2 = lab.copy()
                    lab2[edited] = target          # in place: no embedding moves
                    r_acc = vote(lab2[nbr], pivotal=truth)
                    r_hit = vote(lab2[nbr], pivotal=target)
                    b_hit, w_hit = bounds(r_hit, target)
                    m = score_mask
                    ones = np.ones(m.sum())
                    acc, alo, ahi = donor_ratio_ci(
                        (r_acc.pred[m] == truth[m]).astype(float), ones, donor[m], BOOT_SEED)
                    hit, hlo, hhi = donor_ratio_ci(
                        (r_hit.pred[m] == target).astype(float), ones, donor[m], BOOT_SEED)
                    # Share of this celltype's retrieval traffic the edit captured,
                    # which is the quantity "5% of the celltype" hides.
                    hits = np.bincount(nbr[m].ravel(), minlength=n_ref)
                    cap = hits[edited].sum() / max(hits[pool].sum(), 1)
                    # Per-query indicators, named exactly as Bench.measure
                    # names them so the two arms' donor tables concatenate.
                    # `retrieval` is the with_access analogue of "a poisoned
                    # record entered the top-k": no record was ADDED, so it asks
                    # whether an EDITED one was retrieved. traffic_captured is a
                    # slot SHARE and answers a different question; it is not a
                    # per-query rate and stays where it is.
                    is_edited = np.zeros(n_ref, bool)
                    is_edited[edited] = True
                    in_topk = is_edited[nbr]
                    got = (r_hit.pred == target).astype(float)
                    ok = (r_acc.pred == truth).astype(float)
                    ind = {
                        "accuracy": ok,
                        "baseline_accuracy": clean_ok,
                        "accuracy_delta": ok - clean_ok,
                        "retrieval": in_topk.any(axis=1).astype(float),
                        "poison_in_topk": in_topk.sum(axis=1).astype(float),
                        "attack_success": got,
                        "clean_target_rate": ct,
                        "net_attack_success": got - ct,
                    }
                    # Rank donors are excluded from BOTH scopes -- the whole
                    # point of the split is that the attacker's own donors are
                    # never scored. They still get a row, at 0/0, so the donor
                    # roster is identical in every condition and the threshold
                    # bootstrap can hold one draw fixed across the ladder.
                    g_sel = np.flatnonzero(m)
                    w_sel = np.flatnonzero(m & (truth == source))
                    donor_frames.append(pd.DataFrame({
                        "donor_id": donors,
                        **by_donor("within", w_sel, donor, donors, **ind),
                        **by_donor("global", g_sel, donor, donors, **ind),
                    }).assign(
                        attack="relabel_inplace", selection=sel,
                        split_seed=split_seed, source=source, target=target,
                        rate=rate, seed=split_seed, n_edited=int(k),
                    ))

                    rows.append({
                        THREAT_MODEL: ThreatModel.WITH_ACCESS, "threat_variant": sel,
                        "selection": sel, "split_seed": split_seed,
                        "source": source, "target": target, "ontology_dist": odist,
                        "rate": rate, "n_edited": int(k), "n_pool": int(len(pool)),
                        "traffic_captured": float(cap),
                        "n_rank_donors": len(rank_donors),
                        "n_score_donors": int(len(donors) - len(rank_donors)),
                        "n_scored_queries": int(m.sum()),
                        "accuracy": acc, "acc_lo": alo, "acc_hi": ahi,
                        "attack_success": hit, "hit_lo": hlo, "hit_hi": hhi,
                        "attack_success_best": float(b_hit[m].mean()),
                        "attack_success_worst": float(w_hit[m].mean()),
                        "tie_rate": float((r_hit.n_tied[m] > 1).mean()),
                        "ambiguous_rate": float(r_hit.pivotal_in_tie[m].mean()),
                        "tie_break": TIE_BREAK, "k": nbr.shape[1],
                    })
                    n += 1
            if n % 90 == 0:
                print(f"[{time.time()-t0:5.0f}s] {n} conditions", file=sys.stderr)
    write(rows, OUT)
    write_by_donor(donor_frames, OUT_BY_DONOR)
    print(f"[{time.time()-t0:.0f}s] wrote {len(rows)} rows to {OUT.name}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
