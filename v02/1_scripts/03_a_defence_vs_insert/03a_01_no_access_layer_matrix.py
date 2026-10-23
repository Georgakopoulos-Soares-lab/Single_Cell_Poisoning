"""Step 03a_01 — every defence layer against an attacker who can only SUBMIT.

# No `from __future__ import annotations` here: snakemake's script directive
# prepends a preamble, which would push it off line 1 and make it a SyntaxError.

The companion to ``03b_01``, and the arm where defences actually work. v01
measured the three record-property layers taking every insert attack under 5%
-- ``relabel`` 96.1 to 0.6, ``duplicate`` 29.0 to 4.5, ``flood`` 82.4 to 0.6,
``interpolate`` 8.8 to 0.6 -- on four celltypes, under exact search, with the
ingestion cap and the donor column both unusable. This re-measures it on 26
celltypes, graph-faithful, with two corrections.

**THE LAYERS PARTITION THE ATTACK SPACE; THEY DO NOT OVERLAP.** Each attack is
caught by exactly one layer and the other two do nothing, and on v01's numbers
``stack - min(solos)`` is exactly 0.0 on all four attacks -- composition is
precisely ``min``. That is not a weakness. An insertion attacker must fake the
vector, fake the pairing of vector to label, or repeat themselves, and there is
one cheap orthogonal check on each. ``stack_minus_min`` is written as a derived
column so the claim is a number rather than an inspection.

**THE COST IS THE RESULT, NOT A FOOTNOTE.** v01's four celltypes have a
genuine-record rejection rate of 0.035 to 0.115 under label consistency; over
the 26 attacked celltypes the median is 0.586. The layer that carries
``relabel`` -- the only one that does -- was measured where it is nearly free.
Every row here reports the benefit and the cost side by side.

**AND THE THRESHOLD IS A BUDGET.** ``lc_budget`` is the bottom fraction of
GENUINE retrieved records the defender chooses to sacrifice; the threshold is
derived from the global distribution and reported. On this reference a 1%
budget sets it near 0.02, which is above every measured pre-ingestion poison
score for ``relabel``, ``duplicate`` and ``flood`` -- all of which sit at the
0.55th percentile of genuine. If that holds at 26 celltypes the headline is
that the layer costs 1% of the reference, not v01's 13.7%.

**WHAT DIFFERS FROM THE MODIFY ARM, and it is not a detail.** There, editing
records hands the attacker REAL donor ids -- 437 of them from 9,421 edits -- so
``splits`` is a consequence rather than a choice. Here the attacker invents
donor ids at no cost, so ``splits`` is a swept axis, and v01 puts the sign flip
at roughly eight identities: below it a per-donor cap helps, above it the cap
strips the genuine majority (a real neighbourhood holds only 12-17 distinct
donors) while the attacker keeps their quota. That sweep lives in ``03a_02``;
this step fixes ``splits`` at 1 so the matrix has one donor regime, and says so
in every row.

**Poison label-consistency is scored against the CLEAN reference**, which is
the defender screening a submission against the database as it stands. It is
computed only for the poison rows that are actually RETRIEVED -- a poison
record no query reaches cannot influence a vote, and scoring the whole
submission would cost a kNN over hundreds of thousands of vectors per
condition for rows that never appear. ``02b_30`` puts retrieved poison at 0.78
slots per query, so the set is small. The assumption this encodes is batch
screening before ingestion; ``03a_03`` sweeps the ingestion fraction, where
``duplicate``'s copies start corroborating each other.
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
from celltype_sweep import (Bench, seed_of, write,  # noqa: E402
                            write_by_donor)
from dataset_config import cache_root, load  # noqa: E402
from defended_sweep import evaluate_defense  # noqa: E402
from defense import DefenseConfig  # noqa: E402
from modify import knn_of_vectors, reference_knn_rows  # noqa: E402
import retrieval_store  # noqa: E402
from schemas import (ACCESS, GEOMETRY, K_NEIGHBORS, LABEL_CHANGE,  # noqa: E402
                     POISON_EMB, POISON_LABEL, SELECTION, THREAT_MODEL,
                     Access, ThreatModel)
from provenance import emit  # noqa: E402
from vote import TIE_BREAK  # noqa: E402

BASE = _HERE.parents[2]
CFG = load(BASE)
DS = CFG["dataset"]
CACHE = cache_root(BASE, CFG)
TAB = BASE / "2_results" / DS / "03_a_defence_vs_insert"
OUT = TAB / "03a_01_no_access_layer_matrix.parquet"
OUT_BY_DONOR = TAB / "03a_01_no_access_layer_matrix_by_donor.parquet"
IDX = BASE.parent / CFG["paths"]["reference_dir"] / "annotation" / "labelled_kNN.bin"

#: TWO rungs, not four. At 170 s per condition a four-dose grid over 26
#: celltypes and four attacks is ~40 hours; these two span the range where the
#: insert attacks actually work (02b_12: relabel 38.9% at 0.25, 39.3% at 1.0,
#: with the above-parity jump at 1.25 needing a budget larger than the celltype
#: itself). v01's matrix started at 0.05, where the undefended attack reaches
#: 1.2% and there is nothing to defend.
#:
#: Four rungs. The insert arm's dose response is the OPPOSITE shape to the
#: modify arm's: 02b_12 puts ``relabel`` at 1.1% net attack success at dose
#: 0.002 and 38.9% only by 0.25, with the jump above parity at 1.25. v01's
#: defence matrix started at 0.05, where the undefended attack reaches just
#: 1.2% and there is nothing for a defence to do; its dramatic numbers all come
#: from dose 2.0. This ladder spans the range where the attack actually works.
#: The dose ladder. 0.25 and 1.0 by default -- 02b_03 puts the attack's
#: plateau between them and 03a_01's own numbers agree, giving identical
#: undefended rates at both for every attack.
#:
#: OVERRIDABLE because two rungs cannot draw a curve, and the attack ladder was
#: run at 14 doses precisely so the defence could be measured along it. Setting
#: BIORAG_DOSES runs the SAME step at other rungs and tags the output, so the
#: pieces are one step's family rather than a second step that would have to
#: duplicate this file's loop and could drift from it. Added 2026-09-30.
DOSES = tuple(float(x) for x in
              os.environ.get("BIORAG_DOSES", "0.25,1.0").split(","))
#: Set when the ladder is not the default, so a run at other rungs cannot
#: overwrite the committed table.
#: Any axis override tags the output, so a partial-grid run can never
#: overwrite the committed default-grid table. The tag names the run, not the
#: axis, because three axes can move together.
_ov = [k for k in ("BIORAG_ATTACKS", "BIORAG_SELECTIONS", "BIORAG_DOSES",
                   "BIORAG_FLOOD_ANCHORS", "BIORAG_LABEL_CHANGE")
       if k in os.environ]
#: BIORAG_TAG names the run explicitly and is preferred. The fallback keeps
#: "dose" for a DOSES-only override so the committed
#: 03a_01_no_access_layer_matrix_dose.parquet stays reproducible from the same
#: invocation that produced it.
DOSE_TAG = ("" if not _ov else
            "_" + os.environ.get("BIORAG_TAG",
                                 "dose" if _ov == ["BIORAG_DOSES"]
                                 else "x".join(k.split("_")[1][:3].lower()
                                               for k in _ov)))

#: The four insert attacks, at the rung each occupies. ``relabel`` and
#: ``duplicate`` submit VERBATIM reference embeddings (rung 0, vectors genuine,
#: only the label forged); ``interpolate`` and ``flood`` write coordinates no
#: count matrix produces (rung 1). That split is exactly what ``recompute``
#: gates on, and it is why the layer reads 0.0 on the first two and near-total
#: on the last two.
#: OVERRIDABLE, like DOSES below. Full parity with the attack grid needs each
#: attack at the SELECTION 02b_01/02b_02 ran it under -- relabel at hub and
#: uniform, duplicate and flood at concentrated, interpolate at uniform -- and
#: those are not a cross product, so they are driven one combination per run.
#: Added 2026-10-01.
ATTACKS = tuple(os.environ.get("BIORAG_ATTACKS",
                               "relabel,duplicate,interpolate,flood").split(","))
VERBATIM = {"relabel": True, "duplicate": True,
            "interpolate": False, "flood": False}

#: HUB ONLY. 02b_01 already characterises the selection axis on the attack
#: side -- 02b_05 measures the hub-vs-random gap at ~190x -- so repeating it
#: here would double a 20-hour grid to re-derive a known result. The defence
#: question is whether the layers catch the attack, and testing them against
#: the STRONGEST selection is the conservative direction: a layer that holds
#: under hub holds under uniform, and one that fails under hub has failed.
#: HUB by default and overridable. Hub is the STRONGEST selection -- 02b_05
#: puts the hub-vs-random gap at ~190x -- so a layer that holds under hub holds
#: under the others, which is why the matrix was measured there first. Parity
#: with the attack grid needs uniform and concentrated as well.
SELECTIONS = tuple(os.environ.get("BIORAG_SELECTIONS", "hub").split(","))

#: Flood's geometry. ``build_attack`` refuses to default these, correctly --
#: 02b_17 measures the committed default of spread 0.99 delivering 0.2894
#: against 0.4732 at 0.999, so every flood number in the project was produced
#: at 61% strength. This uses the MEASURED optimum from 02b_13's grid, which is
#: spread 0.999 with NO jitter (1.0 = no displacement; jitter never helps at
#: any level). Defending a weakened flood would overstate the defence.
#: OVERRIDABLE as of 2026-10-01, and the largest untested axis in the grid.
#: 02b_02 swept n_anchors over 64/256/1024/4096/16384; this step fixed it at
#: the strongest, and CLAUDE.md records the setting moving net attack success
#: from 0.024 to 0.400 -- a 16x swing. Note `selection` is NOT the flood's
#: axis: build_flood takes none, so a flood at "concentrated" is identical to
#: one at "hub". Parity for this attack is n_anchors, not selection.
FLOOD_ANCHORS = int(os.environ.get("BIORAG_FLOOD_ANCHORS", 16_384))
#: ``relabel`` (default) forges the target label; ``keep`` submits the poison
#: under its own SOURCE label -- the attack grid's volume and geometry
#: controls (SPEC.md §5.4). OVERRIDABLE as of 2026-10-04 so the defence covers
#: them. Under ``keep`` every poisoned slot votes with the poison's own label,
#: never the target, so the vote and label consistency read POISON_LABEL
#: rather than assuming ``target``.
LABEL_CHANGE_RUN = os.environ.get("BIORAG_LABEL_CHANGE", "relabel")
if LABEL_CHANGE_RUN not in ("relabel", "keep"):
    raise ValueError(f"BIORAG_LABEL_CHANGE={LABEL_CHANGE_RUN!r}; expected relabel or keep")
FLOOD_SPREAD = 0.999
FLOOD_JITTER = 1.0

LC_BUDGETS = (0.01, 0.05, 0.10)
DONOR_CAPS = (1, 2, 5)

#: Fabricated attacker identities. FIXED at 1 here so the matrix has one donor
#: regime; 03a_02 sweeps it, because it is the axis on which donor_cap inverts.
SPLITS = 1

SEED = 0
SHARD = int(os.environ.get("BIORAG_SHARD", 0))
N_SHARDS = int(os.environ.get("BIORAG_N_SHARDS", 1))
PREPARE = os.environ.get("BIORAG_PREPARE", "") == "1"
if not 0 <= SHARD < N_SHARDS:
    raise ValueError(f"BIORAG_SHARD={SHARD} out of range for {N_SHARDS} shards")
if DOSE_TAG or N_SHARDS > 1:
    suffix = DOSE_TAG + (f"_shard{SHARD}" if N_SHARDS > 1 else "")
    OUT = OUT.with_name(OUT.stem + suffix + OUT.suffix)
    OUT_BY_DONOR = OUT_BY_DONOR.with_name(
        OUT_BY_DONOR.stem + suffix + OUT_BY_DONOR.suffix)


def configs_budget_free() -> list[tuple[str, DefenseConfig]]:
    """Configurations whose result does not depend on ``lc_budget``.

    Split from the budget-dependent set because only ``label_consistency`` and
    the stack read a threshold; running the others once per budget would
    triple their cost for three identical answers.
    """
    out: list[tuple[str, DefenseConfig]] = [
        ("none", DefenseConfig()),
        ("recompute", DefenseConfig(recompute=True)),
        ("dedup", DefenseConfig(dedup_tau=0.9999)),
    ]
    out += [(f"donor_cap{c}", DefenseConfig(donor_cap=c)) for c in DONOR_CAPS]
    return out


def configs_budget_dep(lc_threshold: float) -> list[tuple[str, DefenseConfig]]:
    """The two configurations that read the label-consistency threshold."""
    return [
        ("label_consistency",
         DefenseConfig(label_consistency=True, lc_threshold=lc_threshold)),
        ("stack", DefenseConfig(recompute=True, label_consistency=True,
                                lc_threshold=lc_threshold, dedup_tau=0.9999)),
    ]


def main() -> None:
    t0 = time.time()
    emit(__file__)
    b = Bench.load(CFG, BASE, t0)

    # Donor ids as INT CODES: the object array of 7.9M strings costs 5.95 GB to
    # load and 16.8 GB cast, which alone decides how many shards fit. donor_cap
    # only tests identity, so codes are equivalent.
    prov = np.load(CACHE / "reference_provenance.npz", allow_pickle=True)
    ref_donor = pd.factorize(prov["sample"])[0].astype(np.int32)
    n_real_donors = int(ref_donor.max()) + 1
    del prov

    knn_path = CACHE / "reference_knn_retrieved.npz"
    ref_knn: dict[int, np.ndarray] = {}
    if knn_path.exists():
        z = np.load(knn_path)
        ref_knn = {int(r): nb for r, nb in zip(z["idx"], z["nbr"])}

    # The GENUINE side's label consistency never changes here -- an insertion
    # attacker rewrites no existing label -- so it is computed ONCE for the
    # whole clean neighbourhood rather than per condition. That is the opposite
    # of 03b_01, where the attacker's edits change what the neighbours say and
    # the score has to be recomputed every time.
    clean_rows = np.unique(b.clean_idx)
    need = np.setdiff1d(clean_rows,
                        np.fromiter(ref_knn, dtype=np.int64, count=len(ref_knn)))
    if PREPARE:
        if len(need):
            t_k = time.time()
            got = reference_knn_rows(str(IDX), b.ref_emb, need, b.n_ref, b.ef,
                                     k=K_NEIGHBORS)
            ref_knn.update({int(r): nb for r, nb in zip(need, got)})
            idx_all = np.fromiter(ref_knn, dtype=np.int64, count=len(ref_knn))
            order = np.argsort(idx_all)
            np.savez(knn_path, idx=idx_all[order],
                     nbr=np.stack([ref_knn[int(r)] for r in idx_all[order]]
                                  ).astype(np.int32))
            print(f"[{time.time()-t0:.0f}s] kNN top-up: {len(need):,} rows in "
                  f"{time.time()-t_k:.0f}s", file=sys.stderr)
        print(f"[{time.time()-t0:.0f}s] PREPARE done", file=sys.stderr)
        return
    if len(need):
        sys.exit(f"{len(need):,} retrieved records have no cached kNN. Run "
                 f"BIORAG_PREPARE=1 once first.")

    genuine_lc_of = {int(r): float((b.ref_label[ref_knn[int(r)]]
                                    == b.ref_label[int(r)]).mean())
                     for r in clean_rows}
    g_lc = np.fromiter(genuine_lc_of.values(), float, len(genuine_lc_of))
    thr_of = {q: float(np.quantile(g_lc, q)) for q in LC_BUDGETS}
    print(f"[{time.time()-t0:.0f}s] genuine LC over {len(g_lc):,} rows; "
          + ", ".join(f"budget {q:.0%} -> {t:.4f}" for q, t in thr_of.items()),
          file=sys.stderr)

    sources = list(b.sources)[SHARD::N_SHARDS]
    n_cond = len(sources) * len(ATTACKS) * len(SELECTIONS) * len(DOSES)
    # GLOBAL row count, not this shard's. merge_shards.py refuses a shard set
    # whose merged length does not equal it -- the check that catches a shard
    # which died halfway and would otherwise look like a complete table with
    # fewer rows.
    n_rows_global = (len(b.sources) * len(ATTACKS) * len(SELECTIONS)
                     * len(DOSES)
                     * (len(configs_budget_free())
                        + len(configs_budget_dep(0.5)) * len(LC_BUDGETS)))
    print(f"[{time.time()-t0:.0f}s] {n_cond} conditions; "
          f"{len(configs_budget_free())} budget-free + "
          f"{len(configs_budget_dep(0.5))} x {len(LC_BUDGETS)} budget-dependent",
          file=sys.stderr)

    rows_out: list[dict] = []
    donor_frames: list[pd.DataFrame] = []
    done = 0
    for source in sources:
        target, odist = b.target_of[source]
        b.clean_target(target)      # warms the per-target cache
        for attack in ATTACKS:
            for selection in SELECTIONS:
                for rate in DOSES:
                    t1 = time.time()
                    cond_seed = seed_of(source, rate, SEED)
                    poison = build_attack(
                        attack, ref_emb=b.ref_emb, ref_labels=b.ref_label,
                        ref_ids=b.ref_id, source=source, target=target,
                        rate=rate, seed=cond_seed, selection=selection,
                        hub_nbr=b.clean_idx, label_change=LABEL_CHANGE_RUN,
                        **({"n_anchors": FLOOD_ANCHORS,
                            "spread": FLOOD_SPREAD,
                            "jitter": FLOOD_JITTER}
                           if attack == "flood" else {}))
                    pe = poison[POISON_EMB]
                    # Each poison record's OWN label: the target under relabel,
                    # its source under keep. Never assume ``target``.
                    p_label = np.asarray(poison[POISON_LABEL]).astype(object)
                    ret = b.retrieve(pe, target)
                    ids, is_p = ret.ids, ret.is_poison
                    if retrieval_store.enabled():
                        retrieval_store.save(
                            CACHE, "03a_01", DOSE_TAG.lstrip("_"),
                            {"source": source, "attack": attack,
                             "selection": selection, "rate": rate,
                             "label_change": LABEL_CHANGE_RUN,
                             "n_anchors": (FLOOD_ANCHORS if attack == "flood"
                                           else "na")},
                            ids=ids, cos=ret.topk_cos,
                            meta={"condition_seed": int(cond_seed),
                                  "target": target, "n_poison": int(len(pe)),
                                  "poison_digest": retrieval_store.digest(pe),
                                  "n_ref": int(b.n_ref), "k": K_NEIGHBORS,
                                  "ef": int(b.ef), "seed": SEED,
                                  "flood_spread": (FLOOD_SPREAD if attack == "flood" else None),
                                  "flood_jitter": (FLOOD_JITTER if attack == "flood" else None)})
                    t_ret = time.time() - t1

                    labels = b.ref_label[np.where(is_p, 0, ids)].astype(object)
                    labels[is_p] = p_label[ids[is_p] - b.n_ref]
                    # Poison vectors live at ids >= n_ref, indexing into pe.
                    emb_k = np.where(is_p[:, :, None],
                                     pe[np.where(is_p, ids - b.n_ref, 0)],
                                     b.ref_emb[np.where(is_p, 0, ids)])
                    # Fabricated donor ids, offset past every real one so a
                    # poisoned record can never collide with a genuine donor.
                    rng = np.random.default_rng(cond_seed)
                    p_donor = n_real_donors + rng.integers(
                        0, SPLITS, size=len(pe)).astype(np.int32)
                    donor_k = np.where(is_p,
                                       p_donor[np.where(is_p, ids - b.n_ref, 0)],
                                       ref_donor[np.where(is_p, 0, ids)])
                    recompute_ok = (np.ones(labels.shape, bool)
                                    if VERBATIM[attack] else ~is_p)

                    # Label consistency for everything RETRIEVED, against
                    # the CLEAN reference, in ONE index load.
                    #
                    # Two sets need it. The retrieved POISON: a submitted
                    # record no query reaches cannot move a vote, and scoring
                    # the whole submission would cost a kNN over hundreds of
                    # thousands of vectors per condition. And the GENUINE rows
                    # the attacked graph reached that the clean one did not --
                    # inserting poison rewires traversal, so the attacked
                    # neighbourhood is not a subset of the clean one. Caching
                    # only the clean rows and defaulting the rest to
                    # "consistent" would ADMIT records the defence never
                    # examined, which is the defect this step exists to avoid
                    # reporting around.
                    t2 = time.time()
                    p_rows = (np.unique(ids[is_p] - b.n_ref) if is_p.any()
                              else np.empty(0, np.int64))
                    g_rows = np.unique(ids[~is_p])
                    g_miss = np.array(
                        [r for r in g_rows if int(r) not in genuine_lc_of],
                        dtype=np.int64)
                    if len(p_rows) or len(g_miss):
                        vecs = np.concatenate(
                            [pe[p_rows], b.ref_emb[g_miss]]).astype(np.float32)
                        selfr = np.concatenate(
                            [np.full(len(p_rows), -1, np.int64), g_miss])
                        nb = knn_of_vectors(str(IDX), vecs, b.n_ref, b.ef,
                                            k=K_NEIGHBORS, self_rows=selfr)
                        if len(p_rows):
                            p_lc = (b.ref_label[nb[:len(p_rows)]]
                                    == p_label[p_rows][:, None]).mean(axis=1)
                        for i, gr in enumerate(g_miss):
                            genuine_lc_of[int(gr)] = float(
                                (b.ref_label[nb[len(p_rows) + i]]
                                 == b.ref_label[int(gr)]).mean())
                    lc_k = np.ones(labels.shape)
                    if len(p_rows):
                        pos = np.searchsorted(p_rows, ids - b.n_ref)
                        lc_k = np.where(
                            is_p, p_lc[np.clip(pos, 0, len(p_rows) - 1)], 1.0)
                    gi = ~is_p
                    lc_k[gi] = np.fromiter(
                        (genuine_lc_of[int(v)] for v in ids[gi]), float,
                        int(gi.sum()))
                    t_lc = time.time() - t2

                    qi = b.qi_of[source]
                    plan = [(float("nan"), float("nan"), n, c)
                            for n, c in configs_budget_free()]
                    for budget in LC_BUDGETS:
                        plan += [(budget, thr_of[budget], n, c)
                                 for n, c in configs_budget_dep(thr_of[budget])]

                    for budget, thr, name, cfg in plan:
                        res = evaluate_defense(
                            labels=labels, is_poison=is_p, y_true=b.truth,
                            target=target, cfg=cfg, recompute_ok=recompute_ok,
                            label_consistency=lc_k, donor=donor_k,
                            cos=ret.topk_cos, emb=emb_k, per_query=True)
                        got = np.asarray(res["hit"], float)
                        ok = np.asarray(res["accurate"], float)  # accuracy under attack, not "correct" (poison removed)
                        ind = {"accuracy": ok, "baseline_accuracy": b.clean_ok,
                               "accuracy_delta": ok - b.clean_ok,
                               "attack_success": got,
                               "retrieval": is_p.any(axis=1).astype(float),
                               "poison_in_topk": is_p.sum(axis=1).astype(float)}
                        # Bench.per_donor, not a local rebuild: it names the
                        # columns exactly as every attack table names them, so
                        # the defended and undefended donor frames concatenate
                        # and a paired bootstrap can be taken across them.
                        donor_frames.append(
                            b.per_donor(ind, qi).assign(
                                defense=name, lc_budget=budget, source=source,
                                rate=rate, attack=attack, selection=selection,
                                seed=SEED, condition_seed=cond_seed))
                        rows_out.append({
                            THREAT_MODEL: str(ThreatModel.NO_ACCESS),
                            ACCESS: str(Access.INSERT),
                            GEOMETRY: str(poison.get("geometry", "")),
                            LABEL_CHANGE: LABEL_CHANGE_RUN,
                            SELECTION: selection, "attack": attack,
                            "retrieval_path": "hnsw", "ef_search": b.ef,
                            "k": K_NEIGHBORS, "tie_break": TIE_BREAK,
                            "defense": name, "lc_budget": budget,
                            "lc_threshold": thr, "splits": SPLITS,
                            "source": source, "target": target,
                            "ontology_dist": odist, "rate": rate,
                            "dose_basis": "celltype", "seed": SEED,
                            "condition_seed": cond_seed,
                            "n_conditions": n_rows_global,
                            "n_anchors": (FLOOD_ANCHORS if attack == "flood" else None),
                            # The flood's geometry, so a reader cannot mistake this
                            # spread-0.999 flood for 02b_02's spread-0.99 one.
                            "flood_spread": (FLOOD_SPREAD if attack == "flood" else None),
                            "flood_jitter": (FLOOD_JITTER if attack == "flood" else None),
                            "n_poison": int(len(pe)),
                            "n_poison_retrieved": int(len(p_rows)),
                            "within_accuracy": float(ok[qi].mean()),
                            "within_baseline_accuracy": float(b.clean_ok[qi].mean()),
                            "within_attack_success": float(got[qi].mean()),
                            "within_retrieval": float(
                                is_p[qi].any(axis=1).mean()),
                            "global_accuracy": float(ok.mean()),
                            "global_attack_success": float(got.mean()),
                            "false_positive": float(res["false_positive"]),
                            "no_survivors": float(res["abstention"]),
                            "clean_accuracy_defended": float(res["clean_accuracy"]),
                            "seconds": round(time.time() - t1, 1),
                        })
                    done += 1
                    print(f"[{time.time()-t0:7.0f}s] {done}/{n_cond} {source} "
                          f"{attack}/{selection} dose {rate:<6g} "
                          f"n_p {len(pe):>8,} retr {len(p_rows):>6,} "
                          f"(ret {t_ret:.0f}s lc {t_lc:.0f}s)", file=sys.stderr)
                    del poison, pe, emb_k, labels

    TAB.mkdir(parents=True, exist_ok=True)
    write(rows_out, OUT)
    write_by_donor(donor_frames, OUT_BY_DONOR)


if __name__ == "__main__":
    main()
