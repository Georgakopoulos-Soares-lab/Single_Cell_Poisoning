"""Step 03b_01 — every defence layer against an attacker who can OVERWRITE.

# No `from __future__ import annotations` here: snakemake's script directive
# prepends a preamble, which would push it off line 1 and make it a SyntaxError.

**This step exists to report a null, and the null is the headline.** v01
measured the same six configurations against `relabel_inplace` on four
celltypes and found the best layer worth about -2 pp, with 45 of 130
(celltype, defence) combinations NET HARMFUL. Nothing has been re-measured
under v02's graph retrieval, on 26 celltypes, with the unified vote. If the
null survives that, it is the strongest defence result the project has, and
CLAUDE.md 1 requires it be reported with the same weight as any success.

**Why every layer is expected to fail here, stated in advance so the result is
falsifiable rather than fitted.** An in-place edit leaves a GENUINE reference
record in place:

``recompute``   the vector is real, so re-deriving it from counts reproduces
                it. v01 measured max |diff| = 0.0000000000 across 1,248
                matched conditions on ``relabel_inplace``. Structural, not
                incidental -- the layer inspects the vector and the attack is
                in the label. It should bite on ``shifted``/``synthesised``,
                whose replacements no count matrix produces.
``dedup``       no twin is created, so there is nothing to collapse. Its only
                movement in v01 came from collapsing GENUINE near-duplicates,
                of which the shipped reference already holds 6.5%.
``donor_cap``   the attacker INHERITS real sample ids with the records they
                edit -- editing the top 192 B-cell hub records hands them 30
                distinct genuine donors without fabricating anything. So
                ``splits`` is not a free parameter on this arm; it is a
                consequence of which records were selected, and it is reported
                as ``n_donor_edited`` rather than swept.
``label_cons``  the only layer with a mechanism, and it is a budget tax: it
                works at low dose and inverts once the attacker has relabelled
                enough of a neighbourhood that consensus shifts to THEIR term.
                v01 put the crossover at 1% of the celltype for NK, 5% for
                B cell, 10% for macrophage, 20% for classical monocyte.

**The comparison is PAIRED, and that is what makes the null quotable.**
``add_items`` is multi-threaded and order-dependent, so two insertions of the
same poison give two different graphs -- 02b_34 measures the band at up to
0.0168 on ``within_accuracy``. v01's modify-arm deltas were -2.4 to 0.0 pp,
which sits INSIDE that band. Retrieving once per condition and scoring every
configuration off that one neighbour list makes the defended-minus-undefended
difference exact rather than merely within the band: both sides see a
byte-identical top-k. This is CLAUDE.md 7's "one retrieval regime per
comparison" one layer up -- there it was two retrievers, here it would be two
insertions.

**The label-consistency threshold is set by a BUDGET, not by a constant.**
v01 used ``lc_threshold = 0.5``, which 3.9 itself flags as arbitrary and which
rejects anywhere from 3.5% to 100% of genuine records depending on celltype.
Here the swept parameter is ``lc_budget`` -- the bottom fraction of GENUINE
records the defender chooses to sacrifice -- and the threshold is derived from
the genuine score distribution as a reported column. Three reasons: it makes
the cost an input rather than a surprise, it removes a constant the attacker
partly controls, and it transfers to a second dataset, whose LC distribution
will differ so that 0.5 would not mean the same thing.

**Calibration is GLOBAL, over all genuine retrieved records, never
per-celltype.** Measured on this reference, eight of the fourteen scored
attacked celltypes have a 1st-percentile LC of exactly 0.00: "drop the worst
1% of this celltype" would reject nothing at all for them, while poison scores
0.0007-0.0125 before ingestion. Per-celltype calibration therefore hands the
attacker a menu -- pick the celltype with the most permissive threshold. Same
structure as the ``donor_cap``/``splits`` failure: the defender's knob has a
floor set by the data and the attacker chooses which floor to face.

Writes one row per (celltype, attack, dose, configuration) with the benefit and
the cost side by side, plus a by-donor frame for the paired bootstrap.
"""
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "00_common"))
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from celltype_sweep import (by_donor, eligible_sources,  # noqa: E402
                            pick_targets, seed_of, write, write_by_donor)
from dataset_config import cache_root, load, query_cache  # noqa: E402
from defended_sweep import evaluate_defense  # noqa: E402
import retrieval_store  # noqa: E402
from defense import DefenseConfig  # noqa: E402
from modify import (DEEP_K, build_modify, clean_graph_topk_ids,  # noqa: E402
                    graph_topk_ids, reference_knn_rows)
from selection import choose  # noqa: E402
from ontology import Ontology  # noqa: E402
from provenance import emit  # noqa: E402
from schemas import (ACCESS, GEOMETRY, K_NEIGHBORS, LABEL_CHANGE,  # noqa: E402
                     POISON_EMB, POISON_LABEL, SELECTION, THREAT_MODEL,
                     Geometry, LabelChange, Selection, ThreatModel)
from vote import TIE_BREAK, vote  # noqa: E402

BASE = _HERE.parents[2]
CFG = load(BASE)
DS = CFG["dataset"]
CACHE = cache_root(BASE, CFG)
QCACHE = query_cache(BASE, CFG)
TAB = BASE / "2_results" / DS / "03_b_defence_vs_modify"
OUT = TAB / "03b_01_with_access_layer_matrix.parquet"
OUT_BY_DONOR = TAB / "03b_01_with_access_layer_matrix_by_donor.parquet"

#: GRAPH ONLY, and there is no environment switch. 02b_11 offers ``exact``
#: because it predates the decision; this step does not, because CLAUDE.md 6
#: requires that no published number rest on the merge shortcut and nothing
#: here needs to. The cost is one index load per condition.
IDX = BASE.parent / CFG["paths"]["reference_dir"] / "annotation" / "labelled_kNN.bin"
EF = int(CFG.get("hnsw", {}).get("ef_search", 100))
#: No ``exact`` switch. 02b_11 offers one because it predates the decision;
#: CLAUDE.md 6 requires that no published number rest on the merge shortcut and
#: nothing here needs to. ``retrieval_path`` is written PER ROW, because the
#: two geometries genuinely take different paths -- see TYPES.

#: Four rungs, chosen from the measured dose response rather than for symmetry
#: with the insert arm. 02b_12 puts ``relabel_inplace`` at 13.7% net attack
#: success by dose 0.002 and SATURATED at 49.8% from 0.25 onward, so rungs
#: above a quarter buy the attacker nothing and would only cost retrievals.
#: The insert arm's ladder is the opposite shape and lives in 03a_01.
#: OVERRIDABLE for parity with the attack grid, which ran 02b_11 at 13 doses
#: against this step's 4. Added 2026-10-01.
DOSES = tuple(float(x) for x in
              os.environ.get("BIORAG_DOSES", "0.002,0.01,0.05,0.25").split(","))

#: (geometry, label_change, selection). ``unmoved`` is the rung-0 attack --
#: label only, vectors untouched -- and it is the one no v01 layer moved.
#:
#: THE TWO GEOMETRIES TAKE DIFFERENT RETRIEVAL PATHS, and this is not a
#: shortcut. ``unmoved`` moves no embedding, so the index is byte-identical to
#: the clean one and the clean neighbours ARE the attacked neighbours -- only
#: the labels read off them change. ``build_modify`` refuses the geometry for
#: exactly this reason. That makes the arm ``cached_clean``: one retrieval for
#: every condition, exact by construction, and -- since nothing is inserted --
#: with NO ``add_items`` non-determinism, so its numbers carry no band at all
#: while the geometry-editing rows carry CLAUDE.md 1's +/-0.02.
#:
#: The geometry types are carried so ``recompute`` has something it SHOULD
#: catch. A matrix reading zero in every cell cannot distinguish "the defence
#: failed" from "the harness is not wired up".
TYPES = (
    (Geometry.UNMOVED, LabelChange.RELABEL, Selection.HUB),
    (Geometry.UNMOVED, LabelChange.RELABEL, Selection.UNIFORM),
    (Geometry.SHIFTED, LabelChange.RELABEL, Selection.HUB),
    (Geometry.SYNTHESISED, LabelChange.RELABEL, Selection.HUB),
)

#: 02b_15's ladder puts ``shifted``'s peak at alpha 0.05 -- median retrieval
#: 0.8738 and net attack success 0.4055 on hub selection -- and its collapse at
#: 0.5, where retrieval is 0.0005 and the records are not near anyone. 02b_11
#: fixed 0.5 and reported the attack inert; that was one setting off the end of
#: its own ladder. Defending an attack at a setting where it does not work
#: would report a defence that was never tested.
ALPHA = 0.05

#: ``synthesised`` geometry's parameters. ``build_modify`` refuses to default
#: them -- correctly -- and this step CRASHED on 2026-09-29 after four
#: conditions because it passed only ``alpha_range``. The smoke check had
#: exercised ``unmoved`` and ``shifted`` but not this third type, which is
#: precisely the one that broke: a smoke check that covers some of the grid
#: tests some of the grid.
#:
#: Values match 02b_11, whose table this defends, so the defended and
#: undefended measurements describe the same attack.
N_ANCHORS, SPREAD, JITTER = 1024, 0.99, 1.0

#: The bottom fraction of GENUINE retrieved records the defender sacrifices.
#: 0.01 sets a threshold of about 0.02 on this reference, which is above every
#: measured pre-ingestion poison score for ``relabel``, ``duplicate`` and
#: ``flood`` (all at the 0.55th percentile of genuine). The larger budgets are
#: carried to show what the extra cost does or does not buy.
LC_BUDGETS = (0.01, 0.05, 0.10)

#: Per-donor vote caps. 0 is OFF -- ``defense.py`` gates on truthiness, so the
#: filter is skipped entirely, not applied with a cap of zero. 1 is the old
#: ``donor_vote``, kept as a rung of a continuum rather than a named layer so
#: the retracted case is visible beside its neighbours instead of standing
#: alone. Above ~12-17 the cap cannot bind on a genuine neighbourhood at all.
DONOR_CAPS = (1, 2, 5)

SEED = 0
SHARD = int(os.environ.get("BIORAG_SHARD", 0))
N_SHARDS = int(os.environ.get("BIORAG_N_SHARDS", 1))

#: Which retrieval path this process runs. The two arms have very different
#: memory footprints -- ``cached_clean`` never loads the index and peaks at
#: 10.7 GB, ``graph`` reloads it per condition and peaks at 19.9 GB -- so on a
#: 64 GB machine they admit 5 and 2 concurrent shards respectively. Running
#: them as separate phases is what makes that difference usable; a mixed shard
#: would have to be budgeted at the graph figure throughout.
#: ``BIORAG_PREPARE=1`` runs the shared, once-only work and exits: the clean
#: graph retrieval and the reference-kNN top-up. BOTH load the 9.3 GB index,
#: and five shards doing it at once would need 100 GB and would race to write
#: the same cache file. Run prepare, then the shards, which read what it left
#: and never open the index on the cached_clean arm at all.
PREPARE = os.environ.get("BIORAG_PREPARE", "") == "1"
PHASE = os.environ.get("BIORAG_PHASE", "all")
if PHASE not in ("all", "cached_clean", "graph"):
    raise ValueError(f"BIORAG_PHASE={PHASE!r}; expected all, cached_clean or graph")
if PHASE != "all":
    TYPES = tuple(x for x in TYPES
                  if (x[0] == Geometry.UNMOVED) == (PHASE == "cached_clean"))
#: Restrict TYPES to named "geometry:selection" pairs. The committed grid runs
#: shifted and synthesised at HUB only, but 02b_11 attacked both at hub AND
#: uniform, so parity needs the uniform arms that this step never had. Named
#: rather than crossed, because the attack grid's combinations are not a cross
#: product. Added 2026-10-01.
_want = os.environ.get("BIORAG_TYPES", "")
if _want:
    # CONSTRUCTS the requested combinations rather than filtering the default
    # list. 02b_11 attacked shifted and synthesised at BOTH hub and uniform,
    # but this step's default TYPES only ever carried their hub arms, so a
    # filter can return nothing for a combination that genuinely needs running.
    # The default list is left untouched so an unset run reproduces the
    # committed tables exactly.
    _geo = {str(g): g for g in Geometry}
    _sel = {str(s): s for s in Selection}
    _built = []
    # "geometry:selection[:label_change]"; label_change defaults to relabel.
    # keep (2026-10-04) moves the embedding but leaves the record's own label,
    # the attack 02b_11 runs as shifted_inplace/keep at hub and uniform.
    _lc = {str(c): c for c in LabelChange}
    for _p in _want.split(","):
        _parts = _p.split(":")
        _g, _s = _parts[0], (_parts[1] if len(_parts) > 1 else "")
        _c = _parts[2] if len(_parts) > 2 else str(LabelChange.RELABEL)
        if _g not in _geo or _s not in _sel or _c not in _lc or len(_parts) > 3:
            raise ValueError(f"BIORAG_TYPES: {_p!r} -- geometry must be one of "
                             f"{sorted(_geo)}, selection one of {sorted(_sel)}, "
                             f"optional label_change one of {sorted(_lc)}")
        _built.append((_geo[_g], _lc[_c], _sel[_s]))
    TYPES = tuple(_built)
    # PHASE still decides which retrieval path runs, so a request that crosses
    # the split would silently drop half. Refuse instead.
    _un = [x for x in TYPES if x[0] == Geometry.UNMOVED]
    if _un and len(_un) != len(TYPES):
        raise ValueError("BIORAG_TYPES mixes unmoved with moved geometries; "
                         "they take different retrieval paths -- run them as "
                         "separate BIORAG_PHASE invocations")
if not 0 <= SHARD < N_SHARDS:
    raise ValueError(f"BIORAG_SHARD={SHARD} out of range for {N_SHARDS} shards")
_GTAG = os.environ.get("BIORAG_TAG", "")
if N_SHARDS > 1 or PHASE != "all" or _GTAG:
    tag = f"_{PHASE}" if PHASE != "all" else ""
    tag += f"_{_GTAG}" if _GTAG else ""
    tag += f"_shard{SHARD}" if N_SHARDS > 1 else ""
    OUT = OUT.with_name(OUT.stem + tag + OUT.suffix)
    OUT_BY_DONOR = OUT_BY_DONOR.with_name(OUT_BY_DONOR.stem + tag + OUT_BY_DONOR.suffix)


def configs_budget_free() -> list[tuple[str, DefenseConfig]]:
    """Configurations whose result does not depend on ``lc_budget``.

    Split from the budget-dependent set because only ``label_consistency`` and
    the stack read a threshold. Running the other six once per budget would
    triple their cost for three identical answers -- 29 s against 87 s per
    condition, measured.

    Leave-one-out is deliberately ABSENT. On v01's insert arm
    ``stack - min(solos)`` is exactly 0.0 on all four attacks, so composition
    is precisely ``min`` and removing a layer returns its attack to the best of
    what remains -- fully predicted by the solo columns. The informative
    quantity is that difference itself, which this set already supports as a
    derived column, so four extra configurations would buy nothing.

    ``donor_cap`` appears at several rungs because it is the one layer whose
    sign flips: on the insert arm it inverts past roughly eight attacker
    identities. Here identities are inherited rather than invented, so the rung
    that matters is whichever one the edited records' own donor count lands on,
    which is reported per condition as ``n_donor_edited``.
    """
    out: list[tuple[str, DefenseConfig]] = [
        ("none", DefenseConfig()),
        ("recompute", DefenseConfig(recompute=True)),
        ("dedup", DefenseConfig(dedup_tau=0.9999)),
    ]
    out += [(f"donor_cap{c}", DefenseConfig(donor_cap=c)) for c in DONOR_CAPS]
    return out


def configs_budget_dep(lc_threshold: float) -> list[tuple[str, DefenseConfig]]:
    """Configurations that read the label-consistency threshold."""
    return [
        ("label_consistency",
         DefenseConfig(label_consistency=True, lc_threshold=lc_threshold)),
        ("stack", DefenseConfig(recompute=True, label_consistency=True,
                                lc_threshold=lc_threshold, dedup_tau=0.9999)),
    ]


def lc_for_rows(rows: np.ndarray, ref_knn: dict, labels: np.ndarray) -> np.ndarray:
    """Label-consistency of each reference row against its OWN neighbourhood.

    Recomputed rather than read from 01a_05, because the attacker has REWRITTEN
    labels and a record's consistency depends on the labels around it. A
    genuine record whose neighbours were all relabelled to the attacker's term
    becomes inconsistent through no fault of its own -- that is the mechanism
    behind 3.39's crossover, and reading a pre-attack score would hide it.

    Args:
        rows: (m,) reference row ids to score.
        ref_knn: row id -> (k,) its own nearest reference rows, self excluded.
        labels: (n_ref,) the CURRENT labels, post-edit.

    Returns:
        (m,) float64 in [0, 1]; 1.0 where the row has no cached neighbourhood,
        which admits it. Admitting an unscored record is the conservative
        direction for a DEFENCE paper: it can only understate the layer.
    """
    out = np.ones(len(rows), dtype=np.float64)
    for i, r in enumerate(rows):
        nb = ref_knn.get(int(r))
        if nb is not None:
            out[i] = float((labels[nb] == labels[r]).mean())
    return out


def main() -> None:
    t0 = time.time()
    emit(__file__)

    ref = np.load(CACHE / "reference.npz", allow_pickle=True)
    ref_emb, ref_lab, ref_id = ref["emb"], ref["label"], ref["cell_id"]
    n_ref = len(ref_lab)
    # Donor ids as INT CODES, not strings. ``sample`` is an object array of
    # 7.9M Python strings: 5.95 GB to load and 16.8 GB once cast with
    # ``.astype(str)``, which alone decides how many shards fit in 64 GB.
    # ``donor_cap`` only ever tests identity -- ``survivors`` uses the value as
    # a dict key -- so codes are equivalent, and ``test_defense`` pins the two
    # forms equal. 31.7 MB for 2,414 distinct donors.
    prov = np.load(CACHE / "reference_provenance.npz", allow_pickle=True)
    ref_donor = pd.factorize(prov["sample"])[0].astype(np.int32)
    del prov

    deep_path = QCACHE / "clean_neighbors_deep.npz"
    if not deep_path.exists():
        sys.exit(f"{deep_path} is missing; hub selection needs the depth-{DEEP_K} list.")
    deep_idx = np.load(deep_path)["idx"]

    q = np.load(QCACHE / "query_emb.npy")
    meta = pd.read_parquet(QCACHE / "query_meta.parquet")
    truth = meta["y_true"].to_numpy().astype(str)
    donor = meta["donor_id"].to_numpy()
    donors = np.array(sorted(pd.unique(donor)))

    # The reference's own kNN, for label consistency. v01 cached 170,398 rows
    # from EXACT retrieval; v02's graph reaches 195,972, so 95.8% of neighbour
    # slots are covered and the rest are filled here once.
    # The reference's own kNN, for label consistency. v01 cached 170,398 rows
    # from EXACT retrieval; v02's graph reaches 195,972, so ~14% of the
    # distinct records in a v02 neighbourhood had no entry and defaulted to
    # "consistent", which ADMITS them and understates the layer. Topped up
    # here against the rows this run will actually score, then persisted so
    # the next run pays nothing.
    knn_path = CACHE / "reference_knn_retrieved.npz"
    ref_knn: dict[int, np.ndarray] = {}
    if knn_path.exists():
        z = np.load(knn_path)
        ref_knn = {int(r): nb for r, nb in zip(z["idx"], z["nbr"])}
    print(f"[{time.time()-t0:.0f}s] reference kNN cache: {len(ref_knn):,} rows",
          file=sys.stderr)

    sources = eligible_sources(ref_lab, truth)
    onto = Ontology.from_obo(BASE.parent / CFG["ontology"]["obo"])
    target_of = pick_targets(sources, onto)
    sources = [s for s in sources if s in target_of]
    mine = sources[SHARD::N_SHARDS]

    # The clean side, through the SAME retriever, once per run rather than per
    # condition -- CLAUDE.md 7. Subtracting an exhaustive baseline from a graph
    # measurement makes every delta straddle two regimes, which 01a_02 bounds
    # at 0.110 on CL:0002062. 02a_01 pays this same cost at its lines 94-96.
    clean_path = QCACHE / "clean_graph_topk.npz"
    if PREPARE or not clean_path.exists():
        t_c = time.time()
        clean_nbr, clean_cos = clean_graph_topk_ids(str(IDX), q, n_ref, EF,
                                                    k=K_NEIGHBORS)
        np.savez(clean_path, idx=clean_nbr, cos=clean_cos)
        print(f"[{time.time()-t0:.0f}s] clean graph retrieval: "
              f"{time.time()-t_c:.0f}s -> {clean_path.name}", file=sys.stderr)
    else:
        z = np.load(clean_path)
        clean_nbr, clean_cos = z["idx"], z["cos"]
    clean_ok = (vote(ref_lab[clean_nbr], pivotal=truth).pred == truth).astype(float)
    hub_nbr = deep_idx[:, :K_NEIGHBORS]

    # Top up the kNN cache for every record this run can score. PREPARE ONLY:
    # it opens the index, and concurrent shards would both spike memory and
    # race on the same output file.
    need = np.setdiff1d(np.unique(clean_nbr),
                        np.fromiter(ref_knn, dtype=np.int64, count=len(ref_knn)))
    if PREPARE:
        if len(need):
            t_k = time.time()
            got = reference_knn_rows(str(IDX), ref_emb, need, n_ref, EF,
                                     k=K_NEIGHBORS)
            ref_knn.update({int(r): nb for r, nb in zip(need, got)})
            idx_all = np.fromiter(ref_knn, dtype=np.int64, count=len(ref_knn))
            order = np.argsort(idx_all)
            np.savez(knn_path, idx=idx_all[order],
                     nbr=np.stack([ref_knn[int(r)] for r in idx_all[order]]
                                  ).astype(np.int32))
            print(f"[{time.time()-t0:.0f}s] kNN top-up: {len(need):,} rows in "
                  f"{time.time()-t_k:.0f}s -> cache now {len(ref_knn):,}",
                  file=sys.stderr)
        print(f"[{time.time()-t0:.0f}s] PREPARE done; rerun without "
              f"BIORAG_PREPARE to sweep", file=sys.stderr)
        return
    if len(need):
        sys.exit(f"{len(need):,} retrieved records have no cached kNN. Run "
                 f"BIORAG_PREPARE=1 once first -- scoring them as 'consistent' "
                 f"would admit records the defence never actually examined.")

    n_cond = len(mine) * len(TYPES) * len(DOSES)
    # GLOBAL row count for this PHASE, not this shard's. merge_shards refuses a
    # shard set whose merged length does not equal it -- the check that catches
    # a shard which died halfway and would otherwise look like a complete table
    # with fewer rows. TYPES is already phase-filtered at module level, so this
    # is the count for the file this run writes. Its absence is why Run 37 had
    # to supply --expect by hand for both graph tables. Added 2026-09-30.
    n_rows_global = (len(sources) * len(TYPES) * len(DOSES)
                     * (len(configs_budget_free())
                        + len(configs_budget_dep(0.5)) * len(LC_BUDGETS)))
    print(f"[{time.time()-t0:.0f}s] {n_cond} conditions; "
          f"{len(configs_budget_free())} budget-free configs + "
          f"{len(configs_budget_dep(0.5))} x {len(LC_BUDGETS)} budget-dependent",
          file=sys.stderr)

    rows_out: list[dict] = []
    donor_frames: list[pd.DataFrame] = []
    done = 0
    for source in mine:
        target, odist = target_of[source]
        for geometry, label_change, selection in TYPES:
            for dose in DOSES:
                t1 = time.time()
                cond_seed = seed_of(source, dose, SEED)
                if geometry == Geometry.UNMOVED:
                    # cached_clean: relabel in place and re-read the SAME
                    # neighbour list. No index touched, so no insertion
                    # variance and nothing to reload.
                    path = "cached_clean"
                    pool = np.flatnonzero(ref_lab == source)
                    n_p = max(1, int(round(dose * len(pool))))
                    edited = choose(selection, pool, n_p,
                                    np.random.default_rng(cond_seed),
                                    hub_nbr=hub_nbr, n_ref=n_ref)
                    new_label = np.full(len(edited), target)
                    nbr, cos = clean_nbr, clean_cos
                    access = str(ThreatModel.WITH_ACCESS)
                else:
                    path = "graph"
                    p = build_modify(
                        ref_emb=ref_emb, ref_labels=ref_lab, ref_ids=ref_id,
                        source=source, target=target, dose=dose, seed=cond_seed,
                        selection=selection, geometry=geometry,
                        label_change=label_change,
                        alpha_range=(ALPHA, ALPHA), hub_nbr=hub_nbr,
                        **({"n_anchors": N_ANCHORS, "spread": SPREAD,
                            "jitter": JITTER}
                           if geometry == Geometry.SYNTHESISED else {}))
                    edited, new_label = p["rows"], p[POISON_LABEL]
                    access = p["access"]
                    # ONE retrieval. Every configuration below reads this same
                    # neighbour list, so a defended-minus-undefended difference
                    # carries no insertion variance at all.
                    nbr, cos = graph_topk_ids(str(IDX), edited, p[POISON_EMB],
                                              new_label, q, n_ref, EF,
                                              k=K_NEIGHBORS)
                    if retrieval_store.enabled():
                        retrieval_store.save(
                            CACHE, "03b_01", (f"{PHASE}_{_GTAG}" if _GTAG else PHASE),
                            {"source": source, "geometry": geometry,
                             "selection": selection, "label_change": label_change,
                             "rate": dose},
                            ids=nbr, cos=cos,
                            meta={"condition_seed": int(cond_seed),
                                  "target": target,
                                  "n_edited": int(len(edited)),
                                  "poison_digest": retrieval_store.digest(p[POISON_EMB]),
                                  "n_ref": int(n_ref), "k": K_NEIGHBORS,
                                  "ef": int(EF), "seed": SEED, "alpha": ALPHA,
                                  "flood_spread": (SPREAD if geometry == Geometry.SYNTHESISED else None),
                                  "flood_jitter": (JITTER if geometry == Geometry.SYNTHESISED else None),
                                  "n_anchors": (N_ANCHORS if geometry == Geometry.SYNTHESISED else None)},
                            extra={"edited": np.asarray(edited, np.int64),
                                   "new_label": np.asarray(new_label).astype(str)})

                post_lab = ref_lab.copy()
                post_lab[edited] = new_label
                is_edited = np.zeros(n_ref, bool)
                is_edited[edited] = True

                lab_k = post_lab[nbr]
                is_new = is_edited[nbr]
                # An in-place relabel leaves a genuine embedding, so the record
                # IS reproducible from real counts and re-encode admits it.
                # Only a geometry edit writes a vector no count matrix makes.
                recompute_ok = (np.ones(lab_k.shape, bool)
                                if geometry == Geometry.UNMOVED else ~is_new)
                donor_k = ref_donor[nbr]
                emb_k = ref_emb[nbr]

                uniq = np.unique(nbr)
                lc_u = lc_for_rows(uniq, ref_knn, post_lab)
                lc_uncached = float(np.mean([int(r) not in ref_knn for r in uniq]))
                lc_k = lc_u[np.searchsorted(uniq, nbr)]

                g_sel = np.arange(len(q))
                w_sel = np.flatnonzero(truth == source)
                n_donor_edited = int(len(np.unique(ref_donor[edited])))

                genuine_lc = lc_u[~is_edited[uniq]]
                plan: list[tuple[float, float, str, DefenseConfig]] = [
                    (float("nan"), float("nan"), n, c)
                    for n, c in configs_budget_free()
                ]
                for budget in LC_BUDGETS:
                    thr = float(np.quantile(genuine_lc, budget))
                    plan += [(budget, thr, n, c)
                             for n, c in configs_budget_dep(thr)]

                for budget, thr, name, cfg in plan:
                    r = evaluate_defense(
                        labels=lab_k, is_poison=is_new, y_true=truth,
                        target=target, cfg=cfg, recompute_ok=recompute_ok,
                        label_consistency=lc_k, donor=donor_k,
                        cos=cos, emb=emb_k, per_query=True)
                    got = np.asarray(r["hit"], float)
                    ok = np.asarray(r["accurate"], float)  # accuracy under attack, not "correct" (poison removed)
                    ind = {
                        "accuracy": ok, "baseline_accuracy": clean_ok,
                        "accuracy_delta": ok - clean_ok,
                        "retrieval": is_new.any(axis=1).astype(float),
                        "attack_success": got,
                    }
                    donor_frames.append(pd.DataFrame({
                        "donor_id": donors,
                        **by_donor("within", w_sel, donor, donors, **ind),
                        **by_donor("global", g_sel, donor, donors, **ind),
                    }).assign(defense=name, lc_budget=budget, source=source,
                              rate=dose, attack=f"{geometry}_inplace",
                              # selection is NOT redundant with attack: the
                              # cached_clean phase runs one geometry across two
                              # selections, so without it hub and uniform rows
                              # are indistinguishable on the natural key and
                              # condition_seed does not separate them either.
                              # Omitting it left all 194,688 rows of that
                              # table duplicated; 03a_09 recovers the pairing
                              # from row order and verifies it, which it should
                              # not have to do. Added 2026-09-30.
                              selection=str(selection),
                              seed=SEED, condition_seed=cond_seed))
                    rows_out.append({
                        THREAT_MODEL: str(ThreatModel.WITH_ACCESS),
                        ACCESS: access, GEOMETRY: str(geometry),
                        LABEL_CHANGE: str(label_change),
                        SELECTION: str(selection),
                        "attack": f"{geometry}_inplace",
                        "retrieval_path": path, "k": K_NEIGHBORS,
                        "tie_break": TIE_BREAK, "defense": name,
                        "lc_budget": budget, "lc_threshold": thr,
                        "source": source, "target": target,
                        "ontology_dist": odist, "rate": dose,
                        # geometry of a synthesised edit, named as 02b_11 names it
                        "flood_spread": (SPREAD if geometry == Geometry.SYNTHESISED else None),
                        "flood_jitter": (JITTER if geometry == Geometry.SYNTHESISED else None),
                        "dose_basis": "celltype", "seed": SEED,
                        "condition_seed": cond_seed,
                        "n_conditions": n_rows_global,
                        "alpha_lo": ALPHA if geometry != Geometry.UNMOVED else None,
                        "n_poison": int(len(edited)),
                        "n_donor_edited": n_donor_edited,
                        "lc_uncached": lc_uncached,
                        "within_accuracy": float(ok[w_sel].mean()),
                        "within_attack_success": float(got[w_sel].mean()),
                        "within_baseline_accuracy": float(clean_ok[w_sel].mean()),
                        "global_accuracy": float(ok.mean()),
                        "global_attack_success": float(got.mean()),
                        "global_baseline_accuracy": float(clean_ok.mean()),
                        # the COST side, without which the benefit is not a result
                        "false_positive": float(r["false_positive"]),
                        "no_survivors": float(r["abstention"]),
                        "clean_accuracy_defended": float(r["clean_accuracy"]),
                        "seconds": round(time.time() - t1, 1),
                    })
                done += 1
                print(f"[{time.time()-t0:7.0f}s] {done}/{n_cond} {source} "
                      f"{geometry}/{selection} dose {dose:<6g} "
                      f"n_p {len(edited):>8,} donors {n_donor_edited:>4} "
                      f"({time.time()-t1:.0f}s)", file=sys.stderr)

    TAB.mkdir(parents=True, exist_ok=True)
    write(rows_out, OUT)
    write_by_donor(donor_frames, OUT_BY_DONOR)


if __name__ == "__main__":
    main()
