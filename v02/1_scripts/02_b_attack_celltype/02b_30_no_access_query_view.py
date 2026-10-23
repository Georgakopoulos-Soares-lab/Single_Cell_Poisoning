"""Step 02b_30 — the same attacks, seen from the QUERY side: which cell was taken.

Every measurement in ``02_b`` so far describes the REFERENCE. ``02b_01`` inserts
poison, queries the graph, scores 49,456 queries into a rate, and throws the
per-query answer away; ``02b_06`` and ``02b_08`` map where the poison LANDS in
reference space. Nothing in ``v02`` records which query cell was taken, so no
figure can show the query dataset before and after — v01 had three (its fig17,
fig18, fig19) and v02 has no table to redraw them from.

This step is that table. It runs the SAME retrieval ``02b_01`` runs — poison
appended to the shipped ``labelled_kNN.bin`` and the graph re-queried at ef=100,
through ``Bench.retrieve`` — and keeps the per-query result instead of reducing
it. ``retrieval_path`` is ``hnsw`` on every row and is asserted before the table
is written; nothing here uses the merge shortcut.

**Both sides come from the graph.** ``y_pred_clean`` is ``vote()`` over the
clean HNSW neighbours ``01a_01`` committed at the same ef and k, which is the
same array ``Bench.load`` scores its clean baseline from. So a per-query
difference between the two prediction columns is an effect of the poison and not
of two retrievers (CLAUDE.md §7, "one retrieval regime per comparison").

WHAT IS HERE THAT 02b_01 CANNOT ANSWER
--------------------------------------
A pooled rate says how MANY queries moved. It cannot say WHICH, and at parity
that is the whole question. Read from
``02b_01_no_access_celltype_sweep_graph.parquet``, ``within_net_attack_success``
for ``relabel``/``unmoved``/``relabel`` on the two celltypes below:

                       dose 0.25   dose 1.0   dose 1.25
  CL:0000623  hub         0.8488     0.8515      0.9580
  CL:0000623  uniform     0.0048     0.8504      0.8972
  CL:0000057  hub         0.1659     0.1731      0.9832
  CL:0000057  uniform     0.0000     0.1779      0.8702

At dose 0.25 hub and uniform are far apart. At dose 1.0 they are level to within
0.005 on both celltypes.

**And that levelling is arithmetic, not a finding — measured here on
2026-09-28, after this step had run.** At a dose of exactly 1.0 the budget
equals the celltype, so every selection rule draws the whole of it: the two
poison sets are THE SAME SET, Jaccard 1.0000 on both celltypes, differing only
in the order the records are handed to ``add_items``. The selection axis is
degenerate at parity. It is not degenerate elsewhere — at 0.25 the two sets
overlap at Jaccard 0.1416, and at 1.25 at 0.7139, where uniform's
with-replacement draw covers only 134,519 of ``CL:0000623``'s 188,421 records
against hub's 188,421 (HANDOFF §3.32, "coverage beats volume").

This corrects the reasoning that chose conditions 3 and 7 below, not the rows
they produced. They remain in the table, and they are worth more than the
comparison they were picked for: they are the project's only REPLICATE of an
identical poison set through the graph, so the gap between them and conditions
2 and 6 measures the run-to-run noise floor directly rather than by inference.

What the per-query table still answers, and the pooled tables cannot: WHICH
queries a given attack takes. Two attacks with equal rates can capture
equal-sized, disjoint sets of cells, and a rate cannot tell them apart.

These four numbers are this pair of celltypes at these doses. They are not a
statement about the other 24 attackable celltypes, whose curves differ (CLAUDE.md
§1); ``02b_28`` carries the full per-celltype ladder.

THE CONDITIONS, AND WHY EACH
----------------------------
Eight, the approved budget, as two celltypes x four conditions. Held fixed
across all eight: attack ``relabel``, ``access=insert``, ``geometry=unmoved``,
``label_change=relabel``, one seed. Only the attacked celltype, the selection
and the dose vary, so a panel difference has one cause.

The two celltypes are chosen to behave oppositely under hub at sub-parity dose,
from the table above: ``CL:0000623`` (natural killer cell -> ``CL:0000084``
T cell) is already at 0.8488 by dose 0.25, while ``CL:0000057`` (fibroblast ->
``CL:0000097`` mast cell) is at 0.1659 despite being the larger celltype
(196,242 reference records against 188,421) and the better classified one
(clean within-celltype accuracy 0.9663 against 0.9236). Both figures are
``within_`` — over the queries whose own true label is the attacked celltype.

  1. ``hub`` at 0.25 — the working attack below parity, and the one dose where
     the two celltypes disagree most.
  2. ``hub`` at 1.0 — parity. Pooled, indistinguishable from (1) on both
     celltypes (0.8488 -> 0.8515, 0.1659 -> 0.1731). Per query, unknown.
  3. ``uniform`` at 1.0 — chosen as the selection contrast at the dose where
     selection stops mattering to the pooled rate. **It is not that**: at
     parity it draws the identical set (2) draws, so it is a REPLICATE of (2)
     under a different insertion order. Kept and read as one, because that is
     the only direct measurement of the noise floor in the project. See above.
  4. ``hub`` at 1.25 — above parity, where ``within_accuracy`` reaches exactly
     0.0000 on both celltypes and the ambiguous-tie rate collapses (0.5205 ->
     0.0016 on ``CL:0000623``). The vote is no longer a coin flip, so the
     before/after map is unambiguous rather than tie-broken.

``uniform`` at 0.25 is NOT run. It measures 0.0048 and 0.0000 — no query moves,
and the unmoved picture is already on every row as ``y_pred_clean``. In
hindsight it is the dose at which the selection axis is live (Jaccard 0.1416
against hub), so a step revisiting this should spend its budget there rather
than at parity.

Above parity (condition 4) the draw goes with replacement, so the poison covers
LESS of the celltype at more records (HANDOFF §3.32). It is a different attack
from (1)-(3), not more of the same one, and ``selection`` and ``rate`` on the
row say so.

THE QUERY-SPACE UMAP
--------------------
Fitted once on the 49,456 x 128 query embeddings and written as x/y per
``query_id``, so a figure joins it to the predictions on that column. Settings
match ``02b_06``/``02b_08`` — ``n_neighbors=15``, ``min_dist=0.30``, cosine,
``init="pca"``, ``random_state=0`` — so the two layouts are read with the same
eye, but they are NOT interchangeable: this one is fitted on QUERY cells and
``02b_08``'s on REFERENCE cells, and a coordinate in one means nothing in the
other. Nothing is projected into this layout; poison never enters it, because
poison is a reference record and this is the query dataset's own map.

``init="pca"``: the spectral initialiser fails on this data, as ``02b_06``
records — the cosine neighbour graph comes apart and the eigensolver reports too
small an eigengap, after which UMAP falls back to a random start.

THE TIE-BREAK DECIDES HALF OF THIS, AND THE TABLE SAYS SO
---------------------------------------------------------
At parity an inserted twin ties with the record it copied, the tally ties, and
``config.vote.tie_break`` -- the lowest CL id -- decides. ``02b_01`` reports
``within_ambiguous_rate_success`` 0.5205 for ``CL:0000623``/hub/1.0 and 0.6058
for ``CL:0000057``/hub/1.0: on more than half of the attacked celltype's
queries, the answer is a coin the tie-break flipped, not a result. HANDOFF §3.34
is the measurement; CLAUDE.md §7 is the rule -- publish the interval, not the
point.

A pooled table can express that as best/worst columns. A per-query table cannot,
because a single cell has no interval; what it has is a FLAG. So every row
carries ``n_tied`` and, per metric, whether the tie is what decided it:
``tie_decides_accuracy``, ``tie_decides_attack_success``, and the clean side's
own ``n_tied_clean`` / ``tie_decides_accuracy_clean``. A figure that colours a
cell by ``y_pred_poisoned`` without shading those cells is publishing one side
of a coin flip as a fact.

The flags are metric-specific and the count is not, deliberately: only a tie
CONTAINING the pivotal label is ambiguous. A three-way tie among labels that are
all wrong is already determined.

MIN_POISON_SCORE IS COMPUTED, NOT READ OFF THE GRAPH
----------------------------------------------------
``schemas.RAW_RESULT_DOC`` defines it as the score of the NEAREST poison whether
or not it entered the top-k. A graph index cannot answer that: it returns k
neighbours and says nothing about what it did not return, so
``Retrieved.nearest_topk_cos`` is ``-inf`` wherever no poison came back. The
column is therefore computed by brute force over the whole poison set
(:func:`nearest_poison_cosine`), which is a property of the poison and the query
and carries no retrieval regime at all. It is used for no ranking and enters no
vote; the merged-neighbour shortcut is not involved.

``poison_survives_qc`` keeps its schema name because the column is committed,
but read it as what it measures: whether the nearest poisoned record sits within
SCimilarity's search distance bar OF THIS QUERY. It is a query-to-poison
distance. SCimilarity ships no reference-side check of any kind — every filter it
has inspects the cells the caller brings — so nothing ever looks at the poison,
and that absence is the finding (CLAUDE.md §7, ``schemas.SCORE_QC``).

COST
----
Eight conditions, each paying a fresh load of the shipped 9.3 GB index because
hnswlib has no true removal; ~40-85 s apiece in ``02b_01``'s own ``seconds``
column for these celltypes. Peak memory ~15 GB. Approved in conversation
2026-09-28 for these eight conditions and no others.

Writes, under ``2_results/<dataset>/02_b_attack_celltype/``:
  ``02b_30_no_access_per_query.parquet``      8 x 49,456 rows
  ``02b_30_no_access_query_umap_xy.parquet``  49,456 rows
"""
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "00_common"))
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from attacks import build_attack  # noqa: E402
from celltype_sweep import KEY_COLUMNS, Bench, require_tiers, seed_of  # noqa: E402
from dataset_config import load, query_cache  # noqa: E402
from provenance import emit  # noqa: E402
from schemas import (ACCESS, ATTACK, ATTACK_TIER, DONOR_ID,  # noqa: E402
                     EMB_DTYPE, GEOMETRY, K_NEIGHBORS, LABEL_CHANGE,
                     MIN_POISON_SCORE, N_POISON_IN_TOPK, ONTOLOGY_DIST,
                     POISON_EMB, POISON_SURVIVES_QC, QUERY_ID,
                     RATE, RAW_RESULT_DTYPES, SCORE_QC, SEED, SELECTION,
                     THREAT_MODEL, Y_PRED_CLEAN, Y_PRED_POISONED, Y_TRUE,
                     Access, Attack, Geometry, LabelChange, Selection,
                     ThreatModel, scimilarity_score)
from vote import TIE_BREAK, VoteResult, vote  # noqa: E402

BASE = _HERE.parents[2]
CFG = load(BASE)
DS = CFG["dataset"]
OUT = BASE / "2_results" / DS / "02_b_attack_celltype"
OUT_PER_QUERY = OUT / "02b_30_no_access_per_query.parquet"
OUT_UMAP = OUT / "02b_30_no_access_query_umap_xy.parquet"

#: The regime every row of this step was retrieved under. Written as a column
#: AND asserted before the table lands, because a filename is not evidence and
#: the column is (``00_pipeline/check_retrieval.py``).
RETRIEVAL_PATH = "hnsw"

#: ``(source, selection, rate)``. See the docstring for why each one is here and
#: why ``uniform`` at 0.25 is not. Everything else about the attack is fixed.
CONDITIONS: tuple[tuple[str, str, float], ...] = (
    ("CL:0000623", Selection.HUB, 0.25),
    ("CL:0000623", Selection.HUB, 1.0),
    ("CL:0000623", Selection.UNIFORM, 1.0),
    ("CL:0000623", Selection.HUB, 1.25),
    ("CL:0000057", Selection.HUB, 0.25),
    ("CL:0000057", Selection.HUB, 1.0),
    ("CL:0000057", Selection.UNIFORM, 1.0),
    ("CL:0000057", Selection.HUB, 1.25),
)

#: Held fixed across all eight, so the only things that vary are the celltype,
#: the selection and the dose.
ATTACK_NAME = Attack.RELABEL
GEOMETRY_NAME = Geometry.UNMOVED
LABEL_CHANGE_NAME = LabelChange.RELABEL

#: One seed, matching ``02b_01``: its seed-to-seed spread was negligible beside
#: the donor-to-donor spread, and a second seed here costs another 8 index loads.
SEED_VALUE = 0

#: ``02b_06``/``02b_08`` settings, so the query map and the reference map are
#: read with the same eye. They are still different spaces -- see the docstring.
UMAP_SEED = 0
N_NEIGHBORS = 15
MIN_DIST = 0.30

#: Block sizes for :func:`nearest_poison_cosine`. The (q_block, p_block) cosine
#: matrix is the only large temporary and is discarded after each running max,
#: so peak memory is 4096 x 32768 x 4 B = 537 MB regardless of how big the
#: poison set is. Arithmetically inert.
Q_BLOCK = 4096
P_BLOCK = 32768


def nearest_poison_cosine(q: np.ndarray, pe: np.ndarray,
                          q_block: int = Q_BLOCK,
                          p_block: int = P_BLOCK) -> np.ndarray:
    """Cosine from each query to the nearest poisoned vector in the WHOLE set.

    Brute force, because the question has no approximate answer that means
    anything: the graph reports only what it returned, so a query with no poison
    in its top-k would otherwise carry no distance at all. Both inputs are unit
    normalised, so the dot product IS the cosine.

    Args:
        q: (n_q, 128) float32, unit norm.
        pe: (n_p, 128) float32, unit norm.
        q_block: queries per block. p_block: poison vectors per block.

    Returns:
        (n_q,) float32 in [-1, 1].

    Raises:
        ValueError: the poison set is empty, which would leave every query with
            no nearest poison and report the attack as costless.
    """
    if len(pe) == 0:
        raise ValueError("empty poison set — every query would have no nearest "
                         "poison and the condition would score as inert")
    out = np.full(len(q), -np.inf, dtype=EMB_DTYPE)
    for lo in range(0, len(q), q_block):
        hi = min(lo + q_block, len(q))
        for plo in range(0, len(pe), p_block):
            cos = (q[lo:hi] @ pe[plo:plo + p_block].T).astype(EMB_DTYPE)
            np.maximum(out[lo:hi], cos.max(axis=1), out=out[lo:hi])
    return out


def fit_query_layout(emb: np.ndarray) -> np.ndarray:
    """Fit the query dataset's own two-dimensional layout.

    ``umap`` is imported HERE rather than at module scope, as
    ``Bench.open_index`` imports ``hnswlib``: it pulls numba and costs seconds
    to import, and the test that exercises this module's table shapes has no
    use for it.

    Args:
        emb: (n_q, 128) float32, unit norm.

    Returns:
        (n_q, 2) float32 coordinates, one row per query in input order.
    """
    import umap  # noqa: PLC0415

    reducer = umap.UMAP(n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST,
                        n_components=2, metric="cosine", init="pca",
                        random_state=UMAP_SEED).fit(emb)
    return np.asarray(reducer.embedding_, dtype=EMB_DTYPE)


def umap_rows(meta: pd.DataFrame, xy: np.ndarray) -> pd.DataFrame:
    """One row per query: its identity and its coordinate in the query layout.

    Args:
        meta: query metadata, carrying ``query_id``, ``donor_id``, ``y_true``.
        xy: (n_q, 2) float32 from :func:`fit_query_layout`, same row order.

    Returns:
        (n_q, 9) frame. ``threat_model`` is on it because every analysis table
        carries one (``00_pipeline/check_provenance.py``); it names the arm this
        STEP runs, not a property of the layout, which no attacker touches.

    Raises:
        ValueError: coordinates and metadata differ in length.
    """
    if len(xy) != len(meta):
        raise ValueError(f"{len(xy):,} coordinates but {len(meta):,} queries")
    return pd.DataFrame({
        THREAT_MODEL: str(ThreatModel.NO_ACCESS),
        QUERY_ID: meta[QUERY_ID].to_numpy().astype(str),
        DONOR_ID: meta[DONOR_ID].to_numpy().astype(str),
        Y_TRUE: meta[Y_TRUE].to_numpy().astype(str),
        "x": xy[:, 0], "y": xy[:, 1],
        "umap_n_neighbors": np.int16(N_NEIGHBORS),
        "umap_min_dist": EMB_DTYPE(MIN_DIST),
        "umap_seed": np.int16(UMAP_SEED),
    })


def cast_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Narrow every ``raw_result`` column present to its contract dtype.

    Applied per condition AND again after the conditions are concatenated:
    concatenating two categoricals whose categories differ falls back to
    ``object``, so the written table would carry a different dtype from the
    frames it was built out of.

    ``rate`` is exempt (:data:`celltype_sweep.KEY_COLUMNS`) -- it reaches
    ``seed_of`` through ``repr()``, so a float32 round trip names a different
    poison set, and it has to join against ``02b_01``'s rate at the same width.

    Args:
        df: any frame carrying some of the ``raw_result`` columns.

    Returns:
        The same frame with those columns cast; others untouched.
    """
    return df.astype({c: t for c, t in RAW_RESULT_DTYPES.items()
                      if c in df and c not in KEY_COLUMNS})


def query_rows(meta: pd.DataFrame, clean: VoteResult, poisoned: VoteResult,
               hit: VoteResult, n_poison_in_topk: np.ndarray,
               near_cos: np.ndarray, keys: dict[str, object]) -> pd.DataFrame:
    """One condition's per-query outcomes, in the ``raw_result`` column shape.

    TIES ARE CARRIED, NOT RESOLVED AWAY. ``y_pred_poisoned`` is one reading of
    the vote -- the ``lowest_cl_id`` tie-break -- and at parity that reading
    decides about half the attacked celltype's queries (``02b_01`` reports
    ``within_ambiguous_rate_success`` 0.5205 on ``CL:0000623`` at dose 1.0).
    A figure drawn from the point verdict alone would publish one side of a
    coin flip as a fact, which is exactly what CLAUDE.md §7 forbids. So every
    row also says how many labels were tied and whether the tie is what decided
    it, for each metric and for the clean side too, and a figure can shade those
    cells instead of colouring them in.

    Args:
        meta: query metadata, carrying ``query_id``, ``donor_id``, ``y_true``.
        clean: the vote over the CLEAN graph neighbours, pivoted on the true
            label.
        poisoned: the vote after the poison was inserted, pivoted on the true
            label -- the accuracy reading of its ties.
        hit: the same poisoned neighbours voted again, pivoted on the
            attacker's target -- the attack-success reading. ``pred`` is
            identical to ``poisoned.pred`` by construction; only the ambiguity
            flag differs.
        n_poison_in_topk: (n_q,) int, poisoned records among the returned k.
        near_cos: (n_q,) float32 cosine to the nearest poison anywhere in the
            set, from :func:`nearest_poison_cosine`.
        keys: the condition's scalar identity columns, broadcast to every row.

    Returns:
        (n_q, len(keys) + 13) frame. ``rate`` is deliberately NOT narrowed to
        float32 (``celltype_sweep.KEY_COLUMNS``): it reaches ``seed_of`` through
        ``repr()``, so a float32 round trip yields a different key and a
        different poison set, and it has to join against ``02b_01``'s rate at
        the same width.

    Raises:
        ValueError: any per-query array disagrees with the metadata in length,
            or the two poisoned votes disagree on the prediction, which would
            mean they were taken over different neighbours.
    """
    n_q = len(meta)
    for name, arr in (("y_pred_clean", clean.pred),
                      ("y_pred_poisoned", poisoned.pred),
                      ("hit.pred", hit.pred),
                      ("n_poison_in_topk", n_poison_in_topk),
                      ("near_cos", near_cos)):
        if len(arr) != n_q:
            raise ValueError(f"{name} has {len(arr):,} rows, queries have {n_q:,}")
    if not np.array_equal(poisoned.pred, hit.pred):
        raise ValueError("the accuracy vote and the attack-success vote "
                         "returned different predictions; they must be two "
                         "pivots of ONE vote over one set of neighbours")
    score = scimilarity_score(np.asarray(near_cos))
    df = pd.DataFrame({
        **keys,
        QUERY_ID: meta[QUERY_ID].to_numpy().astype(str),
        DONOR_ID: meta[DONOR_ID].to_numpy().astype(str),
        Y_TRUE: meta[Y_TRUE].to_numpy().astype(str),
        Y_PRED_CLEAN: np.asarray(clean.pred).astype(str),
        Y_PRED_POISONED: np.asarray(poisoned.pred).astype(str),
        N_POISON_IN_TOPK: np.asarray(n_poison_in_topk).astype(np.int16),
        MIN_POISON_SCORE: score,
        POISON_SURVIVES_QC: score >= SCORE_QC,
        # how many labels shared the top count, and -- per metric -- whether
        # the arbitrary tie-break is what decided this query. A tie among
        # labels that are all wrong is determined, not ambiguous, so the flags
        # are pivot-specific and the count is not.
        "n_tied": np.asarray(poisoned.n_tied).astype(np.int8),
        "tie_decides_accuracy": np.asarray(poisoned.pivotal_in_tie),
        "tie_decides_attack_success": np.asarray(hit.pivotal_in_tie),
        "n_tied_clean": np.asarray(clean.n_tied).astype(np.int8),
        "tie_decides_accuracy_clean": np.asarray(clean.pivotal_in_tie),
    })
    return cast_raw(df)


def main() -> None:
    t0 = time.time()
    emit(__file__)          # which code ran, HEAD or dirty tree
    require_tiers((ATTACK_NAME,))
    meta = pd.read_parquet(query_cache(BASE, CFG) / "query_meta.parquet")

    b = Bench.load(CFG, BASE, t0)
    # The metadata is read a second time here, for the ids and donors Bench
    # does not keep. Equal LENGTH is not enough -- the per-query table joins a
    # prediction at row i to the query_id at row i, so a reordered frame would
    # attach every prediction to the wrong cell and every column would still
    # look sane. Ground truth is compared element for element.
    if len(meta) != len(b.truth):
        raise ValueError(f"{len(meta):,} metadata rows but Bench holds "
                         f"{len(b.truth):,} queries")
    if not np.array_equal(meta[Y_TRUE].to_numpy().astype(str), b.truth):
        raise ValueError("query_meta.parquet and Bench disagree on y_true row "
                         "for row; the two are not in the same order and every "
                         "prediction would be attached to the wrong cell")
    missing = sorted({s for s, _, _ in CONDITIONS} - set(b.sources))
    if missing:
        raise ValueError(f"not attackable in this dataset: {missing}")

    # The layout first: it needs no index, so a failure in the expensive half
    # does not cost it, and it is the half a figure cannot be started without.
    xy = fit_query_layout(b.q)
    umap_df = umap_rows(meta, xy)
    OUT.mkdir(parents=True, exist_ok=True)
    umap_df.to_parquet(OUT_UMAP, index=False)
    print(f"[{time.time()-t0:.0f}s] query layout fitted on {len(xy):,} cells; "
          f"wrote {OUT_UMAP.name}", file=sys.stderr, flush=True)

    # The clean side, through the SAME vote the poisoned side uses, over the
    # HNSW neighbours 01a_01 committed at this ef and k. Bench.load has already
    # checked that ef and k match, and refuses to build otherwise.
    clean = vote(b.clean_lab, pivotal=b.truth)
    print(f"[{time.time()-t0:.0f}s] clean graph predictions: accuracy "
          f"{(clean.pred == b.truth).mean():.5f}, tie-broken on "
          f"{clean.pivotal_in_tie.mean():.5f} of queries",
          file=sys.stderr, flush=True)

    frames: list[pd.DataFrame] = []
    for i, (source, selection, rate) in enumerate(CONDITIONS, start=1):
        t1 = time.time()
        target, attack_dist = b.target_of[source]
        cond_seed = seed_of(source, rate, SEED_VALUE)
        poison = build_attack(
            ATTACK_NAME, ref_emb=b.ref_emb, ref_labels=b.ref_label,
            ref_ids=b.ref_id, source=source, target=target, rate=rate,
            seed=cond_seed, label_change=LABEL_CHANGE_NAME,
            selection=selection, hub_nbr=b.clean_idx)
        pe = poison[POISON_EMB]
        n_p = len(pe)

        r = b.retrieve(pe, target)
        near_cos = nearest_poison_cosine(b.q, pe)
        del poison

        keys: dict[str, object] = {
            THREAT_MODEL: str(ThreatModel.NO_ACCESS),
            "retrieval_path": RETRIEVAL_PATH, "ef_search": b.ef,
            "k": K_NEIGHBORS, "tie_break": TIE_BREAK,
            ATTACK: str(ATTACK_NAME),
            "attack_tier": str(ATTACK_TIER[ATTACK_NAME]),
            ACCESS: str(Access.INSERT), GEOMETRY: str(GEOMETRY_NAME),
            LABEL_CHANGE: str(LABEL_CHANGE_NAME), SELECTION: str(selection),
            "dose_basis": "celltype",
            "source": source, "target": target,
            # the ATTACK's distance, source -> the attacker's target, fixed
            # before any query is scored. Named by the schema constant because
            # it is exactly RAW_RESULT_DOC's ontology_dist.
            ONTOLOGY_DIST: np.int8(attack_dist),
            RATE: rate, SEED: np.int16(SEED_VALUE),
            "condition_seed": cond_seed,
            # no attack-specific knob column: ``relabel`` has none, and this
            # step runs exactly one attack. ``attacks.params_of`` exists to let
            # one table hold several; six all-null columns here would say
            # nothing and would write as Arrow's null type.
            "n_poison": n_p, "n_celltype": int(len(b.pool_of[source])),
            "n_reference": b.n_ref, "atlas_share": n_p / b.n_ref,
            "condition_index": i - 1, "n_conditions": len(CONDITIONS),
        }
        df = query_rows(meta, clean, r.acc, r.hit,
                        r.is_poison.sum(axis=1), near_cos, keys)
        frames.append(df)

        within = df[Y_TRUE].to_numpy().astype(str) == source
        took = (df[Y_PRED_POISONED].to_numpy().astype(str) == target)
        amb = df["tie_decides_attack_success"].to_numpy()
        print(f"[{time.time()-t0:6.0f}s] {i}/{len(CONDITIONS)} {source} -> "
              f"{target} {selection:<7} dose {rate:<5} n_p {n_p:>7,}  "
              f"queries {len(df):,}  within {int(within.sum()):,}  "
              f"within on target {took[within].mean():.4f}  "
              f"global on target {took.mean():.4f}  "
              f"within tie-decided {amb[within].mean():.4f}  "
              f"({time.time() - t1:.0f}s)", file=sys.stderr, flush=True)

    out = cast_raw(pd.concat(frames, ignore_index=True))
    # The column is the evidence, not the filename: assert the arm before the
    # table lands, so a reader downstream inherits a decision rather than a
    # coincidence (00_pipeline/check_retrieval.py).
    assert (out["retrieval_path"] == RETRIEVAL_PATH).all(), (
        "every row must be retrieved through the shipped graph, not the merge "
        "shortcut")
    out.to_parquet(OUT_PER_QUERY, index=False)
    print(f"\n[{time.time()-t0:.0f}s] wrote {len(out):,} rows "
          f"({len(CONDITIONS)} conditions x {len(meta):,} queries) to "
          f"{OUT_PER_QUERY}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
