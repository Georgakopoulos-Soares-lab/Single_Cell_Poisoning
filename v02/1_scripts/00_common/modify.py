"""Overwriting reference records, embeddings and all. SPEC.md §5.4, ``build 3``.

The last two attack types in the designed grid: ``modify x synthesised`` at
``uniform`` and ``hub`` selection. The attacker does not append records, they
REPLACE existing ones -- the genuine vector is destroyed along with its vote.

**Why this needs its own machinery.** ``02b_05``'s write-access arm is a cheap
re-vote over cached neighbours, and that is exact only because rewriting a LABEL
moves no embedding: the shipped graph is byte-identical and the retrieved
neighbours are the clean ones. Moving an embedding breaks that. A record that
was in a query's top-50 may no longer be, and the record that was 51st should
take its place -- and ``clean_neighbors.npz`` stores exactly 50, so there is
nothing to promote.

**Why not simply update the graph? Because that is a DECISION, not a
limitation -- corrected 2026-09-25.** An earlier version of this docstring said
"hnswlib cannot update a vector". That is FALSE, and it was the stated
justification for everything below. ``add_items`` with an id that already exists
updates the point in place AND rewires its links: demonstrated on a 5,000-node
index by moving the single furthest record (cosine -0.91, rank 4,999) onto the
query, after which it is returned at rank 0. So the poison COULD be written into
the shipped graph, exactly as the insertion arm does.

What is true is that doing so costs an index load per condition -- 18.3 s
measured, single-threaded -- against ~9 s for the whole of this function. Over
the 2,964 conditions of ``02b_11`` and ``02b_15`` that is ~39 h against ~7 h.

So removal-and-merge is a SPEED choice, and it buys exhaustive retrieval rather
than graph retrieval. It is exact for the former -- asserted against a
from-scratch rebuild in ``test_modify`` -- and does NOT reproduce the latter.
Moving a record would relocate its node, and the graph effect of that is not
modelled here. ``01a_02`` bounds the two paths at 99.0-99.6% vote agreement on
CLEAN data; whether poisoning widens that gap is unmeasured, and the rows carry
``retrieval_path = "exact"`` so no reader has to infer it.


**So it needs a DEEPER clean list**, :data:`DEEP_K` per query, retrieved
exhaustively once. Then the surviving originals are the first ``k`` entries of
that list that were not overwritten, and the merge shortcut proceeds as usual
against the new vectors. This is exact whenever at least ``k`` of a query's
``DEEP_K`` survive, which :func:`remerge` CHECKS per query and reports rather
than assuming -- an attacker overwriting a large celltype can exhaust the list
for queries sitting deep inside it, and a silently truncated neighbourhood would
understate the attack.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from flood_common import BLOCK, build_flood, top_poison
from schemas import (EMB_DTYPE, K_NEIGHBORS, POISON_ATTACK, POISON_EMB,
                     POISON_LABEL, POISON_PROVENANCE, Access, Geometry,
                     LabelChange, Selection)
from selection import hub_order

#: Depth of the clean neighbour list this attack needs.
#:
#: 50 is not enough by construction: if j of a query's top-50 are overwritten,
#: ranks 51..50+j must be promoted, and those may be overwritten too. 500 gives
#: room for a query to lose 450 of its nearest records and still be scored
#: exactly. :func:`remerge` reports the queries where even that runs out instead
#: of truncating them quietly.
DEEP_K = 500


def build_modify(*, ref_emb: np.ndarray, ref_labels: np.ndarray,
                 ref_ids: np.ndarray, source: str, target: str, dose: float,
                 seed: int, selection: str,
                 geometry: str = Geometry.SYNTHESISED,
                 n_anchors: int | None = None, spread: float | None = None,
                 jitter: float | None = None,
                 alpha_range: tuple[float, float] | None = None,
                 hub_nbr: np.ndarray | None = None,
                 label_change: str = LabelChange.RELABEL) -> dict[str, Any]:
    """Which records the attacker overwrites, and with what.

    Records are chosen WITHOUT replacement -- a record cannot be overwritten
    twice, which is why ``concentrated`` selection is degenerate under modify
    (SPEC.md §5.4) and only ``uniform`` and ``hub`` are offered here.

    Args:
        ref_emb: (n_ref, 128) float32 unit-norm. ref_labels/(ref_ids): (n_ref,).
        source: celltype whose records are overwritten. target: forged label.
        dose: fraction of the SOURCE CELLTYPE, clipped at 1.0 -- there is no
            such thing as overwriting 125% of a celltype, and silently drawing
            with replacement would report a smaller attack than the dose says.
        seed: passed in, never generated.
        selection: ``uniform`` or ``hub``.
        geometry: ``synthesised`` builds replacements by the flood construction;
            ``shifted`` moves each overwritten record toward the target centroid.
            The two differ in what the replacement is MADE of, and that is the
            axis the write arm was missing -- ``shifted`` keeps a trace of the
            genuine cell, ``synthesised`` does not.
        n_anchors/spread/jitter: ``synthesised`` only, REQUIRED there for the
            reason ``attacks.py`` gives -- none has a defensible default.
        alpha_range: ``shifted`` only, REQUIRED there. ``(a, a)`` for a fixed
            shift; a real range draws per record, as the insertion arm's
            ``interpolate`` does. v01 only ever ran 0.5, which is why
            ``shifted x keep`` reads as inert and why alpha is an axis rather
            than a constant (SPEC.md 5.4, step 5).
        hub_nbr: (n_q, k) observed neighbour rows, ``hub`` only.
        label_change: ``relabel`` forges target; ``keep`` leaves the source term.

    Returns:
        ``{rows, emb, label, provenance, attack, access, geometry,
        label_change, selection, alpha, n_overwritten, dose_clipped}``. ``rows``
        are the reference indices to overwrite; ``emb`` their replacements,
        aligned. ``alpha`` is NaN under ``synthesised``.

    Raises:
        ValueError: unknown selection, ``hub`` without traffic, or a dose that
            overwrites nothing.
    """
    src_idx = np.flatnonzero(ref_labels == source)
    if src_idx.size == 0:
        raise ValueError(f"no reference cells carry the source term {source!r}")

    clipped = dose > 1.0
    n_p = int(round(min(dose, 1.0) * src_idx.size))
    if n_p < 1:
        raise ValueError(
            f"dose {dose} over {src_idx.size:,} {source} cells overwrites nothing")

    rng = np.random.default_rng(seed)
    if selection == Selection.UNIFORM:
        rows = rng.choice(src_idx, size=n_p, replace=False)
    elif selection == Selection.HUB:
        if hub_nbr is None:
            raise ValueError(
                "hub selection requires the observed neighbour table; without "
                "traffic there is nothing to rank and a silent fallback to "
                "uniform would report the wrong attack.")
        rows = src_idx[hub_order(src_idx, hub_nbr, len(ref_labels))[:n_p]]
    else:
        raise ValueError(
            f"selection {selection!r} is not available under modify. "
            "concentrated is degenerate here -- a record cannot be overwritten "
            "twice, so it reduces to uniform at the pool size (SPEC.md 5.4).")

    if geometry == Geometry.SYNTHESISED:
        missing = [n for n, v in (("n_anchors", n_anchors), ("spread", spread),
                                  ("jitter", jitter)) if v is None]
        if missing:
            raise ValueError(
                f"synthesised geometry requires {', '.join(missing)} "
                "explicitly; none has a defensible default (attacks.py).")
        # The flood construction, asked for exactly n_p vectors. Its `rate` is a
        # multiple of the pool it is given, so passing the pool and n_p/len(pool)
        # reproduces the insertion arm's geometry at the same budget -- which is
        # the point: the SAME attack differing only in access.
        pool = ref_emb[src_idx].astype(EMB_DTYPE)
        emb = build_flood(pool, n_anchors, n_p / len(pool), spread, jitter, seed)
        emb = emb[:n_p] if len(emb) >= n_p else np.vstack(
            [emb, emb[: n_p - len(emb)]])
        alpha = np.full(n_p, np.nan, dtype=EMB_DTYPE)
    elif geometry == Geometry.SHIFTED:
        if alpha_range is None:
            raise ValueError(
                "shifted geometry requires alpha_range explicitly. v01 only ever "
                "ran 0.5, which is why shifted x keep reads as inert; a default "
                "here would silently repeat that (SPEC.md 5.4, step 5).")
        tgt_idx = np.flatnonzero(ref_labels == target)
        if tgt_idx.size == 0:
            raise ValueError(f"no reference cells carry the target term {target!r}")
        centroid = ref_emb[tgt_idx].mean(axis=0)
        centroid = (centroid / np.linalg.norm(centroid)).astype(EMB_DTYPE)
        alpha = rng.uniform(*alpha_range, size=n_p).astype(EMB_DTYPE)
        # The OVERWRITTEN records are what move -- ref_emb[rows], not a fresh
        # draw. That is the difference from the insertion arm's `interpolate`,
        # which shifts a COPY and leaves the original in place still voting
        # correctly. Here the original is what gets displaced.
        emb = ((1.0 - alpha)[:, None] * ref_emb[rows]
               + alpha[:, None] * centroid)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    else:
        raise ValueError(
            f"geometry {geometry!r} is not available under modify. `unmoved` "
            "moves no embedding, so it is 02b_05's cheap re-vote and needs "
            "none of this machinery.")

    return {
        "rows": rows,
        POISON_EMB: emb.astype(EMB_DTYPE),
        POISON_LABEL: np.full(n_p, target if label_change == LabelChange.RELABEL
                              else source),
        POISON_PROVENANCE: ref_ids[rows],
        POISON_ATTACK: f"{geometry}_inplace",
        "alpha": alpha,
        "access": str(Access.MODIFY),
        "geometry": str(geometry),
        "label_change": str(label_change),
        "selection": str(selection),
        "n_overwritten": n_p,
        "dose_clipped": clipped,
    }


def remerge(deep_idx: np.ndarray, deep_cos: np.ndarray, rows: np.ndarray,
            new_emb: np.ndarray, new_label: np.ndarray, ref_label: np.ndarray,
            q: np.ndarray, n_ref: int, k: int = K_NEIGHBORS,
            block: int = BLOCK) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The top-k after overwriting, computed exactly where the deep list allows.

    Three steps, in order: drop the overwritten records from each query's deep
    clean list; take the first ``k`` that survive; merge against the new vectors
    scored by brute force. Identical in spirit to the insertion arm's merge
    shortcut, with the removal step that overwriting forces.

    Args:
        deep_idx: (n_q, DEEP_K) int64 clean neighbour rows, nearest first.
        deep_cos: (n_q, DEEP_K) float32 their cosines, descending.
        rows: (n_p,) reference rows that were overwritten.
        new_emb: (n_p, 128) their replacements, aligned with ``rows``.
        new_label: (n_p,) the labels those replacements carry.
        ref_label: (n_ref,) reference labels. q: (n_q, 128) query vectors.
        n_ref: reference size. k: neighbours to return.
        block: poison rows per matmul inside :func:`flood_common.top_poison`.
            ARITHMETICALLY INERT -- it changes only how the same product is
            chunked, never a returned value, which ``test_modify`` asserts by
            equality. It is exposed because the transient buffer is
            ``len(q) x block x 4`` bytes: 3.96 GB at the default against 49,456
            queries, which dominates this function's peak of ~26 GB. A caller
            sharing the machine with other sweeps should lower it.

    Returns:
        ``(labels, is_new, exact)`` -- (n_q, k) labels of the new top-k,
        (n_q, k) bool marking which came from an overwritten record, and (n_q,)
        bool marking the queries whose answer is exact. A False in ``exact``
        means the deep list ran out and that query's row must not be scored.
    """
    overwritten = np.zeros(n_ref, bool)
    overwritten[rows] = True
    mod = overwritten[deep_idx]

    # First k survivors per query, in cosine order. A stable argsort on the
    # boolean brings survivors forward while preserving their ranking.
    order = np.argsort(mod, axis=1, kind="stable")
    keep_idx = np.take_along_axis(deep_idx, order, axis=1)[:, :k]
    keep_cos = np.take_along_axis(deep_cos, order, axis=1)[:, :k]
    n_surv = (~mod).sum(axis=1)
    exact = n_surv >= k
    # Where fewer than k survive, the tail slots still hold overwritten records.
    # Score them -inf so they cannot win the merge; `exact` already flags the row.
    short = np.arange(k)[None, :] >= n_surv[:, None]
    keep_cos = np.where(short, -np.inf, keep_cos)

    new_cos, new_pos = top_poison(q, new_emb, k, block=block)

    cat_cos = np.concatenate([keep_cos, new_cos], axis=1)
    cat_lab = np.concatenate([ref_label[keep_idx], new_label[new_pos]], axis=1)
    cat_new = np.concatenate([np.zeros_like(keep_cos, bool),
                              np.ones_like(new_cos, bool)], axis=1)
    top = np.argsort(-cat_cos, axis=1, kind="stable")[:, :k]
    return (np.take_along_axis(cat_lab, top, axis=1),
            np.take_along_axis(cat_new, top, axis=1), exact)

def graph_topk_ids(idx_path: str, rows: np.ndarray, new_emb: np.ndarray,
                   new_label: np.ndarray, q: np.ndarray, n_ref: int, ef: int,
                   k: int = K_NEIGHBORS) -> tuple[np.ndarray, np.ndarray]:
    """The graph's top-k ROWS and cosines after the edit, for callers that need
    more than labels -- a defence must reach each neighbour's donor, embedding
    and label-consistency score, and only the row id gets you there.

    Split out of :func:`graph_topk` on 2026-09-29 so that ``add_items`` has ONE
    implementation. Two would be two different graphs: insertion is
    multi-threaded and order-dependent (CLAUDE.md 1), so a defended run and its
    undefended baseline must come from the same call, not from two callers that
    look alike.

    The same answer as :func:`remerge`, but from the SHIPPED GRAPH.

    ``remerge`` computes what EXHAUSTIVE retrieval would return. This computes
    what SCimilarity actually returns, by writing the replacements into the
    index: ``add_items`` with ids that already exist updates each point in place
    and rewires its links. Verified on a 5,000-node index by moving the furthest
    record (cosine -0.91, rank 4,999) onto the query and seeing it come back at
    rank 0.

    **The two do not always agree, and the difference is not noise.** Measured
    on four conditions: ``CL:0000235`` matched to 0.0001 at alpha 0.02 and 0.2,
    while ``CL:0000583`` differed by +0.1113 and +0.1240 at the same two --
    celltype-dependent rather than alpha-dependent.

    **CORRECTED 2026-09-27 against all 936 conditions of ``02b_15``.** This
    docstring previously read "the graph was never WEAKER, plausibly because
    updated nodes rewire into hubs that route queries toward themselves, so
    ``remerge`` understates by an unknown per-celltype margin". That was inferred
    from the four conditions above and it is FALSE at scale. On ``within_accuracy``
    the two agree exactly in 41.2% of conditions, and where they differ the
    direction is mixed: the graph attack is stronger in 34.0% and **WEAKER in
    24.8%**. The mean difference is -0.0011, so ``remerge`` does not
    systematically understate -- it disagrees in both directions and nearly
    cancels. The largest single disagreement runs the other way from the claim:
    ``CL:0000583`` shifted/hub at alpha 0.20, exact 0.4343 against graph 0.6628.

    The case for running ``graph`` therefore is NOT that it is uniformly stronger.
    It is that the disagreement is concentrated and unpredictable: 2 of 26
    celltypes exceed 0.10 and 11 of 26 stay under 0.01, and you cannot tell in
    advance which a celltype will be. Nor does it track HNSW's own clean recall
    (Spearman -0.23 against ``01a_02``'s top-50 overlap): ``CL:0002062``'s
    disagreement is consistent with its poor recall of 0.885, but ``CL:0000583``'s
    LARGER one comes with an ordinary 0.975. Every one of the six biggest
    disagreements is ``shifted`` x ``hub`` -- moving the busiest records -- which
    is consistent with rewiring being the mechanism, at the opposite sign from
    what was guessed. ``02b_16`` regenerates all of this from the committed
    tables.

    Costs an index load per condition (18.3 s, single-threaded) plus the update
    (0.77 ms per record, 15.4x threaded). Roughly 10x ``remerge``.

    Args:
        idx_path: the shipped ``labelled_kNN.bin``.
        rows: (n_p,) reference rows to overwrite.
        new_emb: (n_p, 128) their replacements. new_label: (n_p,) forged labels.
        ref_label: (n_ref,) reference labels. q: (n_q, 128) queries.
        n_ref: reference size. ef: hnswlib search width. k: neighbours.

    Returns:
        ``(labels, is_new, exact)`` matching :func:`remerge`'s shape. ``exact``
        is all True: the graph IS the answer, so there is nothing to be
        inexact about.
    """
    import hnswlib  # noqa: PLC0415

    ix = hnswlib.Index(space="cosine", dim=new_emb.shape[1])
    ix.load_index(idx_path, max_elements=n_ref)
    if ix.get_current_count() != n_ref:
        raise ValueError(f"index holds {ix.get_current_count():,} vectors but "
                         f"the reference has {n_ref:,}; ids would not correspond")
    ix.add_items(new_emb, rows)        # UPDATE in place; hnswlib rewires links
    ix.set_ef(ef)
    ids, dist = ix.knn_query(q, k=k)
    del ix
    return ids.astype(np.int64), (1.0 - dist).astype(np.float32)


def graph_topk(idx_path: str, rows: np.ndarray, new_emb: np.ndarray,
               new_label: np.ndarray, ref_label: np.ndarray, q: np.ndarray,
               n_ref: int, ef: int, k: int = K_NEIGHBORS
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Labels of the graph's top-k after the edit. See :func:`graph_topk_ids`.

    Returns:
        ``(labels, is_new, exact)`` matching :func:`remerge`'s shape. ``exact``
        is all True: the graph IS the answer, so there is nothing to be
        inexact about.
    """
    ids, _ = graph_topk_ids(idx_path, rows, new_emb, new_label, q, n_ref, ef, k)
    lab = ref_label.copy()
    lab[rows] = new_label
    edited = np.zeros(n_ref, bool)
    edited[rows] = True
    return lab[ids], edited[ids], np.ones(len(q), bool)


def clean_graph_topk(idx_path: str, ref_label: np.ndarray, q: np.ndarray,
                     n_ref: int, ef: int, k: int = K_NEIGHBORS) -> np.ndarray:
    """The UNPOISONED top-k labels from the shipped graph. Same retriever as
    :func:`graph_topk`, minus the ``add_items``.

    Exists because a clean baseline must come from the same retriever as the
    attacked side it is subtracted from (CLAUDE.md 7, "One retrieval regime per
    comparison"). ``02b_11`` and ``02b_15`` took the attacked side from
    :func:`graph_topk` under ``BIORAG_RETRIEVAL=graph`` but the clean side from
    ``deep_idx[:, :k]``, which is EXHAUSTIVE -- so ``accuracy_delta`` and
    ``net_attack_success`` straddled two regimes. Under ``exact`` both sides came
    from the deep list and the gap cancelled, which is why making the attacked
    side faithful is what exposed it.

    ``01a_02`` bounds what that cost: over 34 celltypes the median |delta
    accuracy| between HNSW and exhaustive is 0.000000 and 32 of 34 sit under
    0.01, but ``CL:0002062`` reaches 0.110429, with top-50 overlap falling to
    0.327 and vote agreement to 0.333 on the worst celltype. That is the same
    order as the graph-vs-exact divergence the graph run exists to measure, so it
    does not round away.

    One index load per CALL, not per condition -- the caller hoists it out of the
    condition loop, exactly as ``02a_01`` does at its lines 94-96.

    Args:
        idx_path: ``labelled_kNN.bin``. Must hold exactly ``n_ref`` vectors, or
            ids would not correspond to ``ref_label``.
        ref_label: (n_ref,) the genuine reference labels, unmodified.
        q: (n_q, 128) float32 unit-norm queries.
        n_ref: reference size, asserted against the index.
        ef: hnswlib search width. Must match the attacked side's, or the two
            differ by search width as well as by poison.
        k: neighbours to return.

    Returns:
        (n_q, k) labels, as ``graph_topk``'s first element but with nothing
        modified.
    """
    import hnswlib  # noqa: PLC0415

    if ef < k:
        raise ValueError(f"ef {ef} < k {k}: hnswlib requires ef >= k")
    ix = hnswlib.Index(space="cosine", dim=q.shape[1])
    ix.load_index(idx_path, max_elements=n_ref)
    if ix.get_current_count() != n_ref:
        raise ValueError(f"index holds {ix.get_current_count():,} vectors but "
                         f"the reference has {n_ref:,}; ids would not correspond")
    ix.set_ef(ef)
    ids, _ = ix.knn_query(q, k=k)
    del ix
    return ref_label[ids.astype(np.int64)]


def clean_graph_topk_ids(idx_path: str, q: np.ndarray, n_ref: int, ef: int,
                         k: int = K_NEIGHBORS) -> tuple[np.ndarray, np.ndarray]:
    """The UNMODIFIED graph's top-k rows and cosines.

    :func:`clean_graph_topk` returns labels, which is all an attack sweep needs.
    A DEFENCE needs the row ids: every layer reaches for something the label
    does not carry -- the neighbour's donor, its embedding, its own
    label-consistency score.

    It also serves the whole ``unmoved`` arm. A label-only edit moves no vector,
    so the index is byte-identical to this one and these ARE the attacked
    neighbours; only the labels read off them change. That makes the arm
    ``cached_clean`` rather than ``graph``: exact by construction, one retrieval
    for every condition, and -- because nothing is ever inserted -- free of the
    ``add_items`` non-determinism that CLAUDE.md 1 bounds at 0.0168.
    """
    import hnswlib  # noqa: PLC0415

    if ef < k:
        raise ValueError(f"ef {ef} < k {k}: hnswlib requires ef >= k")
    ix = hnswlib.Index(space="cosine", dim=q.shape[1])
    ix.load_index(idx_path, max_elements=n_ref)
    if ix.get_current_count() != n_ref:
        raise ValueError(f"index holds {ix.get_current_count():,} vectors but "
                         f"the reference has {n_ref:,}; ids would not correspond")
    ix.set_ef(ef)
    ids, dist = ix.knn_query(q, k=k)
    del ix
    return ids.astype(np.int64), (1.0 - dist).astype(np.float32)


def reference_knn_rows(idx_path: str, ref_emb: np.ndarray, rows: np.ndarray,
                       n_ref: int, ef: int, k: int = K_NEIGHBORS
                       ) -> np.ndarray:
    """Each of ``rows``'s own nearest reference rows, self excluded.

    The input a label-consistency defence needs: a record's score is the share
    of ITS OWN neighbours carrying ITS label, so the filter cannot run on a
    record whose neighbourhood is unknown.

    v01 cached this for the 170,398 rows EXACT search retrieved. v02's graph
    reaches 195,972, so 14% of the distinct records in a v02 neighbourhood have
    no cached entry. Defaulting those to "consistent" ADMITS them, which
    understates the layer -- the conservative direction for a step reporting a
    null, but still a number the defence never actually computed. This fills
    the gap instead.

    Queries ``k + 1`` and drops the self-match, exactly as ``01a_05`` does: a
    record is always its own nearest neighbour in the index it lives in, and
    keeping that would add one guaranteed agreement to every score.
    """
    import hnswlib  # noqa: PLC0415

    ix = hnswlib.Index(space="cosine", dim=ref_emb.shape[1])
    ix.load_index(idx_path, max_elements=n_ref)
    ix.set_ef(max(ef, k + 1))
    ids, _ = ix.knn_query(ref_emb[rows], k=k + 1)
    del ix
    out = np.empty((len(rows), k), dtype=np.int64)
    for i, r in enumerate(rows):
        nb = ids[i][ids[i] != r][:k]
        out[i] = nb if len(nb) == k else np.pad(nb, (0, k - len(nb)), mode="edge")
    return out


def knn_of_vectors(idx_path: str, vecs: np.ndarray, n_ref: int, ef: int,
                   k: int = K_NEIGHBORS, self_rows: np.ndarray = None
                   ) -> np.ndarray:
    """Nearest CLEAN reference rows for arbitrary vectors, in one index load.

    :func:`reference_knn_rows` scores records that are already in the index.
    This scores anything -- a submitted poison vector that was never inserted,
    or a reference row that is -- so a defence can compute label consistency
    for a mixed batch without opening the index twice.

    The index load is ~19 s and dominates; batching the genuine misses and the
    retrieved poison into one call is what keeps a per-condition LC lookup
    affordable.

    Args:
        vecs: (m, 128) float32 unit-norm vectors to score.
        self_rows: (m,) reference row of each vector, or -1 where the vector is
            not in the index. A record in the index is always its own nearest
            neighbour, and counting that would add one guaranteed agreement to
            every score -- the advantage poison does not get, because poison is
            scored BEFORE it is inserted.

    Returns:
        (m, k) int64 reference rows, self-match excluded where applicable.
    """
    import hnswlib  # noqa: PLC0415

    ix = hnswlib.Index(space="cosine", dim=vecs.shape[1])
    ix.load_index(idx_path, max_elements=n_ref)
    ix.set_ef(max(ef, k + 1))
    ids, _ = ix.knn_query(vecs, k=k + 1)
    del ix
    if self_rows is None:
        return ids[:, :k].astype(np.int64)
    out = np.empty((len(vecs), k), dtype=np.int64)
    for i, s in enumerate(self_rows):
        nb = ids[i][ids[i] != s][:k] if s >= 0 else ids[i][:k]
        out[i] = nb if len(nb) == k else np.pad(nb, (0, k - len(nb)), mode="edge")
    return out
