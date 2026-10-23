"""Stage 6 — check the merge shortcut against an honestly rebuilt index.

# No `from __future__ import annotations` here: snakemake's script directive
# prepends a preamble, which would push it off line 1 and make it a SyntaxError.

The benchmark retrieves clean neighbours ONCE and reuses them for every
condition, merging each poison set in afterwards. Under exhaustive search that
is provably exact: adding records can only push clean ones down the ranking, so
any clean record in the true merged top-k already ranks in the clean top-k
(``SPEC.md`` §4.2).

Under HNSW the proof fails. Poisoned records are not just extra entries — they
are new nodes with new edges, so inserting them changes which path the search
walks and therefore which clean records it reaches at all. This stage rebuilds
the index with the poison inside it, queries it directly, and reports how far
the shortcut drifts.

``CLAUDE.md`` §6: publishing sweep results from an approximate retriever
without this is not permitted.
"""
# 1_scripts cannot be a package (leading digit), so the shared library in
# 00_common is put on the path and imported flat.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_common"))

import numpy as np

from schemas import EMB_DIM, EMB_DTYPE


def topk_overlap(a_idx: np.ndarray, b_idx: np.ndarray) -> np.ndarray:
    """Fraction of neighbours two top-k lists share, per query.

    Args:
        a_idx: (n_q, k) int, one path's neighbour ids.
        b_idx: (n_q, k) int, the other's.

    Returns:
        (n_q,) float, |intersection| / k.

    Raises:
        ValueError: shapes differ.

    Note:
        Set overlap, not rank agreement: the vote is a majority over the top-k,
        so order within it does not affect the answer.
    """
    if a_idx.shape != b_idx.shape:
        raise ValueError(f"top-k lists must have the same shape, got {a_idx.shape} and {b_idx.shape}")
    k = a_idx.shape[1]
    return np.array(
        [len(set(a.tolist()) & set(b.tolist())) / k for a, b in zip(a_idx, b_idx)],
        dtype=np.float64,
    )


def agreement_report(
    shortcut_idx: np.ndarray,
    rebuild_idx: np.ndarray,
    shortcut_pred: np.ndarray,
    rebuild_pred: np.ndarray,
    shortcut_n_poison: np.ndarray,
    rebuild_n_poison: np.ndarray,
) -> dict:
    """Summarise how far the shortcut drifts from an honest rebuild.

    Args:
        shortcut_idx: (n_q, k) neighbour ids from merge-and-rerank.
        rebuild_idx: (n_q, k) neighbour ids from the rebuilt index.
        shortcut_pred: (n_q,) labels voted under the shortcut.
        rebuild_pred: (n_q,) labels voted under the rebuild.
        shortcut_n_poison: (n_q,) poison in the shortcut's top-k.
        rebuild_n_poison: (n_q,) poison in the rebuild's top-k.

    Returns:
        ``mean_overlap`` — average share of neighbours in common.
        ``vote_agreement`` — share of queries where both paths voted alike.
        ``mean_poison_delta`` — signed mean of rebuild minus shortcut poison
        counts. Signed on purpose: a systematic excess under rebuild means the
        shortcut *understates* retrieval failure, which is a different finding
        from symmetric noise.
        ``exact`` — every neighbourhood identical. Required under exhaustive
        search; any failure there is a bug, not approximation error.

    Note:
        Both overlap and vote agreement are reported because they can diverge.
        The shortcut can retrieve different cells and still land on the same
        label by luck; quoting vote agreement alone would hide that.
    """
    overlap = topk_overlap(shortcut_idx, rebuild_idx)
    return {
        "mean_overlap": float(overlap.mean()),
        "min_overlap": float(overlap.min()),
        "vote_agreement": float((np.asarray(shortcut_pred) == np.asarray(rebuild_pred)).mean()),
        "mean_poison_delta": float(
            (np.asarray(rebuild_n_poison) - np.asarray(shortcut_n_poison)).mean()
        ),
        "exact": bool(np.isclose(overlap, 1.0).all()),
        "n_queries": int(len(overlap)),
    }


def rebuild_index_with_poison(
    ref_emb: np.ndarray,
    poison_emb: np.ndarray,
    query_emb: np.ndarray,
    k: int,
    m: int = 80,
    ef_construction: int = 200,
    ef_search: int = 200,
    base_index_path: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Put the poison inside a graph index and query it honestly.

    Args:
        ref_emb: (n_ref, 128) float32, unit norm.
        poison_emb: (n_p, 128) float32, unit norm. Inserted AFTER the reference,
            so ids >= n_ref identify poison.
        query_emb: (n_q, 128) float32, unit norm.
        k: neighbours to retrieve.
        m: graph degree, used only when building from scratch.
        ef_construction: build-time candidate list size.
        ef_search: query-time candidate list size.
        base_index_path: an existing index to load and extend. Strongly
            preferred — see Note.

    Returns:
        idx: (n_q, k) int64 neighbour ids; ``>= len(ref_emb)`` means poison.
        cos: (n_q, k) float32 cosine similarity.

    Note:
        Building a 7.9M-node graph at M=80 from scratch takes hours per
        condition, which would make stage 6 unrunnable at the scale it matters
        for. SCimilarity ships that graph already built, so the honest rebuild
        is: load it, grow it, insert the poison, query. Inserting nodes is
        exactly the operation that perturbs traversal, which is the effect this
        stage exists to measure — so extending the shipped index tests the same
        thing a from-scratch build would, at minutes rather than hours.

        It also matches deployment: a real atlas maintainer accepting new
        submissions adds them to the existing index; nobody rebuilds 7.9M nodes
        per upload.
    """
    import hnswlib

    n_ref = len(ref_emb)
    n_p = len(poison_emb)
    index = hnswlib.Index(space="cosine", dim=EMB_DIM)

    if base_index_path:
        index.load_index(base_index_path, max_elements=n_ref + n_p)
        if index.get_current_count() != n_ref:
            raise ValueError(
                f"index holds {index.get_current_count():,} vectors but the "
                f"reference has {n_ref:,}; ids would not line up"
            )
    else:
        index.init_index(max_elements=n_ref + n_p, M=m, ef_construction=ef_construction)
        index.add_items(ref_emb, np.arange(n_ref))

    index.add_items(poison_emb, np.arange(n_ref, n_ref + n_p))
    # hnswlib requires ef >= k and nothing more, so the floor is k. It was
    # `max(ef_search, 4 * k)`, which at k=50 floors ef at 200 and silently
    # discards any configured value below it -- including the 100 that
    # CellAnnotation.get_predictions_knn actually uses. SCimilarity applies no
    # floor at all (cell_search_knn.py: a bare `self.knn.set_ef(ef)`), so
    # matching it means honouring the config and guarding only the hard
    # requirement.
    if ef_search < k:
        raise ValueError(f"ef_search {ef_search} < k {k}: hnswlib requires ef >= k")
    index.set_ef(ef_search)

    labels, dist = index.knn_query(query_emb, k=k)
    # hnswlib returns cosine DISTANCE for space="cosine"; convert to similarity.
    return labels.astype(np.int64), (1.0 - dist).astype(EMB_DTYPE)


def shortcut_topk(
    clean_cos: np.ndarray,
    clean_idx: np.ndarray,
    clean_lab: np.ndarray,
    query_emb: np.ndarray,
    poison_emb: np.ndarray,
    poison_lab: np.ndarray,
    n_ref: int,
    k: int,
    block_size: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the merge shortcut, keeping the neighbour IDS.

    Stage 4 discards the merged index list once it has counted poison and
    voted, so stage 6 recomputes it here rather than changing stage 4's output
    contract for a diagnostic that runs on a handful of conditions.

    Args:
        clean_cos: (n_q, k) float32 cosine to clean neighbours, descending.
        clean_idx: (n_q, k) int64 reference row ids.
        clean_lab: (n_q, k) str labels of those neighbours.
        query_emb: (n_q, 128) float32, unit norm.
        poison_emb: (n_p, 128) float32, unit norm.
        poison_lab: (n_p,) str attacker-chosen labels.
        n_ref: reference size, so poison ids start at ``n_ref`` and match the
            ids :func:`rebuild_index_with_poison` assigns.
        k: neighbours in the vote.
        block_size: queries per block.

    Returns:
        idx: (n_q, k) int64 merged neighbour ids; ``>= n_ref`` means poison.
        lab: (n_q, k) str.
        n_poison: (n_q,) int16 poison count in the merged top-k.
    """
    n_q = len(query_emb)
    out_idx = np.empty((n_q, k), dtype=np.int64)
    out_lab = np.empty((n_q, k), dtype=object)
    n_pois = np.empty(n_q, dtype=np.int16)
    take = min(k, len(poison_emb))

    for lo in range(0, n_q, block_size):
        hi = min(lo + block_size, n_q)
        cos = (query_emb[lo:hi] @ poison_emb.T).astype(EMB_DTYPE)
        top = np.argpartition(-cos, take - 1, axis=1)[:, :take]
        p_cos = np.take_along_axis(cos, top, axis=1)

        merged_cos = np.concatenate([clean_cos[lo:hi], p_cos], axis=1)
        merged_idx = np.concatenate([clean_idx[lo:hi], top + n_ref], axis=1)
        merged_lab = np.concatenate([clean_lab[lo:hi], poison_lab[top]], axis=1)

        order = np.argsort(-merged_cos, axis=1, kind="stable")[:, :k]
        sel = np.take_along_axis(merged_idx, order, axis=1)
        out_idx[lo:hi] = sel
        out_lab[lo:hi] = np.take_along_axis(merged_lab, order, axis=1)
        n_pois[lo:hi] = (sel >= n_ref).sum(axis=1)

    return out_idx, out_lab, n_pois


def _vote(labels: np.ndarray) -> np.ndarray:
    """Majority label per row, ties to the lowest ontology id."""
    out = np.empty(len(labels), dtype=object)
    for i, row in enumerate(labels):
        terms, counts = np.unique(row.astype(str), return_counts=True)
        out[i] = terms[counts.argmax()]
    return out


# ── snakemake entry point ────────────────────────────────────────────────────
if "snakemake" in globals():  # pragma: no cover - exercised by the workflow
    import sys
    import time

    import pandas as pd

    from schemas import K_NEIGHBORS

    cfg = snakemake.config  # noqa: F821
    ref = np.load(snakemake.input.reference)  # noqa: F821
    nbr = np.load(snakemake.input.neighbors, allow_pickle=True)  # noqa: F821
    q = np.load(snakemake.input.query_emb)  # noqa: F821
    pois = np.load(snakemake.input.poison, allow_pickle=True)  # noqa: F821
    w = snakemake.wildcards  # noqa: F821

    n_ref = len(ref["emb"])
    t0 = time.time()
    s_idx, s_lab, s_np = shortcut_topk(
        nbr["cos"], nbr["idx"], nbr["label"], q,
        pois["emb"], pois["label"], n_ref, K_NEIGHBORS,
    )
    print(f"shortcut recomputed in {time.time()-t0:.0f}s", file=sys.stderr)

    t0 = time.time()
    r_idx, _ = rebuild_index_with_poison(
        ref["emb"], pois["emb"], q, K_NEIGHBORS,
        m=int(cfg["hnsw"]["M"]),
        ef_construction=int(cfg["hnsw"]["ef_construction"]),
        ef_search=int(cfg["hnsw"]["ef_search"]),
        base_index_path=f'{cfg["paths"]["reference_dir"]}/annotation/labelled_kNN.bin',
    )
    print(f"index rebuilt + queried in {time.time()-t0:.0f}s", file=sys.stderr)

    all_lab = np.concatenate([ref["label"], pois["label"]])
    rep = agreement_report(
        shortcut_idx=s_idx, rebuild_idx=r_idx,
        shortcut_pred=_vote(s_lab), rebuild_pred=_vote(all_lab[r_idx]),
        shortcut_n_poison=s_np, rebuild_n_poison=(r_idx >= n_ref).sum(axis=1),
    )
    rep.update(attack=w.attack, pair=w.pair, rate=float(w.rate), seed=int(w.seed))
    print(f"  overlap {rep['mean_overlap']:.4f}  votes {rep['vote_agreement']:.4f}  "
          f"poison delta {rep['mean_poison_delta']:+.3f}", file=sys.stderr)
    pd.DataFrame([rep]).to_parquet(snakemake.output[0], index=False)  # noqa: F821
