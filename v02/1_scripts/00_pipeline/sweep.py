"""Stage 4 — merge a poison set into precomputed clean neighbours and vote.

This is the stage the whole architecture exists to make cheap. The clean
neighbour list is retrieved once in stage 2 and reused by every condition; only
the (small) poison set is compared by brute force here. The reference is never
re-embedded and the index is never rebuilt (``CLAUDE.md`` §1).

Retrieving exactly ``k`` clean neighbours is sufficient, not approximate: any
member of the true merged top-k is either clean or poisoned, and in either case
already ranks within its own top-k. See ``SPEC.md`` §4.2.
"""
# 1_scripts cannot be a package (leading digit), so the shared library in
# 00_common is put on the path and imported flat.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_common"))

# No `from __future__ import annotations` here: snakemake's script directive
# prepends a preamble, which would push it off line 1 and make it a SyntaxError.
# Python 3.12 evaluates the annotations used below natively.

import numpy as np
import pandas as pd

from schemas import (
    EMB_DTYPE,
    MIN_POISON_SCORE,
    N_POISON_IN_TOPK,
    POISON_SURVIVES_QC,
    QUERY_ID,
    SCORE_QC,
    Y_PRED_CLEAN,
    Y_PRED_POISONED,
    Y_TRUE,
    scimilarity_score,
)


def _vote(labels: np.ndarray) -> np.ndarray:
    """Majority label per row, ties broken by lowest Cell Ontology id.

    Args:
        labels: (n, k) str, the labels of each query's top-k neighbours.

    Returns:
        (n,) str. Deterministic and seed-free: sorting the candidates and
        taking the first maximal count makes the tie-break the lowest id.
    """
    out = np.empty(len(labels), dtype=object)
    for i, row in enumerate(labels):
        terms, counts = np.unique(row, return_counts=True)  # np.unique sorts
        out[i] = terms[counts.argmax()]
    return out


def run_sweep(
    query_emb: np.ndarray,
    query_ids: np.ndarray,
    y_true: np.ndarray,
    clean_cos: np.ndarray,
    clean_labels: np.ndarray,
    poison_emb: np.ndarray,
    poison_labels: np.ndarray,
    k: int,
    block_size: int = 5000,
) -> pd.DataFrame:
    """Merge poison into clean neighbours and record per-query outcomes.

    Args:
        query_emb: (n_q, d) float32, unit norm.
        query_ids: (n_q,) str, cell barcodes.
        y_true: (n_q,) str, author-provided Cell Ontology terms.
        clean_cos: (n_q, k) float32, cosine to each clean neighbour, descending.
        clean_labels: (n_q, k) str, labels of those neighbours.
        poison_emb: (n_p, d) float32, unit norm.
        poison_labels: (n_p,) str, attacker-chosen terms.
        k: neighbours in the vote.
        block_size: queries per block. The (block, n_p) cosine matrix is
            discarded after the per-row top-k, so peak memory is independent of
            the poison-set size. Arithmetically inert.

    Returns:
        One row per query, long format, carrying the per-cell columns of the
        ``raw_result`` contract. Condition keys are attached by the caller.

    Raises:
        ValueError: the poison set is empty. A condition that poisons nothing
            would report perfect robustness.
    """
    if len(poison_emb) == 0:
        raise ValueError("empty poison set — nothing to merge")

    n_q = len(query_emb)
    n_pois = np.empty(n_q, dtype=np.int16)
    near_score = np.empty(n_q, dtype=EMB_DTYPE)
    pred = np.empty(n_q, dtype=object)

    take = min(k, len(poison_emb))
    for lo in range(0, n_q, block_size):
        hi = min(lo + block_size, n_q)
        cos = (query_emb[lo:hi] @ poison_emb.T).astype(EMB_DTYPE)  # (b, n_p)

        # Nearest poison overall — recorded whether or not it enters the top-k.
        near_score[lo:hi] = scimilarity_score(cos.max(axis=1))

        # Poison can contribute at most k candidates; the rest cannot displace.
        top = np.argpartition(-cos, take - 1, axis=1)[:, :take]
        p_cos = np.take_along_axis(cos, top, axis=1)
        p_lab = poison_labels[top]

        merged_cos = np.concatenate([clean_cos[lo:hi], p_cos], axis=1)
        merged_lab = np.concatenate([clean_labels[lo:hi], p_lab], axis=1)
        is_pois = np.concatenate(
            [np.zeros(clean_cos[lo:hi].shape, bool), np.ones(p_cos.shape, bool)], axis=1
        )

        # TIE POLICY — INCUMBENT. `duplicate` copies reference vectors verbatim,
        # so a poisoned record ties EXACTLY with the record it copied. Clean
        # neighbours are concatenated first and the sort is stable, so on an
        # exact tie the genuine record keeps the higher rank. This is a
        # benchmark decision, not a measurement: in the protein arm the opposite
        # choice moved pooled attack success by a third. Pinned by
        # test_exact_ties_resolve_to_the_genuine_record.
        order = np.argsort(-merged_cos, axis=1, kind="stable")[:, :k]
        n_pois[lo:hi] = np.take_along_axis(is_pois, order, axis=1).sum(axis=1)
        pred[lo:hi] = _vote(np.take_along_axis(merged_lab, order, axis=1))

    return pd.DataFrame({
        QUERY_ID: query_ids,
        Y_TRUE: y_true,
        Y_PRED_CLEAN: _vote(clean_labels),
        Y_PRED_POISONED: pred,
        N_POISON_IN_TOPK: n_pois,
        MIN_POISON_SCORE: near_score,
        POISON_SURVIVES_QC: near_score >= SCORE_QC,
    })


# ── snakemake entry point ────────────────────────────────────────────────────
if "snakemake" in globals():  # pragma: no cover - exercised by the workflow
    import numpy as _np
    import pandas as _pd

    from schemas import (  # noqa: E402
        ATTACK,
        K_NEIGHBORS,
        ONTOLOGY_DIST,
        PAIR,
        RATE,
        RAW_RESULT_DTYPES,
        SEED,
    )

    _q = _np.load(snakemake.input.query_emb)  # noqa: F821
    _n = _np.load(snakemake.input.neighbors)  # noqa: F821
    _p = _np.load(snakemake.input.poison, allow_pickle=True)  # noqa: F821
    _meta = _pd.read_parquet(snakemake.input.query_meta)  # noqa: F821
    _w = snakemake.wildcards  # noqa: F821
    _pair = {p["name"]: p for p in snakemake.config["pairs"]}[_w.pair]  # noqa: F821

    _df = run_sweep(
        query_emb=_q,
        query_ids=_meta["query_id"].to_numpy(),
        y_true=_meta["y_true"].to_numpy(),
        clean_cos=_n["cos"],
        clean_labels=_n["label"],
        poison_emb=_p["emb"],
        poison_labels=_p["label"],
        k=K_NEIGHBORS,
        block_size=int(snakemake.config["sweep"]["query_block_size"]),  # noqa: F821
    )
    _df[ATTACK] = _w.attack
    _df[PAIR] = _w.pair
    _df[RATE] = _np.float32(_w.rate)
    _df[SEED] = _np.int16(_w.seed)
    _df[ONTOLOGY_DIST] = _np.int8(_pair["dist"])
    _df = _df.astype({c: t for c, t in RAW_RESULT_DTYPES.items() if c in _df})
    _df.to_parquet(snakemake.output[0], index=False)  # noqa: F821
