"""Stage 1 — turn held-out query cells into unit-norm embeddings.

The SCimilarity encoder runs *here and nowhere else*. Once this stage has run,
the rest of the pipeline works purely in embedding space: no stage re-embeds
anything and no sweep touches the encoder (``CLAUDE.md`` §1).

Preprocessing is scimilarity's own, in the order its API requires — duplicate
symbols consolidated, gene space aligned to the encoder's expected order, then
log-normalised. ``get_embeddings`` documents its input as "gene space aligned
and log normalized"; doing those two in the other order silently produces
embeddings from the wrong numbers.
"""
# 1_scripts cannot be a package (leading digit), so the shared library in
# 00_common is put on the path and imported flat.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_common"))

# No `from __future__ import annotations` here: snakemake's script directive
# prepends a preamble, which would push it off line 1 and make it a SyntaxError.
# Python 3.12 evaluates the annotations used below natively.

from typing import Callable, Protocol

import numpy as np
import pandas as pd

from schemas import DONOR_ID, EMB_DIM, EMB_DTYPE, QUERY_ID, Y_TRUE

Encoder = Callable[[np.ndarray], np.ndarray]
"""Anything that maps a (n_cells, n_genes) expression matrix to (n_cells, 128).

Injected rather than constructed here so the validation logic is testable
without the 30 GB model."""


class _AnnDataLike(Protocol):
    X: object
    obs: pd.DataFrame
    var: pd.DataFrame


def preprocess_for_encoder(adata: _AnnDataLike, gene_order: list[str]) -> _AnnDataLike:
    """Apply scimilarity's required preprocessing, in its required order.

    Args:
        adata: raw counts, (n_cells, n_genes).
        gene_order: the encoder's expected gene symbols, in order.

    Returns:
        An AnnData ready for ``CellEmbedding.get_embeddings``.

    Note:
        Not unit-tested: every step is scimilarity's own function and retesting
        them here would only assert that we called them. What *is* tested is
        everything downstream, in :func:`embed_and_validate`.
    """
    from scimilarity.utils import align_dataset, consolidate_duplicate_symbols, lognorm_counts

    adata = consolidate_duplicate_symbols(adata)   # duplicates summed
    adata = align_dataset(adata, gene_order)       # align BEFORE lognorm
    return lognorm_counts(adata)


def embed_and_validate(
    X: np.ndarray,
    obs: pd.DataFrame,
    encoder: Encoder,
    label_column: str,
    drop_unlabeled: bool = False,
    donor_column: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Embed query cells and enforce every invariant the rest of the pipeline assumes.

    Args:
        X: (n_cells, n_genes) preprocessed expression matrix.
        obs: (n_cells, ...) per-cell metadata; index holds the barcodes.
        encoder: maps X to (n_cells, 128).
        label_column: column of ``obs`` holding the Cell Ontology term.
        drop_unlabeled: discard cells with no term instead of refusing. Must be
            requested explicitly — see Raises.
        donor_column: column of ``obs`` identifying the subject a cell came
            from. Carried through so stage 5 can resample donors: cells from one
            subject are not independent observations.

    Returns:
        emb: (n_cells, 128) float32, unit norm, row-aligned with ``meta``.
        meta: query_id and y_true, one row per cell, in input order.

    Raises:
        KeyError: ``label_column`` is absent.
        ValueError: the encoder returned the wrong dimension, any cell
            embedded to the origin, or cells carry no ontology term while
            ``drop_unlabeled`` is False. All are refused here rather than
            allowed to become nan neighbours several stages later.

    Note:
        Unlabelled cells are a real case, not a hypothetical: the GSE136831
        query set has 544 multiplets — two cells sequenced as one — which
        therefore have no true cell type. Kept, they become the string "nan",
        count as wrong in every condition, and inflate both the clean error
        rate and the outcome failure rate. Dropping them is the right call but
        it changes the denominator, so it must be asked for rather than
        assumed.
    """
    if label_column not in obs.columns:
        raise KeyError(f"{label_column!r} is not a column of obs; have {list(obs.columns)}")

    labels = obs[label_column]
    unlabeled = labels.isna().to_numpy()
    if unlabeled.any():
        if not drop_unlabeled:
            raise ValueError(
                f"{unlabeled.sum()} cell(s) carry no ontology label in "
                f"{label_column!r}; pass drop_unlabeled=True to discard them "
                f"(this changes the denominator of every rate)"
            )
        keep = ~unlabeled
        X, obs = X[keep], obs.loc[keep]

    emb = np.asarray(encoder(X), dtype=EMB_DTYPE)
    if emb.shape[1] != EMB_DIM:
        raise ValueError(f"encoder returned dimension {emb.shape[1]}, expected {EMB_DIM}")

    norms = np.linalg.norm(emb, axis=1)
    dead = np.flatnonzero(norms == 0)
    if dead.size:
        raise ValueError(
            f"{dead.size} cell(s) produced a zero-norm embedding (first at row "
            f"{dead[0]}); cosine similarity is undefined for them"
        )

    emb = (emb / norms[:, None]).astype(EMB_DTYPE)
    cols = {
        QUERY_ID: obs.index.to_numpy(dtype=str),
        Y_TRUE: obs[label_column].to_numpy(dtype=str),
    }
    if donor_column:
        if donor_column not in obs.columns:
            raise KeyError(
                f"{donor_column!r} is not a column of obs; have {list(obs.columns)}"
            )
        cols[DONOR_ID] = obs[donor_column].to_numpy(dtype=str)
    meta = pd.DataFrame(cols)
    return emb, meta


# ── snakemake entry point ────────────────────────────────────────────────────
if "snakemake" in globals():  # pragma: no cover - exercised by the workflow
    import sys

    import anndata as _ad
    import numpy as _np

    from scimilarity import CellEmbedding as _CE

    _cfg = snakemake.config  # noqa: F821
    _adata = _ad.read_h5ad(snakemake.input.h5ad)  # noqa: F821
    _ce = _CE(model_path=snakemake.input.model)  # noqa: F821
    _prepped = preprocess_for_encoder(_adata, list(_ce.gene_order))

    # obs comes from the PREPROCESSED object, never the original file: if any
    # preprocessing step drops or reorders cells, using the original obs would
    # attach every prediction to the wrong ground-truth label.
    _emb, _meta = embed_and_validate(
        _prepped.X,
        _prepped.obs,
        _ce.get_embeddings,
        _cfg["query"]["label_column"],
        drop_unlabeled=bool(_cfg["query"]["drop_unlabeled"]),
        donor_column=_cfg["query"].get("donor_column"),
    )
    _dropped = _adata.n_obs - len(_emb)
    print(f"queries embedded: {len(_emb):,} (dropped {_dropped:,} unlabelled)", file=sys.stderr)
    if _dropped != _cfg["query"]["expected_n_unlabeled"]:
        print(
            f"  NOTE: expected {_cfg['query']['expected_n_unlabeled']:,} unlabelled; "
            f"the input has changed and this run needs a fresh look",
            file=sys.stderr,
        )
    _np.save(snakemake.output.emb, _emb)  # noqa: F821
    _meta.to_parquet(snakemake.output.meta, index=False)  # noqa: F821
