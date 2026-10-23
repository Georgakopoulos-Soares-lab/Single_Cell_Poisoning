"""Stage 2 — retrieve each query's clean top-k neighbours, exactly once.

This runs ONCE per profile. All 1,260 sweep conditions reuse its output, which
is what makes the benchmark laptop-feasible: the expensive search happens a
single time and only the small poison set is compared afterwards.

Retrieving exactly ``k`` is sufficient, not approximate — FOR INSERTION.
Adding poison to a reference can only push clean records DOWN the ranking, never
promote one, so any clean record in the true merged top-k already ranks in the
clean top-k. See ``SPEC.md`` §4.2. Under HNSW that argument fails, because an
inserted node changes graph traversal; stage 6 quantifies the gap.

**The argument also fails under MODIFY, and for a different reason.** Removing a
record -- which is what overwriting its embedding does to the old vector --
PROMOTES the next one, and a depth-k list has nothing to promote. The
``modify x synthesised`` attacks (SPEC.md §5.4) therefore need a deeper list,
built by ``retrieve_clean_deep.py`` at ``modify.DEEP_K``. This is not a
shortcoming of stage 2; it is the sufficiency argument being about insertion,
which is all that existed when it was written.
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

from schemas import EMB_DTYPE


def subsample_reference(n_total: int, n_keep: int, seed: int) -> np.ndarray:
    """Choose which reference rows the `smoke` profile uses.

    Args:
        n_total: rows in the full reference.
        n_keep: rows to retain.
        seed: passed in, never generated — the subsample must be re-derivable.

    Returns:
        (n_keep,) int64, sorted ascending and without repeats. Sorted so that
        reading the vectors out of the index is roughly sequential; unique so
        that a 500k subsample really contains 500k distinct cells.

    Raises:
        ValueError: ``n_keep`` exceeds ``n_total``.
    """
    if n_keep > n_total:
        raise ValueError(f"cannot keep {n_keep:,} rows from a reference of {n_total:,}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n_keep, replace=False))


def retrieve_exhaustive(
    query_emb: np.ndarray,
    ref_emb: np.ndarray,
    ref_labels: np.ndarray,
    k: int,
    block_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact top-k cosine neighbours by brute force.

    Args:
        query_emb: (n_q, d) float32, unit norm.
        ref_emb: (n_ref, d) float32, unit norm.
        ref_labels: (n_ref,) str.
        k: neighbours to return.
        block_size: queries per block. The (block, n_ref) cosine matrix is
            discarded after the per-row top-k, so peak memory stays flat.
            Arithmetically inert.

    Returns:
        cos: (n_q, k) float32, descending.
        labels: (n_q, k) str, aligned with ``cos``.
        idx: (n_q, k) int64, reference row indices.

    Raises:
        ValueError: ``k`` exceeds the reference size. Returning fewer than k
            silently would make stage 4's merge wrong in a way nothing
            downstream could detect.
    """
    n_ref = len(ref_emb)
    if k > n_ref:
        raise ValueError(f"k={k} exceeds the {n_ref:,}-row reference")

    n_q = len(query_emb)
    cos = np.empty((n_q, k), dtype=EMB_DTYPE)
    idx = np.empty((n_q, k), dtype=np.int64)

    for lo in range(0, n_q, block_size):
        hi = min(lo + block_size, n_q)
        c = (query_emb[lo:hi] @ ref_emb.T).astype(EMB_DTYPE)   # (b, n_ref)
        part = np.argpartition(-c, k - 1, axis=1)[:, :k]        # unordered top-k
        pc = np.take_along_axis(c, part, axis=1)
        order = np.argsort(-pc, axis=1, kind="stable")          # then order them
        idx[lo:hi] = np.take_along_axis(part, order, axis=1)
        cos[lo:hi] = np.take_along_axis(pc, order, axis=1)

    return cos, ref_labels[idx], idx


def load_reference(
    model_path: str, indices: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Read reference vectors and labels out of the SCimilarity model.

    Args:
        model_path: directory holding ``annotation/labelled_kNN.bin`` and
            ``annotation/reference_labels.tsv``.
        indices: rows to read. ``None`` reads all 7.9M — only do that for the
            ``full`` profile. The ``smoke`` profile passes its subsample so it
            never materialises vectors it will not use.

    Returns:
        emb: (n, 128) float32. Already unit norm in the shipped index, verified
            on load rather than assumed.
        labels: (n,) str, cell-type NAMES. The reference labels by name; stage 0
            maps those onto ontology ids.

    Raises:
        ValueError: a vector is not unit norm, which would silently break the
            cosine geometry every later stage depends on.

    Note:
        Loads an 8.6 GB index (~10 GB resident, ~20 s). ``CLAUDE.md`` §1
        requires explicit human approval for any index over 2 GB; this must
        never run from an unattended path without it. Not unit-tested: it is
        I/O over a file no fixture can stand in for.
    """
    import hnswlib

    from schemas import EMB_DIM

    knn_path = f"{model_path}/annotation/labelled_kNN.bin"
    label_path = f"{model_path}/annotation/reference_labels.tsv"

    index = hnswlib.Index(space="cosine", dim=EMB_DIM)
    index.load_index(knn_path)
    n_total = index.get_current_count()

    rows = np.arange(n_total) if indices is None else np.asarray(indices)
    emb = np.asarray(
        index.get_items(rows.tolist(), return_type="numpy"), dtype=EMB_DTYPE
    )

    norms = np.linalg.norm(emb, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError(
            f"reference vectors are not unit norm (min {norms.min():.5f}, "
            f"max {norms.max():.5f}); cosine geometry assumes they are"
        )

    with open(label_path) as fh:
        all_labels = np.array([line.split("\t")[0].strip() for line in fh])
    if len(all_labels) != n_total:
        raise ValueError(
            f"{len(all_labels):,} labels for {n_total:,} vectors — the index "
            f"and label file disagree"
        )

    return emb, all_labels[rows]


# ── snakemake entry point ────────────────────────────────────────────────────
if "snakemake" in globals():  # pragma: no cover - exercised by the workflow
    import sys

    import numpy as _np

    from schemas import K_NEIGHBORS

    _ref = _np.load(snakemake.input.reference)  # noqa: F821
    _q = _np.load(snakemake.input.query_emb)  # noqa: F821
    _mode = snakemake.config["retrieval"]  # noqa: F821
    if _mode != "exhaustive":
        raise ValueError(
            f"retrieval={_mode!r} is not implemented; the HNSW path belongs to "
            f"the full profile and stage 6"
        )
    _cos, _lab, _idx = retrieve_exhaustive(
        _q, _ref["emb"], _ref["label"], k=K_NEIGHBORS,
        block_size=int(snakemake.config["retrieve"]["query_block_size"]),  # noqa: F821
    )
    print(f"retrieved top-{K_NEIGHBORS} for {len(_cos):,} queries", file=sys.stderr)
    _np.savez(snakemake.output[0], cos=_cos, label=_lab, idx=_idx)  # noqa: F821
