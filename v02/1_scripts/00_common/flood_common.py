"""Helpers shared by the cluster-flood attack and the defences built against it.

Kept in their own module so a defence driver can import them without
executing the attack sweep as a side effect of the import.
"""
import pathlib
import re

import numpy as np

from schemas import EMB_DTYPE, N_BOOTSTRAP

CI = 95
BLOCK = 20_000


def donor_tables(tab: pathlib.Path) -> list[pathlib.Path]:
    """One ``by_donor`` table per step, preferring the GRAPH-FAITHFUL variant.

    A plain ``glob("*_by_donor.parquet")`` reads the WRONG table for the two
    steps that have both. ``02b_11`` and ``02b_15`` write their graph run to
    ``<stem>_by_donor_graph.parquet`` -- the ``_graph`` lands AFTER ``_by_donor``,
    so the pattern misses it and matches the ``exact`` table instead. That table
    is the ``modify.remerge`` shortcut, which understates the attack by an unknown
    per-celltype margin (up to 0.1240 on ``CL:0000583``; see HANDOFF.md 5). An
    aggregator would then have waited hours for the faithful numbers and silently
    summarised the understating ones.

    Rules, in order:

    1. Consider ``*_by_donor*.parquet``, not ``*_by_donor.parquet``.
    2. Drop ``_shard<n>`` files. Those are unmerged intermediates; ``merge_shards``
       has no cleanup, so they persist beside the canonical table and would
       double-count the conditions they were merged into.
    3. Group by the step id -- everything before ``_by_donor`` -- and within a
       group prefer the variant whose suffix mentions ``graph``.

    The caller is expected to LOG the returned names, so which variant won is
    visible in the run log rather than inferred.

    Args:
        tab: a ``2_results/<dataset>/<stage>`` directory.

    Returns:
        One path per step, sorted by step id.
    """
    picked: dict[str, tuple[pathlib.Path, bool]] = {}
    for p in sorted(tab.glob("*_by_donor*.parquet")):
        if re.search(r"_shard\d+$", p.stem):
            continue
        step, _, rest = p.stem.partition("_by_donor")
        graph = "graph" in rest
        if step not in picked or (graph and not picked[step][1]):
            picked[step] = (p, graph)
    return [picked[s][0] for s in sorted(picked)]


def donor_ratio_ci(num, den, donor, seed):
    """Ratio of per-query counts, resampling donors."""
    n, d = np.asarray(num, float), np.asarray(den, float)
    if d.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    uniq, inv = np.unique(donor, return_inverse=True)
    ns = np.bincount(inv, weights=n, minlength=len(uniq))
    ds = np.bincount(inv, weights=d, minlength=len(uniq))
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(uniq), size=(N_BOOTSTRAP, len(uniq)))
    db = ds[pick].sum(axis=1)
    ok = db > 0
    boot = ns[pick].sum(axis=1)[ok] / db[ok]
    lo, hi = np.percentile(boot, [(100 - CI) / 2, 100 - (100 - CI) / 2])
    return float(n.sum() / d.sum()), float(lo), float(hi)


def donor_boot_ratios(num, den, seed):
    """Bootstrap ratio distributions for SEVERAL statistics on ONE donor draw.

    ``donor_ratio_ci`` above answers "what is the interval on this one rate",
    and resamples the donors itself. That is the wrong primitive for a PAIRED
    comparison: a defended rate and its undefended twin must be recomputed on
    the SAME resampled donors, or the difference picks up variance that the
    pairing is there to remove. Two independent calls to ``donor_ratio_ci``
    and a subtraction would give an interval far too wide, and wrongly
    centred whenever the two rates are correlated -- which, sharing a
    neighbour list, they overwhelmingly are.

    So this returns the bootstrap DISTRIBUTIONS and lets the caller combine
    them. One donor draw drives every row, which makes level intervals and
    paired-difference intervals mutually consistent by construction.

    Args:
        num: (n_stat, n_donor) float numerators, one row per statistic, all
            sharing the same donor axis in the same order.
        den: (n_stat, n_donor) float denominators, aligned with ``num``.
        seed: passed to ``default_rng``; never generated here.

    Returns:
        ``(point, boot)``. ``point`` is (n_stat,) -- ``num.sum()/den.sum()``
        per row, the same estimator ``donor_ratio_ci`` reports. ``boot`` is
        (n_stat, N_BOOTSTRAP), each column one resample of the donors applied
        to every statistic. Rows whose denominator is zero are ``nan`` in both.
    """
    n = np.atleast_2d(np.asarray(num, float))
    d = np.atleast_2d(np.asarray(den, float))
    if n.shape != d.shape:
        raise ValueError(f"num {n.shape} and den {d.shape} must match")
    n_don = n.shape[1]
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, n_don, size=(N_BOOTSTRAP, n_don))
    # (n_stat, N_BOOTSTRAP): sum the resampled donors for every statistic at
    # once. Transient cost is n_stat x N_BOOTSTRAP x n_donor floats.
    nb = n[:, pick].sum(axis=2)
    db = d[:, pick].sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = np.where(db > 0, nb / db, np.nan)
        tot_d = d.sum(axis=1)
        point = np.where(tot_d > 0, n.sum(axis=1) / tot_d, np.nan)
    return point, boot


def boot_ci(boot):
    """Percentile interval of a (..., N_BOOTSTRAP) bootstrap distribution.

    Args:
        boot: bootstrap draws along the LAST axis, ``nan`` where undefined.

    Returns:
        ``(lo, hi)``, each with ``boot``'s leading shape. All-``nan`` rows give
        ``nan`` rather than raising, because a condition with no queries for
        the attacked celltype is a real condition and must survive to the table.
    """
    b = np.asarray(boot, float)
    ok = np.isfinite(b).any(axis=-1)
    lo = np.full(b.shape[:-1], np.nan)
    hi = np.full(b.shape[:-1], np.nan)
    if ok.any():
        q = np.nanpercentile(b[ok], [(100 - CI) / 2, 100 - (100 - CI) / 2], axis=-1)
        lo[ok], hi[ok] = q[0], q[1]
    return lo, hi


def top_poison(qe: np.ndarray, pe: np.ndarray, k: int,
               block: int = BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """Best k poison cosines per query, computed in blocks over the poison set.

    Args:
        qe: (n_q, d) queries. pe: (n_p, d) poison. k: how many to keep.
        block: poison rows per matmul. The transient cost is
            n_q x block x 4 bytes, so a caller running many workers
            in parallel should lower it.

    Returns:
        (n_q, k) cosines and their poison indices, descending.
    """
    n_q = len(qe)
    best_c = np.full((n_q, k), -np.inf, EMB_DTYPE)
    best_i = np.zeros((n_q, k), np.int64)
    for lo in range(0, len(pe), block):
        blk = pe[lo:lo + block]
        c = (qe @ blk.T).astype(EMB_DTYPE)
        kk = min(k, c.shape[1])
        idx = np.argpartition(-c, kk - 1, axis=1)[:, :kk]
        cc = np.take_along_axis(c, idx, axis=1)
        cat_c = np.concatenate([best_c, cc], axis=1)
        cat_i = np.concatenate([best_i, idx + lo], axis=1)
        keep = np.argpartition(-cat_c, k - 1, axis=1)[:, :k]
        best_c = np.take_along_axis(cat_c, keep, axis=1)
        best_i = np.take_along_axis(cat_i, keep, axis=1)
    o = np.argsort(-best_c, axis=1)
    return np.take_along_axis(best_c, o, axis=1), np.take_along_axis(best_i, o, axis=1)


def kmeans_sites(pool: np.ndarray, k: int, seed: int) -> np.ndarray:
    """``k`` unit-norm cluster centres of the attacked celltype.

    MiniBatchKMeans rather than an exact Lloyd's: measured on this machine at
    n=50,000 and k=256, ``scipy.cluster.vq.kmeans2`` takes 37.4 s CPU for ten
    iterations against 5.7 s for a hundred here, and the largest celltype is
    368,537 records. Exactness of the clustering is not a property any result
    depends on -- the centres are attack LOCATIONS, not a measurement -- so the
    cheaper approximation is the right one.

    Args:
        pool: (n_src, d) float32 unit vectors of the attacked celltype.
        k: how many centres. seed: passed in, never generated.

    Returns:
        (k, d) float32, unit-normalised. Fewer than ``k`` rows if the pool is
        smaller, which MiniBatchKMeans enforces for us.
    """
    from sklearn.cluster import MiniBatchKMeans  # noqa: PLC0415

    km = MiniBatchKMeans(n_clusters=min(k, len(pool)), random_state=seed,
                         n_init=3, batch_size=4096, max_iter=100).fit(pool)
    cen = km.cluster_centers_.astype(EMB_DTYPE)
    return cen / np.linalg.norm(cen, axis=1, keepdims=True)


def build_flood(pool: np.ndarray, n_anchors: int, rate: float, spread: float,
                jitter: float, seed: int, sites: str = "anchor") -> np.ndarray:
    """The cluster-flood poison set: many small clusters, each nudged at a target.

    Anchors are real source-class embeddings, so the flood inherits the true
    manifold instead of sitting at an implausible centroid. Each anchor is
    rotated ``spread`` of the way toward the source centroid's direction and its
    copies are jittered by ``jitter``, which is what defeats near-duplicate
    filtering: the copies are no more similar to each other than genuine cells
    from one study are.

    Args:
        pool: (n_src, d) float32 unit vectors of the source class.
        n_anchors: how many clusters. rate: poison count as a multiple of the
            source class. spread: cosine retained toward the anchor.
            jitter: cosine retained after per-copy noise. seed: RNG seed.
        sites: ``anchor`` places clusters on real reference cells, ``kmeans_centres``
            on cluster centres of the celltype. See :class:`schemas.Sites`. The
            ``anchor`` path consumes the RNG exactly as it did before this
            parameter existed, so every committed flood condition reproduces
            byte for byte.

    Returns:
        (n_p, d) float32 unit-normalised poison embeddings.
    """
    aim = pool.mean(axis=0)
    aim = (aim / np.linalg.norm(aim)).astype(EMB_DTYPE)
    rng = np.random.default_rng(seed)
    if sites == "anchor":
        pick = rng.choice(len(pool), min(n_anchors, len(pool)), replace=False)
        cen = pool[pick].astype(EMB_DTYPE)
    elif sites == "kmeans_centres":
        cen = kmeans_sites(pool, n_anchors, seed)
    else:
        raise ValueError(f"unknown sites {sites!r}; see schemas.Sites")
    n_p = int(round(rate * len(pool)))
    per = np.full(len(cen), n_p // len(cen), dtype=int)
    per[: n_p - per.sum()] += 1
    pe = np.repeat(cen, per, axis=0)
    t = np.broadcast_to(aim, pe.shape)
    w = t - np.einsum("ij,ij->i", t, pe)[:, None] * pe
    nrm = np.linalg.norm(w, axis=1, keepdims=True)
    w = np.divide(w, nrm, out=np.zeros_like(w), where=nrm > 1e-8)
    pe = spread * pe + np.sqrt(1.0 - spread ** 2) * w
    g = rng.normal(size=pe.shape).astype(EMB_DTYPE)
    g -= np.einsum("ij,ij->i", g, pe)[:, None] * pe
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    pe = jitter * pe + np.sqrt(1.0 - jitter ** 2) * g
    return (pe / np.linalg.norm(pe, axis=1, keepdims=True)).astype(EMB_DTYPE)
