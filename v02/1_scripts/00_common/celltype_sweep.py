"""Machinery shared by the per-celltype graph sweeps.

Two steps ask different questions of the same experiment. ``02b_01`` varies the
ATTACK at a fixed dose ladder; ``02b_02`` varies flood's ANCHOR COUNT at a fixed
attack. Everything between "here is a poison set" and "here is a scored row" is
identical, and has to stay identical or the two tables cannot be read beside each
other -- which is the whole point of running them.

So it lives here once. The two drivers own their grid, their identity columns and
their docstring; they own no scoring. A change to how a vote is counted, a tie is
bounded or an interval is resampled lands in both by construction rather than by
someone remembering to copy it.

:class:`Bench` holds everything invariant across conditions -- queries, labels,
the ontology tables, the clean baseline -- so a driver loads once and then calls
:meth:`Bench.measure` per condition. It reads files but computes no result on its
own; every number a driver writes comes from ``vote.py`` and ``flood_common``
through here.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

from dataset_config import cache_root, query_cache
from flood_common import donor_ratio_ci
from ontology import Ontology
from schemas import (ATTACK_TIER, DEN_SUFFIX, EMB_DIM, EMB_DTYPE, K_NEIGHBORS,
                     N_QUERIES, NUM_SUFFIX, SCORE_QC, scimilarity_score)
from vote import VoteResult, bounds, vote

MIN_CELLTYPE = 2000       # reference records needed to be worth attacking
MIN_QUERY_CELLS = 100     # query cells needed to measure anything
TARGET_DIST = 4           # ontology distance each source's target aims for
DEFAULT_EF = 100          # SCimilarity's own default on the annotation path
UNMAPPED = "UNMAPPED"     # reference label carrying no ontology id

#: Flood's geometry, passed explicitly by the drivers rather than left to
#: ``attacks.py``'s defaults, which reproduce no v01 condition.
#:
#: ``build_flood``'s ``jitter`` is the cosine RETAINED after per-copy noise, so
#: 1.0 is no noise and small values are large noise. ``attacks.py`` defaults it
#: to 0.25, which retains only cosine 0.25 with the anchor and leaves the poison
#: ~97% random direction. Measured on real fibroblasts that lands at cosine 0.244
#: to the source centroid, against a fiftieth genuine neighbour at ~0.980: it
#: would never be retrieved, and flood would score as inert.
#:
#: The 0.25 is a misread of v01. ``run_cluster_flood.py`` builds plans as
#: ``[("anchor directed", m, 0.99, r) for r in RATES ...]`` and unpacks them
#: ``for mode, m, spread, rate in plans`` -- the fourth element is the RATE, and
#: that script applies no per-copy noise at all, which is why
#: ``cluster_flood.parquet`` carries no jitter column.
#:
#: These values reproduce it. v01's other flood scripts (``run_flood_dose.py``,
#: the defence matrix) used 0.999/0.999: near-identical geometry with a whisker
#: of noise to survive near-duplicate screening, which matters only once a dedup
#: defence is in front of it. Nothing in 02_b is defended.
FLOOD_SPREAD = 0.99
FLOOD_JITTER = 1.0


def seed_of(source: str, rate: float, seed: int) -> int:
    """Stable seed for one condition. Verbatim from v01, so the poison matches.

    ``hash()`` is salted per process and would not reproduce. The ATTACK is
    deliberately absent from the key, so ``relabel`` and ``interpolate`` draw the
    same source records and interpolation's effect is the vectors' movement
    alone.
    """
    key = f"{source}|{rate!r}|{seed}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")


#: Why every sweep also WRITES the value this returns, in ``condition_seed``.
#:
#: ``rate!r`` above makes the key depend on the float's repr, which makes
#: reproduction depend on carrying the rate at its original width (see
#: :data:`KEY_COLUMNS`). Storing the derived integer removes that dependence
#: outright: a reader seeds the generator from the column and rebuilds the poison
#: without re-deriving anything. The recipe stays published for auditing; the
#: column is what makes it unnecessary.


def require_tiers(attacks: tuple[str, ...]) -> None:
    """Fail in the first second if any attack has no :class:`schemas.AttackTier`.

    An unlabelled ceiling reported beside realistic ones turns "attacks are
    possible" into a number that reads as measured risk. ``schemas.py`` is
    orchestrator-owned, so this reports the gap rather than filling it in.

    Raises:
        ValueError: naming every attack missing from ``ATTACK_TIER``.
    """
    missing = [str(a) for a in attacks if a not in ATTACK_TIER]
    if missing:
        raise ValueError(
            f"schemas.ATTACK_TIER has no entry for {missing}; add one there "
            "(orchestrator-owned) before running. Emitting a null tier would "
            "publish a ceiling result that looks like a realistic one.")


def eligible_sources(ref_label: np.ndarray, truth: np.ndarray) -> list[str]:
    """Celltypes with enough reference records to poison and enough queries to
    measure on.

    Args:
        ref_label: (n_ref,) str reference Cell Ontology terms.
        truth: (n_q,) str query ground-truth terms.

    Returns:
        Sorted CL ids passing both floors. Unmapped reference rows are excluded:
        poisoning one spends budget on a record no ontology statement can be made
        about.
    """
    qn = pd.Series(truth).value_counts()
    terms, counts = np.unique(ref_label[ref_label != UNMAPPED], return_counts=True)
    return [str(t) for t, c in zip(terms, counts)
            if c >= MIN_CELLTYPE and qn.get(t, 0) >= MIN_QUERY_CELLS]


def pick_targets(sources: list[str], onto: Ontology) -> dict[str, tuple[str, int]]:
    """One target per source, at a controlled ontology distance.

    Reproduces v01's ``run_celltype_sweep`` choice exactly, so the graph arm
    attacks the same pairs the exact arm did. Each source takes the eligible
    celltype whose tree distance is closest to :data:`TARGET_DIST`, ties broken
    on the CL id string. Giving every source a near/mid/far ladder instead would
    triple the cost, and HANDOFF §3.17 showed distance is confounded with which
    celltype is attacked; holding it roughly fixed while the source varies is
    what isolates the source.

    Args:
        sources: eligible CL ids. onto: the Cell Ontology graph.

    Returns:
        ``{source: (target, realised distance)}``. Sources with no reachable
        target are absent rather than given a fabricated one.
    """
    out: dict[str, tuple[str, int]] = {}
    for s in sources:
        cand = [(abs(d - TARGET_DIST), t, d) for t in sources if t != s
                for d in [onto.distance(s, t)] if d is not None]
        if cand:
            _, t, d = min(cand, key=lambda x: (x[0], x[1]))
            out[s] = (t, int(d))

    # A target that is an ANCESTOR or DESCENDANT of its source would make
    # "attack success" ambiguous: a vote moving from CD4 T cell to T cell is the
    # attacker's label AND a correct answer at coarser granularity, so outcome
    # failure would count a classifier that is arguably still right.
    #
    # On the 26 eligible celltypes of GSE136831 this never happens -- measured,
    # 0 of 26, at distances 4 (x22) and 5 (x4) -- because minimising |d - 4|
    # lands across a branch rather than along a lineage. But that is a property
    # of THIS celltype set, not of the rule. A different dataset, or a different
    # MIN_CELLTYPE, could produce one and nothing downstream would notice.
    bad = [(s, t) for s, (t, _) in out.items()
           if onto.is_ancestor_or_descendant(s, t)]
    if bad:
        raise ValueError(
            "pick_targets chose a target in its source's own lineage, which "
            "makes attack success indistinguishable from a coarser correct "
            f"answer: {bad}. Raise TARGET_DIST or exclude the pair; do not "
            "score it.")
    return out


def tie_columns(acc: VoteResult, hit: VoteResult, truth: np.ndarray,
                target: str) -> dict[str, np.ndarray]:
    """Per-query tie bounds for both metrics, from two votes over the same
    neighbours.

    ``vote`` marks a tie ambiguous only when the PIVOTAL label is in it, and the
    pivotal label differs by metric, so the same neighbourhoods are voted twice
    -- once pivoted on each query's own true label, once on the attacker's
    target. ``pred`` and ``n_tied`` are identical between the two by
    construction; only the ambiguity flag differs. ``bounds`` requires the vote
    it is given to have been pivoted on the label it is passed and does not check
    that, so the two are paired here rather than at a call site.

    Args:
        acc: vote with ``pivotal=truth``. hit: vote with ``pivotal=target``.
        truth: (n_q,) str ground truth. target: the attacker's label.

    Returns:
        Per-query 1.0/0.0 indicators: accuracy and attack success under each
        reading of the ties, the tie rate, and the two ambiguous rates.
    """
    acc_best, acc_worst = bounds(acc, truth)
    hit_best, hit_worst = bounds(hit, target)
    return {
        "acc_best": acc_best, "acc_worst": acc_worst,
        "attack_success_best": hit_best, "attack_success_worst": hit_worst,
        "tie_rate": (acc.n_tied > 1).astype(float),
        "ambiguous_rate_accuracy": acc.pivotal_in_tie.astype(float),
        "ambiguous_rate_success": hit.pivotal_in_tie.astype(float),
    }


def scope(prefix: str, sel: np.ndarray, donor: np.ndarray, boot_seed: int,
          **num: np.ndarray | tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """Donor-paired point estimate and 95% interval for every indicator, on one
    scope of queries.

    Donors are the unit of independence: cells from one subject share genotype,
    batch, dissociation and sequencing run, so the bootstrap resamples whole
    donors rather than cells. Indicators whose name ends ``_delta`` or ``_net``
    are already per-query DIFFERENCES between the poisoned and clean verdict, so
    each donor supplies both sides and donor-to-donor variation in baseline
    accuracy cancels instead of swamping the effect.

    Every column is a ratio estimator -- numerator and denominator summed over
    resampled donors, then divided. A rate is that with a 0/1 numerator and a
    denominator of 1, which is what a bare array means. An explicit denominator
    gives a MEAN over a subset: ``mean_error_dist`` supplies the hop count as
    numerator and reachability as denominator, so unreachable pairs leave both
    sums rather than entering the average as a sentinel. Resampling stays at the
    donor either way.

    Args:
        prefix: ``within`` or ``global``, prepended to every emitted column.
        sel: (m,) int indices of the queries in scope.
        donor: (n_q,) donor id per query. boot_seed: passed in, never generated.
        num: ``{name: (n_q,) indicator}`` for a rate, or
            ``{name: ((n_q,) numerator, (n_q,) denominator)}`` for a mean.

    Returns:
        ``{f"{prefix}_{name}": value, f"{prefix}_{name}_lo": ..., "_hi": ...}``.
    """
    d = donor[sel]
    ones = np.ones(len(sel))
    out: dict[str, float] = {}
    for name, v in num.items():
        n, den = v if isinstance(v, tuple) else (v, None)
        est, lo, hi = donor_ratio_ci(
            np.asarray(n)[sel].astype(float),
            ones if den is None else np.asarray(den)[sel].astype(float),
            d, boot_seed)
        out[f"{prefix}_{name}"] = est
        out[f"{prefix}_{name}_lo"] = lo
        out[f"{prefix}_{name}_hi"] = hi
    return out


def by_donor(prefix: str, sel: np.ndarray, donor: np.ndarray,
             vocab: np.ndarray,
             **num: np.ndarray | tuple[np.ndarray, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-donor numerator and denominator sums for every indicator, unreduced.

    The same ``ind`` dict :func:`scope` collapses to a point and an interval,
    kept at the donor instead. ``scope`` answers "what is the rate and how
    uncertain is it at THIS dose"; this answers "what did each donor contribute",
    which is the only form a bootstrap can re-pool across doses under one fixed
    donor draw. That pairing across doses is what SPEC.md 5.4c's threshold
    interval rests on, and it is unavailable from any already-aggregated table.

    Emits SUMS, never rates. Dividing here would force the downstream bootstrap
    to average per-donor rates, weighting a donor with 9 queries equally with one
    contributing 4,000 -- a different estimator from the ratio estimator every
    published interval in this project already uses. See ``schemas.BY_DONOR_DOC``.

    Rows are emitted for EVERY donor in ``vocab``, not only those present in
    ``sel``. An absent donor gets 0/0, a structural zero rather than a missing
    value, so that ``within_`` and ``global_`` align row-for-row and can share
    one row per the SCOPE exception.

    Args:
        prefix: ``within`` or ``global``, prepended to every emitted column.
        sel: (m,) int indices of the queries in scope.
        donor: (n_q,) donor id per query.
        vocab: (n_d,) sorted unique donor ids -- the full roster, fixed across
            conditions so every condition emits the same donors in the same
            order.
        num: as :func:`scope`.

    Returns:
        ``{f"{prefix}_{name}_num": (n_d,) float64, ..._den: (n_d,) float64}``
        plus ``f"{prefix}_n_queries"`` as an (n_d,) int64 query count.
    """
    idx = np.searchsorted(vocab, donor[sel])
    n_d = len(vocab)
    ones = np.ones(len(sel))
    out: dict[str, np.ndarray] = {
        f"{prefix}_{N_QUERIES}": np.bincount(idx, minlength=n_d).astype(np.int64)
    }
    for name, v in num.items():
        n, den = v if isinstance(v, tuple) else (v, None)
        # float64 throughout: these are SUMS that a bootstrap re-pools 2000
        # times, and the project's float32 rule is about the 23.4M x 128
        # embedding arrays, not about 78 accumulators.
        out[f"{prefix}_{name}{NUM_SUFFIX}"] = np.bincount(
            idx, weights=np.asarray(n)[sel].astype(float), minlength=n_d)
        out[f"{prefix}_{name}{DEN_SUFFIX}"] = np.bincount(
            idx, weights=ones if den is None
            else np.asarray(den)[sel].astype(float), minlength=n_d)
    return out


@dataclass(frozen=True, eq=False)
class CleanTarget:
    """The clean graph's rate for one attacker target, and its tie bounds.

    Depends on the target but not on the condition, so a driver builds it once
    per source rather than once per condition.
    """

    hit: np.ndarray
    best: np.ndarray
    worst: np.ndarray


@dataclass(frozen=True, eq=False)
class Retrieved:
    """One condition's raw graph answer, before anything is reduced to a rate.

    :meth:`Bench.measure` used to compute all of this inline and keep only the
    rates, so the per-query predictions -- which query was taken, and what it
    was taken to -- were thrown away at the end of every condition. They are the
    only thing a query-side figure can be drawn from, and 02b_30 needs them, so
    the retrieval and the two votes live here and ``measure`` reduces what this
    returns. One retrieval, one vote, two readers: the per-query table and the
    rate table cannot disagree about what the graph answered.

    Attributes:
        is_poison: (n_q, k) bool, whether each returned neighbour is a poisoned
            record (hnswlib id >= ``n_ref``).
        topk_cos: (n_q, k) float32 cosine to each returned neighbour,
            descending. hnswlib's cosine space returns ``1 - cos``; this is
            already converted, because the SCimilarity score is defined on the
            similarity and not on the distance.
        acc: the vote pivoted on each query's TRUE label -- the accuracy
            reading of the ties.
        hit: the same vote pivoted on the attacker's target -- the attack
            success reading. ``pred`` and ``n_tied`` are identical between the
            two by construction; only the ambiguity flag differs.
    """

    is_poison: np.ndarray
    topk_cos: np.ndarray
    acc: VoteResult
    hit: VoteResult
    ids: np.ndarray | None = None
    """(n_q, k) int64 hnswlib ids the graph returned, descending by similarity.

    The IDENTITY of each neighbour, which ``is_poison`` and ``topk_cos`` reduce
    away: two retrievals that return different records at the same distances are
    indistinguishable from those two arrays alone. ``02b_34`` needs the identity,
    because the question it asks is how much the GRAPH moves between two
    insertions of the same poison set, separately from how much the ANSWER moves.

    Populated by :meth:`Bench.retrieve` on every real retrieval. ``None`` only
    where a caller constructs a :class:`Retrieved` by hand -- the tests do -- and
    no consumer of the rates reads it.
    """

    def nearest_topk_cos(self) -> np.ndarray:
        """(n_q,) float32 cosine to the nearest poison AMONG THE RETURNED k.

        ``-inf`` where the graph returned no poisoned record at all, which is
        not the same statement as "no poison is near this query": the graph is
        asked for k neighbours and says nothing about anything it did not
        return. A caller wanting the nearest poison in the whole set has to
        compute it, and 02b_30 does.
        """
        return np.where(self.is_poison, self.topk_cos, -np.inf).max(axis=1)


@dataclass(frozen=True, eq=False)
class Bench:
    """Everything invariant across conditions, loaded once.

    Holds the queries, the reference, the ontology tables and the clean graph
    baseline. A driver constructs one with :meth:`load`, then calls
    :meth:`measure` and :meth:`summarise` per condition.
    """

    idx_path: pathlib.Path
    ef: int
    boot_seed: int
    q: np.ndarray
    truth: np.ndarray
    truth_code: np.ndarray
    donor: np.ndarray
    ref_label: np.ndarray
    ref_emb: np.ndarray
    ref_id: np.ndarray
    n_ref: int
    vocab: np.ndarray
    lin: np.ndarray
    dist: np.ndarray
    clean_lab: np.ndarray
    clean_idx: np.ndarray
    """(n_q, k) int64 reference rows the CLEAN graph returned.

    The attacker's observed traffic, and the only thing ``hub`` selection needs.
    Held here rather than re-read per condition because it is the same table the
    clean baseline above was scored from -- one file, one read, no chance of the
    ranking and the scoring disagreeing about which retrieval they describe.
    """

    clean_ok: np.ndarray
    clean_lin: np.ndarray
    clean_err: np.ndarray
    clean_reach: np.ndarray
    clean_best: np.ndarray
    clean_worst: np.ndarray
    sources: list[str]
    target_of: dict[str, tuple[str, int]]
    qi_of: dict[str, np.ndarray]
    pool_of: dict[str, np.ndarray]

    @classmethod
    def load(cls, cfg: dict, base: pathlib.Path, t0: float) -> Bench:
        """Read the reference, queries, ontology and clean baseline.

        Args:
            cfg: parsed ``config/config.yaml``. base: the version root (``v02``).
            t0: start time, for the progress lines.

        Returns:
            A populated :class:`Bench`.

        Raises:
            ValueError: query embeddings and metadata disagree in length, or the
                committed clean baseline was retrieved at a different ``ef``/``k``
                from the one this sweep will use -- in which case its rates are
                not the counterfactual of the poisoned ones.
        """
        cache = cache_root(base, cfg)        # shared: reference.npz, poison/
        qcache = query_cache(base, cfg)      # this dataset only
        baseline = base / "2_results" / cfg["dataset"] / "01_a_baseline"
        ef = int(cfg.get("hnsw", {}).get("ef_search", DEFAULT_EF))

        ref = np.load(cache / "reference.npz", allow_pickle=True)
        ref_label = ref["label"].astype(str)
        n_ref = len(ref_label)
        q = np.load(qcache / "query_emb.npy").astype(EMB_DTYPE)
        meta = pd.read_parquet(qcache / "query_meta.parquet")
        truth = meta["y_true"].to_numpy().astype(str)
        donor = meta["donor_id"].to_numpy()
        if len(q) != len(truth):
            raise ValueError(f"{len(q):,} embeddings but {len(truth):,} metadata rows")

        sources = eligible_sources(ref_label, truth)
        onto = Ontology.from_obo(base.parent / cfg["ontology"]["obo"])
        target_of = pick_targets(sources, onto)
        sources = [s for s in sources if s in target_of]
        print(f"[{time.time()-t0:.0f}s] {len(sources)} attackable celltypes, covering "
              f"{int(np.isin(truth, sources).sum()):,} of {len(truth):,} queries",
              file=sys.stderr)

        # Graded tolerance and hop counts, over every term either side can
        # produce. Exact id equality scores 'alveolar macrophage' against
        # 'macrophage' as a failure when it is two annotators disagreeing about
        # granularity. Sorted, so a predicted label indexes in with searchsorted.
        vocab = np.unique(np.concatenate([np.unique(ref_label), np.unique(truth)]))
        n_v = len(vocab)
        lin = np.zeros((n_v, n_v), dtype=bool)
        dist = np.full((n_v, n_v), -1, dtype=np.int16)
        for i, a in enumerate(vocab.tolist()):
            dist[i, i] = 0
            for j, b in enumerate(vocab.tolist()):
                lin[i, j] = a == b or onto.is_ancestor_or_descendant(a, b)
                if j > i:
                    hops = onto.distance(a, b)
                    dist[i, j] = dist[j, i] = -1 if hops is None else hops
        truth_code = np.searchsorted(vocab, truth)
        n_unreach = int((dist < 0).sum())
        print(f"[{time.time()-t0:.0f}s] vocabulary {n_v} terms; lineage and distance "
              f"tables built ({n_unreach:,} of {n_v*n_v:,} ordered pairs unreachable, "
              f"all of them involving a term with no ontology id)", file=sys.stderr)

        # The clean graph baseline, read rather than recomputed: step 01a_01
        # already queried this index unpoisoned at the same ef and k. Put through
        # the SAME vote() as the poisoned side, so neither half of a difference
        # is scored by a rule the other half did not use.
        bs = pd.read_parquet(baseline / "01a_01_no_access_hnsw_summary.parquet").iloc[0]
        if int(bs["ef_search"]) != ef or int(bs["k"]) != K_NEIGHBORS:
            raise ValueError(
                f"baseline was retrieved at ef={bs['ef_search']}, k={bs['k']} but "
                f"this sweep runs at ef={ef}, k={K_NEIGHBORS}; the clean rates "
                "would not be the counterfactual of the poisoned ones")
        _clean = np.load(baseline / "01a_01_no_access_hnsw_neighbors.npz")
        clean_lab, clean_idx = _clean["label"], _clean["idx"].astype(np.int64)
        clean_acc = vote(clean_lab, pivotal=truth)
        clean_best, clean_worst = bounds(clean_acc, truth)
        clean_ok = (clean_acc.pred == truth).astype(float)
        clean_code = np.searchsorted(vocab, clean_acc.pred)
        clean_lin = lin[truth_code, clean_code].astype(float)
        clean_d = dist[truth_code, clean_code]
        clean_reach = (clean_d >= 0).astype(float)
        # numerator carries 0 where there is no path; the denominator drops the
        # row, so it leaves the mean rather than entering it as a sentinel
        clean_err = np.where(clean_d >= 0, clean_d, 0).astype(float)
        print(f"[{time.time()-t0:.0f}s] clean graph baseline: accuracy "
              f"{clean_ok.mean():.5f} exact ({clean_worst.mean():.5f}-"
              f"{clean_best.mean():.5f} over ties), {clean_lin.mean():.5f} "
              f"lineage-tolerant, mean error "
              f"{clean_err.sum()/clean_reach.sum():.3f} hops over "
              f"{int(len(truth) - clean_reach.sum()):,} unreachable pairs",
              file=sys.stderr)

        return cls(
            idx_path=(base.parent / cfg["paths"]["reference_dir"] / "annotation"
                      / "labelled_kNN.bin"),
            ef=ef, boot_seed=int(cfg["bootstrap_seed"]),
            q=q, truth=truth, truth_code=truth_code, donor=donor,
            ref_label=ref_label, ref_emb=ref["emb"], ref_id=ref["cell_id"],
            n_ref=n_ref, vocab=vocab, lin=lin, dist=dist,
            clean_lab=clean_lab, clean_idx=clean_idx, clean_ok=clean_ok, clean_lin=clean_lin,
            clean_err=clean_err, clean_reach=clean_reach,
            clean_best=clean_best, clean_worst=clean_worst,
            sources=sources, target_of=target_of,
            qi_of={s: np.flatnonzero(truth == s) for s in sources},
            pool_of={s: np.flatnonzero(ref_label == s) for s in sources},
        )

    def clean_target(self, target: str) -> CleanTarget:
        """The clean graph's rate for ``target``, with its tie bounds.

        A second vote of the same clean neighbours, pivoted on the target.
        Giving the attacker every tie on the poisoned side while denying them
        the clean ties would mix two tie-break rules inside one difference.
        """
        v = vote(self.clean_lab, pivotal=target)
        best, worst = bounds(v, target)
        return CleanTarget((v.pred == target).astype(float), best, worst)

    def open_index(self, pe: np.ndarray):
        """Open the shipped graph and append the poison.

        Args:
            pe: (n_p, 128) float32 unit-norm poison vectors.

        Returns:
            A loaded ``hnswlib.Index`` holding ``n_ref + n_p`` vectors, poison at
            ids ``>= n_ref``. NEVER built: ``init_index`` on 7.9M nodes at M=80
            takes hours and would yield a DIFFERENT graph from the one
            SCimilarity ships, voiding the claim that this is their reference.

        Raises:
            ValueError: ``ef < k``, or the index holds a different number of
                vectors from ``reference.npz`` -- in which case the ids would not
                correspond and every label would be wrong.
        """
        import hnswlib  # noqa: PLC0415

        if self.ef < K_NEIGHBORS:
            raise ValueError(
                f"ef {self.ef} < k {K_NEIGHBORS}: hnswlib requires ef >= k")
        ix = hnswlib.Index(space="cosine", dim=EMB_DIM)
        ix.load_index(str(self.idx_path), max_elements=self.n_ref + len(pe))
        if ix.get_current_count() != self.n_ref:
            raise ValueError(
                f"index holds {ix.get_current_count():,} vectors but "
                f"reference.npz has {self.n_ref:,}; ids would not correspond")
        ix.add_items(pe, np.arange(self.n_ref, self.n_ref + len(pe)))
        ix.set_ef(self.ef)
        return ix

    def retrieve(self, pe: np.ndarray, target: str) -> Retrieved:
        """Insert one poison set, query the graph, and vote. No scoring.

        Everything between "here is a poison set" and "here are the neighbours
        and what they voted for". Split out of :meth:`measure` so that a step
        wanting the per-query answer gets it from the same retrieval and the
        same vote the rates are computed from, rather than repeating either.

        Args:
            pe: (n_p, 128) float32 unit-norm poison vectors, all carrying
                ``target`` as their label. target: the attacker's label.

        Returns:
            A :class:`Retrieved` for these queries against the poisoned index.
        """
        ix = self.open_index(pe)
        ids, dst = ix.knn_query(self.q, k=K_NEIGHBORS)
        del ix
        ids = ids.astype(np.int64)

        # Poison sits at ids >= n_ref and every poisoned record carries the
        # attacker's label. Clipping and overwriting rather than concatenating a
        # labels array avoids copying 7.9M strings per condition to look up 2.5M.
        is_p = ids >= self.n_ref
        labels = self.ref_label[np.where(is_p, 0, ids)]
        labels[is_p] = target

        acc_v = vote(labels, pivotal=self.truth)
        hit_v = vote(labels, pivotal=target)
        del labels

        # hnswlib's cosine space returns 1 - cos; the QC bar is defined on the
        # score, so convert rather than comparing distances.
        return Retrieved(is_poison=is_p, topk_cos=(1.0 - dst).astype(EMB_DTYPE),
                         acc=acc_v, hit=hit_v, ids=ids)

    def measure(self, pe: np.ndarray, target: str,
                ct: CleanTarget) -> tuple[dict[str, Any], np.ndarray]:
        """Insert one poison set, query the graph, and score every query.

        Retrieval and scoring in one call, which is what every sweep driver
        wants. A caller needing to HOLD the retrieval -- to compare two of them
        against each other, as ``02b_34`` does -- calls :meth:`retrieve` and
        :meth:`score` instead and gets the identical numbers: this is those two
        in sequence and nothing else.

        Args:
            pe: (n_p, 128) float32 unit-norm poison vectors, all carrying
                ``target`` as their label. target: the attacker's label.
            ct: the clean-side rates for ``target``, from :meth:`clean_target`.

        Returns:
            ``(indicators, reach)``. ``indicators`` maps a metric name to a
            per-query array, or to a ``(numerator, denominator)`` pair where the
            metric is a mean over a subset. ``reach`` is the (n_q,) 1.0/0.0 mask
            of queries whose predicted term has an ontology path to the truth.
        """
        return self.score(self.retrieve(pe, target), target, ct)

    def score(self, r: Retrieved, target: str,
              ct: CleanTarget) -> tuple[dict[str, Any], np.ndarray]:
        """Score one already-retrieved condition. No retrieval, no randomness.

        Split out of :meth:`measure` so that a retrieval can be scored and ALSO
        kept. Under a multi-threaded ``add_items`` the graph is not reproducible
        (CLAUDE.md §1), so a step that re-runs a condition to measure that
        variation must compare the two retrievals themselves, not merely the two
        rates they reduce to -- and calling ``measure`` twice would throw the
        neighbours away both times.

        Pure in ``r``: given the same :class:`Retrieved` it returns the same
        indicators, so every number here is a property of the retrieval it was
        handed and of the clean baseline, never of when it was called.

        Args:
            r: one graph answer, from :meth:`retrieve`.
            target: the attacker's label, the one ``r.hit`` was pivoted on.
            ct: the clean-side rates for ``target``, from :meth:`clean_target`.

        Returns:
            As :meth:`measure`.
        """
        is_p, acc_v, hit_v = r.is_poison, r.acc, r.hit
        pred = acc_v.pred
        near = r.nearest_topk_cos()
        ok = (pred == self.truth).astype(float)
        got = (pred == target).astype(float)
        pred_code = np.searchsorted(self.vocab, pred)
        # hops from the predicted term to the true one: 0 on an exact match,
        # -1 where the is_a graph has no path between them
        err = self.dist[self.truth_code, pred_code]
        reach = (err >= 0).astype(float)
        err_num = np.where(err >= 0, err, 0).astype(float)
        both = reach * self.clean_reach      # a paired delta needs both sides

        ind: dict[str, Any] = dict(
            accuracy=ok,
            lineage=self.lin[self.truth_code, pred_code].astype(float),
            # LOWER IS BETTER, unlike every other accuracy column here
            mean_error_dist=(err_num, reach),
            baseline_mean_error_dist=(self.clean_err, self.clean_reach),
            mean_error_dist_delta=((err_num - self.clean_err) * both, both),
            # binary tolerance bands, v01's convention: an unreachable prediction
            # is certainly not within t hops, so it counts as a failure here
            # rather than being dropped as it is from the mean above
            within1=(reach * (err <= 1)).astype(float),
            within2=(reach * (err <= 2)).astype(float),
            within3=(reach * (err <= 3)).astype(float),
            baseline_accuracy=self.clean_ok,
            baseline_lineage=self.clean_lin,
            accuracy_delta=ok - self.clean_ok,
            retrieval=is_p.any(axis=1).astype(float),
            poison_in_topk=is_p.sum(axis=1).astype(float),
            qc_pass=(scimilarity_score(near) >= SCORE_QC).astype(float),
            attack_success=got,
            clean_target_rate=ct.hit,
            net_attack_success=got - ct.hit,
            baseline_acc_best=self.clean_best,
            baseline_acc_worst=self.clean_worst,
            clean_target_best=ct.best,
            clean_target_worst=ct.worst,
            **tie_columns(acc_v, hit_v, self.truth, target),
        )
        # net under each reading, matched side to side: a reader who gives the
        # attacker every tie must give them the clean ties too, or the difference
        # mixes two tie-break rules
        ind["net_best"] = ind["attack_success_best"] - ct.best
        ind["net_worst"] = ind["attack_success_worst"] - ct.worst
        return ind, reach

    def summarise(self, ind: dict[str, Any], reach: np.ndarray,
                  qi: np.ndarray) -> dict[str, float]:
        """Both scopes of one condition, plus the unreachable counts.

        ``within_`` covers the queries whose true label IS the attacked celltype
        -- efficacy, how completely the target is captured. ``global_`` covers
        every query -- impact, what one poisoned celltype costs the whole
        dataset. They answer different questions and a curve that mixes them
        answers neither.
        """
        out = dict(scope("within", qi, self.donor, self.boot_seed, **ind))
        out.update(scope("global", np.arange(len(self.truth)), self.donor,
                         self.boot_seed, **ind))
        # counts, not rates: how many queries mean_error_dist could not score
        # because the vote landed on a term with no ontology id
        out["within_n_unreachable_pairs"] = int((reach[qi] == 0).sum())
        out["global_n_unreachable_pairs"] = int((reach == 0).sum())
        return out


    def per_donor(self, ind: dict[str, Any], qi: np.ndarray) -> pd.DataFrame:
        """Both scopes of one condition, kept at the donor instead of pooled.

        The unreduced twin of :meth:`summarise`. One row per donor in the full
        roster, carrying ``within_*`` and ``global_*`` on the SAME row per the
        SCOPE exception -- they are two views of one condition, and splitting
        them would make a paired scope difference a self-join.

        A donor with no query of the attacked celltype still gets a row, with
        ``within_*_den = 0``. That is a structural zero: it keeps the donor
        roster identical across every condition, which is precisely what lets
        the threshold bootstrap hold ONE donor draw fixed across all doses
        (SPEC.md 5.4c).

        Args:
            ind: per-query indicators from :meth:`measure`.
            qi: (m,) indices of queries whose true label is the attacked
                celltype.

        Returns:
            (n_donors, 2 + 2*2*len(ind)) frame; ``donor_id`` is the first column.
        """
        vocab = np.unique(self.donor)
        cols: dict[str, np.ndarray] = {"donor_id": vocab}
        cols.update(by_donor("within", qi, self.donor, vocab, **ind))
        cols.update(by_donor("global", np.arange(len(self.truth)), self.donor,
                             vocab, **ind))
        return pd.DataFrame(cols)


#: Attack-specific columns that are null wherever the attack does not have them,
#: with the dtype they must take anyway.
#:
#: Without this they write as Arrow's ``null`` type in the table where every row
#: is None -- ``n_anchors`` is null-typed in ``02b_01`` and int64 in ``02b_02``,
#: ``alpha_lo`` the other way round -- and ``pyarrow.concat_tables`` rejects the
#: pair outright. The two steps are meant to be read together, so the schema is
#: pinned here rather than left to whichever rows a given driver happened to
#: produce. Nullable ``Int64``, not float: an anchor count is a count.
OPTIONAL_DTYPES: Final[dict[str, str]] = {
    "condition_seed": "Int64",
    "n_anchors": "Int64",
    "n_anchors_effective": "Int64",
    "n_anchors_used": "Int64",
    "flood_spread": "float32",
    "flood_sites": "string",
    "flood_jitter": "float32",
    "alpha_lo": "float32",
    "alpha_hi": "float32",
}


#: Columns that IDENTIFY a condition rather than measure it, and are therefore
#: exempt from the blanket float32 cast below.
#:
#: ``rate`` reaches :func:`seed_of` through ``repr()``, so narrowing it changes
#: the key: the sweep passes the Python float ``0.7`` and gets ``"0.7"``, while a
#: float32 round-trip yields ``"0.699999988079071"`` and a different seed
#: entirely. Anyone re-deriving the poison from a stored rate would silently
#: build a DIFFERENT poison set and conclude the run does not reproduce.
#: float32 is right for the 150-odd measurement columns and wrong for a key.
KEY_COLUMNS: Final[frozenset[str]] = frozenset({"rate"})


def write(rows: list[dict], path: pathlib.Path) -> pd.DataFrame:
    """Cast to the shared schema and write the table.

    Called after each source rather than once at the end, so a truncated run is
    still readable; ``condition_index`` and ``n_conditions`` let a reader tell a
    partial table from a complete one.

    Measurement columns are narrowed to float32; :data:`KEY_COLUMNS` are not.
    """
    df = pd.DataFrame(rows)
    for c in df.select_dtypes("float64").columns:
        if c in KEY_COLUMNS:
            continue
        df[c] = df[c].astype(EMB_DTYPE)
    for c, dt in OPTIONAL_DTYPES.items():
        if c in df.columns:
            df[c] = df[c].astype(dt)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def write_by_donor(frames: list[pd.DataFrame], path: pathlib.Path) -> pd.DataFrame:
    """Concatenate per-condition donor frames and write them.

    Like :func:`write`, called after each source rather than once at the end, so
    a truncated run is still readable.

    NOTHING is narrowed to float32 here, and that is the difference from
    :func:`write`. These are SUMS a bootstrap re-pools 2000 times; ``rate`` is a
    key that reaches ``seed_of`` through ``repr()`` (:data:`KEY_COLUMNS`), and a
    threshold read off a float32 ladder point cannot be compared against the
    ladder it came from. The table is ~85k rows, so the memory the cast would
    save is not worth the precision it would cost.

    Reports the resolved path and the row count to stderr, as every other
    progress line in this module does. That report lives HERE and not in the
    callers on purpose: of the nine, ``02b_13`` and ``02b_14`` printed nothing,
    so their donor tables -- read downstream by ``02b_12`` -- sat on disk with no
    log line anywhere naming them as output. Their mtimes match their main tables
    to the minute, and an mtime is not provenance. One report in the writer
    cannot drift from a second one in a caller, and a tenth caller gets it free.

    Args:
        frames: one frame per condition, each already carrying its key columns.
        path: destination ``*_by_donor.parquet``.

    Returns:
        The concatenated frame as written.
    """
    df = pd.concat(frames, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"[by_donor] wrote {len(df):,} donor rows to {path.resolve()}",
          file=sys.stderr, flush=True)
    return df
