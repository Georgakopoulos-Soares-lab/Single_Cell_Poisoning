"""Stage 5 — collapse per-cell raw results into per-condition rates.

Two rates are reported side by side and never merged: retrieval failure is a
property of the retriever, outcome failure a property of the aggregator
(``CLAUDE.md`` §1). ``clean_error_rate`` accompanies them so the classifier's
own errors stay separable from poisoning-induced ones.
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
    ATTACK_SUCCESS_RATE,
    QUERY_ID,
    CI_HI,
    CI_LO,
    CLEAN_ERROR_RATE,
    CLEAN_TARGET_RATE,
    DONOR_ID,
    FOCUS_SOURCE,
    EMB_DTYPE,
    GROUP_KEYS,
    N_BOOTSTRAP,
    NET_ATTACK_SUCCESS,
    N_POISON_IN_TOPK,
    PAIR,
    N_SEEDS,
    ONTOLOGY_DIST,
    OUTCOME_FAIL_RATE,
    POISON_SURVIVES_QC,
    QC_EVASION_FOCUS,
    RETRIEVAL_FAIL_FOCUS,
    QC_EVASION_RATE,
    RETRIEVAL_FAIL_RATE,
    SEED,
    Y_PRED_CLEAN,
    Y_PRED_POISONED,
    Y_TRUE,
)


def _bootstrap_ci_paired(
    grp: pd.DataFrame,
    hit_p: str,
    hit_c: str,
    donor_column: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile bootstrap on the WITHIN-DONOR attack effect.

    Args:
        grp: rows of one group, restricted to the attacked cell type.
        hit_p: boolean column, poisoned vote landed on the target.
        hit_c: boolean column, clean vote landed on the target.
        donor_column: donor identifier.
        rng: seeded generator.

    Returns:
        (ci_lo, ci_hi) on ``poisoned - clean``.

    Note:
        Each donor supplies BOTH rates from its own cells, so the quantity
        resampled is a difference and donor-to-donor variation in baseline
        accuracy cancels rather than dominating. Bracketing the unpaired
        outcome rate gave a width of ~7.4 percentage points that barely moved
        across four orders of magnitude of contamination, because it was
        measuring how much subjects differ in baseline accuracy — not what the
        attack did.
    """
    seeds = np.asarray(grp[SEED].unique())
    donors = grp[donor_column].astype(str).to_numpy()
    uniq = np.unique(donors)
    s_idx = {s: i for i, s in enumerate(seeds)}
    d_idx = {d: i for i, d in enumerate(uniq)}
    si = np.array([s_idx[s] for s in grp[SEED].to_numpy()])
    di = np.array([d_idx[d] for d in donors])

    hp = np.zeros((len(seeds), len(uniq)))
    hc = np.zeros((len(seeds), len(uniq)))
    cn = np.zeros((len(seeds), len(uniq)))
    np.add.at(hp, (si, di), grp[hit_p].to_numpy().astype(np.float64))
    np.add.at(hc, (si, di), grp[hit_c].to_numpy().astype(np.float64))
    np.add.at(cn, (si, di), 1.0)

    rs = rng.integers(0, len(seeds), size=(N_BOOTSTRAP, len(seeds)))
    rd = rng.integers(0, len(uniq), size=(N_BOOTSTRAP, len(uniq)))
    sel = (rs[:, :, None], rd[:, None, :])
    c = cn[sel].sum(axis=(1, 2))
    diff = (hp[sel].sum(axis=(1, 2)) - hc[sel].sum(axis=(1, 2)))
    means = np.divide(diff, c, out=np.full(N_BOOTSTRAP, np.nan), where=c > 0)
    return float(np.nanpercentile(means, 2.5)), float(np.nanpercentile(means, 97.5))


def _bootstrap_ci_donor(
    grp: pd.DataFrame, flag: str, donor_column: str, rng: np.random.Generator
) -> tuple[float, float]:
    """Percentile bootstrap resampling BOTH seeds and donors.

    Args:
        grp: all rows of one (attack, pair, rate) group.
        flag: boolean column to average.
        donor_column: column holding the donor identifier.
        rng: seeded generator.

    Returns:
        (ci_lo, ci_hi) over ``N_BOOTSTRAP`` replicates.

    Note:
        Two independent sources of variation, both resampled. Seeds capture
        which records the attacker happened to poison. DONORS capture
        biological and technical variation: cells from one subject share
        genotype, batch, dissociation protocol and sequencing run, so 17,571
        macrophage queries drawn from 78 subjects are closer to 78 independent
        observations than to 17,571. Resampling cells instead would report an
        interval far narrower than the experiment supports.

        The estimator is preserved: each replicate pools the cells of the drawn
        donors and takes the same cell-weighted mean that the point estimate
        uses, so the interval belongs to the number actually reported.
    """
    seeds = np.asarray(grp[SEED].unique())
    donors = grp[donor_column].astype(str).to_numpy()
    uniq = np.unique(donors)
    vals = grp[flag].to_numpy().astype(np.float64)
    seed_of = grp[SEED].to_numpy()

    # Per (seed, donor): how many cells, and how many of them fired. A
    # replicate's rate is then sum(selected hits) / sum(selected counts), which
    # is exactly the cell-weighted mean over the pooled cells of the drawn
    # donors -- the same estimator, computed without materialising the pool.
    s_idx = {s: i for i, s in enumerate(seeds)}
    d_idx = {d: i for i, d in enumerate(uniq)}
    hits = np.zeros((len(seeds), len(uniq)))
    cnts = np.zeros((len(seeds), len(uniq)))
    si = np.array([s_idx[s] for s in seed_of])
    di = np.array([d_idx[d] for d in donors])
    np.add.at(hits, (si, di), vals)
    np.add.at(cnts, (si, di), 1.0)

    rng_s = rng.integers(0, len(seeds), size=(N_BOOTSTRAP, len(seeds)))
    rng_d = rng.integers(0, len(uniq), size=(N_BOOTSTRAP, len(uniq)))
    h = hits[rng_s[:, :, None], rng_d[:, None, :]].sum(axis=(1, 2))
    c = cnts[rng_s[:, :, None], rng_d[:, None, :]].sum(axis=(1, 2))
    means = np.divide(h, c, out=np.full(N_BOOTSTRAP, np.nan), where=c > 0)
    return float(np.nanpercentile(means, 2.5)), float(np.nanpercentile(means, 97.5))


def _bootstrap_ci(per_seed: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile bootstrap over seeds.

    Args:
        per_seed: (n_seeds,) float, one rate per replicate.
        rng: seeded generator; seeds are passed in, never generated.

    Returns:
        (ci_lo, ci_hi), the 2.5th and 97.5th percentiles of the resampled means.

    Note:
        Resampling is over SEEDS, not cells. Cells within a seed share one
        poison set and are not independent; treating them as replicates would
        report an interval far narrower than the experiment supports. With a
        single seed every resample is identical and the interval collapses onto
        the point estimate — an honest statement that one replicate carries no
        information about variability.
    """
    draws = rng.choice(per_seed, size=(N_BOOTSTRAP, len(per_seed)), replace=True)
    means = draws.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate_raw(
    raw: pd.DataFrame,
    bootstrap_seed: int,
    pair_meta: dict[str, list[dict[str, str]]] | None = None,
    donor_column: str | None = None,
) -> pd.DataFrame:
    """Aggregate raw per-cell results over seeds.

    Args:
        raw: long-format ``raw_result`` rows, one per query cell per condition.
        bootstrap_seed: seed for the resampling generator.
        pair_meta: ``{pair_name: [{"source": CL id, "target": CL id}, ...]}``.
            A single attack has one member; a simultaneous multi-type attack has
            several. When given, ``attack_success_rate`` and
            ``clean_target_rate`` are computed per member: among queries whose
            TRUE label is that member's source, the fraction voted as that
            member's target, poisoned and clean. A combo therefore yields one
            row per member, so a member's success under a combo is directly
            comparable to the same member attacked alone.
        donor_column: when given, the bootstrap resamples donors as well as
            seeds. Without it the interval reflects only which records the
            attacker drew, not biological variation between subjects, and is
            therefore too narrow.

            This is the metric that survives annotation-granularity mismatch.
            ``outcome_fail_rate`` compares y_pred against y_true by exact
            ontology id, and the reference annotates at a finer granularity than
            the query set (macrophage vs alveolar macrophage), so it sits near
            50% before any poisoning and has almost no headroom. Whether the
            vote landed on the attacker's chosen label has no such problem.

    Returns:
        One row per (attack, pair, rate), carrying the ``summary`` contract
        columns. ``ci_lo``/``ci_hi`` bracket :data:`~contracts.schemas.CI_TARGET`.
    """
    rng = np.random.default_rng(bootstrap_seed)
    work = raw.copy()

    # Compare labels as plain strings. The schema stores them as `category` to
    # keep 1,260 x 49,456 rows small, but each column carries its OWN category
    # set -- y_true never sees the attacker's target, y_pred_poisoned does --
    # and pandas refuses to compare categoricals whose categories differ.
    def _as_str(col: str) -> np.ndarray:
        return work[col].astype(str).to_numpy()

    y_true = _as_str(Y_TRUE)
    work["_retrieval"] = work[N_POISON_IN_TOPK] > 0
    work["_outcome"] = _as_str(Y_PRED_POISONED) != y_true
    work["_clean"] = _as_str(Y_PRED_CLEAN) != y_true
    work["_qc"] = work[POISON_SURVIVES_QC]

    if pair_meta:
        work["_pred_p"] = _as_str(Y_PRED_POISONED)
        work["_pred_c"] = _as_str(Y_PRED_CLEAN)
        work["_y"] = y_true
        for name, members in pair_meta.items():
            sub = work[work[PAIR].astype(str) == name]
            if sub.empty:
                continue
            for m in members:
                if not (sub["_y"] == m["source"]).any():
                    raise ValueError(
                        f"pair {name!r}: no query cells carry source term "
                        f"{m['source']!r}; attack success would be a fraction of "
                        f"zero and read as perfect robustness"
                    )

    rows = []
    for keys, grp in work.groupby(list(GROUP_KEYS), observed=True, sort=True):
        per_seed = grp.groupby(SEED, observed=True)[
            ["_retrieval", "_outcome", "_clean", "_qc"]
        ].mean()
        base = {
            ONTOLOGY_DIST: grp[ONTOLOGY_DIST].iloc[0],
            RETRIEVAL_FAIL_RATE: per_seed["_retrieval"].mean(),
            OUTCOME_FAIL_RATE: per_seed["_outcome"].mean(),
            CLEAN_ERROR_RATE: per_seed["_clean"].mean(),
            QC_EVASION_RATE: per_seed["_qc"].mean(),
            N_SEEDS: len(per_seed),
        }
        # ci_lo/ci_hi bracket NET_ATTACK_SUCCESS (contracts.schemas.CI_TARGET),
        # paired within donor. Filled per member below when pair_meta is given;
        # without it there is no target to define an effect against.
        base[CI_LO] = base[CI_HI] = float("nan")

        if not pair_meta:
            rows.append(dict(zip(GROUP_KEYS, keys, strict=True), **base))
            continue

        pair_name = dict(zip(GROUP_KEYS, keys, strict=True))[PAIR]
        for m in pair_meta[str(pair_name)]:
            src_rows = grp[grp["_y"] == m["source"]].copy()
            # Boolean columns first, then a plain mean per seed: groupby.apply
            # with a lambda both warns and hides the operation being a mean.
            src_rows["_hit_p"] = src_rows["_pred_p"] == m["target"]
            src_rows["_hit_c"] = src_rows["_pred_c"] == m["target"]
            src_rows["_qc_f"] = src_rows[POISON_SURVIVES_QC]
            src_rows["_ret_f"] = src_rows[N_POISON_IN_TOPK] > 0
            by_seed = src_rows.groupby(SEED, observed=True)[
                ["_hit_p", "_hit_c", "_qc_f", "_ret_f"]
            ].mean()
            a = float(by_seed["_hit_p"].mean())
            c = float(by_seed["_hit_c"].mean())
            if donor_column and donor_column in src_rows.columns:
                lo, hi = _bootstrap_ci_paired(
                    src_rows, "_hit_p", "_hit_c", donor_column, rng
                )
            else:
                lo = hi = float("nan")
            rows.append(dict(
                zip(GROUP_KEYS, keys, strict=True),
                **{k: v for k, v in base.items() if k not in (CI_LO, CI_HI)},
                **{
                    FOCUS_SOURCE: m["source"],
                    ATTACK_SUCCESS_RATE: a,
                    CLEAN_TARGET_RATE: c,
                    NET_ATTACK_SUCCESS: a - c,
                    QC_EVASION_FOCUS: float(by_seed["_qc_f"].mean()),
                    RETRIEVAL_FAIL_FOCUS: float(by_seed["_ret_f"].mean()),
                    CI_LO: lo,
                    CI_HI: hi,
                },
            ))

    out = pd.DataFrame(rows)
    numeric = [RETRIEVAL_FAIL_RATE, OUTCOME_FAIL_RATE, CLEAN_ERROR_RATE,
               QC_EVASION_RATE, CI_LO, CI_HI]
    if pair_meta:
        numeric += [RETRIEVAL_FAIL_FOCUS, ATTACK_SUCCESS_RATE, CLEAN_TARGET_RATE,
                    NET_ATTACK_SUCCESS, QC_EVASION_FOCUS]
    for col in numeric:
        out[col] = out[col].astype(EMB_DTYPE)
    out[N_SEEDS] = out[N_SEEDS].astype("int16")
    return out


# ── snakemake entry point ────────────────────────────────────────────────────
if "snakemake" in globals():  # pragma: no cover - exercised by the workflow
    import collections
    import re
    import sys

    import pandas as _pd

    _cfg_pairs = {p["name"]: p for p in snakemake.config["pairs"]}  # noqa: F821
    _meta = {
        name: (
            [{"source": _cfg_pairs[mem]["source"], "target": _cfg_pairs[mem]["target"]}
             for mem in p["members"]]
            if "members" in p
            else [{"source": p["source"], "target": p["target"]}]
        )
        for name, p in _cfg_pairs.items()
    }

    # Aggregate ONE GROUP AT A TIME rather than concatenating every raw file.
    # A group is (attack, pair, rate) across seeds, and no statistic crosses a
    # group boundary, so this is exactly equivalent -- pinned by
    # test_streaming_by_group_matches_aggregating_everything_at_once.
    #
    # It is not an optimisation, it is what makes the stage runnable: 7,672
    # files x 49,456 rows is 379M rows, and loading them together reached 18 GB
    # resident with 27 GB of swap before being killed. Streaming holds one
    # group -- a few seeds -- at a time.
    _pat = re.compile(r"(.+)_(.+)_([0-9.eE+-]+)_(\d+)\.parquet$")
    _groups = collections.defaultdict(list)
    for _f in snakemake.input.raw:  # noqa: F821
        _m = _pat.match(str(_f).split("/")[-1])
        if not _m:
            raise ValueError(f"cannot parse condition from filename: {_f}")
        _attack, _rest, _rate, _seed = _m.groups()
        # `pair` may contain underscores; the attack is the leading token.
        _attack, _pair = str(_f).split("/")[-1].split("_", 1)[0], _rest
        _groups[(_attack, _pair, _rate)].append(_f)

    # Donor comes from query_meta, joined on query_id, rather than being stored
    # in all 8,374 raw files -- adding a column there would mean re-running every
    # sweep for information that is a property of the query set alone.
    _qm = _pd.read_parquet(snakemake.input.query_meta)  # noqa: F821
    _donor = dict(zip(_qm[QUERY_ID].astype(str), _qm[DONOR_ID].astype(str))) \
        if DONOR_ID in _qm.columns else {}
    print(f"aggregating {len(_groups):,} groups · "
          f"{len(set(_donor.values())):,} donors joined from query_meta", file=sys.stderr)
    _out = []
    for _i, (_key, _files) in enumerate(sorted(_groups.items()), 1):
        _raw = _pd.concat([_pd.read_parquet(f) for f in _files], ignore_index=True)
        if _donor:
            _raw[DONOR_ID] = _raw[QUERY_ID].astype(str).map(_donor)
        _out.append(
            aggregate_raw(
                _raw,
                bootstrap_seed=int(snakemake.config["bootstrap_seed"]),  # noqa: F821
                pair_meta=_meta,
                donor_column=DONOR_ID,
            )
        )
        del _raw
        if _i % 200 == 0:
            print(f"  {_i:,}/{len(_groups):,} groups", file=sys.stderr)

    _pd.concat(_out, ignore_index=True).to_parquet(
        snakemake.output[0], index=False  # noqa: F821
    )
    print(f"wrote {sum(len(o) for o in _out):,} summary rows", file=sys.stderr)
