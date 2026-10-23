"""Step 02b_34 — how far does the SAME condition move when you run it again?

``Bench.open_index`` calls ``hnswlib.add_items(pe, ids)`` without
``num_threads``, so insertion runs on every core and the resulting graph depends
on thread scheduling. The same poison set inserted twice therefore gives a
different graph, and every graph-path table in this project -- ``02a_01``,
``02b_01``, ``02b_02``, ``02b_11_graph``, ``02b_13``, ``02b_14``,
``02b_15_graph`` -- is one realisation of a distribution rather than a value
anyone can re-derive (CLAUDE.md §1).

**What was missing is the SIZE of that distribution.** It had been seen twice and
measured never: a controlled 20k+4k build disagreed with itself on 10.2% of
neighbour slots, and ``02b_30`` re-ran 8 of ``02b_01``'s conditions and moved
``within_accuracy`` by up to 0.019. Two observations of a difference are not a
band. One of them could be typical, a tail or a coincidence, and nothing in the
repository could say which -- which is why CLAUDE.md currently forbids quoting
any graph number to better than ±0.02, a bound chosen because the real one was
unknown.

This step measures it. Four conditions, six repeats each, IDENTICAL inputs every
time: same ``condition_seed``, same poison set, same clean baseline, same
bootstrap seed. Everything that could legitimately move a number is held fixed,
so whatever spread comes out is the insertion and nothing else.

WHAT IS HELD FIXED, AND HOW THAT IS PROVEN RATHER THAN ASSERTED
---------------------------------------------------------------
Three controls run inside the loop, because "identical inputs" is a claim and
each of these is the evidence for one part of it:

* ``poison_digest`` -- sha256 of the poison matrix's bytes, written on every row
  and checked against the condition's first repeat. If the attack construction
  drifted, the digest moves and the run stops. The seed is passed in, never
  generated (``seed_of(source, rate, SEED_VALUE)``), so it should not.
* the CLEAN side -- ``baseline_accuracy``, ``baseline_lineage`` and the clean
  target rates are read from ``01a_01``'s committed neighbours and inserted
  nothing. They appear in the spread table like every other metric and their
  standard deviation must be exactly 0. That is the control that localises the
  variation to ``add_items``: if the clean columns moved too, the cause would be
  somewhere else entirely.
* the BOOTSTRAP -- ``Bench.boot_seed`` comes from ``config.bootstrap_seed`` and
  is the same integer on every repeat, so the donor resampling contributes no
  variation to the point estimates and none to the intervals.

RETRIEVAL CHURN AND ANSWER CHURN ARE DIFFERENT SIZES
-----------------------------------------------------
A rate that moves by 0.002 between two runs does not mean the graph moved by
0.002. Most of a query's fifty neighbours can be replaced without the plurality
vote changing at all, and a single swapped neighbour can flip a query that was
tied. So each repeat records BOTH, against the condition's first repeat:

* ``*_topk_overlap_mean`` -- the mean fraction of a query's k returned ids that
  are also in the first repeat's k, as a SET. This is how much the graph moved.
* ``*_topk_ordered_identical`` / ``*_topk_set_identical`` / ``*_nn1_identical``
  -- the fraction of queries whose neighbour list is unchanged in order,
  unchanged as a set, and whose single nearest neighbour is unchanged.
* ``*_pred_changed`` -- the fraction of queries whose predicted celltype differs
  from the first repeat's. This is how much the ANSWER moved, and it is the one
  that reaches a published number.

The gap between the first group and the last is the point: a large graph churn
with a small answer churn means the vote absorbs the noise, and the opposite
means it does not.

Each is emitted on both scopes, ``within_`` and ``global_``, on one row per the
SCOPE exception. That is not tidiness. A first run of this step reported the
pooled overlap at 0.9999 on a condition where 5.3% of the attacked celltype's
predictions moved between repeats, and the two are reconcilable only once it is
clear that one is an average over 49,456 queries and the other over 1,858. The
poison is concentrated in one celltype's neighbourhoods, so the pooled figure is
mostly a measurement of the 27 celltypes nobody attacked.

THE CONDITIONS, AND WHY THESE FOUR
-----------------------------------
Budget 24 retrievals, approved as 4 x 6. Each retrieval reloads the shipped
9.3 GB index, so the choice of four is a choice not to have eight.

Held fixed across all four: attack ``relabel``, ``access=insert``,
``geometry=unmoved``, ``label_change=relabel``, ``selection=hub``, seed 0. Only
the attacked celltype and the dose vary, so the design is a 2 x 2 and a
difference between conditions has one cause.

The two celltypes are the pair ``02b_30`` already ran, which is deliberate --
those eight rows become a ninth and tenth reading of two of these four cells --
and they sit at opposite ends of exposure. Recomputed from
``02b_01_no_access_celltype_sweep_graph.parquet`` on 2026-09-28,
``relabel``/``hub``, ``within_`` scope:

                       dose 0.25                  dose 1.00
                accuracy  attack  ambig     accuracy  attack  ambig
  CL:0000623      0.0818  0.8531  0.5188      0.0797  0.8558  0.5205
  CL:0000057      0.7981  0.1659  0.5913      0.7909  0.1731  0.6058

``CL:0000623`` (natural killer cell -> ``CL:0000084`` T cell) is already taken
at a quarter dose; ``CL:0000057`` (fibroblast -> ``CL:0000097`` mast cell)
resists at both. If the variation scales with the effect, the two rows will not
carry the same band, and a single band quoted for the project would be wrong in
one direction on one of them.

Both doses are on the ladder ``02b_01`` ran, one BELOW parity and one AT it.
Parity is where a twin ties with the record it copied: the ``ambig`` column above
says the tie-break decides 52% and 61% of the attacked celltype's queries there.
A tie is exactly where one swapped neighbour flips an answer, so if the
insertion noise is going to show up anywhere it is at 1.0. Below parity is the
control for that: same celltype, same attack, fewer ties.

These four cells are four cells. They are not a statement about the other 24
attackable celltypes or the other twelve rungs of the dose ladder, whose bands
are unmeasured (CLAUDE.md §1).

COST
----
24 retrievals. ``02b_01``'s own ``seconds`` column for these four conditions
reads 27.4 / 57.2 / 39.1 / 61.7 s, so the loop is about 19 minutes plus one
``Bench.load``. Peak memory ~15 GB: the shipped index, the reference arrays, and
one (n_q, k) int64 id table per condition held as the comparison reference.

Writes, under ``2_results/<dataset>/02_b_attack_celltype/``:
  ``02b_34_no_access_repeats.parquet``  one row per (condition, repeat), 24 rows
  ``02b_34_no_access_spread.parquet``   one row per (condition, metric)

The second is derived from the first and from nothing else, so a reader can
recompute every band in it with a groupby.
"""
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "00_common"))
import hashlib  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from attacks import build_attack  # noqa: E402
from celltype_sweep import Bench, require_tiers, seed_of, write  # noqa: E402
from dataset_config import load  # noqa: E402
from provenance import emit  # noqa: E402
from schemas import (ACCESS, ATTACK_TIER, EMB_DTYPE, GEOMETRY,  # noqa: E402
                     K_NEIGHBORS, LABEL_CHANGE, POISON_EMB, SELECTION,
                     THREAT_MODEL, Access, Attack, Geometry, LabelChange,
                     Selection, ThreatModel)
from vote import TIE_BREAK  # noqa: E402

BASE = _HERE.parents[2]
CFG = load(BASE)
OUT = (BASE / "2_results" / CFG["dataset"] / "02_b_attack_celltype"
       / "02b_34_no_access_repeats.parquet")
OUT_SPREAD = OUT.with_name("02b_34_no_access_spread.parquet")

#: ``(source, rate)``. Two celltypes at opposite ends of exposure x two doses,
#: one below parity and one at it. See the docstring for why each.
CONDITIONS: tuple[tuple[str, float], ...] = (
    ("CL:0000623", 0.25),
    ("CL:0000623", 1.0),
    ("CL:0000057", 0.25),
    ("CL:0000057", 1.0),
)

#: Repeats per condition. Six is the approved budget of 24 retrievals divided by
#: four conditions. It gives a sample standard deviation on 5 degrees of freedom
#: -- enough to size a band, not enough to characterise its tails, and the table
#: reports min and max alongside so a reader is never handed only the sd.
N_REPEATS = 6

#: The ceiling approved in conversation on 2026-09-28. Asserted rather than
#: trusted: each retrieval is another ~40 s of shipped-index load, so a fifth
#: condition or a seventh repeat is real unapproved compute, not a rounding
#: error.
BUDGET = 24

#: Held fixed across every condition and every repeat.
ATTACK_NAME = Attack.RELABEL
GEOMETRY_NAME = Geometry.UNMOVED
LABEL_CHANGE_NAME = LabelChange.RELABEL
SELECTION_NAME = Selection.HUB
SEED_VALUE = 0

#: Suffixes of the interval columns ``Bench.summarise`` emits beside each point
#: estimate. The spread table covers point estimates only: an interval's own
#: spread across repeats is a second-order quantity nobody reads, and including
#: it would triple the table.
INTERVAL_SUFFIXES = ("_lo", "_hi")

#: How much a repeat's RETRIEVAL differs from its condition's first repeat,
#: before the ``within_`` / ``global_`` prefix each one is emitted under.
#:
#: BOTH SCOPES, on one row, per CLAUDE.md §7's SCOPE exception -- they are two
#: views of one retrieval, not two observations. The scope is not a refinement
#: here, it is the whole reading: the attacked celltype's queries are the ones
#: whose neighbourhoods fill with poison, and pooling them into 49,456 queries
#: divides their churn by twenty-seven. A first run of this step reported the
#: pooled top-50 overlap as 0.9999 on a condition where 5.3% of the attacked
#: celltype's predictions moved, which is not a small effect described
#: conservatively -- it is two different populations reported as one.
#:
#: Every one is 1.0 / 0.0 by construction on the first repeat itself, which is a
#: comparison with itself and not a measurement, so :func:`spread` computes them
#: over the other repeats only and says so in its ``n_repeats`` column.
CHURN_METRICS = ("topk_overlap_mean", "topk_overlap_min", "topk_set_identical",
                 "topk_ordered_identical", "nn1_identical", "pred_changed")

#: The scoped column names :func:`agreement` emits and :func:`spread` treats as
#: comparisons against repeat 1.
CHURN_COLUMNS = tuple(f"{p}_{m}" for p in ("within", "global")
                      for m in CHURN_METRICS)

#: Numeric columns that IDENTIFY a repeat or a condition rather than measure
#: it, and so have no band to report. ``n_poison`` and ``atlas_share`` are
#: deliberately NOT here: they are constant by construction and their zero
#: standard deviation is a control on the poison, exactly as the ``baseline_``
#: columns are a control on the clean side.
NOT_METRICS = frozenset({
    "repeat", "n_repeats", "elapsed_seconds", "seed", "condition_seed",
    "ef_search", "k", "attack_ontology_dist", "n_reference", "n_celltype",
    "n_queries_within", "n_queries_global",
})

#: Multiplier that packs a (row, id) pair into one int64 key in
#: :func:`row_overlap`. Must exceed every hnswlib id, which is bounded by
#: ``n_ref + n_poison`` < 8.2e6 + 2.5e5 here; 2**34 leaves four orders of
#: headroom and still fits 49,456 rows in int64 (4.9e4 * 1.7e10 = 8.5e14).
#: Arithmetically inert -- checked against the real bound before use.
ROW_KEY_STRIDE = np.int64(1) << 34


def poison_digest(pe: np.ndarray) -> str:
    """sha256 over a poison matrix's bytes, so two builds can be compared.

    Args:
        pe: (n_p, 128) float32, unit norm.

    Returns:
        64 hex characters. Depends on dtype, shape and byte order, all of which
        are part of "the same poison set".
    """
    h = hashlib.sha256()
    h.update(f"{pe.dtype}|{pe.shape}|".encode())
    h.update(np.ascontiguousarray(pe).tobytes())
    return h.hexdigest()


def row_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per query, the fraction of ``a``'s returned ids that are also in ``b``'s.

    A SET comparison, ignoring order: a graph that returns the same fifty
    records in a different sequence has returned the same neighbourhood, and the
    vote cannot tell the difference. Order is reported separately by
    :func:`agreement`.

    Args:
        a: (n_q, k) int64 hnswlib ids. b: (n_q, k) int64, same shape.

    Returns:
        (n_q,) float32 in [0, 1]. Symmetric in its arguments, because both rows
        hold k distinct ids and the shape is shared.

    Raises:
        ValueError: shapes differ, an id is negative or at/above
            :data:`ROW_KEY_STRIDE` (the packing would collide across rows), or a
            row repeats an id (membership would count it twice and the fraction
            would exceed the true overlap).
    """
    if a.shape != b.shape:
        raise ValueError(f"{a.shape} against {b.shape}: not the same retrieval "
                         "shape, so there is no per-query comparison to make")
    n_q, k = a.shape
    hi = max(int(a.max()), int(b.max()))
    if min(int(a.min()), int(b.min())) < 0 or hi >= ROW_KEY_STRIDE:
        raise ValueError(f"ids span 0..{hi}, which does not fit the "
                         f"{int(ROW_KEY_STRIDE)} row stride; the packed keys "
                         "would collide between queries")
    for name, arr in (("a", a), ("b", b)):
        if bool((np.diff(np.sort(arr, axis=1), axis=1) == 0).any()):
            raise ValueError(f"{name} repeats an id within a query's k "
                             "neighbours; hnswlib returns k distinct records "
                             "and membership counting assumes it")

    # One int64 key per (query, id) so a single np.isin answers all n_q rows at
    # once. Without the row offset a neighbour of query 3 would match a
    # neighbour of query 900 and every overlap would read too high.
    rows = (np.arange(n_q, dtype=np.int64) * ROW_KEY_STRIDE)[:, None]
    ka = (rows + a.astype(np.int64)).ravel()
    kb = (rows + b.astype(np.int64)).ravel()
    return (np.isin(ka, kb).reshape(n_q, k).sum(axis=1) / k).astype(EMB_DTYPE)


def agreement(a: np.ndarray, b: np.ndarray, pred_a: np.ndarray,
              pred_b: np.ndarray, within: np.ndarray) -> dict[str, float]:
    """How much two retrievals of the same condition differ, on both levels.

    The graph level and the answer level are kept apart on purpose. They are not
    the same size -- a plurality of fifty absorbs a good deal of neighbour churn
    -- and only the second one reaches a number anyone publishes. Each is
    reported on both scopes, ``within_`` (the attacked celltype's own queries)
    and ``global_`` (all of them), because the poison is concentrated in one
    celltype's neighbourhoods and a pooled figure reports the other 27 celltypes'
    stability as if it were the attacked one's.

    Args:
        a: (n_q, k) int64 ids from the repeat being scored.
        b: (n_q, k) int64 ids from its condition's FIRST repeat.
        pred_a: (n_q,) str predicted celltype from the repeat being scored.
        pred_b: (n_q,) str predicted celltype from the first repeat.
        within: (m,) int indices of the queries whose true label is the
            attacked celltype.

    Returns:
        ``{column: value}`` for :data:`CHURN_COLUMNS`, all dimensionless
        fractions in [0, 1].

    Raises:
        ValueError: any array disagrees with the others in length, or the
            attacked celltype has no queries -- in which case the ``within_``
            columns would be means over nothing and would land as NaN.
    """
    if not (len(a) == len(b) == len(pred_a) == len(pred_b)):
        raise ValueError(
            f"ids {len(a):,}/{len(b):,} and predictions {len(pred_a):,}/"
            f"{len(pred_b):,} are not all the same query set")
    if len(within) == 0:
        raise ValueError("no query carries the attacked celltype's label, so "
                         "the within_ scope has nothing to average over")
    ov = row_overlap(a, b)
    same_order = (a == b).all(axis=1)
    same_nn1 = a[:, 0] == b[:, 0]
    changed = pred_a != pred_b

    out: dict[str, float] = {}
    for prefix, s in (("within", within), ("global", np.arange(len(a)))):
        out[f"{prefix}_topk_overlap_mean"] = float(ov[s].mean())
        out[f"{prefix}_topk_overlap_min"] = float(ov[s].min())
        out[f"{prefix}_topk_set_identical"] = float((ov[s] == 1.0).mean())
        out[f"{prefix}_topk_ordered_identical"] = float(same_order[s].mean())
        out[f"{prefix}_nn1_identical"] = float(same_nn1[s].mean())
        out[f"{prefix}_pred_changed"] = float(changed[s].mean())
    return out


def spread(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """The band each metric occupies across repeats of one condition.

    Long format: one row per (condition, metric), because metric is an axis and
    a wide table of 40-odd columns x 4 conditions could not be filtered,
    sorted or plotted on it.

    ``sd`` is the SAMPLE standard deviation, ddof=1 -- these six repeats are
    draws from the insertion's distribution, not the population of runs anyone
    will ever make. On six draws it is itself a noisy estimate, which is why
    ``range`` is emitted beside it: what a reader needs is the width actually
    observed, and the sd only says how that width is distributed.

    :data:`CHURN_COLUMNS` are averaged over the repeats AFTER the first, since
    the first is compared with itself; ``n_repeats`` on the row says which
    count was used, so nothing has to be inferred from the metric's name.

    Args:
        df: the per-repeat table, one row per (condition, repeat).
        keys: the columns identifying a condition.

    Returns:
        (n_conditions * n_metrics, len(keys) + 9) frame, in condition order.
        ``range`` is ``max - min``; ``rel_range`` is that over ``|mean|``, NaN
        where the mean is 0 -- a relative width on a zero baseline is not a
        number. ``first`` is the value of the EARLIEST repeat included, which
        for a :data:`CHURN_COLUMNS` row is repeat 2, not repeat 1.
    """
    metrics = [c for c in df.columns
               if pd.api.types.is_numeric_dtype(df[c])
               and not c.endswith(INTERVAL_SUFFIXES)
               and c not in keys and c not in NOT_METRICS]
    rows: list[dict] = []
    for cond, g in df.groupby(keys, sort=False):
        g = g.sort_values("repeat")
        ident = dict(zip(keys, cond if isinstance(cond, tuple) else (cond,)))
        for m in metrics:
            v = g.loc[g["repeat"] > 1, m] if m in CHURN_COLUMNS else g[m]
            v = v.to_numpy(dtype=float)
            mean = float(v.mean())
            rows.append({
                **ident, "metric": m, "n_repeats": len(v), "mean": mean,
                # ddof=1: six runs are a sample of the insertion's behaviour
                "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
                "min": float(v.min()), "max": float(v.max()),
                "range": float(v.max() - v.min()),
                "rel_range": (float(v.max() - v.min()) / abs(mean)
                              if mean != 0 else float("nan")),
                "first": float(v[0]),
            })
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    emit(__file__)          # which code ran, HEAD or dirty tree
    require_tiers((ATTACK_NAME,))
    n_ret = len(CONDITIONS) * N_REPEATS
    if n_ret > BUDGET:
        raise ValueError(
            f"{len(CONDITIONS)} conditions x {N_REPEATS} repeats is {n_ret} "
            f"retrievals, over the approved ceiling of {BUDGET}; each one "
            "reloads the shipped 9.3 GB index. Get approval, do not raise "
            "BUDGET.")

    b = Bench.load(CFG, BASE, t0)
    missing = sorted({s for s, _ in CONDITIONS} - set(b.sources))
    if missing:
        raise ValueError(f"not attackable in this dataset: {missing}")
    print(f"[{time.time()-t0:.0f}s] {len(CONDITIONS)} conditions x {N_REPEATS} "
          f"repeats = {n_ret} retrievals; identical condition_seed, identical "
          f"poison set and identical bootstrap seed {b.boot_seed} on every one",
          file=sys.stderr, flush=True)

    keys = [THREAT_MODEL, "attack", ACCESS, GEOMETRY, LABEL_CHANGE, SELECTION,
            "source", "target", "rate"]
    rows: list[dict] = []
    for source, rate in CONDITIONS:
        target, attack_dist = b.target_of[source]
        qi, pool = b.qi_of[source], b.pool_of[source]
        ct = b.clean_target(target)
        cond_seed = seed_of(source, rate, SEED_VALUE)

        # The poison is rebuilt every repeat rather than built once and reused,
        # because "the same inputs give different answers" has to include the
        # construction: a set built once and handed round would leave the claim
        # resting on an assumption. poison_digest is the check that it does not.
        ref_digest: str | None = None
        ref_ids: np.ndarray | None = None
        ref_pred: np.ndarray | None = None

        for rep in range(1, N_REPEATS + 1):
            t1 = time.time()
            poison = build_attack(
                ATTACK_NAME, ref_emb=b.ref_emb, ref_labels=b.ref_label,
                ref_ids=b.ref_id, source=source, target=target, rate=rate,
                seed=cond_seed, label_change=LABEL_CHANGE_NAME,
                selection=SELECTION_NAME, hub_nbr=b.clean_idx)
            pe = poison[POISON_EMB]
            n_p = len(pe)
            digest = poison_digest(pe)
            if ref_digest is None:
                ref_digest = digest
            elif digest != ref_digest:
                raise ValueError(
                    f"{source} dose {rate} repeat {rep}: the poison set is not "
                    f"the one repeat 1 built ({digest[:12]} against "
                    f"{ref_digest[:12]}). The seed is passed in, so this is a "
                    "change in the attack construction, and the repeats would "
                    "no longer be measuring the graph.")

            # retrieve and score separately, so the neighbours survive being
            # scored: b.measure(pe, target, ct) is these two in sequence.
            r = b.retrieve(pe, target)
            del poison
            ind, reach = b.score(r, target, ct)
            row: dict = {
                THREAT_MODEL: str(ThreatModel.NO_ACCESS),
                "retrieval_path": "hnsw", "ef_search": b.ef,
                "k": K_NEIGHBORS, "tie_break": TIE_BREAK,
                "attack": str(ATTACK_NAME),
                "attack_tier": str(ATTACK_TIER[ATTACK_NAME]),
                ACCESS: str(Access.INSERT), GEOMETRY: str(GEOMETRY_NAME),
                LABEL_CHANGE: str(LABEL_CHANGE_NAME),
                SELECTION: str(SELECTION_NAME),
                "dose_basis": "celltype",
                "source": source, "target": target,
                "attack_ontology_dist": attack_dist,
                "rate": rate, "seed": SEED_VALUE, "condition_seed": cond_seed,
                # the repeat axis. Everything above is identical across it by
                # construction, and everything below is what moved.
                "repeat": rep, "n_repeats": N_REPEATS,
                "poison_digest": digest,
                "insert_num_threads": "default (-1, all cores)",
                "n_poison": n_p, "n_celltype": int(len(pool)),
                "n_reference": b.n_ref, "atlas_share": n_p / b.n_ref,
                "n_queries_within": int(len(qi)),
                "n_queries_global": len(b.truth),
            }
            row.update(b.summarise(ind, reach, qi))
            if ref_ids is None:
                ref_ids, ref_pred = r.ids, r.acc.pred
            row.update(agreement(r.ids, ref_ids, r.acc.pred, ref_pred, qi))
            row["elapsed_seconds"] = round(time.time() - t1, 1)
            rows.append(row)
            del r

            print(f"[{time.time()-t0:6.0f}s] {len(rows):2d}/{n_ret} {source} "
                  f"dose {rate:<5} rep {rep}/{N_REPEATS} n_p {n_p:>7,}  "
                  f"acc {row['within_accuracy']:.4f}  "
                  f"net {row['within_net_attack_success']:+.4f}  "
                  f"ret {row['within_retrieval']:.4f}  "
                  f"within top{K_NEIGHBORS} overlap "
                  f"{row['within_topk_overlap_mean']:.4f}  "
                  f"within pred moved {row['within_pred_changed']:.4f}  "
                  f"({row['elapsed_seconds']:.0f}s)",
                  file=sys.stderr, flush=True)

        # written after each condition, so a truncated run is still readable
        df = write(rows, OUT)
        spread(df, keys).to_parquet(OUT_SPREAD, index=False)

    df = pd.read_parquet(OUT)
    sp = pd.read_parquet(OUT_SPREAD)
    print(f"\n[{time.time()-t0:.0f}s] wrote {len(df)} repeat rows to {OUT}\n"
          f"[{time.time()-t0:.0f}s] wrote {len(sp)} (condition, metric) rows "
          f"to {OUT_SPREAD}", file=sys.stderr)

    # Read back from the files, not from the in-memory rows: the numbers a
    # reader will quote are the ones on disk after the float32 cast, and a
    # summary computed from anything else is a different number (CLAUDE.md §1).
    head = ["within_accuracy", "within_attack_success",
            "within_net_attack_success", "within_retrieval",
            "within_baseline_accuracy",
            *(f"within_{m}" for m in CHURN_METRICS)]
    show = sp[sp["metric"].isin(head)].copy()
    show["metric"] = pd.Categorical(show["metric"], head, ordered=True)
    print("\nband across repeats, identical inputs "
          "(sd is ddof=1; churn rows compare against repeat 1 and use n-1):",
          file=sys.stderr)
    print(show.sort_values(["source", "rate", "metric"])[
        ["source", "rate", "metric", "n_repeats", "mean", "sd", "min", "max",
         "range"]].to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
