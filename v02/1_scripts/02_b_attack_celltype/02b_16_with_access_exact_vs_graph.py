"""Step 02b_16 — how wrong is the merge shortcut, measured rather than assumed?

``modify.remerge`` computes what EXHAUSTIVE retrieval would return.
``modify.graph_topk`` writes the replacements into the shipped index and
re-queries, which is what SCimilarity actually returns. ``02b_11`` and ``02b_15``
run both, under ``BIORAG_RETRIEVAL``, writing ``<stem>.parquet`` and
``<stem>_graph.parquet``. This step joins each such pair condition by condition
and reports where they disagree.

**It exists because the previous answer came from four conditions and was wrong.**
``graph_topk``'s docstring asserted "the graph was never WEAKER… so ``remerge``
understates by an unknown, per-celltype margin", inferred from ``CL:0000235`` and
``CL:0000583`` at two alphas each. Over all 936 conditions of ``02b_15`` the two
agree EXACTLY in 41.2%, and where they differ the direction is mixed: the graph
attack is stronger in 34.0% and weaker in 24.8%, mean difference -0.0011. The
shortcut does not systematically understate; it disagrees both ways and nearly
cancels. The largest single disagreement runs opposite to the old claim --
``CL:0000583`` shifted/hub at alpha 0.20, exact 0.4343 against graph 0.6628.

That does NOT retire the graph runs, it re-argues them. The disagreement is
concentrated and unpredictable: 2 of 26 celltypes exceed 0.10 and 11 of 26 stay
under 0.01, and nothing says in advance which a celltype will be. It does not
even track HNSW's own clean recall -- Spearman -0.23 against ``01a_02``'s top-50
overlap. ``CL:0002062``'s disagreement fits its poor 0.885 recall; ``CL:0000583``'s
LARGER one comes with an ordinary 0.975.

**LEVELS ONLY, and this is not a detail.** The graph runs also carry the
baseline-regime fix (CLAUDE.md 7): under ``graph`` the clean side now comes from
``clean_graph_topk`` rather than from the exhaustive deep list. So any column
built by subtracting the clean side -- ``accuracy_delta``, ``net_attack_success``,
``*_delta`` -- differs between the two tables for TWO reasons at once and cannot
be read as a retrieval effect. Only quantities computed wholly from the attacked
side are comparable, and :data:`EXCLUDE` is what enforces that.

Reads only committed tables. No index, no cache, seconds to run.

Writes ``02b_16_with_access_exact_vs_graph.parquet``: long format, one row per
(step, condition, metric), carrying the exact value, the graph value and their
difference.
"""
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "00_common"))
import time  # noqa: E402

import pandas as pd  # noqa: E402

from dataset_config import load  # noqa: E402
from provenance import emit  # noqa: E402
from schemas import THREAT_MODEL, ThreatModel  # noqa: E402

BASE = _HERE.parents[2]
CFG = load(BASE)
TAB = BASE / "2_results" / CFG["dataset"] / "02_b_attack_celltype"
OUT = TAB / "02b_16_with_access_exact_vs_graph.parquet"

#: A metric is comparable between the two tables ONLY if it is computed wholly
#: from the attacked side. Anything that subtracts, divides by, or is itself the
#: clean side moved when the baseline-regime fix landed, so a difference in it
#: mixes two changes. Substring match, deliberately blunt -- a new column that
#: happens to contain one of these words should be excluded until someone
#: confirms it is a pure level, not silently compared.
EXCLUDE = ("baseline", "clean_", "_delta", "net_", "_lo", "_hi",
           "_best", "_worst", "ambiguous")

#: Never a measurement: bookkeeping that differs between runs by construction.
NOT_A_METRIC = ("seconds", "condition_index", "n_conditions", "ef_search",
                "k", "n_exact", "n_inexact", "retrieval_path")

#: What identifies a condition. NAMED, not inferred. An earlier version derived
#: the keys as "every shared column that is not a metric", which swept
#: ``within_net_attack_success`` -- excluded from comparison, and therefore not a
#: metric by that test -- into the key set and joined 750 of 936 rows. A join key
#: that is really a measurement silently drops every condition where the two runs
#: disagree, which is exactly the population this step exists to count.
#:
#: Only the columns actually present in a pair are used, so a step with no
#: ``alpha_lo`` is fine; but the intersection must still identify a condition,
#: which ``validate="one_to_one"`` and the row-count check below enforce.
AXES = ("source", "target", "attack", "geometry", "label_change", "selection",
        "rate", "alpha_lo", "alpha_hi", "n_anchors", "flood_spread",
        "flood_jitter", "flood_sites", "seed", "draw", "split_seed")


#: A measurement in this project carries a scope prefix -- ``within_`` for the
#: attacked celltype's own queries, ``global_`` for all of them (CLAUDE.md 7, the
#: SCOPE exception). Requiring one is what separates a measurement from a
#: condition PARAMETER. Without it this step also "compared" ``n_anchors``,
#: ``atlas_share``, ``flood_spread`` and ``condition_seed``: identical between the
#: two runs by construction, so every difference was zero, and a reader could
#: take "n_anchors agreed between retrievers" for a finding.
SCOPES = ("within_", "global_")


def comparable(col: str, exact: pd.DataFrame, graph: pd.DataFrame) -> bool:
    """True if ``col`` is a scope-prefixed measurement of the attacked side."""
    if not col.startswith(SCOPES):
        return False
    if col in NOT_A_METRIC or any(x in col for x in EXCLUDE):
        return False
    if col not in exact.columns or col not in graph.columns:
        return False
    # bool is numeric to pandas, but a boolean column here is a property of the
    # CONDITION (target_in_lineage, poison_survives_qc), not a measurement of the
    # run -- and subtracting two bool Series raises rather than giving a signed
    # difference, which is the right instinct badly timed.
    for d in (exact[col], graph[col]):
        if pd.api.types.is_bool_dtype(d) or not pd.api.types.is_numeric_dtype(d):
            return False
    return True


def pairs(tab: pathlib.Path) -> list[tuple[str, pathlib.Path, pathlib.Path]]:
    """Every step under ``tab`` that has BOTH an exact and a graph table."""
    out = []
    for g in sorted(tab.glob("*_graph.parquet")):
        if "_shard" in g.stem or "_by_donor" in g.stem:
            continue
        e = g.with_name(g.stem[: -len("_graph")] + ".parquet")
        if e.exists():
            out.append((e.stem, e, g))
    return out


def main() -> None:
    t0 = time.time()
    emit(__file__)          # which code ran, HEAD or dirty tree
    found = pairs(TAB)
    if not found:
        sys.exit(f"no <stem>.parquet / <stem>_graph.parquet pair under {TAB}. "
                 f"Run a step under both BIORAG_RETRIEVAL=exact and =graph first.")

    rows: list[pd.DataFrame] = []
    for step, e_path, g_path in found:
        e, g = pd.read_parquet(e_path), pd.read_parquet(g_path)
        keys = [c for c in AXES if c in e.columns and c in g.columns
                and e[c].notna().all() and g[c].notna().all()]
        if not keys:
            raise ValueError(f"{step}: none of AXES present in both tables")
        m = e.merge(g, on=keys, suffixes=("_ex", "_gr"), validate="one_to_one")
        if len(m) != len(e):
            raise ValueError(
                f"{step}: joined {len(m)} rows from {len(e)} exact and {len(g)} "
                f"graph on {keys}; the key set does not identify a condition")

        # A join key is never also a metric: it carries no suffix after the merge,
        # and comparing an axis to itself is meaningless. `rate` is both numeric
        # and an axis, so it reaches here without this.
        metrics = [c for c in e.columns if comparable(c, e, g) and c not in keys]
        print(f"[{time.time()-t0:.0f}s] {step}: {len(m)} conditions, "
              f"{len(metrics)} comparable metrics, keys {keys}", file=sys.stderr)
        for c in metrics:
            rows.append(pd.DataFrame({
                THREAT_MODEL: str(ThreatModel.WITH_ACCESS),
                "step": step, "metric": c,
                **{k: m[k] for k in keys},
                "exact": m[c + "_ex"].astype("float64"),
                "graph": m[c + "_gr"].astype("float64"),
                "diff": (m[c + "_gr"] - m[c + "_ex"]).astype("float64"),
            }))

    df = pd.concat(rows, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    print(f"\n[{time.time()-t0:.0f}s] wrote {len(df):,} rows to {OUT.name}\n",
          file=sys.stderr)
    for (step, metric), s in df.groupby(["step", "metric"], observed=True):
        d = s["diff"]
        if metric not in ("within_accuracy", "within_retrieval"):
            continue
        print(f"  {step[:34]:<34} {metric:<20} n={len(d):<5} "
              f"identical {100*(d.abs() < 1e-9).mean():5.1f}%  "
              f"|d|>0.01 {100*(d.abs() > 0.01).mean():5.1f}%  "
              f"mean {d.mean():+.5f}  "
              f"graph stronger {100*(d < -1e-9).mean():5.1f}% / "
              f"weaker {100*(d > 1e-9).mean():5.1f}%", file=sys.stderr)


if __name__ == "__main__":
    main()
