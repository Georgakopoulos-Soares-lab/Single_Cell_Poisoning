"""Step 03b_03 — the add_items band, measured over 208 conditions instead of 4.

# No `from __future__ import annotations`: snakemake's script directive
# prepends a preamble, which would push it off line 1 and make it a SyntaxError.

``02b_34`` established that a graph result is one realisation of a distribution
rather than a re-derivable value, and sized that distribution on FOUR
conditions at six repeats. CLAUDE.md §1 turns its answer into the rule every
graph table is quoted under: outcome numbers good to about ±0.02 within-celltype
and ±0.001 global. Four conditions is a thin basis for a bound that governs
seven tables.

This widens it. ``03b_01``'s graph arm was run a SECOND time with identical
inputs -- same ``condition_seed``, same poison construction, same bootstrap seed,
same defence configurations -- and the two realisations are kept side by side
rather than one replacing the other. Replacing would have swapped one arbitrary
draw for another at zero epistemic gain; keeping both turns the same four hours
into 208 conditions x 12 configurations of paired evidence.

THE CENTRAL DISTINCTION IS THE THING TO WATCH. ``02b_34`` found that retrieval
rates reproduce EXACTLY while outcome rates do not, and that this is mechanism
rather than coincidence: insertion order perturbs which neighbours come back,
but whether ANY poisoned record is in the list is stable, so the vote absorbs
the churn and only predictions near parity flip. If that holds here it holds
across a far wider grid than it was established on. If it does not, the rule in
CLAUDE.md §1 needs rewriting, and finding that out is worth more than
confirming it.

THE CONTROLS ARE WHAT MAKE THE SPREAD ATTRIBUTABLE. ``within_baseline_accuracy``
and ``global_baseline_accuracy`` are the CLEAN side: nothing is inserted to
compute them, so they must reproduce exactly. ``n_poison`` and
``n_donor_edited`` describe the edit itself and must also be identical. They are
reported in the same table as every other metric, and if any of them moves the
cause is not ``add_items`` and the whole measurement means something else.
"""
import os
import pathlib
import sys
import time

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "00_common"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dataset_config import load  # noqa: E402
from provenance import emit  # noqa: E402

BASE = _HERE.parents[2]
CFG = load(BASE)
DS = CFG["dataset"]
TAB = BASE / "2_results" / DS / "03_b_defence_vs_modify"
STEM = "03b_01_with_access_layer_matrix"
#: Opt-in (added 2026-10-09): BIORAG_TAG=<tag> compares the tagged 03b_01 graph
#: table with its second realisation ``<tag>_rep2`` (or BIORAG_REP2_TAG) and
#: writes a tagged output. Untagged, the original pair is read as before.
TAG = os.environ.get("BIORAG_TAG", "")
REP2_TAG = os.environ.get("BIORAG_REP2_TAG", f"{TAG}_rep2" if TAG else "rep2")
_t1 = f"_{TAG}" if TAG else ""
REP1 = TAB / f"{STEM}_graph{_t1}.parquet"
REP2 = TAB / f"{STEM}_graph_{REP2_TAG}.parquet"
BD1 = TAB / f"{STEM}_by_donor_graph{_t1}.parquet"
BD2 = TAB / f"{STEM}_by_donor_graph_{REP2_TAG}.parquet"
OUT = TAB / f"03b_03_with_access_repeatability{_t1}.parquet"

#: Condition identity. `selection` is absent from rep1 (written before the
#: 2026-09-30 schema fix) and constant at `hub` for this phase, so it is not a
#: key here; `attack` already separates the two geometries.
KEYS = ("source", "attack", "rate", "defense", "lc_budget")
#: Metrics that SHOULD move, being downstream of the vote.
OUTCOME = ("within_accuracy", "within_attack_success", "global_accuracy",
           "global_attack_success", "false_positive", "clean_accuracy_defended")
#: Controls that must NOT move. The clean side inserts nothing; the edit
#: descriptors are fixed by the seed. If any of these moves, the spread below
#: is not attributable to add_items.
CONTROL = ("within_baseline_accuracy", "global_baseline_accuracy",
           "n_poison", "n_donor_edited")
#: Derived from the donor frames, which carry the num/den pair the main table
#: does not. This is the metric 02b_34 found reproduces exactly.
RETRIEVAL = ("within_retrieval", "global_retrieval")


def cfg_key(d: pd.DataFrame) -> pd.DataFrame:
    """Normalise the join key. See 03a_09 on why lc_budget is formatted."""
    d = d.copy()
    d["lc_key"] = [f"{float(b):.4g}" if pd.notna(b) else ""
                   for b in d.lc_budget]
    return d


def retrieval_rates(path: pathlib.Path) -> pd.DataFrame:
    """Per-condition retrieval rates from a donor frame's num/den columns."""
    bd = cfg_key(pd.read_parquet(path))
    gk = [k for k in KEYS if k != "lc_budget"] + ["lc_key"]
    out = bd.groupby(gk, observed=True, dropna=False).agg(
        **{f"{m}_num": (f"{m}_num", "sum") for m in RETRIEVAL},
        **{f"{m}_den": (f"{m}_den", "sum") for m in RETRIEVAL}).reset_index()
    for m in RETRIEVAL:
        out[m] = np.where(out[f"{m}_den"] > 0,
                          out[f"{m}_num"] / out[f"{m}_den"], np.nan)
    return out[gk + list(RETRIEVAL)]


def main() -> None:
    t0 = time.time()
    emit(__file__)
    for p in (REP1, REP2, BD1, BD2):
        if not p.exists():
            sys.exit(f"missing {p.name} -- run 03b_01's graph phase twice first")

    a, b = cfg_key(pd.read_parquet(REP1)), cfg_key(pd.read_parquet(REP2))
    gk = [k for k in KEYS if k != "lc_budget"] + ["lc_key"]
    for name, f in (("rep1", a), ("rep2", b)):
        if f.duplicated(gk).any():
            sys.exit(f"{name}: join key is not unique")
    if len(a) != len(b):
        sys.exit(f"rep1 has {len(a):,} rows, rep2 has {len(b):,}")

    metrics = [m for m in OUTCOME + CONTROL if m in a.columns and m in b.columns]
    j = a[gk + metrics].merge(b[gk + metrics], on=gk, suffixes=("_1", "_2"))
    if len(j) != len(a):
        sys.exit(f"joined {len(j):,} of {len(a):,} rows")

    ra, rb = retrieval_rates(BD1), retrieval_rates(BD2)
    jr = ra.merge(rb, on=gk, suffixes=("_1", "_2"))
    if len(jr) != len(ra):
        sys.exit(f"retrieval joined {len(jr):,} of {len(ra):,} rows")

    rows: list[pd.DataFrame] = []
    for m, kind in ([(m, "outcome") for m in metrics if m in OUTCOME]
                    + [(m, "control") for m in metrics if m in CONTROL]
                    + [(m, "retrieval") for m in RETRIEVAL]):
        src = jr if kind == "retrieval" else j
        v1, v2 = src[f"{m}_1"].astype(float), src[f"{m}_2"].astype(float)
        rows.append(pd.DataFrame({
            **{k: src[k] for k in gk}, "metric": m, "kind": kind,
            "rep1": v1, "rep2": v2, "abs_diff": (v1 - v2).abs(),
            "signed_diff": v2 - v1}))
    res = pd.concat(rows, ignore_index=True)
    res["threat_model"] = "with_access"
    res["access"] = "overwrite"
    res["retrieval_path"] = "graph"
    res["n_realisations"] = 2
    res["seed"] = 0
    TAB.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT, index=False)

    print(f"[{time.time()-t0:.0f}s] wrote {len(res):,} rows to {OUT.name}",
          file=sys.stderr)
    summ = (res.groupby(["kind", "metric"], observed=True).abs_diff
               .agg(n="size", median="median",
                    p95=lambda s: float(np.nanpercentile(s, 95)),
                    max="max",
                    exact=lambda s: int((s < 1e-12).sum()))
               .reset_index().sort_values(["kind", "max"], ascending=[True, False]))
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.6f}"),
          file=sys.stderr)

    bad = summ[(summ.kind == "control") & (summ["max"] > 1e-9)]
    if len(bad):
        print("\n!! A CONTROL MOVED. The spread above is not attributable to "
              "add_items alone:\n" + bad.to_string(index=False), file=sys.stderr)
    else:
        print("\ncontrols all exact -- the spread is add_items and nothing else",
              file=sys.stderr)


if __name__ == "__main__":
    main()
