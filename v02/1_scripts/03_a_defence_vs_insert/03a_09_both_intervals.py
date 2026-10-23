"""Step 03a_09 — donor-paired intervals for every defence layer, both arms.

# No `from __future__ import annotations`: snakemake's script directive
# prepends a preamble, which would push it off line 1 and make it a SyntaxError.

The layer matrices report point estimates only, so `03a_04` and `03b_02` draw
distributions over conditions and say in their captions that no interval is
available. This supplies it, reading the `_by_donor` twins the three runs
already wrote and never re-running an attack.

WHY A PAIRED INTERVAL AND NOT TWO SEPARATE ONES. The question a defender asks
is not "what is the interval on the defended rate" but "is the DIFFERENCE from
no defence real". A defended rate and its undefended twin come from the same
condition, the same poison set and -- because `03a_01` and `03b_01` retrieve
once per condition and hand every configuration the same neighbour list -- the
same neighbours. They are almost perfectly correlated. Bootstrapping them
independently and subtracting would inflate the interval by roughly the sum of
two variances when the pairing removes nearly all of it, and would wrongly
centre the difference. So the donors are resampled ONCE per condition and every
configuration is recomputed on that same draw, which is what
`flood_common.donor_boot_ratios` exists to make possible.

THE UNIT OF INDEPENDENCE IS THE DONOR, 78 of them. Not the query: queries from
one donor share a subject, a dissociation batch and a sequencing run, so
treating 49,456 of them as independent would give intervals far too narrow.

ONE RECOVERED KEY, VERIFIED RATHER THAN ASSUMED. `03b_01`'s cached_clean
`_by_donor` table does not carry its `selection` column, so its hub and uniform
rows are indistinguishable on the natural key and `condition_seed` does not
separate them either -- every one of its 194,688 rows is a duplicate. The
pairing is recovered from row order and then CHECKED against the main table,
which does carry `selection`: under `occ 0 = hub, occ 1 = uniform` the
reaggregated rates match to 3.0e-08, float32 rounding, on all 624 comparisons,
while the swapped assignment is off by up to 0.978. That is a measurement with
a seven-order margin, not an inference, and the check runs on every execution
rather than having been done once by hand. If it ever fails the step exits.
`03b_01` NOW EMITS `selection` (fixed 2026-09-30), so a future run of that arm
needs none of this. The committed table predates the fix and is not being
re-run: the cached_clean arm is deterministic and would reproduce byte-
identical numbers, so ~85 minutes of compute would buy only a column that is
already recovered and verified. The recovery below stays because the table it
reads still lacks the column, and it is written to NOT fire when the column is
present -- `selection` then joins the key and no duplication arises.

WHAT THIS DOES NOT DO. It does not re-derive any point estimate. Every `_value`
column here is the same `num.sum()/den.sum()` the layer matrix reports, and the
step asserts that against the committed tables before writing.
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
from flood_common import boot_ci, donor_boot_ratios  # noqa: E402
from provenance import emit  # noqa: E402
from schemas import N_BOOTSTRAP  # noqa: E402

BASE = _HERE.parents[2]
CFG = load(BASE)
DS = CFG["dataset"]
RES = BASE / "2_results" / DS
OUT = RES / "03_a_defence_vs_insert" / "03a_09_both_intervals.parquet"

#: (arm label, stage, by_donor table, main table, threat_model, access,
#:  retrieval_path). The main table is read to verify the point estimates and
#: to recover any key the donor frame dropped.
ARMS = (
    ("insert", "03_a_defence_vs_insert",
     "03a_01_no_access_layer_matrix_by_donor.parquet",
     "03a_01_no_access_layer_matrix.parquet", "no_access", "insert", "hnsw"),
    ("modify_graph", "03_b_defence_vs_modify",
     "03b_01_with_access_layer_matrix_by_donor_graph.parquet",
     "03b_01_with_access_layer_matrix_graph.parquet",
     "with_access", "overwrite", "graph"),
    ("modify_cached_clean", "03_b_defence_vs_modify",
     "03b_01_with_access_layer_matrix_by_donor_cached_clean.parquet",
     "03b_01_with_access_layer_matrix_cached_clean.parquet",
     "with_access", "overwrite", "cached_clean"),
)
#: BIORAG_TAG (2026-10-04) points the step at ONE tagged layer matrix from the
#: full-parity grid (RUNLOG Run 48) instead of the three default tables, and
#: tags the output, so a tagged run can never overwrite the committed default
#: intervals. The arm is read off which tagged pair exists: an insert tag
#: (relhub, flo64, ...) names a 03a_01 pair, a modify tag (modhub, moduni) a
#: 03b_01 graph pair. Unset, the step is unchanged. A tagged cached_clean pair
#: (2026-10-10, the label-only arm re-run on the fixed accuracy) is the third
#: kind; its output names the arm, since its tag ("v3") is not one of the
#: per-block tags and would otherwise read as covering every v3 table.
TAG = os.environ.get("BIORAG_TAG", "").strip()
if TAG:
    _tagged = (
        ("insert", "03_a_defence_vs_insert",
         f"03a_01_no_access_layer_matrix_by_donor_{TAG}.parquet",
         f"03a_01_no_access_layer_matrix_{TAG}.parquet",
         "no_access", "insert", "hnsw"),
        ("modify_graph", "03_b_defence_vs_modify",
         f"03b_01_with_access_layer_matrix_by_donor_graph_{TAG}.parquet",
         f"03b_01_with_access_layer_matrix_graph_{TAG}.parquet",
         "with_access", "overwrite", "graph"),
        ("modify_cached_clean", "03_b_defence_vs_modify",
         f"03b_01_with_access_layer_matrix_by_donor_cached_clean_{TAG}.parquet",
         f"03b_01_with_access_layer_matrix_cached_clean_{TAG}.parquet",
         "with_access", "overwrite", "cached_clean"),
    )
    ARMS = tuple(a for a in _tagged
                 if (RES / a[1] / a[2]).exists() and (RES / a[1] / a[3]).exists())
    if len(ARMS) != 1:
        sys.exit(f"BIORAG_TAG={TAG}: expected exactly one tagged layer-matrix "
                 f"pair (main + by_donor), found {len(ARMS)}")
    OUT = OUT.with_name(f"03a_09_both_intervals_cached_clean_{TAG}.parquet"
                        if ARMS[0][0] == "modify_cached_clean"
                        else f"03a_09_both_intervals_{TAG}.parquet")
#: Columns that identify a CONDITION -- everything except the defence.
COND = ("source", "attack", "rate", "selection", "geometry")
#: The defence axis. lc_budget splits label_consistency and stack into three.
CFG_COLS = ("defense", "lc_budget")
#: Metrics to interval, where the donor frame carries the num/den pair.
METRICS = ("within_attack_success", "within_accuracy", "within_retrieval",
           "global_attack_success", "global_accuracy", "global_retrieval")
BASELINE_CFG = "none"
#: Tolerance on the point-estimate cross-check. The donor frames are float32
#: sums, so agreement is to about 1e-7, not to machine epsilon.
TOL = 1e-6


def cfg_label(defense, lc_budget) -> str:
    """``defense`` plus its budget where one applies, e.g. ``stack@0.1``.

    The budget is formatted through ``float(...)`` with an explicit precision,
    which is not fussiness. The donor frames store ``lc_budget`` as float64 and
    the layer matrices as float32, and an f-string on a ``np.float32`` widens
    it: ``f"{np.float32(0.1)}"`` is ``0.10000000149011612`` while the float64
    is ``0.1``. Labelling both the same way therefore produced two disjoint
    sets of names for the six budget-bearing configurations, and the join that
    checks these intervals against the committed point estimates silently
    matched only the six budget-free ones -- half the rows, with no error.
    ``.astype(str)`` on a float32 Series does NOT do this, which is why the
    figure scripts were unaffected and why the difference is worth naming here.
    """
    if not pd.notna(lc_budget):
        return str(defense)
    return f"{defense}@{float(lc_budget):.4g}"


def recover_selection(bd: pd.DataFrame, mn: pd.DataFrame, keys: list[str],
                      t0: float) -> pd.DataFrame:
    """Rebuild a ``selection`` column the donor frame never wrote.

    Verifies the recovery against ``mn``, which does carry it, and exits rather
    than guessing if the check fails. See the module docstring.
    """
    order = sorted(mn.selection.unique())
    bd = bd.copy()
    bd["occ"] = bd.groupby(keys, observed=True, dropna=False).cumcount()
    if bd.occ.max() != len(order) - 1:
        sys.exit(f"cannot recover selection: {bd.occ.max()+1} rows per key, "
                 f"{len(order)} selections")
    bd["selection"] = bd.occ.map(dict(enumerate(order)))
    gk = [k for k in keys if k != "donor_id"] + ["selection"]
    agg = (bd.groupby(gk, observed=True, dropna=False)
             .agg(num=("within_attack_success_num", "sum"),
                  den=("within_attack_success_den", "sum")).reset_index())
    agg["ratio"] = np.where(agg.den > 0, agg.num / agg.den, np.nan)
    chk = agg.merge(mn[gk + ["within_attack_success"]], on=gk, how="inner")
    diff = (chk.ratio - chk.within_attack_success).abs()
    worst = float(np.nanmax(diff))
    if worst > 1e-5:
        sys.exit(f"selection recovery FAILED: worst |diff| {worst:.3e} over "
                 f"{len(chk)} comparisons. Do not guess -- fix 03b_01 to emit "
                 f"selection in its donor frame and re-run that arm.")
    print(f"[{time.time()-t0:.0f}s]   selection recovered from row order and "
          f"verified: worst |diff| {worst:.2e} over {len(chk)} comparisons",
          file=sys.stderr)
    return bd.drop(columns="occ")


def main() -> None:
    t0 = time.time()
    emit(__file__)
    boot_seed = int(CFG["bootstrap_seed"])
    out: list[pd.DataFrame] = []

    for arm, stage, bd_name, mn_name, threat, access, path in ARMS:
        bd = pd.read_parquet(RES / stage / bd_name)
        mn = pd.read_parquet(RES / stage / mn_name)
        print(f"[{time.time()-t0:.0f}s] {arm}: {len(bd):,} donor rows, "
              f"{len(mn):,} conditions", file=sys.stderr)

        keys = [c for c in ("source", "attack", "rate", "selection", "geometry")
                if c in bd.columns] + list(CFG_COLS) + ["donor_id"]
        if bd.duplicated(keys, keep=False).any():
            if "selection" in mn.columns and "selection" not in bd.columns:
                bd = recover_selection(bd, mn, keys, t0)
                keys = keys[:-3] + ["selection"] + list(CFG_COLS) + ["donor_id"]
            if bd.duplicated(keys, keep=False).any():
                sys.exit(f"{arm}: donor key still not unique after recovery")
        # geometry and selection identify the sub-arm but are carried on the
        # main frame only. Attach them so the output is self-describing AND so
        # the concatenated result has no all-null key column: the arms are
        # concatenated at the end, and a column present for one arm and absent
        # for another becomes NaN there, which then silently joins against
        # nothing. That is how the first run of this step matched 0 of 2,496
        # graph rows. Constant columns are filled with the constant; a column
        # that varies is derived from `attack`, which determines it.
        for col in ("geometry", "selection"):
            if col in bd.columns or col not in mn.columns:
                continue
            if mn[col].nunique() == 1:
                bd[col] = mn[col].iloc[0]
            elif mn.groupby("attack", observed=True)[col].nunique().max() == 1:
                m = mn.drop_duplicates("attack").set_index("attack")[col]
                bd[col] = bd.attack.map(m)
            else:
                sys.exit(f"{arm}: cannot attach '{col}' -- it varies within an "
                         f"attack, so the donor frame cannot be keyed on it. "
                         f"Emit it from the producing step.")
            if bd[col].isna().any():
                sys.exit(f"{arm}: '{col}' is null for "
                         f"{int(bd[col].isna().sum()):,} donor rows")

        cond_cols = [c for c in COND if c in bd.columns]
        bd["cfg"] = [cfg_label(d, b) for d, b in zip(bd.defense, bd.lc_budget)]
        mn["cfg"] = [cfg_label(d, b) for d, b in zip(mn.defense, mn.lc_budget)]
        metrics = [m for m in METRICS if f"{m}_num" in bd.columns]

        n_cond = 0
        for cond, g in bd.groupby(cond_cols, observed=True):
            piv_n = g.pivot_table(index="cfg", columns="donor_id",
                                  values=[f"{m}_num" for m in metrics],
                                  aggfunc="sum", observed=True)
            piv_d = g.pivot_table(index="cfg", columns="donor_id",
                                  values=[f"{m}_den" for m in metrics],
                                  aggfunc="sum", observed=True)
            cfgs = list(piv_n.index)
            if BASELINE_CFG not in cfgs:
                sys.exit(f"{arm} {cond}: no '{BASELINE_CFG}' row to pair against")
            base_i = cfgs.index(BASELINE_CFG)

            # ONE donor draw for the whole condition: every metric and every
            # configuration is recomputed on it, which is what makes the
            # paired difference below a paired statistic.
            num = np.stack([piv_n[f"{m}_num"].to_numpy(float) for m in metrics])
            den = np.stack([piv_d[f"{m}_den"].to_numpy(float) for m in metrics])
            n_met, n_cfg, n_don = num.shape
            pt, bt = donor_boot_ratios(num.reshape(-1, n_don),
                                       den.reshape(-1, n_don), boot_seed)
            pt = pt.reshape(n_met, n_cfg)
            bt = bt.reshape(n_met, n_cfg, N_BOOTSTRAP)
            lo, hi = boot_ci(bt)
            dlo, dhi = boot_ci(bt - bt[:, base_i:base_i + 1, :])
            delta = pt - pt[:, base_i:base_i + 1]

            row = {c: v for c, v in zip(cond_cols, cond if isinstance(cond, tuple) else (cond,))}
            f = pd.DataFrame({**row, "cfg": cfgs, "n_donors": n_don})
            for i, m in enumerate(metrics):
                f[m] = pt[i]
                f[f"{m}_lo"], f[f"{m}_hi"] = lo[i], hi[i]
                f[f"{m}_delta"] = np.where(np.arange(n_cfg) == base_i, np.nan, delta[i])
                f[f"{m}_delta_lo"] = np.where(np.arange(n_cfg) == base_i, np.nan, dlo[i])
                f[f"{m}_delta_hi"] = np.where(np.arange(n_cfg) == base_i, np.nan, dhi[i])
            f["arm"], f["threat_model"] = arm, threat
            f["access"], f["retrieval_path"] = access, path
            if TAG:
                # A flood tag fixes n_anchors for the whole table; carry it so
                # the output says which point on that axis it is.
                f["tag"] = TAG
                if "n_anchors" in mn.columns and mn.n_anchors.notna().any():
                    anch = mn.n_anchors.dropna().unique()
                    if len(anch) != 1:
                        sys.exit(f"{arm}: n_anchors varies within tag {TAG}")
                    f["n_anchors"] = int(anch[0])
            out.append(f)
            n_cond += 1
        print(f"[{time.time()-t0:.0f}s]   {n_cond} conditions x {len(metrics)} "
              f"metrics intervalled", file=sys.stderr)

    res = pd.concat(out, ignore_index=True)
    res["n_bootstrap"] = N_BOOTSTRAP
    res["bootstrap_seed"] = boot_seed
    res["seed"] = 0

    # The point estimates must equal the committed layer matrices. This is the
    # check that stops a silent regrouping error producing plausible intervals
    # around the wrong centre.
    n_checked = 0
    for arm, stage, _, mn_name, *_ in ARMS:
        mn = pd.read_parquet(RES / stage / mn_name)
        mn["cfg"] = [cfg_label(d, b) for d, b in zip(mn.defense, mn.lc_budget)]
        a = res[res.arm == arm]
        # Belt and braces: never key on a column that is entirely null for
        # this arm, whatever attached it.
        on = [c for c in COND if c in mn.columns and c in a.columns
              and a[c].notna().any()] + ["cfg"]
        for m in ("within_attack_success", "within_accuracy"):
            if m not in mn.columns:
                continue
            j = a.merge(mn[on + [m]], on=on, suffixes=("", "_main"))
            if len(j) != len(a):
                sys.exit(f"{arm}/{m}: joined {len(j)} of {len(a)} rows")
            d = (j[m] - j[f"{m}_main"]).abs()
            if float(np.nanmax(d)) > TOL:
                sys.exit(f"{arm}/{m}: point estimate differs from the layer "
                         f"matrix by {float(np.nanmax(d)):.3e}")
            n_checked += len(j)
    print(f"[{time.time()-t0:.0f}s] point estimates agree with the committed "
          f"matrices on {n_checked:,} comparisons (tol {TOL:.0e})", file=sys.stderr)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT, index=False)
    print(f"[{time.time()-t0:.0f}s] wrote {len(res):,} rows to {OUT.name}",
          file=sys.stderr)

    sig = res[res.within_attack_success_delta_hi < 0]
    print(f"  {len(sig):,} of {len(res):,} (condition, layer) rows have a "
          f"within_attack_success reduction whose 95% interval excludes zero",
          file=sys.stderr)


if __name__ == "__main__":
    main()
