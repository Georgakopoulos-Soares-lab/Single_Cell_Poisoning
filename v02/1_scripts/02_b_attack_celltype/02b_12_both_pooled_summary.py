"""Step 02b_12 — one curve per attack type, pooled across all 26 celltypes.

The headline. Every other table in this module is PER CELLTYPE: ``02b_01``
reports each source separately, ``02b_10`` finds each source's threshold. Nothing
in v02 answers "what does this attack do to the atlas, with an interval" -- which
is the first number a reader looks for and the one a dose-response figure plots.

This is v01's ``global_summary``/``source_summary`` family, rebuilt. Two of
v01's levels do not survive the port, and that is a design difference rather
than an omission: v01 gave each source 3-6 targets, so ``pair_summary`` and
``source_summary`` were genuinely different poolings. ``pick_targets`` gives
each source exactly ONE target (SPEC.md §5.3), so those two levels coincide, and
``02b_01``'s own rows already are them -- one row per (type, source, dose) with
donor-paired intervals. Only the pooling ACROSS sources was missing.

**Not called ``global_summary``.** "Global" already means two other things here:
the atlas-scope module (``02_a``), where the budget is drawn from the whole
reference, and the ``global_`` column prefix, which means "scored over every
query" as opposed to ``within_``. A third sense in a table name is the offence
SPEC.md §5.5 exists to catch. This pools across CELLTYPES, so it is
``pooled_``, and both scopes are reported inside it.

**Reads ``*_by_donor.parquet`` only.** Pooling cannot be done from the
per-celltype tables: they are already aggregated past the donor, and the pooled
interval needs one donor draw held fixed across every celltype at once -- the
same requirement ``02b_10`` has along the dose axis. Different celltypes'
``within_`` queries are disjoint sets drawn from the SAME 78 donors, so a donor
who is unusual carries that through every celltype simultaneously; resampling
celltypes independently would let that cancel and understate the interval.

Writes ``02b_12_both_pooled_summary.parquet``: one row per
(attack type, dose, metric), carrying ``within_*`` and ``global_*`` per the
SCOPE exception.
"""
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "00_common"))
import time  # noqa: E402

import pandas as pd  # noqa: E402

from dataset_config import load  # noqa: E402
from flood_common import donor_ratio_ci, donor_tables  # noqa: E402
from provenance import emit  # noqa: E402
from schemas import (ATTACK_AXES, DEN_SUFFIX, NUM_SUFFIX,  # noqa: E402
                     THREAT_MODEL, ThreatModel)

BASE = _HERE.parents[2]
CFG = load(BASE)
TAB = BASE / "2_results" / CFG["dataset"] / "02_b_attack_celltype"
OUT = TAB / "02b_12_both_pooled_summary.parquet"

#: What identifies an attack TYPE -- the axes, never the name.
#:
#: ``source`` and ``target`` are deliberately absent: pooling over them is the
#: whole point. ``seed`` is absent for the same reason a seed is not a finding.
TYPE_KEYS = ("attack", "geometry", "label_change", "selection", "n_anchors",
             "split_seed")

BOOT_SEED = 0


def metrics_of(cols: list[str], scope: str) -> list[str]:
    """Indicator names present for one scope, from the ``_num``/``_den`` pairs.

    Derived from the table rather than listed, so an indicator added to the
    sweep appears here without this file being edited -- and one removed
    disappears instead of raising.

    Args:
        cols: the donor table's columns. scope: ``within`` or ``global``.

    Returns:
        Indicator names having BOTH a numerator and a denominator column.
    """
    pre = f"{scope}_"
    num = {c[len(pre):-len(NUM_SUFFIX)] for c in cols
           if c.startswith(pre) and c.endswith(NUM_SUFFIX)}
    den = {c[len(pre):-len(DEN_SUFFIX)] for c in cols
           if c.startswith(pre) and c.endswith(DEN_SUFFIX)}
    return sorted(num & den)


def pooled(g: pd.DataFrame, scope: str, metric: str) -> dict[str, float]:
    """One metric pooled over every celltype in ``g``, with a donor interval.

    Sums each donor's numerator and denominator ACROSS celltypes first, then
    resamples donors. Summing first is what makes the draw paired: a donor
    enters or leaves the replicate for all celltypes at once, which is the
    correlation that actually exists -- the same cells contribute to every
    celltype's ``global_`` scope, and one donor's peculiarity is not 26
    independent accidents.

    Args:
        g: donor rows for one (attack type, dose). scope/metric: what to pool.

    Returns:
        ``{pooled, ci_lo, ci_hi, n_donors, n_sources}``; the estimate is NaN
        where the denominator is empty, which happens for ``within_`` on a
        celltype with no query of its own.
    """
    stem = f"{scope}_{metric}"
    by = g.groupby("donor_id", observed=True)[
        [stem + NUM_SUFFIX, stem + DEN_SUFFIX]].sum()
    est, lo, hi = donor_ratio_ci(by[stem + NUM_SUFFIX].to_numpy(float),
                                 by[stem + DEN_SUFFIX].to_numpy(float),
                                 by.index.to_numpy(), BOOT_SEED)
    return {f"{scope}_pooled": est, f"{scope}_ci_lo": lo, f"{scope}_ci_hi": hi,
            f"{scope}_n_donors": int((by[stem + DEN_SUFFIX] > 0).sum())}


def main() -> None:
    t0 = time.time()
    emit(__file__)          # which code ran, HEAD or dirty tree
    # donor_tables, not a bare glob: 02b_11 and 02b_15 write their graph run to
    # <stem>_by_donor_graph.parquet, and "*_by_donor.parquet" misses it and
    # matches the EXACT table instead -- the remerge shortcut, which understates
    # by up to 0.1240 on CL:0000583. Pooling the understating table would have
    # been invisible in the output.
    files = donor_tables(TAB)
    if not files:
        sys.exit(f"no *_by_donor*.parquet under {TAB} -- run the sweeps first. "
                 f"Pooling cannot be done from the per-celltype tables: they "
                 f"are aggregated past the donor, and the pooled interval needs "
                 f"one donor draw held across every celltype at once.")
    print(f"[{time.time()-t0:.0f}s] {len(files)} donor tables: "
          f"{', '.join(f.name for f in files)}", file=sys.stderr)

    rows: list[dict] = []
    for f in files:
        d = pd.read_parquet(f)
        keys = [c for c in TYPE_KEYS if c in d.columns]
        scopes = [s for s in ("within", "global")
                  if metrics_of(list(d.columns), s)]
        for kv, g in d.groupby([*keys, "rate"], observed=True, sort=False):
            kv = dict(zip([*keys, "rate"], kv, strict=True))
            for metric in metrics_of(list(d.columns), scopes[0]):
                # Per ROW, not per file: this step pools across sweeps, so one
                # value for the table would be wrong, but every row comes from
                # one attack type and so has a definite threat model.
                # insert <-> no_access, modify <-> with_access.
                acc = ATTACK_AXES.get(str(kv.get("attack", "")), {}).get("access")
                tm = (ThreatModel.WITH_ACCESS if acc == "modify"
                      else ThreatModel.NO_ACCESS)
                row = {THREAT_MODEL: str(tm), **kv, "metric": metric,
                       "n_sources": int(g.source.nunique()),
                       "source_table": f.name}
                for scope in scopes:
                    row.update(pooled(g, scope, metric))
                rows.append(row)
        print(f"[{time.time()-t0:5.0f}s] {f.name}: {len(rows)} rows so far",
              file=sys.stderr)

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\n[{time.time()-t0:.0f}s] wrote {len(df)} rows to {OUT.name}\n"
          f"  {df.metric.nunique()} metrics x {df.rate.nunique()} doses, "
          f"pooled over {df.n_sources.max()} celltypes and "
          f"{df.within_n_donors.max()} donors", file=sys.stderr)


if __name__ == "__main__":
    main()
