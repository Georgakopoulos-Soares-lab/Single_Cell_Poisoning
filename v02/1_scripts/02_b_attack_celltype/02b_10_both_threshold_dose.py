"""Step 02b_10 — the dose at which each attack starts working, with an interval.

The payoff of the shared ladder (SPEC.md §5.4b). Fourteen rates per condition is
not a result; it is fourteen numbers with the turning point left to the reader's
eye. This reduces each condition to ONE dose -- the smallest at which the
response exceeds a stated level -- and puts a donor-paired bootstrap interval on
it.

``both``, because it is the only step that spans the two arms. That is the whole
reason they were put on one ladder, and it is also why this step could not have
been written earlier: until ``02b_05`` gained ``retrieval`` and
``net_attack_success`` it reported ``attack_success`` over a GLOBAL denominator
of 26,055 queries while the insertion arm's headline was ``within_``, the
attacked celltype's own 120 to 17,571. A threshold read off the two tables as
they stood would have compared a within-celltype rate against a whole-dataset
one and called the difference an arm effect.

**Reads only ``*_by_donor.parquet``.** Not the graph tables: those are already
aggregated past the donor, and a threshold interval needs one donor draw held
fixed across every dose on the ladder. Two percentiles per dose cannot
reconstruct the joint distribution across doses that a crossing point depends
on.

**Every statistic is in ``00_common/threshold.py``** and tested there on curves
whose answer is known by hand. This file owns the grid, the identity columns and
the file discovery, and computes nothing.

**Censoring is a result.** A condition whose curve never reaches tau by dose
1.25 is right-censored and reported as such -- the honest form of "this attack
does not work" -- and one already above tau at 0.002 is left-censored, which is
the expected reading for hub-selected modify. Neither is a missing value and
neither may be filled in. A NULL censor means something different again: the
scope held no scored query at all.

Writes ``02b_10_both_threshold_dose.parquet``: one row per
(condition, response, tau), carrying ``within_*`` and ``global_*`` per the SCOPE
exception.
"""
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "00_common"))
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dataset_config import load  # noqa: E402
from flood_common import donor_tables  # noqa: E402
from provenance import emit  # noqa: E402
from schemas import (ATTACK_AXES, DEN_SUFFIX, GEOMETRY,  # noqa: E402
                     THREAT_MODEL, ThreatModel,
                     LABEL_CHANGE, NUM_SUFFIX, SELECTION,
                     SELECTION_AXES, TAU_LEVELS, THRESHOLD_RESPONSES)
from threshold import threshold_curve  # noqa: E402

BASE = _HERE.parents[2]
CFG = load(BASE)
TAB = BASE / "2_results" / CFG["dataset"] / "02_b_attack_celltype"
OUT = TAB / "02b_10_both_threshold_dose.parquet"

#: Columns that identify a CONDITION, i.e. a curve to find a threshold on.
#:
#: Named rather than inferred, because two columns that look like keys vary WITH
#: the dose and would put every rung in a group of its own: ``condition_seed``
#: is blake2b of ``source|rate|seed``, and ``n_edited`` is the dose in records.
#: An inferred identity would silently produce one-point curves and a table full
#: of censored rows.
#: ``geometry`` and ``label_change`` are here because THE ATTACK NAME DOES NOT
#: IDENTIFY A CONDITION -- the offence SPEC.md 5.5 names, and schemas.py says it
#: outright: "relabel and duplicate are ONE construction at two selection
#: settings, and nothing in the name says so." 02b_01 runs `relabel` at both
#: keep and relabel label-change; 02b_11 runs `shifted_inplace` at both. Without
#: these two, those collapse into one group, 56,784 rows of 02b_01 alone become
#: duplicates, and the per-donor pivot raises rather than silently averaging two
#: different attacks into one curve. Added 2026-09-27 on this step's first run.
IDENTITY = ("attack", "geometry", "label_change", "source", "target",
            "selection", "split_seed", "n_anchors", "seed")

BOOT_SEED = 0


def curves(d: pd.DataFrame, scope: str, response: str
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One condition's per-donor counts, shaped (n_doses, n_donors).

    Args:
        d: rows for a single condition, one per (rate, donor).
        scope: ``within`` or ``global``. response: an indicator name.

    Returns:
        ``(dose, num, den)`` with ``dose`` strictly increasing.
    """
    stem = f"{scope}_{response}"
    num = d.pivot(index="rate", columns="donor_id", values=stem + NUM_SUFFIX)
    den = d.pivot(index="rate", columns="donor_id", values=stem + DEN_SUFFIX)
    return (num.index.to_numpy(dtype=float),
            num.to_numpy(dtype=float), den.to_numpy(dtype=float))


def axes_of(keys: dict[str, object]) -> dict[str, str]:
    """The taxonomy axes (SPEC.md §5.4) a condition occupies.

    The sweeps write an attack NAME; the comparison this table exists for is
    between AXES -- insert against modify, unmoved against shifted. Becomes
    redundant when SPEC.md §5.5 has the sweeps write the axes directly.

    Args:
        keys: the groupby key dict for one condition. Read for ``attack``, and
            for ``selection``, ``geometry`` and ``label_change`` where the sweep
            writes them.

    Returns:
        ``{access, geometry, label, selection, knowledge}``.
    """
    attack = str(keys.get("attack", ""))
    sel_term = keys.get(SELECTION)
    sel_term = None if sel_term is None or str(sel_term) == "nan" else str(sel_term)

    if sel_term in SELECTION_AXES:
        sel, knowledge = SELECTION_AXES[sel_term]
    elif attack in SELECTION_AXES:
        sel, knowledge = SELECTION_AXES[attack]
    else:
        raise KeyError(
            f"no selection axis for attack={attack!r} selection={sel_term!r}; "
            f"add it to schemas.SELECTION_AXES rather than guessing here")

    if attack not in ATTACK_AXES:
        raise KeyError(
            f"no attack axes for {attack!r}; add it to schemas.ATTACK_AXES")
    ax = dict(ATTACK_AXES[attack])

    # PREFER WHAT THE SWEEP WROTE. ATTACK_AXES keys on a NAME, and a name cannot
    # carry an axis the sweep varies: 02b_11 runs both relabel and keep under the
    # single name shifted_inplace, so the table's own label_change column is the
    # truth and the dict entry is only a default. Same for geometry. This is the
    # "becomes redundant" in the docstring arriving one column at a time.
    for col, axis in ((GEOMETRY, "geometry"), (LABEL_CHANGE, "label")):
        v = keys.get(col)
        if v is not None and str(v) != "nan":
            ax[axis] = str(v)

    return {**ax, "selection": sel, "knowledge": knowledge}


def main() -> None:
    t0 = time.time()
    emit(__file__)          # which code ran, HEAD or dirty tree
    # donor_tables, not a bare glob: 02b_11 and 02b_15 write their graph run to
    # <stem>_by_donor_graph.parquet, and "*_by_donor.parquet" misses it and
    # matches the EXACT table instead -- the remerge shortcut, which understates
    # by up to 0.1240 on CL:0000583. Names are logged below so the variant that
    # won is in the run log rather than inferred.
    files = donor_tables(TAB)
    if not files:
        sys.exit(f"no *_by_donor*.parquet under {TAB} -- run the sweeps first "
                 f"(02b_01, 02b_02, 02b_05). This step reads nothing else.")
    print(f"[{time.time()-t0:.0f}s] {len(files)} donor tables: "
          f"{', '.join(f.name for f in files)}", file=sys.stderr)

    rows: list[dict] = []
    for f in files:
        d = pd.read_parquet(f)
        ident = [c for c in IDENTITY if c in d.columns]
        n_rate = d.rate.nunique()

        # A THRESHOLD DOSE NEEDS A DOSE LADDER. 02b_13, 02b_14 and 02b_15 hold
        # the dose fixed -- at 1.0, 0.25 and 0.25 -- and sweep a different axis
        # entirely: spread x jitter, site placement, alpha. They are not partial
        # dose sweeps to be censored, they are not dose sweeps at all, and
        # D_tau is undefined on them. Skipped whole rather than per group,
        # because the per-group guard below compares a group to its own table
        # and 1 == 1 passes it.
        if n_rate < 2:
            print(f"  SKIP {f.name}: {n_rate} dose, not a dose sweep",
                  file=sys.stderr)
            continue
        for keys, g in d.groupby(ident, observed=True, sort=False):
            keys = dict(zip(ident, keys if isinstance(keys, tuple) else (keys,),
                            strict=True))
            # A curve needs every rung. A short group means the sweep is partial
            # or IDENTITY is wrong; either way a threshold off it would be a
            # threshold on a different ladder.
            if g.rate.nunique() != n_rate:
                print(f"  SKIP {keys}: {g.rate.nunique()}/{n_rate} doses",
                      file=sys.stderr)
                continue
            ax = axes_of(keys)
            for response in THRESHOLD_RESPONSES:
                for tau in TAU_LEVELS:
                    # Per ROW, not per table: this step pools across sweeps, so
                    # one value for the file would be wrong, but every row comes
                    # from exactly one condition and so has a definite threat
                    # model. insert <-> no_access, modify <-> with_access.
                    tm = (ThreatModel.WITH_ACCESS if ax["access"] == "modify"
                          else ThreatModel.NO_ACCESS)
                    row = {THREAT_MODEL: str(tm), **keys, **ax,
                           "tau": tau, "response": response,
                           "n_doses": int(n_rate), "source_table": f.name}
                    for scope in ("within", "global"):
                        dose, num, den = curves(g, scope, response)
                        out = threshold_curve(dose, num, den, tau, BOOT_SEED)
                        row.update({f"{scope}_{k}": v for k, v in out.items()})
                    rows.append(row)
        print(f"[{time.time()-t0:5.0f}s] {f.name}: {len(rows)} rows so far",
              file=sys.stderr)

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    got = df[df.within_censor == "none"]
    print(f"\n[{time.time()-t0:.0f}s] wrote {len(df)} rows to {OUT.name}\n"
          f"  within-scope: {len(got)} with a threshold, "
          f"{(df.within_censor == 'left').sum()} left-censored, "
          f"{(df.within_censor == 'right').sum()} right-censored, "
          f"{df.within_censor.isna().sum()} with no scored query",
          file=sys.stderr)


if __name__ == "__main__":
    main()
