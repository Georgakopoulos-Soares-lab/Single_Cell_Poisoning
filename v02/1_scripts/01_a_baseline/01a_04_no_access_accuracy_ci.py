"""Step 01a_04 — donor-paired intervals on the unpoisoned accuracies and the granularity gap.

01a_03 reports what the deployed system answered. This reports how much of that
is a property of the system and how much is a property of the 78 people whose
lungs it was measured on.

**Donors are resampled, never cells.** Cells from one donor share genotype,
batch, dissociation protocol and sequencing run, so the 49,456 queries are
closer to 78 observations than to 49,456. A cell bootstrap would report an
interval narrower by roughly the square root of the cells-per-donor ratio and
would be wrong by that factor. ``flood_common.donor_ratio_ci`` is the single
implementation of this in the project; it is imported rather than reimplemented
so that no two steps can disagree about what the interval means.

**The granularity gap is estimated paired.** ``lineage - exact`` is the share of
apparent errors that are really the two annotators disagreeing about
granularity, and it is the quantity a reader needs to decide whether a 53%
accuracy is a broken system or a vocabulary mismatch. Estimating it as the
difference of two independently-bootstrapped intervals would inflate its width
by donor-to-donor variation in baseline accuracy -- which is large, and which
cancels exactly when both rates are taken inside the same resampled donors.
That pairing is free here: the numerator becomes ``lineage_count -
correct_count`` over a shared denominator, so the ratio of the difference is the
difference of the ratios, resample by resample. The same trick gives paired
intervals on each tolerance step.

Point estimates are recomputed from the per-donor counts and checked against
01a_03's table. They must agree to floating-point tolerance; if they do not, the
two steps are reading different data and the run stops rather than publishing a
number whose provenance is ambiguous.

``__all__`` is derived here by summing celltypes within each donor, not read
from a stored row -- a stored pooled row and a derived one can drift, and only
one of them can be right.

**THREAT MODEL: none.** Clean control, no attacker. ``threat_model`` is written
so the table can be attributed to the arm it feeds.

Reads ``01a_03_no_access_per_donor.parquet`` and ``01a_03_no_access_clean_baseline.parquet``.
Writes ``01a_04_no_access_accuracy_ci.parquet`` -- one row per celltype, plus ``__all__``.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_common"))
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dataset_config import load  # noqa: E402
from flood_common import donor_ratio_ci  # noqa: E402
from provenance import emit  # noqa: E402
from schemas import DONOR_ID, N_BOOTSTRAP, THREAT_MODEL, ThreatModel  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[2]
CFG = load(BASE)
DS = CFG["dataset"]
OUT = BASE / "2_results" / DS / "01_a_baseline"

#: Bootstrap seed. Passed in, never generated (``CLAUDE.md`` §7): an interval
#: that cannot be re-derived is not evidence. Taken from config so every step in
#: the project resamples the same way.
BOOT_SEED = int(CFG["bootstrap_seed"])

#: Rates to bracket, in the order they are reported. ``correct`` is exact match;
#: the rest are the tolerance ladder and the reachability ceiling.
METRICS = ("correct", "lineage", "within1", "within2", "within3", "reachable")

#: Rates whose gap above exact match is reported with a paired interval. Each
#: says how much of the exact-match error is granularity rather than retrieval.
GAPS = ("lineage", "within1", "within2", "within3")

#: Column stem written for each metric, so the table reads as a rate rather than
#: as the indicator it was counted from.
STEM = {"correct": "clean_accuracy", "lineage": "lineage",
        "within1": "within1", "within2": "within2", "within3": "within3",
        "reachable": "reachable"}


def rate_with_ci(num: np.ndarray, den: np.ndarray, donor: np.ndarray,
                 seed: int) -> tuple[float, float, float]:
    """Point estimate and donor-resampled interval for one rate.

    Args:
        num: (n_donors,) float, per-donor count of the numerator event. May be
            negative, which is what makes the paired gaps work: a difference of
            two counts over a shared denominator is a signed numerator.
        den: (n_donors,) float, per-donor query count. Shared across every
            metric of a celltype, which is what pairs them.
        donor: (n_donors,) str, donor ids -- the unit resampled.
        seed: bootstrap seed, passed in.

    Returns:
        ``(value, lo, hi)``, the 2.5th and 97.5th percentiles of
        :data:`~schemas.N_BOOTSTRAP` donor resamples.
    """
    return donor_ratio_ci(num, den, donor, seed)


def main() -> None:
    t0 = time.time()
    emit(__file__)          # which code ran, HEAD or dirty tree
    pd_tbl = pd.read_parquet(OUT / "01a_03_no_access_per_donor.parquet")
    baseline = pd.read_parquet(OUT / "01a_03_no_access_clean_baseline.parquet")
    names = dict(zip(baseline.celltype, baseline.label))

    missing = [m for m in METRICS if m not in pd_tbl.columns]
    if missing:
        raise ValueError(f"per-donor table lacks {missing}; 01a_03 and 01a_04 "
                         "disagree about which rates exist")

    # __all__ is the sum over celltypes within a donor, derived rather than stored.
    pooled = pd_tbl.groupby(DONOR_ID, observed=True)[
        list(METRICS) + ["n_queries"]].sum().reset_index()
    pooled["celltype"] = "__all__"
    tbl = pd.concat([pd_tbl, pooled], ignore_index=True)

    print(f"[{time.time() - t0:.0f}s] {N_BOOTSTRAP:,} donor resamples x "
          f"{tbl.celltype.nunique()} celltypes x "
          f"{len(METRICS) + len(GAPS)} quantities", file=sys.stderr)

    rows = []
    for ct, sub in tbl.groupby("celltype", observed=True):
        den = sub["n_queries"].to_numpy(dtype=float)
        donor = sub[DONOR_ID].to_numpy().astype(str)
        row = {
            THREAT_MODEL: ThreatModel.NO_ACCESS,
            "celltype": ct,
            "label": names.get(ct, ct),
            "n_queries": int(den.sum()),
            "n_donors": int(len(np.unique(donor))),
        }
        for m in METRICS:
            v, lo, hi = rate_with_ci(sub[m].to_numpy(dtype=float), den, donor,
                                     BOOT_SEED)
            row[STEM[m]] = v
            row[f"{STEM[m]}_lo"] = lo
            row[f"{STEM[m]}_hi"] = hi
        # Paired: same denominator, same donors, same seed, so the resampled
        # picks are identical and the donor-level baseline cancels.
        correct = sub["correct"].to_numpy(dtype=float)
        for m in GAPS:
            v, lo, hi = rate_with_ci(sub[m].to_numpy(dtype=float) - correct, den,
                                     donor, BOOT_SEED)
            row[f"{m}_gap"] = v
            row[f"{m}_gap_lo"] = lo
            row[f"{m}_gap_hi"] = hi
        rows.append(row)

    out = pd.DataFrame(rows)

    # Provenance check: these rates are 01a_03's rates, reached by a different
    # route. A disagreement means the two steps read different inputs.
    chk = out.merge(baseline[["celltype"] + [STEM[m] for m in METRICS]],
                    on="celltype", suffixes=("", "_03"))
    if len(chk) != len(out):
        raise ValueError(f"{len(out)} celltypes here but {len(chk)} matched in "
                         "01a_03's baseline table")
    for m in METRICS:
        delta = np.abs(chk[STEM[m]].to_numpy() - chk[f"{STEM[m]}_03"].to_numpy())
        if delta.max() > 1e-9:
            worst = chk.celltype.iloc[int(delta.argmax())]
            raise ValueError(
                f"{STEM[m]} disagrees with 01a_03 by {delta.max():.3g} on "
                f"{worst!r}: the two steps are not describing the same run"
            )

    out.to_parquet(OUT / "01a_04_no_access_accuracy_ci.parquet", index=False)

    a = out[out.celltype == "__all__"].iloc[0]
    print(f"\n[{time.time() - t0:.0f}s] pooled over {int(a.n_queries):,} queries, "
          f"{int(a.n_donors)} donors:", file=sys.stderr)
    for m in METRICS:
        s = STEM[m]
        print(f"  {s:<15} {a[s]:.4f} [{a[f'{s}_lo']:.4f}, {a[f'{s}_hi']:.4f}]",
              file=sys.stderr)
    for m in GAPS:
        print(f"  {m + ' gap':<15} {a[f'{m}_gap']:+.4f} "
              f"[{a[f'{m}_gap_lo']:+.4f}, {a[f'{m}_gap_hi']:+.4f}]  paired",
              file=sys.stderr)
    print(f"[{time.time() - t0:.0f}s] wrote "
          f"{OUT / '01a_04_no_access_accuracy_ci.parquet'} ({len(out)} rows)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
