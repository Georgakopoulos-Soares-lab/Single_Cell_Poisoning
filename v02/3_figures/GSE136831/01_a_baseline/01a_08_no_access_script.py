"""Step 01a_08 — what a label-consistency filter costs before it catches anything.

Label consistency is the obvious defence: reject a record whose label disagrees
with the labels of its own neighbours. Scored against poison it looks free,
because poison is mislabelled by construction. 01a_05 scores the other side of
the ledger -- the 74,548 GENUINE reference records that the 49,456 queries
actually retrieve, each against SCimilarity's own shipped graph with its
self-match dropped. This draws that distribution, which is the false-positive
floor any LC threshold inherits before it has rejected a single poisoned record.

The ceiling is 0.98, not 1.00, because the self-match is excluded: a record can
agree with at most 49 of the 50 slots it is scored over. 45.5% of the records
sit exactly there, and the median is 0.96. So most genuine records are
uncontroversial -- and that is exactly why the tail is expensive. A threshold at
0.9, which sounds permissive against a distribution piled at 0.98, throws away
36.17% of the genuine records a query would have retrieved.

The second panel weights each record by how many query-neighbour slots it
actually fills, because rejecting a record that answers 3,229 queries and one
that answers a single query are not the same act. Pooled, weighting makes the
filter look slightly cheaper (31.29% of slots at the 0.9 threshold against
36.17% of records) -- the heavily-retrieved records are, on average, the
self-consistent ones. Per celltype that reverses: for macrophage (CL:0000235)
the weighted cost at 0.9 is 45.81% against 37.15% unweighted, and for classical
monocyte (CL:0000860) 40.34% against 17.86%. The records a filter would remove
from macrophage answers are the ones macrophage queries lean on hardest.

No confidence intervals anywhere in this figure: these are reference records,
the shipped annotation bundle carries no donor id for them, and every other
interval in this project resamples donors. 01a_05 states this and the figure
inherits it.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "1_scripts" / "00_common"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotnine as p9  # noqa: E402
import yaml  # noqa: E402

from biorag_theme import INK_SOFT, biorag_theme, write_caption  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[3]
CFG = yaml.safe_load((BASE / "config" / "config.yaml").read_text())
DS = CFG["dataset"]
STAGE = "01_a_baseline"
NAME = "01a_08_no_access"

#: Both tables are 01a_05's. The per-record scores give the distribution; the
#: summary gives the rejection rates, which are READ rather than recomputed so
#: the annotation cannot drift from the table it claims to report.
READS: tuple[str, ...] = ("01a_05",)
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE

POOLED = "__all__"

#: The thresholds 01a_05 tabulates. Nothing else can be annotated without
#: recomputing a rate the table does not hold.
THRESHOLDS = (0.5, 0.7, 0.9)

#: k = 50, so every score is a multiple of 1/50. Binning at anything else would
#: alias two adjacent achievable values into one bar and split a third.
K = 50

PANELS = ("one row per genuine record", "weighted by query-neighbour slots")

#: Bands between the tabulated thresholds, low to high.
BANDS = ("below 0.5", "0.5 to 0.7", "0.7 to 0.9", "0.9 and above")


def main() -> None:
    scores = pd.read_parquet(TAB / "01a_05_no_access_genuine_lc_scores.parquet")
    summ = pd.read_parquet(TAB / "01a_05_no_access_genuine_lc_summary.parquet")
    pooled = summ[summ.ref_celltype == POOLED].iloc[0]

    if len(scores) != pooled.n_records:
        raise ValueError(f"{len(scores)} scored rows against "
                         f"{pooled.n_records} in the summary")

    # Exact bin identity: the float32 score is k/50, so recover k and bin there.
    bin_k = np.rint(scores.label_consistency.to_numpy() * K).astype(int)
    lc = bin_k / K
    band = pd.cut(lc, [-0.01, *THRESHOLDS, 1.01], labels=list(BANDS),
                  right=False)

    per = pd.DataFrame({"lc": lc, "band": band,
                        "slots": scores.n_retrievals.to_numpy()})
    counts = (per.groupby(["lc", "band"], observed=True)
                 .agg(records=("slots", "size"), slots=("slots", "sum"))
                 .reset_index())

    n_rec, n_slot = counts.records.sum(), counts.slots.sum()
    if n_slot != pooled.n_retrieval_slots:
        raise ValueError(f"{n_slot} slots against {pooled.n_retrieval_slots}")

    bars = pd.concat([
        counts.assign(panel=PANELS[0], share=counts.records / n_rec,
                      denom=n_rec),
        counts.assign(panel=PANELS[1], share=counts.slots / n_slot,
                      denom=n_slot),
    ])
    bars["panel"] = pd.Categorical(bars.panel, PANELS, ordered=True)
    bars["band"] = pd.Categorical(bars.band, BANDS, ordered=True)

    # Rejection rates are READ from the summary, never recounted here: the
    # figure's job is to show the distribution the table describes, and a
    # second implementation of the same count is a second thing to disagree.
    ann = pd.DataFrame([
        {"panel": panel, "t": t,
         "rate": float(pooled[f"reject_at_{t:g}"] if panel == PANELS[0]
                       else pooled[f"reject_weighted_at_{t:g}"])}
        for panel in PANELS for t in THRESHOLDS
    ])
    ann["panel"] = pd.Categorical(ann.panel, PANELS, ordered=True)
    ann["label"] = [f"reject < {r.t:g}\n{r.rate * 100:.1f}% lost"
                    for r in ann.itertuples()]
    # Stacked down from the top so three annotations on one panel do not
    # collide; the tallest bar is the 0.98 spike at the right edge.
    top = bars.share.max()
    ann["y"] = [top * f for f in (0.95, 0.62, 0.35)] * len(PANELS)

    # Quoted in the caption. Pulled from the summary here rather than typed in,
    # so a rerun of 01a_05 moves the prose with the table.
    named = summ.set_index("ref_celltype")

    def cost(cl: str, t: float = 0.9) -> str:
        """Weighted and unweighted rejection cost of one reference celltype.

        Args:
            cl: Cell Ontology id of the reference celltype.
            t: rejection threshold; must be one 01a_05 tabulated.

        Returns:
            ``"<weighted>% of its slots against <unweighted>% of its records"``.
        """
        row = named.loc[cl]
        return (f"{row[f'reject_weighted_at_{t:g}'] * 100:.2f}% of its slots "
                f"against {row[f'reject_at_{t:g}'] * 100:.2f}% of its records")

    plot = (
        p9.ggplot(bars, p9.aes("lc", "share", fill="band"))
        + p9.geom_col(width=1 / K, size=0)
        + p9.geom_vline(p9.aes(xintercept="t"), data=ann, color=INK_SOFT,
                        linetype="dashed", size=0.35)
        + p9.geom_text(p9.aes("t", "y", label="label"), data=ann,
                       inherit_aes=False, ha="right", va="top", size=6,
                       nudge_x=-0.015, color=INK_SOFT, lineheight=0.9)
        + p9.facet_wrap("panel", nrow=1)
        + p9.scale_fill_cmap_d(cmap_name="viridis")
        + p9.scale_x_continuous(breaks=[0, 0.25, 0.5, 0.7, 0.9, 1.0],
                                limits=(-0.02, 1.02))
        # Square root, not linear: 45.5% of the records sit in one bin and on a
        # linear axis every other bar is a line 1 px tall. The tail is what the
        # figure is about, so it has to be visible; the axis says so.
        + p9.scale_y_sqrt(labels=lambda v: [f"{x:.0%}" for x in v],
                          breaks=[0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                          expand=(0, 0, 0.05, 0))
        + p9.labs(x="label consistency of a genuine reference record "
                    "(share of its own 49 neighbours carrying its label)",
                  y="share of the panel's total (square-root scale)",
                  fill="threshold band")
        + p9.ggtitle("An LC filter throws away a third of the genuine records "
                     "queries rely on")
        + biorag_theme()
        + p9.theme(figure_size=(10.0, 4.6), legend_position="right")
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)

    out = pd.concat([
        bars[["panel", "lc", "band", "share", "records", "slots", "denom"]],
        ann.assign(lc=ann.t, band="annotation (read from 01a_05 summary)",
                   share=ann.rate, records=pd.NA, slots=pd.NA,
                   denom=pd.NA)[["panel", "lc", "band", "share", "records",
                                 "slots", "denom"]],
    ])
    out.to_csv(TAB / f"{NAME}.csv", index=False)

    write_caption(
        FIG, NAME,
        "An LC filter throws away a third of the genuine records queries rely on",
        f"{DS}; no attacker. {n_rec:,} genuine reference records -- those the "
        f"49,456 held-out queries actually retrieve, not the whole reference -- "
        f"scored against SCimilarity's shipped labelled_kNN.bin at ef=100, k=50, "
        f"with each record's self-match excluded, so the ceiling is 0.98 rather "
        f"than 1.00. Bars are the share of the panel's total in each achievable "
        f"1/50 bin on a SQUARE-ROOT y axis, because 45.5% of records fall in "
        f"the single 0.98 bin and a linear axis flattens the tail this figure "
        f"is about; left panel counts records, right panel weights each record "
        f"by the {n_slot:,} query-neighbour slots it fills. Dashed lines and the "
        f"percentages beside them are the rejection rates tabulated by 01a_05, "
        f"read from its summary rather than recounted here. No intervals: these "
        f"are reference records and the shipped bundle carries no donor id for "
        f"them.",
        f"""The distribution is piled at the ceiling and has a real tail. The
        median is {pooled.lc_median:.2f}, {(lc >= 0.98).mean() * 100:.1f}% of
        records sit exactly at 0.98, and {(lc < 0.5).mean() * 100:.1f}% disagree
        with more than half of their own neighbourhood.

        That tail is the price list. Rejecting below 0.5 removes
        {pooled['reject_at_0.5'] * 100:.2f}% of the genuine records a query
        would retrieve; below 0.7, {pooled['reject_at_0.7'] * 100:.2f}%; below
        0.9, {pooled['reject_at_0.9'] * 100:.2f}%. These are false positives by
        construction -- no attacker exists in this step.

        Weighted by retrieval, the pooled cost is slightly lower at every
        threshold ({pooled['reject_weighted_at_0.5'] * 100:.2f}%,
        {pooled['reject_weighted_at_0.7'] * 100:.2f}%,
        {pooled['reject_weighted_at_0.9'] * 100:.2f}%), so the records queries
        lean on hardest are on average the self-consistent ones.

        The pooled direction reverses inside individual celltypes, and it
        reverses for large ones. At the 0.9 threshold macrophage (CL:0000235)
        loses {cost("CL:0000235")}, and classical monocyte (CL:0000860)
        {cost("CL:0000860")}. Alveolar macrophage (CL:0000583), the
        most-retrieved reference celltype at
        {named.loc["CL:0000583"].n_retrieval_slots:,} slots, goes the pooled way
        instead and loses {cost("CL:0000583")}. A threshold tuned on the pooled
        number is not tuned to what it would cost the celltypes a reader cares
        about.

        Two reference celltypes are removed almost entirely at 0.9: CL:0002397
        loses {cost("CL:0002397")} and CL:0000738 {cost("CL:0000738")}. Neither
        is attacked anywhere in this project; they are genuine records whose
        annotation disagrees with the neighbourhood they sit in, which is the
        whole reason this false-positive floor has to be measured before an LC
        threshold is called a defence.""")
    print(f"{NAME}: {len(out)} plotted values -> pdf, png, csv, txt",
          file=sys.stderr)


if __name__ == "__main__":
    main()
