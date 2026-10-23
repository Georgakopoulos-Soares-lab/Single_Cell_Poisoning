"""Step 01a_07 — the denominator every attack is measured against is not one number.

Net attack success subtracts a clean baseline, so the baseline is inside every
effect size this project reports. Pooled over 49,456 held-out queries the
deployed system answers the author's exact term for 46.37% of them
[42.82, 50.14], donor-paired. Quoting that alone would be the wrong summary:
across the 33 true celltypes the same system runs from 0.000 to 0.9665, median
0.3132, and 6 celltypes sit at exactly zero.

Zero does not always mean the same thing. Two of the six -- cytotoxic T cell
(2,705 queries) and fibroblast of lymphatic vessel (317) -- have a true term the
labelled reference never says, so no retrieval and no defence could have scored
above zero; their ``reachable`` is 0.0 and the figure says so on the row. The
other four are reachable and still missed.

For most of the rest the miss is a vocabulary disagreement rather than a
retrieval failure, which is why the lineage-tolerant rate is drawn beside the
exact one. Counting a prediction correct when it is an ancestor or descendant of
the true term lifts the pooled rate from 0.4637 to 0.8849, a paired gap of
0.4212 [0.3826, 0.4568]. Per celltype the gap is extreme where the annotators
disagreed about granularity: lung endothelial cell scores 0.000 exact and 0.9269
lineage-tolerant, and lung multiciliated epithelial cell 0.0391 against 0.8635.

The reading for an attacker's number: a flipped label on macrophage (exact
0.2827) starts from a very different floor than one on mast cell (0.9665), and
an attack reported against the pooled 0.4637 is reported against a celltype
almost none of the queries belong to.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "1_scripts" / "00_common"))
import pandas as pd  # noqa: E402
import plotnine as p9  # noqa: E402
import yaml  # noqa: E402

from biorag_theme import INK_SOFT, biorag_theme, write_caption  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[3]
CFG = yaml.safe_load((BASE / "config" / "config.yaml").read_text())
DS = CFG["dataset"]
STAGE = "01_a_baseline"
NAME = "01a_07_no_access"

#: 01a_04 carries the intervals and is the table drawn. 01a_03 is opened only to
#: confirm that the point estimates 01a_04 recomputed still match the ones it
#: was derived from -- a figure that silently drew a drifted baseline would put
#: the wrong denominator under every attack figure in the project.
READS: tuple[str, ...] = ("01a_03", "01a_04")
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE

POOLED = "__all__"

#: Two rates, in the order they are read: what the system answered, and what it
#: answered once granularity disagreements are forgiven.
EXACT = "exact match"
LINEAGE = "ancestor or descendant counted correct"
SERIES = (EXACT, LINEAGE)

#: Sampled from viridis at 0.05 and 0.55 rather than taking its two endpoints:
#: the top of viridis is #fde725, which is unreadable as a 1.5 pt point on the
#: white panel this theme sets. Same colormap, legible ends.
SERIES_COLORS = {EXACT: "#472d7b", LINEAGE: "#2c9e8f"}

#: Vertical offset of each series within a celltype's row, in row units.
DODGE = 0.20


def main() -> None:
    d = pd.read_parquet(TAB / "01a_04_no_access_accuracy_ci.parquet")
    base = pd.read_parquet(TAB / "01a_03_no_access_clean_baseline.parquet")

    chk = d.merge(base[["celltype", "clean_accuracy", "lineage"]],
                  on="celltype", how="inner", suffixes=("", "_03"))
    for col in ("clean_accuracy", "lineage"):
        off = (chk[col] - chk[f"{col}_03"]).abs().max()
        if off > 1e-12:
            raise ValueError(f"01a_03 and 01a_04 disagree on {col} by {off}")

    pooled = d[d.celltype == POOLED].iloc[0]
    ct = d[d.celltype != POOLED].copy()
    ct["tick"] = ct.label + "  " + ct.celltype
    ct = ct.sort_values("clean_accuracy", ascending=True).reset_index(drop=True)
    # y is numeric, not categorical, so the two series can be offset within a
    # row. Six celltypes have ZERO granularity gap; drawn on one line the later
    # series covers the earlier one exactly and the figure looks as if a series
    # were missing where in fact the two agree.
    ct["row"] = ct.index.astype(float)

    long = pd.concat([
        ct.assign(series=EXACT, value=ct.clean_accuracy, ypos=ct.row - DODGE,
                  lo=ct.clean_accuracy_lo, hi=ct.clean_accuracy_hi),
        ct.assign(series=LINEAGE, value=ct.lineage, ypos=ct.row + DODGE,
                  lo=ct.lineage_lo, hi=ct.lineage_hi),
    ])
    long["series"] = pd.Categorical(long.series, SERIES, ordered=True)

    # The connector: exact to lineage-tolerant within one celltype, so the
    # granularity gap is a length rather than a subtraction the reader does.
    gap = ct.assign(x=ct.clean_accuracy, y=ct.row - DODGE,
                    xend=ct.lineage, yend=ct.row + DODGE)

    # An accuracy of 0 for a term the reference cannot say is a ceiling, not a
    # failure, and the two look identical on this axis unless the figure says so.
    unreach = ct[ct.reachable < 1.0].assign(
        note="true term absent from the labelled reference — 0 is a ceiling",
        x=0.02, ypos=lambda f: f.row)

    ref = pd.DataFrame({
        "series": pd.Categorical(list(SERIES), SERIES, ordered=True),
        "x": [pooled.clean_accuracy, pooled.lineage],
    })

    plot = (
        p9.ggplot()
        + p9.geom_vline(p9.aes(xintercept="x", color="series"), data=ref,
                        linetype="dashed", size=0.4, show_legend=False)
        + p9.geom_segment(p9.aes(x="x", xend="xend", y="y", yend="yend"),
                          data=gap, color=INK_SOFT, size=0.35, alpha=0.6)
        + p9.geom_segment(p9.aes(x="lo", xend="hi", y="ypos", yend="ypos",
                                 color="series"), data=long, size=0.9,
                          alpha=0.85)
        + p9.geom_point(p9.aes("value", "ypos", color="series"), data=long,
                        size=1.5)
        + p9.geom_text(p9.aes("x", "ypos", label="note"), data=unreach,
                       ha="left", size=5, color=INK_SOFT)
        + p9.scale_color_manual(values=SERIES_COLORS, limits=list(SERIES))
        + p9.scale_y_continuous(breaks=ct.row.tolist(), labels=ct.tick.tolist(),
                                limits=(-0.7, len(ct) - 0.3))
        + p9.scale_x_continuous(limits=(0, 1), breaks=[0, 0.25, 0.5, 0.75, 1.0])
        + p9.labs(x="share of the celltype's queries answered correctly",
                  y="", color="")
        + p9.ggtitle("Before any attacker: clean accuracy runs 0.00 to 0.97 "
                     "across celltypes")
        + biorag_theme()
        + p9.theme(figure_size=(9.6, 7.0), legend_position="top",
                   axis_text_y=p9.element_text(size=6.5))
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)

    cols = ["celltype", "label", "series", "n_queries", "n_donors",
            "value", "lo", "hi", "reachable"]
    line = ref.assign(celltype=POOLED, label="all queries (dashed line)",
                      value=ref.x, lo=ref.x, hi=ref.x,
                      n_queries=pooled.n_queries, n_donors=pooled.n_donors,
                      reachable=pooled.reachable)
    out = (pd.concat([long[cols], line[cols]])
             .sort_values(["series", "value"]))
    out.to_csv(TAB / f"{NAME}.csv", index=False)

    write_caption(
        FIG, NAME,
        "Before any attacker: clean accuracy runs 0.00 to 0.97 across celltypes",
        f"{DS}; 49,456 held-out queries, 78 donors, no attacker anywhere in this "
        f"figure. k = 50 vote over SCimilarity's shipped labelled_kNN.bin at "
        f"ef=100 -- the deployed path, not the exact retrieval the sweeps merge "
        f"poison into. One row per true celltype, 33 celltypes, ordered by exact "
        f"accuracy. Bars are donor-paired bootstrap intervals, 2,000 resamples "
        f"over the 78 donors; the grey connector is the granularity gap. Dashed "
        f"lines are the pooled rates over all queries. Ties broken by "
        f"config.vote.tie_break = lowest_cl_id.",
        """Pooled exact-match accuracy is 0.4637 [0.4282, 0.5014]. No celltype is
        near it by accident: the per-celltype values run from 0.000 to 0.9665
        with median 0.3132, and only 4 of 33 reach 0.90.

        Six celltypes score exactly zero. Two of them could not have scored
        anything else -- cytotoxic T cell (2,705 queries) and fibroblast of
        lymphatic vessel (317) have a true term the labelled reference never
        says, reachable = 0.0. The other four are reachable and still missed,
        and every one of them is answered by a related term instead: lung
        endothelial cell (342 queries) scores 0.9269 lineage-tolerant,
        Langerhans cell (56) 0.7321, and the two six-query celltypes, ionocyte
        and pulmonary neuroendocrine cell, 0.8333 and 0.6667.

        Most of the apparent error is vocabulary. Counting an ancestor or
        descendant correct lifts the pooled rate to 0.8849 [0.8749, 0.8944]; the
        paired gap is 0.4212 [0.3826, 0.4568]. Per celltype the largest gaps are
        lung endothelial cell 0.9269 [0.8719, 0.9657], ionocyte 0.8333 [0.7500,
        1.0000] on 6 queries, lung multiciliated epithelial cell 0.8244 [0.7862,
        0.8538] and regulatory T cell 0.7542 [0.6862, 0.8178]. Six celltypes
        have no gap at all.

        Why this matters for every attack number in the project: net attack
        success is an attack rate minus this baseline, so the floor an attack
        starts from is a property of the celltype attacked. Macrophage starts at
        0.2827 [0.2285, 0.3450] and mast cell at 0.9665 [0.9383, 0.9885]. The
        same absolute flip means different things on those two rows, and the
        pooled 0.4637 describes neither.

        The intervals are donor-paired, so they are wide where the celltype is
        carried by few donors: smooth muscle cell spans 0.2366 on 25 donors,
        plasmacytoid dendritic cell 0.2343 on 35. A cell bootstrap would have
        reported these as precise.""")
    print(f"{NAME}: {len(out)} plotted values -> pdf, png, csv, txt",
          file=sys.stderr)


if __name__ == "__main__":
    main()
