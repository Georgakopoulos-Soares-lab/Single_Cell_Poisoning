"""Step 01a_06 — exact and HNSW retrieval agree on average, not on every celltype.

Every sweep in this project merges poison into an EXACT clean top-50, so the
whole benchmark describes the deployed system only to the extent that exact
retrieval and SCimilarity's shipped graph return the same neighbours. 01a_02
measured that on clean data. This draws it per celltype, because the pooled row
is reassuring and the per-celltype rows are not uniformly so.

Pooled over 49,456 queries the shipped graph returns 97.87% of the exact top-50
and the k = 50 vote lands on the same celltype for 99.29% of queries. The mean
over the 33 celltypes is 0.9616, and the spread under it is the finding: two
celltypes fall below 0.90, and for 9 of 33 at least one query shares NO
neighbour with exact search at all (``min_overlap`` = 0).

Only one of those matters for a label. ``CL:1000223`` (pulmonary neuroendocrine
cell) has overlap 0.3267, but it has 6 queries from 4 donors, its interval runs
to zero, and both retrievals score 0.000 accuracy — nothing is being decided.
``CL:0002062`` (pulmonary alveolar type 1 cell) has 163 queries from 40 donors,
overlap 0.8853, vote agreement 0.8896, and an accuracy gap of 0.1104: exact
search answers 0.9571 where the shipped graph answers 0.8466. The benchmark's
exact-retrieval baseline is optimistic about that celltype by 11 accuracy
points, and pooling hides it because 163 of 49,456 queries cannot move a mean.

The third panel is signed and has no interval — 01a_02 writes ``accuracy_delta``
as a difference of two point estimates, so it is drawn as a displacement from
zero rather than as an estimate with error. Three celltypes come out negative,
none by more than 0.0023.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "1_scripts" / "00_common"))
import pandas as pd  # noqa: E402
import plotnine as p9  # noqa: E402
import yaml  # noqa: E402

from biorag_theme import BASELINE, biorag_theme, write_caption  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[3]
CFG = yaml.safe_load((BASE / "config" / "config.yaml").read_text())
DS = CFG["dataset"]
STAGE = "01_a_baseline"
NAME = "01a_06_no_access"

#: 01a_02 carries the agreement itself; 01a_03 carries the human-readable term
#: for each CL id, which 01a_02 does not write. Declared so check_provenance
#: verifies the link rather than trusting the docstring.
#:
#: 01a_01 is here because its three scalars are the SCALE this figure is read
#: against, and until 2026-09-28 no figure or caption in the project used them
#: -- a stage producing a number no slide uses, which CLAUDE.md 3 forbids. They
#: say how close the graph's clean neighbours actually are; without that, a
#: reader cannot tell whether 97.87% overlap means the graph returns slightly
#: different NEAR neighbours or occasionally returns far ones.
READS: tuple[str, ...] = ("01a_01", "01a_02", "01a_03")
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE

#: 01a_02 names its pooled row ``__pooled__``; 01a_03 names its own ``__all__``.
POOLED = "__pooled__"

#: Panel order, left to right: the approximation itself, then what it does to a
#: vote, then what it does to an answer.
PANELS = ("top-50 set overlap", "k-NN vote agreement",
          "accuracy: exact − HNSW")


def main() -> None:
    d = pd.read_parquet(TAB / "01a_02_no_access_exact_vs_hnsw.parquet")
    lab = pd.read_parquet(TAB / "01a_03_no_access_clean_baseline.parquet")
    # One row, three scalars: what the shipped graph returns on CLEAN data.
    hnsw = pd.read_parquet(TAB / "01a_01_no_access_hnsw_summary.parquet").iloc[0]

    pooled = d[d.celltype == POOLED].iloc[0]
    ct = d[d.celltype != POOLED].merge(
        lab[["celltype", "label"]], on="celltype", how="left", validate="1:1")
    if ct.label.isna().any():
        raise ValueError(f"no label for {list(ct.celltype[ct.label.isna()])}")

    # The CL id stays in the tick: the label is what a reader knows, the id is
    # what a grep across the project finds.
    ct["tick"] = ct.label + "  " + ct.celltype
    # Worst agreement at the top, so the two celltypes the pooled row hides are
    # the first thing read. plotnine puts the FIRST category at the bottom.
    order = ct.sort_values("overlap", ascending=False).tick.tolist()
    ct["tick"] = pd.Categorical(ct.tick, categories=order, ordered=True)

    # Two layers, not one: panels 1-2 are estimates with donor-paired intervals,
    # panel 3 is a difference of two point estimates that 01a_02 gives no
    # interval for. Drawing it as a pointrange would invent one.
    est = pd.concat([
        ct.assign(panel=PANELS[0], value=ct.overlap,
                  lo=ct.overlap_lo, hi=ct.overlap_hi),
        ct.assign(panel=PANELS[1], value=ct.vote_agreement,
                  lo=ct.agree_lo, hi=ct.agree_hi),
    ])
    delta = ct.assign(panel=PANELS[2], value=ct.accuracy_delta,
                      lo=ct.accuracy_delta.clip(upper=0.0),
                      hi=ct.accuracy_delta.clip(lower=0.0))
    for frame in (est, delta):
        frame["panel"] = pd.Categorical(frame.panel, PANELS, ordered=True)

    # Pooled reference per panel: where the whole query set sits, so a celltype
    # can be read against it instead of against nothing. Zero for the delta.
    ref = pd.DataFrame({
        "panel": pd.Categorical(list(PANELS), PANELS, ordered=True),
        "x": [pooled.overlap, pooled.vote_agreement, 0.0],
    })

    plot = (
        p9.ggplot(mapping=p9.aes("value", "tick"))
        + p9.geom_vline(p9.aes(xintercept="x"), data=ref,
                        color=BASELINE, linetype="dashed", size=0.4)
        + p9.geom_segment(p9.aes(x="lo", xend="hi", y="tick", yend="tick",
                                 color="n_queries"), data=est, size=0.7)
        + p9.geom_point(p9.aes(color="n_queries"), data=est, size=1.5)
        + p9.geom_point(p9.aes(color="n_queries"), data=delta, size=1.5)
        + p9.facet_wrap("panel", nrow=1, scales="free_x")
        + p9.scale_color_cmap(cmap_name="viridis", trans="log10",
                              breaks=[10, 100, 1000, 10000])
        + p9.labs(x="", y="", color="queries\n(log)")
        + p9.ggtitle("Exact and HNSW retrieval agree on average, not per celltype")
        + biorag_theme()
        + p9.theme(figure_size=(10.4, 7.0), legend_position="right",
                   axis_text_y=p9.element_text(size=6),
                   axis_text_x=p9.element_text(size=6),
                   panel_spacing_x=0.06)
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)

    # The dashed pooled line is a plotted value too, so it goes in the CSV
    # alongside the points rather than living only in the caption.
    cols = ["celltype", "label", "panel", "n_queries", "n_donors",
            "value", "lo", "hi", "min_overlap"]
    line = ref.assign(celltype=POOLED, label="all queries (dashed line)",
                      value=ref.x, lo=ref.x, hi=ref.x,
                      n_queries=pooled.n_queries, n_donors=pooled.n_donors,
                      min_overlap=pooled.min_overlap)
    out = (pd.concat([est[cols], delta[cols], line[cols]])
             .sort_values(["panel", "value"]))
    out.to_csv(TAB / f"{NAME}.csv", index=False)

    write_caption(
        FIG, NAME, "Exact and HNSW retrieval agree on average, not per celltype",
        f"{DS}; 49,456 held-out queries, 78 donors, no attacker. Exact "
        f"brute-force top-50 against SCimilarity's shipped labelled_kNN.bin at "
        f"ef=100, k=50. SCALE, from 01a_01: on clean data that graph returns a "
        f"nearest neighbour at cosine {hnsw.mean_cos_top1:.4f} and a 50th at "
        f"{hnsw.mean_cos_top50:.4f}, so every neighbour it returns is close and "
        f"the disagreement below is about WHICH near neighbours, not about far "
        f"ones. One row per true celltype, 33 celltypes. Points are "
        f"per-celltype estimates, bars are donor-paired bootstrap intervals "
        f"(2,000 resamples over donors, cells are not the unit of "
        f"independence). The dashed line is the pooled value over all queries. "
        f"The third panel is a difference of two point estimates and 01a_02 "
        f"writes no interval for it, so it is drawn as a displacement from zero "
        f"with no error bar. Colour is the query count, log scale.",
        """Pooled, the shipped graph returns 97.87% [97.70, 98.03] of the exact
        top-50 and the vote agrees for 99.29% [99.21, 99.37] of queries, and the
        two paths differ in accuracy by 0.0011 (0.4637 HNSW, 0.4648 exact). At
        that level the merge shortcut describes the deployed system.

        Per celltype it is not uniform. The mean overlap over the 33 celltypes is
        0.9616 and the minimum is 0.3267; two celltypes fall below 0.90; for 9 of
        33, at least one query shares no neighbour at all with exact search.

        Only one of the two low celltypes carries a consequence. CL:1000223
        (pulmonary neuroendocrine cell) has overlap 0.3267 and vote agreement
        0.3333, but 6 queries from 4 donors, an interval running to 0, and 0.000
        accuracy on both paths -- nothing is decided either way. CL:0002062
        (pulmonary alveolar type 1 cell) has 163 queries from 40 donors, overlap
        0.8853 [0.8129, 0.9479], vote agreement 0.8896, and accuracy 0.8466
        [0.7658, 0.9157] on the shipped graph against 0.9571 [0.9143, 0.9845]
        exact -- a gap of 0.1104. For that celltype the exact baseline this
        benchmark attacks is 11 accuracy points better than the system a user
        installs.

        The gap is signed and mostly positive: exact search is more accurate for
        10 of 33 celltypes, less for 3 (worst -0.0023, CL:0000158), and identical
        for 20. Vote agreement is exactly 1.000 for 11 celltypes.

        Direction of the bias: because exact retrieval is the more accurate path
        for the celltypes where the two differ, a poisoning result computed on
        the exact top-50 is measured against a slightly HEALTHIER system than the
        deployed one. Pooled that is 0.0011 and ignorable; for CL:0002062 it is
        not, and any claim about that celltype should be read as a claim about
        exact search.

        Ties are left on argmax order here rather than resolved, and the pooled
        tie rate is 0.0076 -- the bound on how much a different tie rule could
        move vote agreement.""")
    print(f"{NAME}: {len(out)} plotted values -> pdf, png, csv, txt",
          file=sys.stderr)


if __name__ == "__main__":
    main()
