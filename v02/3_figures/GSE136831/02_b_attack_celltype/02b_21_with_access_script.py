"""Step 02b_21 — the merge shortcut is close, two-sided, and concentrated.

METHODS figure. ``modify.remerge`` computes what exhaustive retrieval WOULD
return after the poison is merged in -- the shortcut that makes this project
laptop-feasible. ``modify.graph_topk`` writes the same records into
SCimilarity's shipped HNSW index and re-queries it, which is what the deployed
system actually returns. ``02b_16`` joins the two condition by condition for
``02b_11`` (2,028 conditions) and ``02b_15`` (936), on the two metrics that are
computed wholly from the attacked side and are therefore comparable. This draws
the signed difference, graph minus shortcut, one point per condition.

**A MEAN WOULD SAY "NO DIFFERENCE" AND THAT WOULD BE THE WRONG CONCLUSION.** On
within-celltype accuracy the mean difference is -0.0015 for ``02b_11`` and
-0.0011 for ``02b_15`` -- about a tenth of a percentage point. But the two
retrievers agree to the last bit in only 43.98% and 41.24% of conditions, and
where they differ the direction is MIXED: the graph reads BELOW the shortcut in
33.38% / 33.97% and ABOVE it in 22.63% / 24.79%. The extremes reach -0.117 and
+0.345. The mean is small because the two tails nearly cancel, not because the
retrievers agree.

**THE EARLIER CLAIM THAT "THE GRAPH WAS NEVER WEAKER" IS FALSE AT SCALE.** It
came from four conditions. Pooled over both steps, the graph reads on the
opposite side of the shortcut in 691 of 2,964 accuracy conditions -- 23.3%.

**THE DISAGREEMENT IS CARRIED BY A HANDFUL OF CELLTYPES**, which is why the y
axis is the celltype and not the dose. Ordered by total absolute difference,
``CL:0002062`` alone accounts for 28.3% of it and the top three for 49.3%;
``CL:1001567`` contributes 0.07% and agrees exactly in 95.2% of its conditions.
Nothing in a clean-recall measurement predicts which celltype will be bad -- the
producing step reports Spearman -0.23 against ``01a_02``'s top-50 overlap -- so
the shortcut's error cannot be bounded per celltype in advance, only measured.

LEVELS ONLY. ``02b_16`` excludes every delta, baseline and net column, because
the graph runs also carry the baseline-regime fix and a difference in those
columns would mix two changes at once.
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
STAGE = "02_b_attack_celltype"
NAME = "02b_21_with_access"

#: Declared so check_provenance verifies the link rather than trusting the
#: docstring. A figure's number does not encode its input.
READS: tuple[str, ...] = ("02b_16",)
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE

#: Neutral wording on purpose. ``diff`` is graph minus shortcut, and its SIGN
#: means opposite things for the two metrics -- lower accuracy is a stronger
#: attack, lower retrieval is a weaker one. Naming the direction "stronger"
#: would be wrong in half the panels, so the legend names the quantity.
DIRECTION: dict[str, str] = {
    "below": "graph reads BELOW the shortcut",
    "same": "identical to the last bit",
    "above": "graph reads ABOVE the shortcut",
}
STEP_LABEL: dict[str, str] = {
    "02b_11_with_access_inplace": "02b_11 — dose sweep, 2,028 conditions",
    "02b_15_with_access_alpha_sweep": "02b_15 — alpha sweep, 936 conditions",
}
METRIC_LABEL: dict[str, str] = {
    "within_accuracy": "within-celltype accuracy",
    "within_retrieval": "poison reached the top-50",
}


def main() -> None:
    d = pd.read_parquet(TAB / "02b_16_with_access_exact_vs_graph.parquet")

    g = d[["step", "metric", "source", "target", "attack", "label_change",
           "selection", "rate", "seed", "exact", "graph", "diff"]].copy()
    g["abs_diff"] = g["diff"].abs()
    g["direction"] = [DIRECTION["same"] if v == 0
                      else DIRECTION["below"] if v < 0 else DIRECTION["above"]
                      for v in g["diff"]]
    g["direction"] = pd.Categorical(
        g.direction, [DIRECTION[k] for k in ("below", "same", "above")],
        ordered=True)

    # ONE celltype order across all four panels, from the POOLED total absolute
    # difference. A per-panel order would let the eye read the concentration as
    # four unrelated rankings; the claim is that the same few celltypes carry it
    # everywhere, and a shared axis is what makes that checkable.
    rank = g.groupby("source")["abs_diff"].sum().sort_values()
    g["source"] = pd.Categorical(g.source, list(rank.index), ordered=True)
    g["step"] = pd.Categorical(g.step.map(STEP_LABEL),
                               [STEP_LABEL[s] for s in STEP_LABEL], ordered=True)
    g["metric"] = pd.Categorical(g.metric.map(METRIC_LABEL),
                                 [METRIC_LABEL[m] for m in METRIC_LABEL],
                                 ordered=True)

    # The pile at zero is the headline and overplots into a single blob, so each
    # panel also states it as a count. Read from the same frame that is plotted.
    ann = (g.groupby(["step", "metric"], observed=True)
             .agg(n=("diff", "size"), n_same=("diff", lambda s: int((s == 0).sum())))
             .reset_index())
    ann["label"] = [f"identical in {r.n_same:,} of {r.n:,}  ({r.n_same / r.n:.1%})"
                    for r in ann.itertuples()]
    ann["source"] = rank.index[2]       # third row from the bottom: empty in
    ann["diff"] = -0.215                # every panel, so nothing is covered

    plot = (
        p9.ggplot(g, p9.aes("diff", "source"))
        + p9.geom_vline(xintercept=0, color=INK_SOFT, size=0.35)
        # ONE layer with a dark edge, not a colour layer over a halo layer: two
        # jittered layers draw independent offsets and the halo separates from
        # its marker. The edge is what keeps viridis slot 3, which is
        # near-yellow, visible on a white panel. Jitter is vertical only -- the
        # x position is the measurement and must not be moved.
        + p9.geom_point(p9.aes(fill="direction"), size=1.5, shape="o",
                        color="#3f3f3f", stroke=0.25, alpha=0.8,
                        position=p9.position_jitter(width=0, height=0.28,
                                                    random_state=0))
        + p9.geom_text(ann, p9.aes(x="diff", y="source", label="label"),
                       size=6.5, ha="left", color=INK_SOFT, nudge_y=0.5,
                       inherit_aes=False)
        + p9.scale_fill_cmap_d(cmap_name="viridis")
        + p9.scale_x_continuous(limits=(-0.22, 0.37),
                                breaks=[-0.2, -0.1, 0, 0.1, 0.2, 0.3])
        + p9.facet_grid("metric ~ step")
        + p9.labs(x="graph minus shortcut  (HNSW re-query − merge-and-rerank)",
                  y="attacked celltype, ordered by total absolute difference",
                  fill="")
        + p9.ggtitle("The merge shortcut disagrees both ways, and a few celltypes carry it")
        + biorag_theme()
        + p9.theme(figure_size=(11.5, 9.0), legend_position="bottom",
                   legend_direction="horizontal",
                   axis_text_y=p9.element_text(size=6),
                   axis_text_x=p9.element_text(size=7))
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)
    g.to_csv(TAB / f"{NAME}.csv", index=False)

    write_caption(
        FIG, NAME, "The merge shortcut disagrees both ways, and a few celltypes carry it",
        f"{DS}; every condition of 02b_11 (dose sweep, 2,028) and 02b_15 (alpha "
        f"sweep, 936) scored twice -- once by modify.remerge, which computes what "
        f"exhaustive retrieval would return after the poison is merged in, and "
        f"once by modify.graph_topk, which writes the records into SCimilarity's "
        f"shipped HNSW index at ef=100 and re-queries it; k=50, seed 0, "
        f"with_access (records overwritten in place); x is graph minus shortcut, "
        f"one point per condition, jittered vertically only; LEVELS ONLY -- every "
        f"delta, baseline and net column is excluded by 02b_16 because the graph "
        f"runs also carry the baseline-regime fix.",
        """The two retrievers agree exactly far less often than a mean
        difference suggests. On within-celltype accuracy they are identical in
        892 of 2,028 conditions (43.98%) for 02b_11 and 386 of 936 (41.24%) for
        02b_15. On retrieval, 1,227 of 2,028 (60.50%) and 434 of 936 (46.37%).

        Where they differ the direction is mixed. On accuracy the graph reads
        below the shortcut in 33.38% of 02b_11 conditions and above it in 22.63%;
        for 02b_15 it is 33.97% below and 24.79% above. The mean difference is
        -0.0015 and -0.0011, about a tenth of a percentage point, because the two
        tails nearly cancel -- not because the retrievers agree. Reporting the
        mean alone would licence exactly the wrong conclusion.

        The tail is fat. Accuracy differences run from -0.1166 to +0.3449 in
        02b_11 and -0.1902 to +0.2285 in 02b_15; 232 of 2,028 and 85 of 936
        conditions exceed 0.01 in absolute value. The single largest is
        CL:0002063 to CL:0000158, hub selection at dose 0.85, where the shortcut
        reads within-celltype accuracy 0.1860 and the graph reads 0.5308.

        The earlier claim that the graph was never weaker came from four
        conditions and does not survive. Pooled over both steps the graph lands
        on the opposite side of the shortcut in 691 of 2,964 accuracy conditions,
        23.3%.

        The disagreement is concentrated in celltypes, not doses. Ordered by
        total absolute difference, CL:0002062 alone carries 28.3% of it and the
        top three carry 49.3% -- both over BOTH metrics pooled, which this
        sentence did not say until 2026-09-28; over accuracy alone they are
        36.0% and 60.8%. On retrieval the top three carry 39.4%. At the
        other end CL:1001567 carries 0.07% and is identical in 95.2% of its
        conditions. The producing step reports Spearman -0.23 between a
        celltype's disagreement and HNSW's clean top-50 overlap for it, so which
        celltypes are bad is not predictable from clean recall and the shortcut's
        error can be measured but not bounded in advance.

        Retrieval behaves differently from accuracy and is drawn rather than
        summarised away: it agrees more often but leans one way, with the graph
        below the shortcut in 28.21% of 02b_11 conditions against 11.29% above,
        and 41.35% against 12.29% in 02b_15.

        One seed. This measures the shortcut's error on the conditions run, and
        does not establish how it moves with the seed.""")
    print(f"{NAME}: {len(g)} plotted values -> pdf, png, csv, txt", file=sys.stderr)


if __name__ == "__main__":
    main()
