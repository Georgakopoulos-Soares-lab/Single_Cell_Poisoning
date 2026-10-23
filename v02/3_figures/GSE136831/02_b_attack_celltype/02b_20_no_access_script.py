"""Step 02b_20 — flood does not need real cells to stand on.

``02b_02`` varies how MANY places the flood budget is spread across; ``02b_13``
varies how far each cluster sits from its anchor. This draws ``02b_14``, the last
unswept axis: WHERE the clusters go. ``flood_sites=anchor`` builds each cluster
on a real reference cell of the source celltype; ``kmeans_centres`` builds it on
a synthetic k-means centroid of that celltype's pool. Same budget, same spread,
same jitter, same dose -- only the placement differs, so the two arms are a
matched pair per celltype and the figure is drawn as a paired scatter.

**SYNTHETIC PLACEMENT IS NOT A CONCESSION -- IT IS USUALLY BETTER.** Over the
104 paired conditions (26 celltypes x 4 anchor counts), k-means centres give
higher within-celltype net attack success in 71, tie in 10, and lose in 23;
the median paired difference is +0.0178 and the mean +0.0279. Where the
committed donor-paired intervals are disjoint -- 20 of 104 -- the synthetic arm
is the higher one 17 times and the real-anchor arm 3 times. Seven celltypes
favour synthetic at all four anchor counts; none favours real anchors at all
four. The attacker does not need to obtain real cells of the celltype it is
attacking, which removes the one step in the flood recipe that required data
access rather than arithmetic.

**THE ADVANTAGE IS A LOW-BUDGET-SITE EFFECT.** The median paired gain falls
+0.0261 (64 sites), +0.0304 (256), +0.0211 (1,024), +0.0009 (4,096). With few
clusters the placement is the whole attack and centroids sit in denser parts of
the celltype than a randomly drawn cell does; by 4,096 clusters the budget
covers the celltype either way and the choice stops mattering.

**AND THE TWO ROWS DISSOCIATE AGAIN.** Retrieval climbs monotonically with the
number of sites -- median 0.139 -> 0.153 -> 0.287 -> 0.472 for real anchors --
while net attack success does not: 0.120 -> 0.105 -> 0.112 -> 0.069. Spreading
the same budget over more sites reaches far more queries and flips fewer labels,
because each site is left too thin to win a 50-neighbour vote.

One dose (0.25), one seed, so this fixes the placement effect, not its
seed-to-seed stability.
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
STAGE = "02_b_attack_celltype"
NAME = "02b_20_no_access"

#: Declared so check_provenance verifies the link rather than trusting the
#: docstring. A figure's number does not encode its input.
READS: tuple[str, ...] = ("02b_14",)
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE

#: Plotted metric -> the column family in 02b_14 it comes from. Both carry
#: committed donor-paired ``_lo``/``_hi`` bounds, which is what makes the
#: verdict below a read rather than a recomputation.
METRICS: dict[str, str] = {
    "net attack success": "within_net_attack_success",
    "poison reached the top-50": "within_retrieval",
}
SITES: tuple[str, str] = ("anchor", "kmeans_centres")


def main() -> None:
    d = pd.read_parquet(TAB / "02b_14_no_access_flood_sites.parquet")

    # One row per (celltype, n_anchors, metric), the two placements side by side.
    # The design is matched -- same celltype, same budget, same geometry -- so a
    # paired scatter against y=x is the honest reading and a marginal mean is not.
    rows = []
    for label, col in METRICS.items():
        w = d.pivot_table(index=["source", "n_anchors"], columns="flood_sites",
                          values=[col, f"{col}_lo", f"{col}_hi"])
        w.columns = [f"{a}|{b}" for a, b in w.columns]
        w = w.reset_index()
        rows.append(pd.DataFrame({
            "metric": label,
            "source": w.source,
            "n_anchors": w.n_anchors,
            "anchor": w[f"{col}|anchor"],
            "anchor_lo": w[f"{col}_lo|anchor"],
            "anchor_hi": w[f"{col}_hi|anchor"],
            "kmeans_centres": w[f"{col}|kmeans_centres"],
            "kmeans_centres_lo": w[f"{col}_lo|kmeans_centres"],
            "kmeans_centres_hi": w[f"{col}_hi|kmeans_centres"],
        }))
    g = pd.concat(rows, ignore_index=True)
    g["paired_diff"] = g.kmeans_centres - g.anchor

    # The verdict is read off the intervals already in the table; nothing new is
    # estimated here. "disjoint" is the strict form -- one arm's lo above the
    # other arm's hi -- so it undercounts rather than overclaims.
    km_wins = g.kmeans_centres_lo > g.anchor_hi
    an_wins = g.anchor_lo > g.kmeans_centres_hi
    g["verdict"] = "intervals overlap"
    g.loc[km_wins, "verdict"] = "synthetic higher, intervals disjoint"
    g.loc[an_wins, "verdict"] = "real anchors higher, intervals disjoint"

    order = ["real anchors higher, intervals disjoint", "intervals overlap",
             "synthetic higher, intervals disjoint"]
    g["verdict"] = pd.Categorical(g.verdict, order, ordered=True)
    g["metric"] = pd.Categorical(g.metric, list(METRICS), ordered=True)
    g["sites"] = pd.Categorical(
        [f"{n:,} sites" for n in g.n_anchors],
        categories=[f"{n:,} sites" for n in sorted(g.n_anchors.unique())],
        ordered=True)

    lim = (0.0, 0.9)  # both metrics live in [0, 0.868]; a square scale lets the
    #                   eye read distance from y=x instead of decoding two axes.
    plot = (
        p9.ggplot(g, p9.aes("anchor", "kmeans_centres"))
        + p9.geom_abline(slope=1, intercept=0, linetype="dashed",
                         color=BASELINE, size=0.4)
        # Dark halo first: viridis slot 3 is near-yellow and would vanish on a
        # white panel on its own. Shape is mapped as well, so the verdict never
        # depends on hue alone.
        + p9.geom_point(p9.aes(shape="verdict"), size=2.9, color="#3f3f3f")
        + p9.geom_point(p9.aes(shape="verdict", color="verdict"), size=1.7)
        + p9.scale_color_cmap_d(cmap_name="viridis")
        + p9.scale_shape_manual(values=["v", "o", "^"])
        + p9.facet_grid("metric ~ sites")
        # Identical limits and breaks on both axes, and a figure size chosen so
        # the eight panels come out near-square, rather than coord_fixed or
        # theme(aspect_ratio=): both crash plotnine 0.13.6's tight-layout engine
        # on this facet_grid (deepcopy of a generator in _reduce_width).
        + p9.scale_x_continuous(limits=lim, breaks=[0, 0.3, 0.6, 0.9])
        + p9.scale_y_continuous(limits=lim, breaks=[0, 0.3, 0.6, 0.9])
        + p9.labs(x="placement on REAL source cells (flood_sites = anchor)",
                  y="placement on SYNTHETIC k-means centres",
                  color="donor-paired\nintervals", shape="donor-paired\nintervals")
        + p9.ggtitle("Flood does not need real cells: synthetic placement wins more often")
        + biorag_theme()
        + p9.theme(figure_size=(11.0, 6.0),
                   legend_position="bottom", legend_direction="horizontal",
                   legend_title=p9.element_text(size=7),
                   axis_text=p9.element_text(size=6))
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)
    g.drop(columns=["sites"]).to_csv(TAB / f"{NAME}.csv", index=False)

    write_caption(
        FIG, NAME, "Flood does not need real cells: synthetic placement wins more often",
        f"{DS}; flood only, all 26 celltypes, placement on real source cells vs "
        f"on synthetic k-means centres of the same celltype's pool, crossed with "
        f"n_anchors in 64 / 256 / 1,024 / 4,096, at dose 0.25 of the celltype and "
        f"ONE seed; spread 0.99 and jitter 1.0 throughout; SCimilarity's shipped "
        f"HNSW index over 7,913,892 reference cells at ef=100, k=50, poison "
        f"inserted and the graph re-queried; one point per celltype per panel, "
        f"both axes on the same 0-0.9 scale with the panels drawn near-square, "
        f"dashed line is y=x. Flood is "
        f"CEILING tier -- it submits rotated vectors no count matrix produces.",
        """Synthetic placement is usually the stronger attack. Across the 104
        paired conditions -- 26 celltypes x 4 anchor counts -- k-means centres
        give higher within-celltype net attack success in 71, tie in 10 and lose
        in 23. The median paired difference is +0.0178 and the mean +0.0279,
        spanning -0.1668 to +0.4038.

        Where the committed donor-paired intervals are disjoint the direction is
        lopsided: 20 of 104 conditions separate on net attack success, and the
        synthetic arm is the higher one in 17 of them against 3 for real anchors.
        On retrieval 20 separate as well, 15 favouring synthetic. Seven celltypes
        favour synthetic at all four anchor counts; none favours real anchors at
        all four.

        The gain is a few-sites effect and decays: median paired difference
        +0.0261 at 64 sites, +0.0304 at 256, +0.0211 at 1,024, +0.0009 at 4,096.
        When the budget is concentrated the placement is most of the attack, and
        a centroid sits in a denser part of the celltype than a randomly drawn
        member does. By 4,096 clusters the budget blankets the celltype either
        way.

        The two rows dissociate, which is why both are drawn. Retrieval rises
        monotonically with the number of sites -- median over celltypes 0.139,
        0.153, 0.287, 0.472 for real anchors and 0.144, 0.174, 0.327, 0.478 for
        synthetic -- while net attack success does not: 0.120, 0.105, 0.112,
        0.069 and 0.134, 0.138, 0.151, 0.117. More sites reach far more queries
        and flip fewer labels, because each site is too thin to carry a
        50-neighbour vote. A retrieval detector would call 4,096 sites the worst
        case; an outcome detector would call it the mildest.

        Negative result, stated as such: CL:1001567 scores exactly zero net
        attack success in both arms at all four anchor counts. Its clean
        within-celltype accuracy is 0.0000 over 342 queries, so the pipeline
        already misreads every one of them and there is no label left for the
        attack to flip. Poison does reach its queries -- retrieval 0.0117 for
        synthetic at 4,096 sites against 0.0029 for real anchors -- and changes
        nothing. Its point in the top row sits on the origin because the attack
        is inert there, not because the two placements tie.

        One dose and one seed, so this establishes the placement effect, not its
        seed-to-seed stability.""")
    print(f"{NAME}: {len(g)} plotted values -> pdf, png, csv, txt", file=sys.stderr)


if __name__ == "__main__":
    main()
