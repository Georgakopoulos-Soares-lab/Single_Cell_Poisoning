"""Step 02b_17 — flood's geometry, and the setting that was costing 39%.

``02b_02`` sweeps the anchor COUNT and holds the two geometry knobs fixed at
whatever v01 happened to use. This draws ``02b_13``, which titrated them: how far
each anchor rotates toward the pool centroid (``spread``) and how much random
scatter is added afterwards (``jitter``). Both are expressed against the MEASURED
neighbourhood radius -- 1 - cos = 0.0195 for the genuine top-50, about 11.3
degrees -- so 51% means the record is displaced half a neighbourhood.

**THE OPTIMUM IS INTERIOR, WHICH IS WHY IT WAS MISSED.** Neither extreme is best.
No spread at all (1.0) gives median net attack success 0.2775; 5% of the radius
(0.999) gives 0.4732. Identical vectors compete for the same top-50 slots, so a
point mass wastes most of its budget; a small scatter covers more query
neighbourhoods. The committed default of 0.99 -- 51% of the radius -- yields
0.2894, which is 61% of what 0.999 achieves. Every flood number in the project
was produced at roughly three-fifths strength.

**JITTER IS SELF-DEFEATING AT EVERY LEVEL.** It never helps. Median net attack
success falls from 0.2775 at no jitter to 0.1326 at 5% of the radius and to
0.0000 at 51% and beyond, and at 256% not one of 182 conditions succeeds.

**And the two panels dissociate, which is the point of drawing both.** At jitter
0.99 the poison still reaches 12% of queries and flips nothing: it is retrieved
and outvoted. Across the whole grid 89 of 1,092 conditions put poison in more
than 10% of top-50s -- up to 78% -- and change no label. A detector watching
retrieval fires on all 89; one watching outcomes fires on none.

One dose (1.0), one seed. Intervals are donor-paired over 78 donors but there is
no seed replication here, so this establishes the geometry, not its stability.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "1_scripts" / "00_common"))
import pandas as pd  # noqa: E402
import plotnine as p9  # noqa: E402
import yaml  # noqa: E402

from biorag_theme import biorag_theme, write_caption  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[3]
CFG = yaml.safe_load((BASE / "config" / "config.yaml").read_text())
DS = CFG["dataset"]
STAGE = "02_b_attack_celltype"
NAME = "02b_17_no_access"

#: Declared so check_provenance verifies the link rather than trusting the
#: docstring. A figure's number does not encode its input.
READS: tuple[str, ...] = ("02b_13",)
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE


def main() -> None:
    d = pd.read_parquet(TAB / "02b_13_no_access_flood_geometry.parquet")

    long = pd.concat([
        d.assign(metric="net attack success", value=d.within_net_attack_success),
        d.assign(metric="poison reached the top-50", value=d.within_retrieval),
    ])
    # MEDIAN over the 26 celltypes, not mean. The per-celltype range at the
    # optimum is [0.0000, 0.9039] -- a mean would be dragged by the celltypes
    # where the attack is total, and the claim here is about the GEOMETRY, which
    # the typical celltype shows more honestly than the average one.
    g = (long.groupby(["metric", "flood_spread", "flood_jitter"], observed=True)
             .agg(value=("value", "median"),
                  spread_pct=("spread_radius_frac", "first"),
                  jitter_pct=("jitter_radius_frac", "first"))
             .reset_index())

    # Axis labels carry BOTH the parameter and what it means. "0.999" is opaque;
    # "0.999 / 5%" says the record moved a twentieth of a neighbourhood.
    def lab(par: float, frac: float) -> str:
        return f"{par:g}\n{frac*100:.0f}%"

    sp = {r.flood_spread: lab(r.flood_spread, r.spread_pct)
          for r in g.itertuples()}
    ji = {r.flood_jitter: lab(r.flood_jitter, r.jitter_pct)
          for r in g.itertuples()}
    g["sp"] = pd.Categorical(g.flood_spread.map(sp),
                             categories=[sp[k] for k in sorted(sp, reverse=True)],
                             ordered=True)
    g["ji"] = pd.Categorical(g.flood_jitter.map(ji),
                             categories=[ji[k] for k in sorted(ji)],
                             ordered=True)
    g["metric"] = pd.Categorical(
        g.metric, ["net attack success", "poison reached the top-50"])

    # Text colour flips with the fill. magma is near-black at 0, so a single
    # ink colour makes either the zeros or the peaks unreadable -- and the zeros
    # are half the finding here, not filler.
    g["ink"] = ["black" if v > 0.45 else "white" for v in g.value]

    plot = (
        p9.ggplot(g, p9.aes("ji", "sp", fill="value"))
        + p9.geom_tile(color="white", size=0.4)
        + p9.geom_text(p9.aes(label="value", color="ink"),
                       format_string="{:.2f}", size=6)
        + p9.scale_color_identity()
        + p9.facet_wrap("metric", nrow=1)
        + p9.scale_fill_cmap(cmap_name="magma", limits=(0, 1))
        + p9.labs(x="jitter — random scatter added, as % of the neighbourhood radius",
                  y="spread — rotation toward the centroid, as % of the radius",
                  fill="median\nover 26\ncelltypes")
        + p9.ggtitle("Flood's optimum is interior: a little spread, no jitter")
        + biorag_theme()
        + p9.theme(figure_size=(9.4, 4.4), legend_position="right",
                   axis_text_x=p9.element_text(size=6),
                   axis_text_y=p9.element_text(size=6))
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)
    g.drop(columns=["sp", "ji", "ink"]).to_csv(TAB / f"{NAME}.csv", index=False)

    write_caption(
        FIG, NAME, "Flood's optimum is interior: a little spread, no jitter",
        f"{DS}; flood only, all 26 celltypes, 7 spreads x 6 jitters at "
        f"n_anchors=16,384 and dose 1.0, ONE seed; SCimilarity's shipped HNSW "
        f"index at ef=100, k=50, poison inserted and the graph re-queried; cell "
        f"values are the MEDIAN over 26 celltypes; percentages are displacement "
        f"against the measured neighbourhood radius, 1-cos = 0.0195. Flood is "
        f"CEILING tier -- it submits rotated vectors no count matrix produces.",
        """Spread has an interior optimum. No spread at all gives median net
        attack success 0.2775; 5% of the radius gives 0.4732; 51% gives 0.2894.
        Identical vectors compete for the same top-50 slots, so a point mass
        wastes its budget, while a small scatter reaches more neighbourhoods.

        The committed default is spread 0.99, which is 61% of what 0.999
        achieves. Every flood result in this project was produced at about
        three-fifths strength, and the gap is 1.64x.

        Jitter never helps. Median net attack success falls 0.2775 -> 0.1326 ->
        0.0000 as it grows through 0%, 5% and 51% of the radius, and at 256% not
        one of 182 conditions succeeds.

        The panels dissociate. At jitter 0.99 poison still reaches 12% of queries
        and flips nothing. Over the whole grid, 89 of 1,092 conditions put poison
        into more than 10% of top-50s -- up to 78% -- and change no label. A
        retrieval detector fires on all 89; an outcome detector on none.

        Unexplained: 256% jitter is completely inert (0 of 182) while 513% has 4
        conditions succeeding, max 0.1779. Non-monotone at the tail, and stated
        rather than smoothed over.

        One dose and one seed, so this fixes the geometry, not its seed-to-seed
        stability.""")
    print(f"{NAME}: {len(g)} plotted values -> pdf, png, csv, txt", file=sys.stderr)


if __name__ == "__main__":
    main()
