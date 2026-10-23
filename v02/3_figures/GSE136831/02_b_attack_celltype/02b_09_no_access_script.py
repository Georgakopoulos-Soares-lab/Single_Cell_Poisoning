"""Step 02b_09 — how the reference changes as an attack is dosed up.

The reference atlas in grey, exactly as it ships: 200,000 records drawn uniformly
from all 7,913,892, across all 202 celltypes. Coloured points are the records the
attacker ADDS. Everything grey is untouched.

Attack down the rows, degree across the columns, one layout throughout. The only
thing that changes between panels is what the attacker did, so a cloud that moves
moved in the reference.

WHAT THE ROWS DO DIFFERENTLY
----------------------------
``relabel`` fills the attacked celltype evenly. Each poisoned record is a
different genuine cell with a false label, so the colour spreads across the whole
celltype and, at parity, covers it.

``duplicate`` stacks. It draws from 1% of the celltype and copies, so raising the
dose deepens a few points instead of covering more of the celltype -- the colour
saturates early and then stops changing, which is the geometry behind its measured
plateau at ~13% from rate 0.25 onward.

``interpolate`` leaves. It blends each source vector toward the target's centroid,
so the colour drifts off the celltype toward somewhere in between. It is measurably
further from real cells than the other three -- roughly threefold -- which is why
almost nothing retrieves it.

``flood`` concentrates on anchors. Its anchor count is held FIXED across the row,
so the dose column shows budget alone: more records piled onto the same number of
places.

WHAT THIS FIGURE IS NOT
-----------------------
It is not attack success. Nothing here is scored, and position is not outcome --
``relabel`` and ``flood`` occupy the same regions and score 34% and 26% at parity,
while ``duplicate`` retrieves a third as often as ``relabel`` and wins more. The
vote is in ``02b_03``, the reach curve in ``02b_04``, record diversity in
``02b_07_counts``.

It is also not the whole threat model. A ``with_access`` attacker rewrites LABELS
and moves no embedding, so on a map of embeddings that attack is invisible: every
point would stay exactly where it is and this figure would be blank. That is a
finding, not a gap, and it belongs in its own panel rather than silently inside
this one. v01's ``fig17`` is the view that does show it, by colouring QUERIES
whose answer changed rather than reference records.

Reads ``02b_08_no_access_umap_global_xy.parquet``; computes no statistic. The
record counts in the strip labels are read from that table's ``n_poison`` and
``atlas_share``, which describe the FULL poison set, not the thinned draw.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "1_scripts" / "00_common"))
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import plotnine as p9  # noqa: E402
import yaml  # noqa: E402

from biorag_theme import biorag_theme, write_caption  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[3]
CFG = yaml.safe_load((BASE / "config" / "config.yaml").read_text())
DS = CFG["dataset"]
STAGE = "02_b_attack_celltype"
NAME = "02b_09_no_access"

#: The analysis step this figure draws, declared so the link is CHECKED rather
#: than implied. A figure's number does not encode which step fed it -- 02b_03
#: reads 02b_01 and 02b_04 reads 02b_02, so the offset is not even constant --
#: and a docstring naming the input can drift from the code that opens it.
#: ``check_provenance`` verifies every step named here exists and produced a
#: table, and that every table this script actually opens belongs to a step
#: named here.
READS: tuple[str, ...] = ('02b_08',)
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE

#: Backdrop points per panel, from the 1,000,000-record layout. High enough that
#: the atlas reads as continents rather than wisps, which is what makes "the
#: poison landed HERE" legible at all. A seeded draw; changes no quantity.
ATLAS_DRAW = 60000
DRAW_SEED = 0

#: Scatter layers are rasterised inside the PDF, and this is not cosmetic. At
#: 16 panels x ~160,000 points a vector PDF stores every marker separately and
#: reaches a gigabyte, which is not a figure anyone can open or commit. Text,
#: axes and strips stay vector, so the PDF is still sharp where sharpness
#: carries meaning. plotnine 0.13 has no `raster` parameter, so this reaches the
#: matplotlib collections directly after the plot is built.
RASTER_DPI = 320

ORDER = ["relabel", "duplicate", "interpolate", "flood"]
TIER = {"relabel": "rung 0", "duplicate": "rung 0",
        "interpolate": "rung 1", "flood": "rung 1"}
ATLAS_C = "#d5d3cc"
POISON_C = {"relabel": "#c1440e", "duplicate": "#1f6f8b",
            "interpolate": "#8a6bbe", "flood": "#2e7d32"}


def draw_one(atlas: pd.DataFrame, poison: pd.DataFrame, source: str,
             back: pd.DataFrame) -> int:
    """One celltype's attack-by-dose grid, on the shared reference layout.

    Args:
        atlas: backdrop rows. poison: this source's poison rows.
        source: the attacked CL id. back: the thinned backdrop, drawn once and
            reused across celltypes so every grid shares the same grey.

    Returns:
        Points drawn.
    """
    cost = (poison.groupby(["attack", "rate"], observed=True)
                  .agg(n_poison=("n_poison", "first"),
                       atlas_share=("atlas_share", "first")).reset_index())
    dose_lab = {r: f"{r:g}x the celltype" for r in sorted(poison.rate.unique())}
    row_lab = {a: f"{a}  ({TIER[a]})" for a in ORDER}

    panels = []
    for a in ORDER:
        for r in sorted(poison.rate.unique()):
            panels.append(back.assign(attack=a, rate=r, layer="atlas"))
            panels.append(poison[(poison.attack == a) & (poison.rate == r)]
                          .assign(layer="poison"))
    panel = pd.concat(panels)
    panel["row"] = pd.Categorical(panel.attack.map(row_lab),
                                  [row_lab[a] for a in ORDER])
    panel["col"] = pd.Categorical(panel.rate.map(dose_lab),
                                  [dose_lab[r] for r in sorted(dose_lab)])
    panel["shade"] = [ATLAS_C if k == "atlas" else POISON_C[a]
                      for k, a in zip(panel.layer, panel.attack, strict=True)]

    stem = f"{NAME}_{source.replace(':', '')}"
    plot = (
        p9.ggplot(panel, p9.aes("x", "y", color="shade"))
        + p9.geom_point(size=0.16, alpha=0.45, stroke=0)
        + p9.facet_grid("row ~ col")
        + p9.scale_color_identity()
        + p9.labs(x="", y="")
        + p9.ggtitle(f"How the reference changes as the attack is dosed up "
                     f"— {source}, grey is untouched")
        + biorag_theme()
        + p9.theme(figure_size=(10.0, 10.4), legend_position="none",
                   axis_text=p9.element_blank(), axis_ticks=p9.element_blank(),
                   strip_text=p9.element_text(size=8))
    )
    FIG.mkdir(parents=True, exist_ok=True)
    fig = plot.draw()
    for ax in fig.get_axes():
        for coll in ax.collections:
            coll.set_rasterized(True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=RASTER_DPI, bbox_inches="tight")
    plt.close(fig)
    cost.to_csv(TAB / f"{stem}.csv", index=False)

    lines = "; ".join(
        f"{r.attack} at {r.rate:g}x = {int(r.n_poison):,} records "
        f"({r.atlas_share:.3%} of the atlas)"
        for _, r in cost[cost.attack == "relabel"].iterrows())
    write_caption(
        FIG, stem, "How the reference changes as the attack is dosed up",
        f"{DS}; attacked celltype {source}. One shared reference layout across "
        f"every celltype's grid. Grey is the reference as shipped -- "
        f"{len(atlas):,} records drawn PROPORTIONALLY across all 202 celltypes "
        f"with a floor of one each, thinned to {ATLAS_DRAW:,} per panel for "
        f"rendering. Colour is what the attacker adds, at four budgets, built "
        f"with the same condition seeds the sweeps used and projected into one "
        f"UMAP fitted once on the backdrop (cosine, n_neighbors=15, "
        f"min_dist=0.30, init=pca, random_state=0 -- v01 fig15a's settings, so "
        f"the two versions' maps are comparable). All doses are thinned at ONE "
        f"fraction, so drawn density scales with budget -- but SUBLINEARLY, "
        f"because points overprint. Measured across all 26 celltypes of this "
        f"family, the 5x budget step from 0.25x to 1.25x renders as only "
        f"1.7x to 3.1x more coloured ink (median 2.3x), and the poison covers "
        f"between 0.03% and 0.32% of a panel. Read these panels for WHERE the "
        f"poison sits, not for how much of it there is; the quantity is in the "
        f"budgets below and in 02b_07's counts panel. Flood's anchor count is "
        f"fixed across its row, so the dose column varies budget alone. Budgets, "
        f"identical for every attack: {lines}. Nothing here is scored. The backdrop is a "
        f"subsample of the 7,913,892-record atlas, and that is a CHOICE, not "
        f"a limit: probes on this machine measure 10.5 min / 15.2 GB at 2M "
        f"and ~25 min CPU / 24.4 GB at 4M, an exponent of 1.25, so the full "
        f"fit is about 59 min and 42 GB of 64. v01's extrapolation of ~13 h "
        f"and ~215 GB came from 200k/500k timings on other hardware and does "
        f"not hold here.",
        """Each row is one attack; each column is one budget. Grey is untouched
        reference; colour is what was added.

        relabel spreads evenly and covers the celltype by parity -- every record
        is a different genuine cell wearing a false label.

        duplicate stacks instead of spreading: it draws from 1% of the celltype
        and copies, so the colour saturates early and later columns add depth
        rather than area. That is the geometry behind its measured plateau near
        13% from rate 0.25 onward.

        interpolate leaves the celltype, drifting toward the target's centroid
        and landing about threefold further from any real cell than the other
        three. Almost nothing retrieves it: 0.4% net attack success at every dose.

        flood concentrates on a fixed number of anchors, so its column shows
        budget alone -- more records on the same number of places.

        Position is not outcome. relabel and flood occupy similar regions and
        score 34% and 26% at parity; duplicate retrieves a third as often as
        relabel and wins more often. And a with_access attacker would leave this
        figure entirely blank, because rewriting labels moves no embedding.""")
    return len(panel)


def main() -> None:
    d = pd.read_parquet(TAB / "02b_08_no_access_umap_global_xy.parquet")
    atlas = d[d.kind == "atlas"]
    poison = d[d.kind == "poison"]
    # Drawn ONCE and reused: every celltype's grid then shares the identical
    # grey, so a difference between two grids is the attack and never the draw.
    back = atlas.sample(min(len(atlas), ATLAS_DRAW), random_state=DRAW_SEED)
    for source in sorted(poison.source.dropna().unique()):
        n = draw_one(atlas, poison[poison.source == source], source, back)
        stem = f"{NAME}_{source.replace(':', '')}"
        size = (FIG / f"{stem}.png").stat().st_size / 1e6
        print(f"{stem}: {n:,} points, {size:.1f} MB png", file=sys.stderr)


if __name__ == "__main__":
    main()
