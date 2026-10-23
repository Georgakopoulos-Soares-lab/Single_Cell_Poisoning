"""Step 02b_07 — the four insertion attacks, drawn where they land.

One layout per celltype, fitted on the attacked celltype and its target and
shared by every panel, so a cloud that moves between panels moved in the data
rather than in the projection. Grey is the atlas as it ships; colour is what the
attacker adds.

What the panels show, and it is the same story the rates tell in ``02b_03``:

``relabel`` and ``duplicate`` sit exactly on the genuine cloud, because their
vectors ARE genuine records with a false label attached. Nothing in the geometry
distinguishes them from the atlas, which is why no re-encoding defence can reject
them and why both are rung 0 of the capability ladder.

``interpolate`` is the outlier, and the panel explains its 0.4% net success
better than any number does. It sits in open space between the two celltypes,
because it moves each vector toward the TARGET centroid -- away from the source
queries it has to fool. It is a rung-1 attack, permitted to write any
coordinates it likes, and it spends that freedom aiming at the wrong place.

``flood`` sits on the source cloud like relabel, but it got there by
construction rather than by copying: real anchors rotated toward the celltype's
own centre. That is the rung-1 capability actually paying off.

**The bar panel is the half a scatter cannot draw**, and across all 26
celltypes it is the stronger result. All four attacks add the same number of
records, so the scatters look equally dense. They are nowhere near equally
diverse:

``duplicate`` holds at 1.0% distinct by construction -- it draws from a 1% pool
and copies. The floor is CL:0000160, where 15,100 added records are 150 vectors.

``flood`` has a MEDIAN of 2.6% distinct, and its floor is the finding: against
CL:0000860 the attacker submits 368,537 records that are **64 distinct
vectors**, each repeated about 5,760 times, because the anchor count that won
there was 64 and jitter is 1.0 so no per-copy noise is applied. The strongest
attack in this project, at its own per-celltype optimum, is 64 points. A
near-duplicate filter does not need to be good to remove it.

``relabel`` is 100% distinct in only 16 of 26 celltypes, and falls to 60.6% on
CL:0000186 -- 54,712 records carrying 33,153 distinct vectors. The attacker drew
54,712 DIFFERENT cells; the atlas simply ships byte-identical duplicates of
them (``HANDOFF.md`` 3.5). That cuts the other way from the rest of this panel:
a de-duplication defence tuned to collapse the orange bars would also be
discarding genuine records, and this is the measurement of how many.

Reads ``02b_06_no_access_umap_xy.parquet`` and its counts table; computes no
statistic. The distinct counts are read, not derived here.
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
NAME = "02b_07_no_access"

#: The analysis step this figure draws, declared so the link is CHECKED rather
#: than implied. A figure's number does not encode which step fed it -- 02b_03
#: reads 02b_01 and 02b_04 reads 02b_02, so the offset is not even constant --
#: and a docstring naming the input can drift from the code that opens it.
#: ``check_provenance`` verifies every step named here exists and produced a
#: table, and that every table this script actually opens belongs to a step
#: named here.
READS: tuple[str, ...] = ('02b_06',)
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE

#: Points drawn per cloud per panel. A scatter of 88,000 points renders as a
#: solid block, so position stops being readable well before this limit; the
#: cap is a drawing decision and changes no quantity. Every COUNT in the bar
#: panel comes from the counts table, which is not subsampled.
DRAW_MAX = 6000
DRAW_SEED = 0

ORDER = ["relabel", "duplicate", "interpolate", "flood"]
LABEL = {"relabel": "relabel  (rung 0)", "duplicate": "duplicate  (rung 0)",
         "interpolate": "interpolate  (rung 1)", "flood": "flood  (rung 1)"}
KIND = {"genuine_target": "the target celltype, as shipped",
        "genuine_source": "the attacked celltype, as shipped",
        "poison": "records the attacker adds"}
COLOR = {"the target celltype, as shipped": "#d9d7d0",
         "the attacked celltype, as shipped": "#8a8985",
         "records the attacker adds": "#c1440e"}


def thin(d: pd.DataFrame) -> pd.DataFrame:
    """At most DRAW_MAX rows per (panel, cloud), drawn reproducibly.

    Args:
        d: plotted points, one row each.

    Returns:
        The same frame thinned for rendering; no column is altered.
    """
    return (d.groupby(["attack", "kind"], observed=True, group_keys=False)
             .apply(lambda g: g.sample(min(len(g), DRAW_MAX),
                                       random_state=DRAW_SEED)))


def main() -> None:
    xy = pd.read_parquet(TAB / "02b_06_no_access_umap_xy.parquet")
    counts = pd.read_parquet(TAB / "02b_06_no_access_umap_counts.parquet")

    for source in xy.source.unique():
        d = xy[xy.source == source]
        target = d.target.iloc[0]
        genuine = d[d.attack == "none"]
        panels = []
        for attack in ORDER:
            p = d[d.attack == attack]
            panels.append(pd.concat([genuine.assign(attack=attack), p]))
        panel = pd.concat(panels)
        panel["cloud"] = pd.Categorical(panel.kind.map(KIND), list(COLOR))
        panel["panel"] = pd.Categorical(panel.attack.map(LABEL),
                                        [LABEL[a] for a in ORDER])
        drawn = thin(panel)

        plot = (
            p9.ggplot(drawn, p9.aes("x", "y", color="cloud"))
            + p9.geom_point(size=0.22, alpha=0.45, stroke=0)
            + p9.facet_wrap("panel", nrow=1)
            + p9.scale_color_manual(values=COLOR)
            + p9.labs(x="UMAP 1", y="UMAP 2", color="")
            + p9.ggtitle(f"{source} attacked toward {target} — where the poison lands")
            + biorag_theme()
            + p9.theme(figure_size=(11.0, 3.4), legend_position="bottom",
                       axis_text=p9.element_blank(),
                       axis_ticks=p9.element_blank())
            + p9.guides(color=p9.guide_legend(override_aes={"size": 3,
                                                            "alpha": 1}))
        )
        stem = f"{NAME}_{source.replace(':', '')}"
        FIG.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            plot.save(FIG / f"{stem}.{ext}", verbose=False, dpi=200)
        drawn.to_csv(TAB / f"{stem}.csv", index=False)
        print(f"{stem}: {len(drawn):,} points drawn of {len(panel):,}",
              file=sys.stderr)

    # A RATIO, not two raw bars. Across 26 celltypes the budgets span 10,438 to
    # 368,537 records, so absolute bars would compare celltype sizes rather than
    # attacks; the share of a budget that is distinct is the quantity that is
    # comparable, and it is the one a de-duplication filter acts on. Both
    # components stay in the CSV so the raw counts remain inspectable.
    long = counts[counts.attack != "none"].copy()
    long["panel"] = pd.Categorical(long.attack.map(LABEL),
                                   [LABEL[a] for a in ORDER])
    long["distinct_share"] = long.n_distinct / long.n_poison
    order = (long[long.attack == "flood"]
             .sort_values("distinct_share").source.tolist())
    order += [s for s in long.source.unique() if s not in order]
    long["celltype"] = pd.Categorical(long.source, order)
    bar = (
        p9.ggplot(long, p9.aes("celltype", "distinct_share * 100"))
        + p9.geom_col(fill="#c1440e", width=0.78)
        + p9.geom_hline(yintercept=100, color="#8a8985", size=0.4,
                        linetype="dashed")
        + p9.facet_wrap("panel", nrow=1)
        + p9.coord_flip()
        + p9.scale_y_continuous(limits=(0, 104), breaks=(0, 50, 100))
        + p9.labs(x="", y="% of the added records that are DISTINCT vectors "
                          "(dashed = every record different)")
        + p9.ggtitle("Same budget, very different diversity")
        + biorag_theme()
        + p9.theme(figure_size=(10.4, 6.2), legend_position="none",
                   axis_text_y=p9.element_text(size=6))
    )
    for ext in ("pdf", "png"):
        bar.save(FIG / f"{NAME}_counts.{ext}", verbose=False, dpi=200)
    long.to_csv(TAB / f"{NAME}_counts.csv", index=False)

    write_caption(
        FIG, NAME, "The four insertion attacks, drawn where they land",
        f"{DS}; two celltypes, rate 1.0 (one poison record per genuine record), "
        f"seed 0; UMAP fitted once per celltype on the attacked celltype and its "
        f"target together, cosine metric, n_neighbors=30, min_dist=0.1, "
        f"random_state=0, and every poison set projected into that fixed layout. "
        f"Poison sets are the ones 02b_01 and 02b_02 scored, rebuilt from the "
        f"same condition seed; flood is drawn at the anchor count that was best "
        f"for that celltype. Scatters are thinned to {DRAW_MAX:,} points per "
        f"cloud per panel for rendering; the bar panel is not thinned and comes "
        f"from the counts table.",
        """relabel and duplicate land exactly on the genuine cloud, because their
        vectors are genuine records carrying a false label. No geometric check
        can separate them from the atlas -- that is what rung 0 means.

        interpolate sits in open space between the two celltypes. It is
        permitted to write any coordinates it likes and it spends that on moving
        toward the target centroid, away from the source queries it must fool.
        This is why it measures 0.4% net success at every dose while occupying a
        higher capability rung than the attacks that beat it: a rung is
        permission, not achievement.

        flood lands on the source cloud like relabel, but by construction rather
        than by copying -- real anchors rotated toward the celltype's own centre.

        The bar panel carries the half the scatters cannot, over all 26
        celltypes. duplicate holds at 1.0% distinct by construction. flood has a
        median of 2.6%, and against CL:0000860 its 368,537 records are 64
        distinct vectors repeated -- the anchor count that won there -- because
        jitter is 1.0 and no per-copy noise is applied. The strongest attack in
        the project, at its own optimum, is 64 points.

        relabel is fully distinct in only 16 of 26 celltypes and falls to 60.6%
        on CL:0000186, where 54,712 added records carry 33,153 distinct vectors.
        The attacker drew 54,712 different cells; the atlas ships byte-identical
        duplicates of them. So a de-duplication defence tuned to collapse the
        flood bar would also discard genuine records, and this panel measures
        both sides of that trade at once.

        Celltypes where the attack fails are included for the same reason the
        successes are -- CL:1001567 scores 0.000 for relabel and 0.003 for
        flood, and its poison lands on the celltype exactly as it does where the
        attack works.""")
    print(f"{NAME}_counts: {len(long)} plotted values -> pdf, png, csv, txt",
          file=sys.stderr)


if __name__ == "__main__":
    main()
