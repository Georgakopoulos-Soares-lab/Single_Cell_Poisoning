"""Step 02b_04 — coverage keeps rising; wins stop. Flood's anchor curve.

The anchor count decides how many places a fixed budget is spread across: 64
anchors means a few dense piles, 16,384 means a thin layer everywhere. v01
measured this on four celltypes with exact search and found it monotone, with
16,384 clearly best. Across all 26 celltypes on SCimilarity's own graph it is
not monotone -- it saturates around 256-1,024.

Both panels are the same conditions, which is what makes the point. Retrieval
climbs the whole way, 16% to 62%: more anchors genuinely does put poison into
more neighbourhoods. Net attack success does not follow. Going from 1,024 to
16,384 nearly doubles reach and converts almost none of it.

The mechanism is the same wall relabel hits below parity. Being present in a
top-50 is not a vote; winning needs enough records in ONE neighbourhood to beat
the genuine ones. Spreading thinner buys presence at the cost of local weight,
and past a point the attacker is everywhere and wins nowhere.

**Read the right-hand edge with care.** At 16,384 the request was capped -- by
pool size in 27% of conditions and by budget in 28% -- because build_flood takes
min(n_anchors, pool) and the budget cannot supply more anchors than records. The
lower counts ran as requested. So part of the flattening at 16,384 is a ceiling
on what could be asked for, not only a property of the attack.
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
NAME = "02b_04_no_access"

#: The analysis step this figure draws, declared so the link is CHECKED rather
#: than implied. A figure's number does not encode which step fed it -- 02b_03
#: reads 02b_01 and 02b_04 reads 02b_02, so the offset is not even constant --
#: and a docstring naming the input can drift from the code that opens it.
#: ``check_provenance`` verifies every step named here exists and produced a
#: table, and that every table this script actually opens belongs to a step
#: named here.
READS: tuple[str, ...] = ('02b_02',)
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE


def main() -> None:
    d = pd.read_parquet(TAB / "02b_02_no_access_flood_anchors.parquet")
    long = pd.concat([
        d.assign(metric="poison reached the top-50",
                 value=d.within_retrieval * 100,
                 lo=d.within_retrieval_lo * 100, hi=d.within_retrieval_hi * 100),
        d.assign(metric="vote returned the attacker's label",
                 value=d.within_net_attack_success * 100,
                 lo=d.within_net_attack_success_lo * 100,
                 hi=d.within_net_attack_success_hi * 100),
    ])
    g = (long.groupby(["metric", "n_anchors", "rate"], observed=True)
             [["value", "lo", "hi"]].mean().reset_index())
    g["metric"] = pd.Categorical(g.metric, ["poison reached the top-50",
                                            "vote returned the attacker's label"])
    # ORDERED by the numeric rate. Mapping to a bare string sorted the legend
    # alphabetically -- 0.85 before 0.8, 1.25 before 1 -- because dose became
    # text. The 14-dose union ladder (SPEC 5.4b) made that visible; the older
    # 4-dose grid happened to sort correctly by accident.
    _lab = {r: f"{r:g}x celltype" for r in sorted(g.rate.unique())}
    g["dose"] = pd.Categorical(g.rate.map(_lab),
                               categories=[_lab[r] for r in sorted(_lab)],
                               ordered=True)

    plot = (
        p9.ggplot(g, p9.aes("n_anchors", "value", color="dose"))
        + p9.geom_ribbon(p9.aes(ymin="lo", ymax="hi", fill="dose"),
                         alpha=0.13, color="none")
        + p9.geom_line(size=0.9)
        + p9.geom_point(size=1.5)
        + p9.facet_wrap("metric", nrow=1)
        + p9.scale_x_log10(breaks=sorted(g.n_anchors.unique()),
                           labels=[f"{int(a):,}" for a in sorted(g.n_anchors.unique())])
        + p9.scale_y_continuous(limits=(0, 70), breaks=range(0, 71, 10))
        # viridis, not brewer: YlGnBu caps at 9 colours and the ladder has 14,
        # so five doses -- 0.8, 0.85, 0.9, 1.0, 1.25, the whole top of the
        # range -- rendered as one indistinguishable grey.
        + p9.scale_color_cmap_d(cmap_name="viridis")
        + p9.scale_fill_cmap_d(cmap_name="viridis")
        + p9.labs(x="anchors — how many places the budget is spread across (log scale)",
                  y="% of that celltype's queries", color="dose", fill="dose")
        + p9.ggtitle("Coverage keeps rising; wins stop")
        + biorag_theme()
        + p9.theme(figure_size=(8.6, 4.1), legend_position="right",
                   axis_text_x=p9.element_text(angle=45, ha="right", size=7))
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)
    g.to_csv(TAB / f"{NAME}.csv", index=False)

    write_caption(
        FIG, NAME, "Coverage keeps rising; wins stop",
        f"{DS}; flood only, all 26 celltypes, 5 anchor counts, 14 doses; "
        f"SCimilarity's shipped HNSW index at ef=100, k=50; donor-paired 95% "
        f"intervals over 78 donors; attack success NET of the clean target rate. "
        f"Flood is CEILING tier -- it submits rotated, jittered vectors no count "
        f"matrix produces.",
        """Retrieval rises monotonically with anchors, 15% to 48% averaged over the
        ladder -- though not at every dose: at 0.002x and 0.005x retrieval
        peaks at 4,096 anchors and falls back by 16,384, which is the two
        lowest curves turning down on the left panel. Spreading the budget does reach more neighbourhoods. Net
        attack success saturates by 256-1,024 at about 15% and does not improve
        after. Reach is not a vote.

        THE BEST ANCHOR COUNT INVERTS WITH DOSE, which the 4-dose grid could not
        see because it held only high doses. Counting which anchor count wins per
        (celltype, dose) over 364 pairs -- 26 celltypes x 14 doses -- 64 wins 153,
        256 wins 60, 1,024 wins 69, 4,096 wins 39, 16,384 wins 43. But the total
        hides the structure: at dose 0.005, 64 anchors wins 25 of 26 celltypes; at
        dose 1.25 it wins 4, and 4,096 wins 9. The crossover sits near dose 0.25.
        A small budget spread over 16,384 places puts a fraction of a record in
        each and reaches nobody's top-50; concentrated in 64 it is locally dense
        enough to carry the vote. Only a large budget can afford to spread.

        So "16,384 is best" was a B-cell result, and "concentrate" is a low-dose
        result. Neither generalises; the anchor count has to be read against the
        dose.

        Caveat on the right edge: at 16,384 the request was capped by pool size
        in 27% of conditions and by budget in 60%, because build_flood takes
        min(n_anchors, pool). Lower counts ran as asked.""")
    print(f"{NAME}: {len(g)} plotted values -> pdf, png, csv, txt", file=sys.stderr)


if __name__ == "__main__":
    main()
