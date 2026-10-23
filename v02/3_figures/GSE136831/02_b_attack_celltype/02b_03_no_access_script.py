"""Step 02b_03 — retrieval plateaus at 80%, and the vote waits for parity.

The project's central distinction, drawn from one run on SCimilarity's own
graph. Both panels share a y axis on purpose: the same attack occupies four
fifths of the neighbourhoods while carrying roughly an eighth of the votes, and
free scales would flatter the two curves into looking alike.

`relabel` is the shape to read, and it has THREE regimes, not two. Retrieval
climbs steadily from 23% at dose 0.002 to a plateau near 80% by dose 0.5 and
goes no further. Net attack success climbs with it to 11.5% by dose 0.25 and
then STALLS -- 11.5%, 11.7%, 12.3%, 13.4%, 16.4% across doses 0.25 to 0.9, while
retrieval is already saturated. At parity it jumps to 34.3% and at 1.25 to
47.0%. More retrieval stops buying votes long before parity; what breaks the
stall is outnumbering the genuine records, not reaching more queries. Below
parity the genuine twin is still in the index and still votes, so the attacker
adds a wrong vote without removing a right one.

The efficiency ratio makes it explicit. Votes won per unit of retrieval is
pinned at 0.15 for relabel from dose 0.05 through 0.5, then rises to 0.42 at
parity and 0.59 at 1.25.

`duplicate` is the opposite shape and it is the more efficient attacker.
Retrieval saturates near 20% by dose 0.05 and never improves -- it concentrates
copies on a small pool and cannot reach past the queries already beside it --
but it converts what it does reach at 0.49 to 0.67 votes per unit of retrieval,
three times relabel's rate over the same range. Concentration is what turns
retrieval into votes.

`interpolate` is inert, and geometrically so: it aims at the TARGET centroid,
which is away from the source queries it needs to fool. It never exceeds 2.7%
retrieval or 0.44% net attack success at any dose.

Tier is drawn as linetype, not decoration. interpolate synthesises vectors no
count matrix produces, so it is a CEILING -- an upper bound on damage, not a
threat an attacker could mount by submitting data.
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
NAME = "02b_03_no_access"

#: The analysis step this figure draws, declared so the link is CHECKED rather
#: than implied. A figure's number does not encode which step fed it -- 02b_03
#: reads 02b_01 and 02b_04 reads 02b_02, so the offset is not even constant --
#: and a docstring naming the input can drift from the code that opens it.
#: ``check_provenance`` verifies every step named here exists and produced a
#: table, and that every table this script actually opens belongs to a step
#: named here.
READS: tuple[str, ...] = ('02b_01',)
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE
#: Parity: poison records equal to the attacked celltype's own population.
PARITY = 1.0
TITLE = "Retrieval plateaus at 80%, and the vote waits for parity"


def main() -> None:
    d = pd.read_parquet(TAB / "02b_01_no_access_celltype_sweep_graph.parquet")
    long = pd.concat([
        d.assign(metric="poison reached the top-50",
                 value=d.within_retrieval * 100,
                 lo=d.within_retrieval_lo * 100, hi=d.within_retrieval_hi * 100),
        d.assign(metric="vote returned the attacker's label",
                 value=d.within_net_attack_success * 100,
                 lo=d.within_net_attack_success_lo * 100,
                 hi=d.within_net_attack_success_hi * 100),
    ])
    g = (long.groupby(["metric", "attack", "attack_tier", "rate"], observed=True)
             [["value", "lo", "hi"]].mean().reset_index())
    g["metric"] = pd.Categorical(g.metric, ["poison reached the top-50",
                                            "vote returned the attacker's label"])
    # Ordered categorical so scale_linetype_manual's positional list lines up.
    g["attack_tier"] = pd.Categorical(g.attack_tier.astype(str),
                                      ["realistic", "ceiling"], ordered=True)

    plot = (
        p9.ggplot(g, p9.aes("rate", "value", color="attack"))
        + p9.geom_vline(xintercept=PARITY, color=BASELINE, size=0.4, linetype="dotted")
        # color="none" -- passing None leaves matplotlib's default blue edge,
        # which reads as a fourth series.
        + p9.geom_ribbon(p9.aes(ymin="lo", ymax="hi", fill="attack"),
                         alpha=0.15, color="none")
        # linetype lives on the line alone: geom_ribbon inherits a global
        # aesthetic and matplotlib cannot turn a tier name into dashes.
        + p9.geom_line(p9.aes(linetype="attack_tier"), size=0.9)
        + p9.geom_point(size=1.4)
        + p9.facet_wrap("metric", nrow=1)
        # Explicit: plotnine's default linetype scale emits a dash pattern
        # matplotlib rejects, and a dict of values is not consumed positionally.
        + p9.scale_linetype_manual(values=["solid", "dashed"])
        + p9.scale_color_brewer(type="qual", palette="Dark2")
        + p9.scale_fill_brewer(type="qual", palette="Dark2")
        # Log axis: the ladder runs 0.002 to 1.25 and on a linear scale the
        # first six rungs collapse against the y axis, hiding the climb that
        # the three-regime reading depends on. Breaks thinned -- six rungs
        # crowd between 0.7 and 1.25 and overprint.
        + p9.scale_x_log10(breaks=[0.002, 0.01, 0.05, 0.25, 0.5, 1.0, 1.25],
                           labels=["0.002", "0.01", "0.05", "0.25", "0.5",
                                   "1.0", "1.25"])
        + p9.scale_y_continuous(limits=(0, 100), breaks=range(0, 101, 20))
        + p9.labs(x="poison records, as a multiple of the attacked celltype",
                  y="% of that celltype's queries",
                  color="attack", fill="attack", linetype="tier")
        + p9.ggtitle(TITLE)
        + biorag_theme()
        + p9.theme(figure_size=(8.6, 4.1), legend_position="right",
                   axis_text_x=p9.element_text(angle=45, ha="right", size=7))
    )
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)
    g.to_csv(TAB / f"{NAME}.csv", index=False)

    # Every number below is pulled from the frame that was just plotted. The
    # previous revision of this caption was written against an 8-dose table and
    # survived the rerun to 14 doses unchanged, claiming 0.02% net attack
    # success at dose 0.25 where the graph run gives 11.46% and an "850-fold"
    # parity jump that is 3.0x. A caption holding literals cannot be rechecked
    # by running the script; one holding lookups fails loudly instead.
    def v(metric: str, attack: str, rate: float) -> float:
        m = ("poison reached the top-50" if metric == "ret"
             else "vote returned the attacker's label")
        s = g[(g.metric == m) & (g.attack.astype(str) == attack)
              & (g.rate == rate)].value
        assert len(s) == 1, f"{metric}/{attack}/{rate}: {len(s)} rows"
        return float(s.iloc[0])

    rel = g[(g.attack.astype(str) == "relabel")
            & (g.metric == "poison reached the top-50")]
    n_doses = g.rate.nunique()
    write_caption(
        FIG, NAME, TITLE,
        f"{DS}; 26 celltypes, 3 attacks, {n_doses} doses from 0.002 to 1.25, ONE "
        f"seed; SCimilarity's shipped HNSW index with poison INSERTED via "
        f"add_items and the graph re-queried at ef=100, k=50; donor-paired 95% "
        f"intervals over 78 donors, averaged over the 26 celltypes; attack "
        f"success is NET of the clean rate of the same target label; the dotted "
        f"rule is parity, where poison records equal the attacked celltype's own "
        f"population.",
        f"""Both panels share a y axis deliberately. relabel occupies
        {v('ret', 'relabel', 0.5):.0f}% of neighbourhoods at dose 0.5 and carries
        {v('nas', 'relabel', 0.5):.1f}% of the votes there.

        relabel has THREE regimes, not two. Retrieval climbs from
        {v('ret', 'relabel', 0.002):.1f}% at dose 0.002 to
        {rel.value.max():.1f}% and plateaus. Net attack success climbs with it to
        {v('nas', 'relabel', 0.25):.1f}% at dose 0.25 and then STALLS --
        {v('nas', 'relabel', 0.25):.1f}, {v('nas', 'relabel', 0.5):.1f},
        {v('nas', 'relabel', 0.8):.1f}, {v('nas', 'relabel', 0.9):.1f} across
        doses 0.25 to 0.9 while retrieval is already saturated -- then jumps to
        {v('nas', 'relabel', 1.0):.1f}% at parity and
        {v('nas', 'relabel', 1.25):.1f}% at 1.25. The parity step is
        {v('nas', 'relabel', 1.0) / v('nas', 'relabel', 0.9):.1f}x in one rung.
        More retrieval stops buying votes well before parity; what breaks the
        stall is outnumbering the genuine records, not reaching more queries.
        Below parity the genuine twin is still in the index and still votes, so
        the attacker adds a wrong vote without removing a right one.

        duplicate is the opposite shape and the more efficient attacker.
        Retrieval saturates near {v('ret', 'duplicate', 0.05):.0f}% by dose 0.05
        and never improves -- it concentrates copies on a small pool and cannot
        reach past the queries already beside it -- but it converts what it does
        reach at {v('nas', 'duplicate', 0.25) / v('ret', 'duplicate', 0.25):.2f}
        votes per unit of retrieval at dose 0.25 against relabel's
        {v('nas', 'relabel', 0.25) / v('ret', 'relabel', 0.25):.2f}. Three times
        the conversion on a quarter of the reach. Concentration is what turns
        retrieval into votes, which is the same variable 02a_04 isolates at
        atlas scale.

        interpolate is inert and geometrically so: it aims at the target
        centroid, away from the queries it must fool. It never exceeds
        {g[g.attack.astype(str) == 'interpolate'].value.max():.1f}% on either
        panel at any dose. Reported as the null it is.

        Dashed lines are CEILING-tier attacks, which synthesise vectors no count
        matrix produces. They bound the damage rather than describing a threat.

        ONE SEED. This fixes the shape of the dose response, not its
        seed-to-seed stability. The intervals are over donors, not over draws of
        the poison set.""")
    print(f"{NAME}: {len(g)} plotted values -> pdf, png, csv, txt", file=sys.stderr)


if __name__ == "__main__":
    main()
