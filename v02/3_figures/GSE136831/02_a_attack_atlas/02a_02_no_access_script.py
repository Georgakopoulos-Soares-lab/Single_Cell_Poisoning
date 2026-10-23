"""Step 02a_02 — poison in every neighbourhood, and the answer does not move.

The null, drawn. Scatter the budget at random across the whole 7.9M-record atlas
and give each poisoned record its own random wrong label. At 10% of the atlas --
791,389 fabricated records -- poison is in the top-50 of 99.55% of queries and
accuracy falls 0.137 percentage points.

That is the clearest statement this project can make of the distinction it exists
to defend: RETRIEVAL FAILURE and OUTCOME FAILURE are different questions. Poison
that is retrieved by nearly every query changes nearly no answer, because each
record claims something different and they split their own vote. Concentration is
what makes poisoning work, and this is the control that shows it.

Both series are on one panel with one y axis because the gap IS the result.
Drawing them on free scales, or on separate panels, would make two rising curves
and destroy the comparison.

Measured on SCimilarity's own shipped graph, with the poison inserted into it
rather than merged in afterwards, so this is the deployed system's behaviour.
The poison sets are byte-identical to v01's -- same blake2b seeds, same records,
same forged labels -- so the only difference from v01's exact-search run is the
retrieval path.
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
STAGE = "02_a_attack_atlas"
NAME = "02a_02_no_access"

#: The analysis step this figure draws, declared so the link is CHECKED rather
#: than implied. A figure's number does not encode which step fed it -- 02b_03
#: reads 02b_01 and 02b_04 reads 02b_02, so the offset is not even constant --
#: and a docstring naming the input can drift from the code that opens it.
#: ``check_provenance`` verifies every step named here exists and produced a
#: table, and that every table this script actually opens belongs to a step
#: named here.
READS: tuple[str, ...] = ('02a_01',)
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE


def main() -> None:
    d = pd.read_parquet(TAB / "02a_01_no_access_uniform_graph.parquet")
    base = float(d.baseline_accuracy.iloc[0])
    # Band is the range over the five independent draws, for BOTH series.
    # 02a_01 also publishes donor-paired intervals on the absolute levels. For
    # retrieval that interval is the right one and is tight. For ACCURACY it is
    # not: acc_lo/acc_hi bound the poisoned run's absolute accuracy at about
    # [0.427, 0.500], so carrying it onto the drop would draw +/-3.6 pp of
    # uncertainty around a 0.137 pp effect -- twenty-six times the quantity
    # being measured. That width is donor heterogeneity in the BASELINE, which
    # the clean and poisoned runs share and which a paired reading cancels. The
    # draw range is what actually varies here, and it is what is shaded.
    g = d.groupby("budget", as_index=False).agg(
        retrieval=("retrieval", "mean"), ret_lo=("retrieval", "min"),
        ret_hi=("retrieval", "max"),
        accuracy=("accuracy", "mean"), acc_min=("accuracy", "min"),
        acc_max=("accuracy", "max"), n_poison=("n_poison", "first"),
        pub_acc_lo=("acc_lo", "mean"), pub_acc_hi=("acc_hi", "mean"))
    # Accuracy LOST, so both series read "more is worse" on one axis.
    g["acc_drop"] = (base - g.accuracy) * 100
    g["drop_lo"] = (base - g.acc_max) * 100
    g["drop_hi"] = (base - g.acc_min) * 100
    g["retrieval"] *= 100
    g["ret_lo"] *= 100
    g["ret_hi"] *= 100

    # Annotations computed, never typed. The last redraw of this figure carried
    # "0.16 pp" from a superseded table while the graph rerun had moved it to
    # 0.137 -- a literal in a caption cannot be re-checked by running the script.
    top = g.iloc[-1]
    lab_ret, lab_drop = f"{top.retrieval:.2f}%", f"{top.acc_drop:.3f} pp"

    long = pd.concat([
        g.assign(series="poison reached the top-50",
                 value=g.retrieval, lo=g.ret_lo, hi=g.ret_hi),
        g.assign(series="accuracy lost", value=g.acc_drop, lo=g.drop_lo, hi=g.drop_hi),
    ])
    long["series"] = pd.Categorical(long.series,
                                    ["poison reached the top-50", "accuracy lost"])

    plot = (
        p9.ggplot(long, p9.aes("budget * 100", "value", color="series"))
        + p9.geom_ribbon(p9.aes(ymin="lo", ymax="hi", fill="series"),
                         alpha=0.15, color="none")
        + p9.geom_line(size=1.0)
        + p9.geom_point(size=2.0)
        + p9.scale_x_log10(breaks=[0.1, 0.5, 1, 2, 5, 10],
                           labels=["0.1%", "0.5%", "1%", "2%", "5%", "10%"])
        + p9.scale_y_continuous(limits=(0, 100), breaks=range(0, 101, 20))
        + p9.scale_color_manual(values=["#1f6f8b", BASELINE])
        + p9.scale_fill_manual(values=["#1f6f8b", BASELINE])
        + p9.annotate("text", x=10, y=92, label=lab_ret, size=8,
                      color="#1f6f8b", ha="right")
        + p9.annotate("text", x=10, y=6, label=lab_drop, size=8,
                      color=BASELINE, ha="right")
        + p9.labs(x="poison inserted, as a share of the 7.9M-record atlas (log scale)",
                  y="% of all 49,456 queries", color="", fill="")
        + p9.ggtitle("Scattered poisoning: everywhere, and inert")
        + biorag_theme()
        + p9.theme(figure_size=(6.8, 4.0), legend_position="top")
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)
    long.to_csv(TAB / f"{NAME}.csv", index=False)

    write_caption(
        FIG, NAME, "Scattered poisoning: everywhere, and inert",
        f"{DS}; uniform attack, 6 budgets x 5 draws; poison scattered across the "
        f"whole atlas, each record given its own random wrong label; SCimilarity's "
        f"shipped HNSW index with poison INSERTED, queried at ef=100, k=50; "
        f"shaded band is the range over the five draws; clean graph baseline "
        f"accuracy {base:.5f}.",
        f"""At 10% of the atlas -- {int(top.n_poison):,} fabricated records --
        poison reaches the top-50 of {top.retrieval:.2f}% of queries and costs
        {top.acc_drop:.3f} percentage points of accuracy. Retrieval failure and
        outcome failure are not the same question, and this is the control that
        proves it: poison that disagrees with itself splits its own vote no
        matter how much of it there is.

        The poison sets are byte-identical to v01's exact-search run -- same
        blake2b seeds, same records, same forged labels -- so the only
        difference is the retrieval path. v01 measured 99.45% retrieval and
        0.156 pp against a baseline of 0.464777
        (v01/2_results/{DS}/02_a_attack_global/global_attack.parquet, mode
        uniform); the graph gives {top.retrieval:.2f}% and {top.acc_drop:.3f} pp
        against {base:.6f}. The null holds on the deployed system, and the two
        paths agree to within a quarter of a percentage point on an effect that
        is itself a seventh of a percentage point.

        WHAT THE BAND IS, because it is not the published interval. 02a_01
        carries donor-paired 95% intervals on the absolute levels. For retrieval
        that interval is correct and tight -- [{g.iloc[-1].ret_lo:.2f},
        {g.iloc[-1].ret_hi:.2f}] here is the draw range and the published CI is
        comparable. For ACCURACY it is not usable on this axis: it bounds the
        poisoned run's absolute accuracy at about [{top.pub_acc_lo:.3f},
        {top.pub_acc_hi:.3f}], so transferring it to the drop would draw
        +/-3.6 pp around a {top.acc_drop:.3f} pp effect. That width is donor
        heterogeneity in the BASELINE, shared by the clean and poisoned runs and
        cancelled by reading them paired. Both series are shaded with the range
        over the five draws instead.

        This figure and 02a_04 measure the same submission arm. 02a_04 adds the
        overwrite arm beside it and shows that at uniform selection write access
        buys nothing, while hub selection at a hundredth of this dose takes
        accuracy from 0.4637 to 0.3096.""")
    print(f"{NAME}: {len(long)} plotted values -> pdf, png, csv, txt", file=sys.stderr)


if __name__ == "__main__":
    main()
