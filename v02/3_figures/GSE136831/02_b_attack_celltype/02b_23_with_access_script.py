"""Step 02b_23 — the write-access grid, where net attack success reports a null
and the celltype is destroyed anyway.

``02b_11`` is the six write-arm types whose replacements MOVE: geometry in
{shifted, synthesised} x label in {relabel, keep} x selection in {uniform, hub},
minus synthesised+keep, across the 13-dose ladder. This draws its GRAPH run.

**READ ONLY THE LEFT PANEL AND FOUR OF THE SIX TYPES ARE A NULL.** Every
``shifted`` type sits at 0.000 net attack success at every dose from 0.002 to
1.0. Taken alone that is "moving records does not work".

**The right panel says the celltype was destroyed.** The same four types drive
within-accuracy from a clean 0.361 down to 0.000 -- every query of the attacked
celltype wrong -- and they do it monotonically in dose with no threshold to
cross.

The mechanism is the one ``02b_18`` isolated on the alpha ladder. ``02b_11``
fixes ALPHA = 0.5, far outside the neighbourhood a record came from: 02b_15
measures median retrieval at 0.0005 there against 0.8738 at alpha 0.05. Nothing is being added to any query's top-50. What was
removed is the genuine record that used to vote correctly. The attacker deletes
right answers instead of supplying wrong ones.

**The clinching detail is that relabel and keep CONVERGE.** shifted/relabel/hub
and shifted/keep/hub are 0.333 and 0.334 at dose 0.002 and identical at every
rung after. Forging a label changes nothing once the record is out of retrieval
range, because no vote ever sees it. Two attacks that differ only in the label
producing the same curve is what tells you the label is not the mechanism.

``synthesised`` does both: it reaches 0.464 net attack success at dose 0.85 AND
takes accuracy to 0.000. But it is CEILING tier -- it writes vectors no count
matrix produces, so a defender who re-derives embeddings from raw counts rejects
it outright.

A net-attack-success-only benchmark would report four of these six as harmless.
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
NAME = "02b_23_with_access"

#: 01a_03 is here for the clean baseline rule only. 02b_11 carries no
#: baseline column -- it has exactly three within_ metrics -- so the
#: scale the accuracy panel is read against has to come from the
#: baseline module, restricted to the same 26 celltypes.
READS: tuple[str, ...] = ("02b_11", "01a_03")
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE


def main() -> None:
    # The GRAPH table. The plain file is the `exact` run, and 02b_16 measures
    # the two disagreeing on 44% of exactly these conditions.
    d = pd.read_parquet(TAB / "02b_11_with_access_inplace_graph.parquet")
    assert (d.retrieval_path == "graph").all(), "not the graph run"

    d = d.assign(type=(d.geometry.astype(str) + "/" + d.label_change.astype(str)
                       + "/" + d.selection.astype(str)))
    base = pd.read_parquet(BASE / "2_results" / DS / "01_a_baseline"
                           / "01a_03_no_access_clean_baseline.parquet")
    clean = float(base.loc[base.celltype.astype(str).isin(set(d.source.astype(str))),
                           "clean_accuracy"].median())

    long = pd.concat([
        d.assign(metric="net attack success", value=d.within_net_attack_success),
        d.assign(metric="within-celltype accuracy", value=d.within_accuracy),
        d.assign(metric="poison reached the top-50", value=d.within_retrieval),
    ])
    g = (long.groupby(["metric", "type", "rate"], observed=True)
             .agg(value=("value", "median"),
                  lo=("value", lambda s: s.quantile(0.25)),
                  hi=("value", lambda s: s.quantile(0.75)))
             .reset_index())
    g["metric"] = pd.Categorical(g.metric, ["net attack success",
                                            "within-celltype accuracy",
                                            "poison reached the top-50"])

    # A dashed rule at the clean baseline, on the accuracy panel only. Without
    # it the reader has no scale for "0.12 accuracy" -- the whole claim is the
    # DISTANCE from 0.361, not the value. NOTE this is the MEDIAN over the 26
    # attacked celltypes; their MEAN is 0.4584, and the docstring said 0.46
    # until 2026-09-28 while the figure has always drawn the median.
    rule = pd.DataFrame({"metric": pd.Categorical(["within-celltype accuracy"],
                                                  g.metric.cat.categories),
                         "y": [clean]})

    plot = (
        p9.ggplot(g, p9.aes("rate", "value", color="type", fill="type"))
        + p9.geom_hline(p9.aes(yintercept="y"), data=rule,
                        linetype="dashed", color="#666666", size=0.4)
        + p9.geom_ribbon(p9.aes(ymin="lo", ymax="hi"), alpha=0.12, color="none")
        + p9.geom_line(size=0.9)
        + p9.geom_point(size=1.3)
        + p9.facet_wrap("metric", nrow=1)
        + p9.scale_x_log10(breaks=[0.002, 0.01, 0.05, 0.25, 1.0],
                           labels=["0.002", "0.01", "0.05", "0.25", "1.0"])
        + p9.scale_y_continuous(limits=(0, 1))
        + p9.scale_color_cmap_d(cmap_name="viridis")
        + p9.scale_fill_cmap_d(cmap_name="viridis")
        + p9.labs(x="dose — records overwritten, as a fraction of the celltype (log)",
                  y="median over 26 celltypes, IQR shaded",
                  color="geometry / label / selection",
                  fill="geometry / label / selection")
        + p9.ggtitle("Four of these six report zero attack success and still "
                     "take accuracy to zero")
        + biorag_theme()
        + p9.theme(figure_size=(10.6, 4.2), legend_position="right",
                   axis_text_x=p9.element_text(size=7))
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)
    g.to_csv(TAB / f"{NAME}.csv", index=False)

    write_caption(
        FIG, NAME,
        "Four of these six report zero attack success and still take accuracy to zero",
        f"{DS}; modify only, six types across 13 doses, 26 celltypes, ONE seed; "
        f"the GRAPH run -- replacements written into the shipped index with "
        f"add_items and the graph re-queried; median over celltypes with the IQR "
        f"shaded; the dashed rule is the clean baseline accuracy, {clean:.3f}; "
        f"02b_11 fixes ALPHA = 0.5, where 02b_15 measures median retrieval at "
        f"0.0005 against 0.8738 at alpha 0.05.",
        f"""Every shifted type sits at 0.000 net attack success at every dose from
        0.002 to 1.0. Read that column alone and four of the six are harmless.

        The accuracy panel says otherwise: the same four take the attacked
        celltype from a clean {clean:.3f} to 0.000, monotonically in dose, with no
        threshold to cross.

        The mechanism is deletion, isolated on the alpha ladder in 02b_18 and
        confirmed here across 13 doses. At alpha 0.5 each overwritten record
        lands far outside its own neighbourhood and retrieval is 0.0005.
        Nothing is added to any top-50; what is gone is the genuine record that
        used to vote correctly.

        The clinching detail is that relabel and keep CONVERGE. shifted/relabel/
        hub and shifted/keep/hub read 0.333 and 0.334 at dose 0.002 and are
        identical at every rung after. Forging a label changes nothing once the
        record is out of retrieval range, and two attacks differing only in the
        label producing one curve is what shows the label is not the mechanism.

        synthesised does both -- 0.464 net attack success at dose 0.85 and
        accuracy to 0.000 -- but it is CEILING tier: it writes vectors no count
        matrix produces, and a defender who re-derives embeddings from raw counts
        rejects it outright.

        A benchmark reporting only net attack success would call four of these
        six a null result.

        FLOOR EFFECT, stated because the median hides it: 4 of the 26 attacked
        celltypes already sit below 0.05 clean exact accuracy, and CL:1001567 is
        at exactly 0.0000 before any attacker touches it. For those the drop to
        zero is not the attack's doing. The median curve is over a mixed set and
        the per-celltype IQR is the honest width; 01a_07 draws the clean spread,
        which runs 0.000 to 0.967.""")
    print(f"{NAME}: {len(g)} plotted values, clean baseline {clean:.4f} "
          f"-> pdf, png, csv, txt", file=sys.stderr)


if __name__ == "__main__":
    main()
