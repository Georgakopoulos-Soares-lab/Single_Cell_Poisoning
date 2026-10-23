"""Step 02b_18 — how far to move a record, and two mechanisms that disagree.

``02b_11`` measured ``modify x shifted`` at a single alpha of 0.5 and reported it
inert. ``02b_15`` swept the ladder; this draws its GRAPH run, the one that
inserts the replacements into the shipped index and re-queries.

**Alpha is a displacement, and 0.5 is off the end of the map.** It interpolates
each overwritten record toward the target centroid, and at alpha 0.5 it lands
far outside the neighbourhood it came from: median retrieval collapses from
0.8738 at alpha 0.05 to 0.0005 at 0.5. The records are not near
anyone. 02b_11 did not find shifted weak, it found one setting that throws the
poison out of the graph.

**THE TWO LABEL MODES PEAK AT OPPOSITE ENDS, BY DIFFERENT MECHANISMS.**

``relabel`` redirects votes. Median net attack success on hub selection rises to
0.4055 at alpha 0.05 and collapses to 0.0462 by 0.2, tracking retrieval exactly:
the forged label only helps while the record is still close enough to be
retrieved. Within-accuracy at those settings is 0.0000 -- every query of the
attacked celltype is wrong.

``keep`` deletes. It forges no label, so net attack success is 0 by construction
at every alpha, and reading only that column would call it a null. Its damage is
in accuracy, which falls from 0.3686 to 0.1121 as alpha rises -- WORST where
retrieval is zero. Nothing is being added to any neighbourhood; the genuine
records that used to vote correctly have been displaced out of their own. The
attacker removes right answers instead of supplying wrong ones, and needs no
target label to do it.

That is why the accuracy panel is here and not just net attack success. One
metric makes ``keep`` invisible.

Dose fixed at 0.25, one seed, 26 celltypes x 4 types x 9 alphas.
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
NAME = "02b_18_with_access"

READS: tuple[str, ...] = ("02b_15",)
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE

#: The GRAPH table, never the plain one. The plain file is the `exact` run --
#: modify.remerge over the depth-500 list -- and 02b_16 measures the two
#: disagreeing on 44% of conditions. HANDOFF.md 5 carries the trap.
#: Spelled as a plain literal on purpose. It was built by `.replace("_by_donor",
#: "")` on the by-donor name until 2026-09-28 -- harmless here, since line 64
#: asserts the regime, but check_retrieval.py can only see string LITERALS, so
#: the same construct with `.replace("_graph", "")` would read the shortcut arm
#: past a check written to forbid exactly that. No filename in this lane is
#: assembled at runtime.
SRC = "02b_15_with_access_alpha_sweep_graph.parquet"


def main() -> None:
    d = pd.read_parquet(TAB / SRC)
    assert (d.retrieval_path == "graph").all(), "not the graph run"

    d = d.assign(type=d.label_change.astype(str) + " / " + d.selection.astype(str))
    long = pd.concat([
        d.assign(metric="net attack success", value=d.within_net_attack_success),
        d.assign(metric="within-celltype accuracy", value=d.within_accuracy),
        d.assign(metric="poison reached the top-50", value=d.within_retrieval),
    ])
    g = (long.groupby(["metric", "type", "alpha_lo"], observed=True)
             .agg(value=("value", "median"),
                  lo=("value", lambda s: s.quantile(0.25)),
                  hi=("value", lambda s: s.quantile(0.75)))
             .reset_index())
    g["metric"] = pd.Categorical(g.metric, ["net attack success",
                                            "within-celltype accuracy",
                                            "poison reached the top-50"])

    plot = (
        p9.ggplot(g, p9.aes("alpha_lo", "value", color="type", fill="type"))
        + p9.geom_ribbon(p9.aes(ymin="lo", ymax="hi"), alpha=0.15, color="none")
        + p9.geom_line(size=0.9)
        + p9.geom_point(size=1.6)
        + p9.facet_wrap("metric", nrow=1)
        + p9.scale_x_log10(breaks=sorted(g.alpha_lo.unique()),
                           labels=[f"{a:g}" for a in sorted(g.alpha_lo.unique())])
        + p9.scale_y_continuous(limits=(0, 1))
        + p9.scale_color_cmap_d(cmap_name="viridis")
        + p9.scale_fill_cmap_d(cmap_name="viridis")
        + p9.labs(x="alpha — how far each record is moved toward the target centroid (log)",
                  y="median over 26 celltypes, IQR shaded",
                  color="label / selection", fill="label / selection")
        + p9.ggtitle("Relabel redirects at low alpha; keep DELETES at high alpha")
        + biorag_theme()
        + p9.theme(figure_size=(10.2, 4.0), legend_position="right",
                   axis_text_x=p9.element_text(angle=45, ha="right", size=6))
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)
    g.to_csv(TAB / f"{NAME}.csv", index=False)

    write_caption(
        FIG, NAME, "Relabel redirects at low alpha; keep DELETES at high alpha",
        f"{DS}; modify x shifted only, all 26 celltypes, 4 label/selection types "
        f"x 9 alphas at dose 0.25, ONE seed; the GRAPH run -- replacements written "
        f"into the shipped index with add_items and the graph re-queried, not the "
        f"merge shortcut; lines are the median over 26 celltypes with the IQR "
        f"shaded. The genuine neighbourhood spans 1-cos = 0.0195, measured in "
        f"02b_13 and carried in its neighbourhood_1mcos column; the displacement "
        f"alpha produces is NOT written to any table, so this figure states what "
        f"is measured -- retrieval against alpha -- rather than a ratio of the "
        f"two.",
        """02b_11 measured shifted at alpha 0.5 alone and called it inert. It is
        not: median retrieval falls from 0.8738 at alpha 0.05 to 0.2267 at 0.2 and
        0.0005 at 0.5, so the poison is simply not near anyone. The ladder shows where
        the attack actually lives.

        relabel redirects. Net attack success on hub selection peaks at 0.4055 at
        alpha 0.05 and collapses to 0.0462 by 0.2, tracking retrieval exactly --
        a forged label only helps while the record is still retrieved.
        Within-accuracy at the peak is 0.0000: every query of that celltype is
        wrong.

        keep deletes, and one metric hides it. It forges no label, so net attack
        success is 0 at every alpha by construction. Its damage is in accuracy,
        which falls 0.3686 -> 0.1121 as alpha rises -- WORST where retrieval is
        zero. Nothing is added to any neighbourhood; the genuine records that
        voted correctly have been displaced out of their own. The attacker
        removes right answers rather than supplying wrong ones, and needs no
        target label.

        Selection matters more than alpha for relabel: hub reaches 0.4055 where
        uniform never exceeds 0.0135 at any setting.

        One dose, one seed. This fixes the shape of the alpha response, not its
        seed-to-seed stability.""")
    print(f"{NAME}: {len(g)} plotted values -> pdf, png, csv, txt", file=sys.stderr)


if __name__ == "__main__":
    main()
