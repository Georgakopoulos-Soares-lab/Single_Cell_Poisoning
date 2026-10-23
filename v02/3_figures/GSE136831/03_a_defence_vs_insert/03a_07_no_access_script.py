"""Step 03a_07 — the same defence, total against one attack and worthless against the other.

Four columns of embedding space, two rows. Each row is one attack at dose 1.0
against SCimilarity's shipped graph; each column is what the defender's
neighbourhood looks like after a layer is applied. Read across.

THE TOP ROW IS THE ARGUMENT. `relabel` copies a genuine reference vector and
writes the attacker's label on it. Columns 1 and 2 are therefore nearly
indistinguishable in POSITION -- the poison lands on top of the cells it
imitates, because it IS those cells. Column 3 is the punchline: `recompute`
re-derives every embedding from counts, finds them all authentic, and admits
the entire poison set. Nothing is removed and the panel is identical to
column 2. `03a_01` measures it as 0.8528 attack success both undefended and
under recompute -- the same number twice, not a near miss.

THE BOTTOM ROW IS THE CONTROL. `flood` synthesises vectors around anchors.
They are new points, visible as their own cloud in column 2, and `recompute`
removes every one of them because no count matrix produces those coordinates.
Same access level, same dose, same layer, opposite outcome.

Column 4 is the stack, and it clears both -- but only because
`label_consistency` carries the top row alone. The figure exists to stop a
reader concluding from the bottom row that poisoned records are something you
can see. Position separates a flood; it does not separate a relabel, and the
defence that works on one is worth exactly zero on the other.

COORDINATES ARE READ, NOT FITTED HERE. `03a_06` does the UMAP and writes the
coordinates to `2_results/`, because a fit is a computed quantity and a figure
script that computes a statistic instead of reading one is a violation
(CLAUDE.md 7). This script positions nothing; it draws what that table says.
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
STAGE = "03_a_defence_vs_insert"
NAME = "03a_07_no_access"

READS: tuple[str, ...] = ("03a_06",)
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE
TITLE = "The same defence: total against a flood, worthless against a relabel"

#: The four columns, each a rule for which rows survive into that panel.
#: "clean" drops the poison entirely -- it is the reference as it would be
#: without the attack, not a defence.
PANELS = (("1 · clean reference", lambda d: d.role != "poison"),
          ("2 · after the attack", lambda d: d.index == d.index),
          ("3 · recompute\n(is the vector authentic?)", lambda d: d.survives_recompute),
          ("4 · stack\n(all four layers)", lambda d: d.survives_stack))
ROLE_COL = {"genuine reference": "#c8c7c2", "attacked celltype's queries": "#2c7bb6",
            "poisoned records": "#d7191c"}
PRETTY_ROLE = {"genuine": "genuine reference", "query": "attacked celltype's queries",
               "poison": "poisoned records"}


def main() -> None:
    d = pd.read_parquet(TAB / "03a_06_no_access_umap.parquet")
    d["role_p"] = d.role.map(PRETTY_ROLE)
    # One row per (point, panel it survives into). Long, per CLAUDE.md 7.
    g = pd.concat([d[rule(d)].assign(panel=name) for name, rule in PANELS],
                  ignore_index=True)
    g["panel"] = pd.Categorical(g.panel, [n for n, _ in PANELS], ordered=True)
    g["role_p"] = pd.Categorical(g.role_p, list(ROLE_COL), ordered=True)
    # Poison last so it draws on top of the genuine cloud it is hiding in --
    # otherwise the top row's overlap reads as absence rather than concealment.
    g = g.sort_values(["panel", "role_p"])
    # Short row strip: facet_grid rotates it 90 degrees on the right edge,
    # where a three-line label is unreadable.
    g["row"] = g.attack + "\n" + g.source + " -> " + g.target

    plot = (
        p9.ggplot(g, p9.aes("x", "y", color="role_p"))
        + p9.geom_point(size=0.30, alpha=0.55, stroke=0)
        + p9.facet_grid("row ~ panel", scales="free")
        + p9.scale_color_manual(values=ROLE_COL, name="")
        + p9.guides(color=p9.guide_legend(override_aes={"size": 3.2, "alpha": 1}))
        + p9.labs(x="", y="", color="")
        + p9.ggtitle(TITLE)
        + biorag_theme()
        + p9.theme(figure_size=(11.6, 6.4), legend_position="bottom",
                   axis_text=p9.element_blank(), axis_ticks=p9.element_blank(),
                   panel_grid=p9.element_blank(),
                   strip_text=p9.element_text(size=6.8),
                   legend_title=p9.element_blank(),
                   legend_box_spacing=0.01)
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)
    g[["condition", "source", "target", "attack", "panel", "role", "x", "y",
       "poison_lc", "survives_recompute", "survives_lc", "survives_stack"]
      ].to_csv(TAB / f"{NAME}.csv", index=False)

    def kept(cond_sub: str, col: str) -> int:
        p = d[(d.source == cond_sub) & (d.role == "poison")]
        return int(p[col].sum())

    def tot(cond_sub: str) -> int:
        return int(((d.source == cond_sub) & (d.role == "poison")).sum())

    thr = float(d.lc_threshold.iloc[0])
    write_caption(
        FIG, NAME, TITLE,
        f"{DS}; UMAP of 128-d SCimilarity embeddings, cosine metric, fitted by "
        f"03a_06 with a fixed seed on the attacked celltype's queries, the "
        f"genuine reference records those queries retrieved, and the retrieved "
        f"poison. Both attacks at dose 1.0, hub selection, ONE seed. Axes are "
        f"UMAP coordinates and carry no units; panels are fitted per row.",
        f"Poison records drawn, and how many each layer ADMITS. "
        f"CL:0002062 relabel: {tot('CL:0002062'):,} drawn, recompute admits "
        f"{kept('CL:0002062','survives_recompute'):,} -- all of them, because "
        f"the attacker copied a genuine vector -- and label consistency admits "
        f"{kept('CL:0002062','survives_lc'):,}. CL:0000583 flood: "
        f"{tot('CL:0000583'):,} drawn, recompute admits "
        f"{kept('CL:0000583','survives_recompute'):,}. The layer that removes "
        f"everything in the bottom row removes nothing in the top. 03a_01 "
        f"scores the same two conditions at 0.8528 attack success undefended "
        f"and 0.8528 under recompute for the relabel, and 0.9046 undefended "
        f"against 0.0000 under recompute for the flood. Label consistency uses "
        f"a {float(d.lc_budget.iloc[0]):.0%} budget of genuine records, "
        f"threshold {thr:.4f}, derived from the genuine distribution. Panel 1 "
        f"is the reference WITHOUT the attack, not a defence.")
    print(f"wrote {NAME}.pdf/.png/.txt and {NAME}.csv  ({len(g):,} plotted rows)")


if __name__ == "__main__":
    main()
