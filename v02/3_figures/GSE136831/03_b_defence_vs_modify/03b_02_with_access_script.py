"""Step 03b_02 — what every layer costs, and what it buys, against an overwriter.

The module's headline drawn in full. `03a_04` abbreviates this arm to six
layers to make a cross-arm point; this draws all twelve configurations against
all three geometries, on the two axes a defender actually has to trade off.

Horizontal: attack success, so LEFT is better. Vertical: the share of GENUINE
records the layer rejects, so LOW is better. The dashed rule is the undefended
level. A useful layer sits left of the rule and near the floor. Anything to the
RIGHT of the rule made things worse.

THE THREE PANELS ARE THE SAME ATTACK AT THREE LEVELS OF FORGERY, and the
defence collapses as the forgery shrinks:

  synthesised -- the attacker writes a vector from nothing. recompute is free
  and total: it sits on the floor at the far left.

  shifted -- the attacker moves an existing vector. recompute is still free and
  still total, because a moved vector is reproducible from no count matrix.

  unmoved -- the attacker changes ONLY the label. recompute lands exactly on
  the dashed rule, at zero cost, because it is doing nothing at all. It admits
  every record: the embedding is genuine and re-encoding reproduces it. This is
  the null, and it is exact rather than approximate (the caption reports the
  measured maximum difference over every condition).

WHAT THE COST AXIS ADDS THAT A RATE TABLE CANNOT. label_consistency is the only
layer with any traction on the unmoved arm, and the panel shows what it charges
for it: the three budgets climb the y axis while barely moving left (the
caption gives both numbers). Whether that trade is acceptable is a judgement,
but it is a judgement made against two numbers instead of one.

donor_cap charges nothing and does not help; the caption counts how many of
its points fall right of the rule. `03a_02` explains why: a cap cannot tell poisoned records from genuine
ones, so once the attacker holds more identities than the cap allows it
truncates the defenders' evidence instead.

Redrawn 2026-10-10 on the v3 tables: the FIXED accuracy (182d5c65), full
13-rate ladders. synthesised and shifted pool v3modhub and v3moduni (hub and
uniform); unmoved is the cached_clean v3 re-run (RUNLOG Run 59), the old one
having predated the fix. Axis limits follow the data.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "1_scripts" / "00_common"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotnine as p9  # noqa: E402
import yaml  # noqa: E402

from biorag_theme import BASELINE, biorag_theme, write_caption  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[3]
CFG = yaml.safe_load((BASE / "config" / "config.yaml").read_text())
DS = CFG["dataset"]
STAGE = "03_b_defence_vs_modify"
NAME = "03b_02_with_access"

READS: tuple[str, ...] = ("03b_01",)
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE
TITLE = "Against an overwriter, every layer is free, useless, or both"

SRC = ("03b_01_with_access_layer_matrix_graph_v3modhub.parquet",
       "03b_01_with_access_layer_matrix_graph_v3moduni.parquet",
       "03b_01_with_access_layer_matrix_cached_clean_v3.parquet")
#: Panels ordered by how much the attacker had to forge, most to least. The
#: defence collapses along this axis and the reading depends on the order.
GEOM = ("synthesised", "shifted", "unmoved")
GEOM_LABEL = {"synthesised": "synthesised\nvector written from nothing",
              "shifted": "shifted\nexisting vector moved",
              "unmoved": "unmoved\nLABEL ONLY — vector untouched"}
#: Which mechanism each configuration tests, so the legend groups by the
#: question asked rather than by the config's name.
FAMILY = {"none": "no defence", "recompute": "vector authenticity",
          "dedup": "multiplicity", "donor_cap": "identity volume",
          "label_consistency": "vector–label pairing", "stack": "all four"}


#: The four the caption argues from: the layer that works, the layer that
#: does nothing, the only one with traction on unmoved, and the tightest cap.
#: config -> (y offset, horizontal justification). adjustText is not in the
#: environment and one figure does not justify adding a dependency, so the
#: four labels are placed explicitly. recompute and donor_cap1 both sit at
#: zero cost and would otherwise print on top of each other.
#: config -> (y offset, x offset, ha). stack and label_consistency COINCIDE on
#: the unmoved panel -- the stack there is label_consistency and nothing
#: else, which is the finding -- so they are separated vertically rather
#: than allowed to overprint; recompute is set off to the right of its point.
LABELLED = {"stack@0.1": (0.85, 0.0, "center"),
            "label_consistency@0.1": (-0.85, 0.0, "center"),
            "recompute": (0.55, 1.2, "left"), "donor_cap1": (-0.50, 0.0, "center")}


def family_of(defense: str) -> str:
    return FAMILY["donor_cap" if defense.startswith("donor_cap") else defense]


def main() -> None:
    d = pd.concat([pd.read_parquet(TAB / s) for s in SRC], ignore_index=True)
    d["cfg"] = np.where(d.lc_budget.notna(),
                        d.defense.astype(str) + "@" + d.lc_budget.astype(str),
                        d.defense.astype(str))
    g = (d.groupby(["geometry", "cfg", "defense"], observed=True)
           .agg(attack=("within_attack_success", "mean"),
                cost=("false_positive", "mean"),
                accuracy=("within_accuracy", "mean"),
                n=("within_attack_success", "size")).reset_index())
    g["attack"] *= 100
    g["cost"] *= 100
    g["accuracy"] *= 100
    g["family"] = g.defense.map(family_of)
    g["geometry"] = pd.Categorical(g.geometry.map(GEOM_LABEL),
                                   [GEOM_LABEL[k] for k in GEOM], ordered=True)
    # The undefended level per panel, carried as a column so the rule the eye
    # reads and the number the caption quotes are the same value.
    und = g[g.cfg == "none"].set_index("geometry").attack
    # float(), explicitly: mapping over a Categorical returns object dtype,
    # and plotnine then reads xintercept as a DISCRETE scale and dies in
    # map_position with "RangeDiscrete has no attribute range".
    g["undefended"] = g.geometry.map(und).astype(float)
    # Label a CURATED subset. Eight of the twelve configurations sit at zero
    # cost, so labelling all of them overprints the axis into illegibility
    # and hides the four points the reading actually turns on. The rest are
    # identified by colour, and every value is in the sibling .csv.
    lab = g[g.cfg.isin(LABELLED)].copy()
    lab["ytext"] = lab.cost + lab.cfg.map(lambda c: LABELLED[c][0])
    lab["xtext"] = lab.attack + lab.cfg.map(lambda c: LABELLED[c][1])
    lab["ha"] = lab.cfg.map(lambda c: LABELLED[c][2])

    plot = (
        p9.ggplot(g, p9.aes("attack", "cost"))
        + p9.geom_vline(p9.aes(xintercept="undefended"), color=BASELINE,
                        linetype="dashed", size=0.45)
        + p9.geom_point(p9.aes(color="family"), size=2.6, alpha=0.9)
        + p9.geom_text(lab[lab.ha == "center"], p9.aes("xtext", "ytext", label="cfg"),
                       size=5.2, color="#2b2a28", inherit_aes=False)
        + p9.geom_text(lab[lab.ha == "left"], p9.aes("xtext", "ytext", label="cfg"),
                       size=5.2, color="#2b2a28", ha="left", inherit_aes=False)
        # Explicit limits: the leftmost and rightmost labels overhang their
        # points and were being clipped by the default expansion.
        + p9.scale_x_continuous(limits=(-5, float(g.attack.max()) + 7))
        + p9.scale_y_continuous(limits=(-1.1, float(g.cost.max()) + 1.5))
        + p9.facet_wrap("geometry", nrow=1)
        + p9.scale_color_brewer(type="qual", palette="Dark2", name="what the layer checks")
        + p9.labs(x="attack success, % of the attacked celltype's queries  (left is better)",
                  y="genuine records rejected, %  (low is better)")
        + p9.ggtitle(TITLE)
        + biorag_theme()
        + p9.theme(figure_size=(11.4, 4.6), legend_position="right",
                   strip_text=p9.element_text(size=7))
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)
    g.to_csv(TAB / f"{NAME}.csv", index=False)

    def v(geom: str, cfg: str, col: str = "attack") -> float:
        s = g[(g.geometry == GEOM_LABEL[geom]) & (g.cfg == cfg)][col]
        assert len(s) == 1, f"{geom}/{cfg}: {len(s)} rows"
        return float(s.iloc[0])

    caps = [c for c in g.cfg.unique() if c.startswith("donor_cap")]
    worse = int(sum((v(gm, c) > v(gm, "none")) for gm in GEOM for c in caps))
    n_per_geom = d[d.cfg == "none"].groupby("geometry", observed=True).size()
    k = ["geometry", "selection", "source", "rate"]
    un = d[d.geometry == "unmoved"]
    rc_max = float((un[un.cfg == "recompute"].set_index(k).within_attack_success
                    - un[un.cfg == "none"].set_index(k).within_attack_success)
                   .abs().max()) * 100
    write_caption(
        FIG, NAME, TITLE,
        f"{DS}; {d.source.nunique()} celltypes, "
        + ", ".join(f"{n} conditions for {gm}" for gm, n in n_per_geom.items())
        + f" (hub and uniform, {d.rate.nunique()} doses); 12 configurations. "
        f"Dashed rule is the undefended attack success of that panel. x is "
        f"benefit, y is cost.",
        f"recompute reaches {v('synthesised','recompute'):.2f}% from "
        f"{v('synthesised','none'):.2f}% on synthesised and "
        f"{v('shifted','recompute'):.2f}% from {v('shifted','none'):.2f}% on "
        f"shifted, at {v('synthesised','recompute','cost'):.2f}% and "
        f"{v('shifted','recompute','cost'):.2f}% cost. On unmoved it reaches "
        f"{v('unmoved','recompute'):.2f}% against {v('unmoved','none'):.2f}% "
        f"undefended; max |difference| over its conditions {rc_max:.6f} pp, "
        f"because an in-place relabel leaves a genuine embedding that "
        f"re-encoding reproduces. On unmoved, label_consistency@0.1 reaches "
        f"{v('unmoved','label_consistency@0.1'):.2f}% for a "
        f"{v('unmoved','label_consistency@0.1','cost'):.2f}% rejection rate on "
        f"genuine records. {worse} of the {len(caps)*len(GEOM)} donor_cap "
        f"points lie to the RIGHT of the rule -- worse than no defence. "
        f"Accuracy under each layer is in the sibling .csv. Graph-path "
        f"single-condition levels carry up to ~0.1 of graph-build noise at high "
        f"dose (RUNLOG Run 56); panel means move by under 0.005. unmoved is "
        f"cached_clean and exact by construction.")
    print(f"wrote {NAME}.pdf/.png/.txt and {NAME}.csv  ({len(g)} points)")


if __name__ == "__main__":
    main()
