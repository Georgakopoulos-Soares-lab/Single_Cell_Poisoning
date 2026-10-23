"""Step 02b_27 — does the celltype matter more than the dose? Only if you aim.

v01 asked this as ``fig03`` and answered "which cell type you attack matters
more than how much you poison", from four celltypes. On v02's 26 x 14 grid the
answer is conditional, and the condition is the one this project keeps finding:
SELECTION.

Measured on ``relabel``, whose name covers three types that must not be pooled:

============================  ==========================  =====================
type                          range across CELLTYPES      range across DOSES
                              at a fixed dose             at a fixed celltype
============================  ==========================  =====================
``relabel`` / ``hub``         **0.846**                   0.525
``relabel`` / ``uniform``     **0.007**                   0.392
``keep`` / ``uniform``        0.007                       0.392
============================  ==========================  =====================

**Aimed at hubs, the celltype is the dominant axis** and v01's claim holds:
picking the celltype moves the result further than sweeping dose over three
orders of magnitude. **Spread uniformly, celltypes are indistinguishable** --
a range of 0.007 across all 26 -- and only dose moves anything.

That is not a small caveat on v01's figure, it reverses it for two of the three
types. An earlier draft of THIS figure pooled the three and reported a range of
0.007 against 0.393, which is the uniform answer wearing the label of all three;
the pooled median is a number describing no attacker.

Rows are ordered by mean over the hub panel, so the ordering is the exposed
surface under the attacker who aims. Read the uniform panels against that
ordering: their flatness is the point.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "1_scripts" / "00_common"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotnine as p9  # noqa: E402

from biorag_theme import biorag_theme, write_caption  # noqa: E402
from dataset_config import load  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[3]
CFG = load(BASE)
DS = CFG["dataset"]
STAGE = "02_b_attack_celltype"
NAME = "02b_27_no_access"

READS: tuple[str, ...] = ("02b_01", "01a_03")
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE
ATTACK = "relabel"


def main() -> None:
    d = pd.read_parquet(TAB / "02b_01_no_access_celltype_sweep_graph.parquet")
    assert (d.retrieval_path == "hnsw").all(), "not the graph run"
    d = d[d.attack.astype(str) == ATTACK].copy()
    # The NAME covers three types. Pooling them averages an aimed attacker with
    # two unaimed ones and describes neither -- the defect this figure was
    # redrawn to remove.
    d["type"] = d.label_change.astype(str) + " / " + d.selection.astype(str)
    assert d.groupby(["source", "rate", "type"], observed=True).size().max() == 1

    lab = pd.read_parquet(BASE / "2_results" / DS / "01_a_baseline"
                          / "01a_03_no_access_clean_baseline.parquet")
    g = d[["source", "rate", "type", "within_net_attack_success"]].rename(
        columns={"within_net_attack_success": "value"})
    g = g.merge(lab[["celltype", "label"]], left_on="source", right_on="celltype",
                how="left", validate="many_to_one")
    if g.label.isna().any():
        raise ValueError(f"no label for {sorted(set(g.source[g.label.isna()]))}")
    g["tick"] = g.label.str.slice(0, 26) + "  " + g.source.astype(str)

    AIMED = "relabel / hub"
    order = (g[g.type == AIMED].groupby("tick", observed=True).value.mean()
             .sort_values())
    g["tick"] = pd.Categorical(g.tick, list(order.index), ordered=True)
    g["dose"] = pd.Categorical(g.rate.map("{:g}".format),
                               [f"{r:g}" for r in sorted(g.rate.unique())],
                               ordered=True)
    g["type"] = pd.Categorical(g.type, [AIMED, "relabel / uniform",
                                        "keep / uniform"], ordered=True)

    stat = {}
    for t, s in g.groupby("type", observed=True):
        stat[str(t)] = (
            float(s.groupby("dose", observed=True).value.agg(lambda x: x.max() - x.min()).median()),
            float(s.groupby("tick", observed=True).value.agg(lambda x: x.max() - x.min()).median()))

    g["ink"] = np.where(g.value > 0.45, "black", "white")
    plot = (
        p9.ggplot(g, p9.aes("dose", "tick", fill="value"))
        + p9.geom_tile(color="white", size=0.25)
        + p9.geom_text(p9.aes(label="value", color="ink"),
                       format_string="{:.2f}", size=3.6)
        + p9.scale_color_identity()
        + p9.facet_wrap("type", nrow=1)
        + p9.scale_fill_cmap(cmap_name="magma", limits=(0, 1))
        + p9.labs(x="dose — poison records as a multiple of the attacked celltype",
                  y="", fill="net attack\nsuccess")
        + p9.ggtitle("Celltype dominates only when the attacker aims; spread "
                     "uniformly, all 26 behave alike")
        + biorag_theme()
        + p9.theme(figure_size=(13.6, 7.6), legend_position="right",
                   strip_text=p9.element_text(size=8),
                   axis_text_x=p9.element_text(angle=45, ha="right", size=5.5),
                   axis_text_y=p9.element_text(size=6))
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)
    g.drop(columns=["ink", "celltype"]).to_csv(TAB / f"{NAME}.csv", index=False)

    h_ct, h_dz = stat[AIMED]
    u_ct, u_dz = stat["relabel / uniform"]
    dead = int((order <= 0.001).sum())
    write_caption(
        FIG, NAME,
        "Celltype dominates only when the attacker aims; spread uniformly, all 26 behave alike",
        f"{DS}; the three types the `relabel` NAME covers, 26 celltypes x 14 "
        f"doses, ONE seed; SCimilarity's shipped HNSW index with poison INSERTED "
        f"and the graph re-queried at ef=100, k=50; cells are net attack success; "
        f"rows ordered by the row mean of the HUB panel, so the same ordering is "
        f"used in all three and the uniform panels' flatness is visible against "
        f"it.",
        f"""v01 asked this as fig03, from four celltypes, and answered that the
        celltype matters more than the dose. On 26 celltypes the answer is
        conditional and the condition is selection.

        Aimed at hubs, celltype dominates and v01's claim holds: the range across
        celltypes at a fixed dose is {h_ct:.3f} against {h_dz:.3f} across the
        entire dose ladder within a celltype. Choosing the target moves the
        result further than three orders of magnitude of dose.

        Spread uniformly, it reverses. The range across all 26 celltypes is
        {u_ct:.3f} -- they are indistinguishable -- while dose still moves
        {u_dz:.3f}. keep / uniform behaves the same way.

        THE POOLED ANSWER IS NOT A COMPROMISE, IT IS A FICTION. An earlier draft
        of this figure medianed the three types together and reported
        0.007 against 0.393, which is the uniform answer wearing the label of all
        three. There is no attacker whose behaviour that describes. The same
        trap 02b_19 fell into by comparing two arms at different selections.

        THE UNIFORM PANELS' JUMP AT PARITY IS NOT UNIFORM SELECTION STARTING TO
        WORK -- IT IS THE SELECTION AXIS CEASING TO EXIST. At dose 1.0 the budget
        equals the celltype pool exactly (n_poison == n_celltype: 188,421 for
        CL:0000623, 196,242 for CL:0000057), so hub and uniform both take every
        record and there is nothing left to choose between. Measured on those two
        celltypes the hub-minus-uniform gap collapses from 0.844 and 0.166 at
        dose 0.25 to 0.001 and 0.005 at parity, then REOPENS to 0.061 and 0.113
        at 1.25 where uniform's with-replacement draw no longer covers the pool.
        Read the parity column of the uniform panels as a degenerate case, not as
        a result about spreading poison. Found 2026-09-28 by the agent that built
        02b_30, after I had briefed two conditions whose selection axis was
        degenerate without noticing.

        {dead} of 26 celltypes never exceed 0.001 at any dose even under hub
        selection. They are not defended: 01a_07 shows several start at or near
        zero clean accuracy, so there is no correct answer to take away. And
        02b_26 shows that where the attack does land, below parity it moves the
        answer to a NEARBY ontology term rather than a distant one.

        ONE SEED. This fixes the shape of the surface, not its seed-to-seed
        stability.""")
    print(f"{NAME}: {len(g)} cells; hub range ct {h_ct:.3f} vs dose {h_dz:.3f}; "
          f"uniform ct {u_ct:.3f} vs dose {u_dz:.3f} -> pdf, png, csv, txt",
          file=sys.stderr)


if __name__ == "__main__":
    main()
