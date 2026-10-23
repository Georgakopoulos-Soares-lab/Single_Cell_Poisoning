"""Step 02b_26 — wrong, or differently wrong? It depends on the dose.

v01 asked this as ``fig08`` and v02 had no answer, though it computes one for
every condition: ``02b_01`` carries 49 ontology-tolerance columns -- exact
accuracy, agreement within 1, 2 and 3 Cell Ontology hops, and same-lineage --
and until 2026-09-28 only the CLEAN baseline was ever drawn (``01a_07``), never
the attacked side.

**BELOW PARITY THE ATTACK IS DIFFERENTLY WRONG, NOT WRONG.** For ``relabel``,
exact accuracy falls from 0.364 at dose 0.002 to 0.200 at 0.9 -- a 45% relative
loss -- while agreement within 3 ontology hops moves only 0.965 to 0.797, and
same-lineage 0.866 to 0.703. The answer is being pushed to a NEARBY term. A
clinician reading "alveolar macrophage" where the truth is "macrophage" has been
misled in a way that a strict accuracy metric and a biologist would score very
differently.

**AT PARITY THE CHARACTER OF THE DAMAGE CHANGES.** At dose 1.0 within-3-hops
drops to 0.554 and same-lineage to 0.356; at 1.25 they reach 0.308 and 0.264.
The answers stop being near misses and become distant ones. That is the same
dose at which ``02b_25`` shows the vote tying, and it is not a coincidence: below
parity the genuine records still shape the neighbourhood and the error stays
local, while above it the attacker's label owns the neighbourhood outright.

``interpolate`` is flat at every tolerance -- it never displaces anything.
``duplicate`` degrades mildly and uniformly.

The dashed rules are the clean baseline at each tolerance, restricted to the
same 26 attacked celltypes. Read the GAP from the rule, not the level.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "1_scripts" / "00_common"))
import pandas as pd  # noqa: E402
import plotnine as p9  # noqa: E402

from biorag_theme import biorag_theme, write_caption  # noqa: E402
from dataset_config import load  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[3]
CFG = load(BASE)
DS = CFG["dataset"]
STAGE = "02_b_attack_celltype"
NAME = "02b_26_no_access"

#: 02b_01 holds the attacked tolerances; 01a_04 holds the clean ones at every
#: tolerance level, which 02b_01 does not -- it pairs a baseline for exact and
#: lineage only. Restricted to the 26 attacked celltypes so the rule is the
#: right scale to read the attacked curves against.
READS: tuple[str, ...] = ("02b_01", "01a_04")
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE

#: Ordered, because the whole figure is about movement ALONG this axis.
TOL = [("within_accuracy", "clean_accuracy", "exact match"),
       ("within_within1", "within1", "within 1 hop"),
       ("within_within2", "within2", "within 2 hops"),
       ("within_within3", "within3", "within 3 hops"),
       ("within_lineage", "lineage", "same lineage")]


def main() -> None:
    d = pd.read_parquet(TAB / "02b_01_no_access_celltype_sweep_graph.parquet")
    assert (d.retrieval_path == "hnsw").all(), "not the graph run"
    base = pd.read_parquet(BASE / "2_results" / DS / "01_a_baseline"
                           / "01a_04_no_access_accuracy_ci.parquet")
    src = set(d.source.astype(str))
    base = base[base.celltype.astype(str).isin(src)]
    assert len(base) == len(src), f"{len(base)} baselines for {len(src)} celltypes"

    rows = []
    for acol, bcol, lab in TOL:
        g = (d.groupby(["attack", "rate"], observed=True)[acol].median()
               .reset_index().rename(columns={acol: "value"}))
        rows.append(g.assign(tolerance=lab, clean=float(base[bcol].median())))
    g = pd.concat(rows, ignore_index=True)
    g["tolerance"] = pd.Categorical(g.tolerance, [t[2] for t in TOL], ordered=True)
    g["attack"] = g.attack.astype(str)

    rule = g[["attack", "tolerance", "clean"]].drop_duplicates()

    plot = (
        p9.ggplot(g, p9.aes("rate", "value", color="tolerance"))
        + p9.geom_vline(xintercept=1.0, color="#666666", linetype="dotted", size=0.5)
        + p9.geom_hline(p9.aes(yintercept="clean", color="tolerance"),
                        data=rule, linetype="dashed", size=0.4, alpha=0.55)
        + p9.geom_line(size=0.9)
        + p9.geom_point(size=1.3)
        + p9.facet_wrap("attack", nrow=1)
        + p9.scale_x_log10(breaks=[0.002, 0.01, 0.05, 0.25, 0.5, 1.0, 1.25],
                           labels=["0.002", "0.01", "0.05", "0.25", "0.5",
                                   "1.0", "1.25"])
        + p9.scale_y_continuous(limits=(0, 1))
        + p9.scale_color_cmap_d(cmap_name="viridis")
        + p9.labs(x="dose — poison records as a multiple of the attacked celltype "
                    "(log); dotted rule is parity",
                  y="median over 26 celltypes; dashed = clean at that tolerance",
                  color="how close does\nthe answer have\nto be?")
        + p9.ggtitle("Below parity the attack is differently wrong; at parity it "
                     "becomes wrong")
        + biorag_theme()
        + p9.theme(figure_size=(10.8, 4.3), legend_position="right",
                   axis_text_x=p9.element_text(angle=45, ha="right", size=6))
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)
    g.to_csv(TAB / f"{NAME}.csv", index=False)

    r = g[g.attack == "relabel"].pivot_table(index="rate", columns="tolerance",
                                             values="value", observed=True)
    def at(rate, lab):
        return float(r.loc[rate, lab])
    write_caption(
        FIG, NAME,
        "Below parity the attack is differently wrong; at parity it becomes wrong",
        f"{DS}; three insertion attacks, 26 celltypes, 14 doses, ONE seed; "
        f"SCimilarity's shipped HNSW index with poison INSERTED and the graph "
        f"re-queried at ef=100, k=50; tolerance is distance in the Cell Ontology "
        f"is_a graph between the predicted term and the true one, so 'within 2 "
        f"hops' counts a prediction two edges away as correct; medians over the "
        f"26 attacked celltypes; dashed rules are the clean baseline at each "
        f"tolerance from 01a_04, restricted to those same 26.",
        f"""v01 asked this as fig08 and v02 had no answer, although it computes
        one for every condition: 02b_01 carries 49 ontology-tolerance columns and
        only the clean baseline was ever drawn.

        Below parity the attack is DIFFERENTLY wrong. relabel takes exact
        accuracy from {at(0.002,'exact match'):.3f} at dose 0.002 to
        {at(0.9,'exact match'):.3f} at 0.9, a 45% relative loss, while agreement
        within 3 hops moves only {at(0.002,'within 3 hops'):.3f} to
        {at(0.9,'within 3 hops'):.3f} and same-lineage
        {at(0.002,'same lineage'):.3f} to {at(0.9,'same lineage'):.3f}. The
        answer is pushed to a NEARBY term, not a distant one.

        At parity the character of the damage changes. Within 3 hops falls to
        {at(1.0,'within 3 hops'):.3f} at dose 1.0 and
        {at(1.25,'within 3 hops'):.3f} at 1.25; same-lineage to
        {at(1.0,'same lineage'):.3f} and {at(1.25,'same lineage'):.3f}. That is
        the dose at which 02b_25 shows the vote tying, and the two are the same
        event seen twice: below parity the genuine records still shape the
        neighbourhood and the error stays local; above it the attacker's label
        owns the neighbourhood and the error can go anywhere.

        WHY THIS MATTERS FOR HOW THE BENCHMARK IS READ. A strict accuracy metric
        scores 'alveolar macrophage' for 'macrophage' exactly as harshly as it
        scores 'T cell', and a biologist would not. Reporting exact accuracy
        alone overstates the damage below parity and understates how sharply it
        worsens above it. Both numbers are in the table; only one was being
        drawn.

        interpolate is flat at every tolerance -- it displaces nothing, and its
        curves sit on the clean rules. duplicate degrades mildly and roughly
        uniformly across tolerances.

        THE LEGEND IS ORDERED BUT THE RELATIONS ARE NOT NESTED, which is why
        'same lineage' crosses 'within 2 hops' in the duplicate panel. Hop
        distance counts edges in either direction, so a SIBLING term two edges
        away scores as within-2 while not being in the lineage at all; lineage
        counts any ancestor or descendant however far, so a distant ancestor is
        in-lineage while being more than 3 hops out. They measure different kinds
        of nearness and neither contains the other. Read each curve against its
        own dashed rule rather than against the curve above it.

        ONE SEED. This fixes the shape against dose, not its seed-to-seed
        stability.""")
    print(f"{NAME}: {len(g)} plotted values -> pdf, png, csv, txt", file=sys.stderr)


if __name__ == "__main__":
    main()
