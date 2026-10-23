"""Step 02b_28 — the full ladder per celltype: reached, close enough, believed.

v01 drew this as ``fig10``, titled "QC through to the answer". **The middle rung
is not QC and this figure does not call it that** (CLAUDE.md §7). SCimilarity
ships no reference-side filter; ``qc_pass`` measures whether a QUERY's nearest
POISONED record falls within the search distance bar its own retrieval uses. It
is a property of the retrieval, not a verdict on the poison, and nothing in the
tool ever inspects the reference. That absence is the finding.

**The funnel does not narrow where a defender would expect.** At dose 0.25 under
``relabel``/``hub``, median over 26 celltypes: poison reaches the top-50 of
0.866 of queries, 0.755 have their nearest poison inside the distance bar, and
0.270 actually return the attacker's label. Two thirds of the loss happens at
the VOTE -- the 49 genuine neighbours outvoting the poison -- and almost none of
it at retrieval or at the distance bar.

A filter placed at either of the first two rungs would therefore be acting on a
population that is already overwhelmingly harmless, and would have to be
near-perfect to beat simply counting votes.

**The per-celltype spread is the reason this is not drawn pooled.** At the same
dose the distance-bar rung runs 0.023 to 0.994 across celltypes. A single
pooled number for that rung describes almost none of them.

Right panel: what the ladder costs. Accuracy before against after, one segment
per celltype, so the celltypes with nothing to lose are visible as short
segments near zero rather than hidden inside a mean.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "1_scripts" / "00_common"))
import pandas as pd  # noqa: E402
import plotnine as p9  # noqa: E402

from biorag_theme import BASELINE, biorag_theme, write_caption  # noqa: E402
from dataset_config import load  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[3]
CFG = load(BASE)
DS = CFG["dataset"]
STAGE = "02_b_attack_celltype"
NAME = "02b_28_no_access"

READS: tuple[str, ...] = ("02b_01", "01a_03")
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE

#: One dose, so the rungs are comparable across celltypes. 0.25 is mid-ladder:
#: high enough that hub selection is working, low enough to be well below the
#: parity cliff 02b_25 and 02b_26 both locate at 1.0.
DOSE = 0.25
RUNGS = [("within_retrieval", "1. poison reached\nthe top-50"),
         ("within_qc_pass", "2. nearest poison within\nthe search distance bar"),
         ("within_attack_success", "3. vote returned\nthe attacker's label")]


def main() -> None:
    d = pd.read_parquet(TAB / "02b_01_no_access_celltype_sweep_graph.parquet")
    assert (d.retrieval_path == "hnsw").all(), "not the graph run"
    h = d[(d.attack.astype(str) == "relabel")
          & (d.selection.astype(str) == "hub")
          & (d.label_change.astype(str) == "relabel")
          & (d.rate == DOSE)].copy()
    assert len(h) == 26, f"{len(h)} rows, expected one per celltype"

    lab = pd.read_parquet(BASE / "2_results" / DS / "01_a_baseline"
                          / "01a_03_no_access_clean_baseline.parquet")
    h = h.merge(lab[["celltype", "label"]], left_on="source", right_on="celltype",
                how="left", validate="one_to_one")
    h["tick"] = h.label.str.slice(0, 26) + "  " + h.source.astype(str)
    h["tick"] = pd.Categorical(h.tick, h.sort_values("within_attack_success").tick,
                               ordered=True)

    lad = pd.concat([h.assign(rung=name, value=h[col]) for col, name in RUNGS])
    lad["rung"] = pd.Categorical(lad.rung, [n for _, n in RUNGS], ordered=True)
    lad["panel"] = "the ladder — share of the celltype's queries"

    acc = pd.concat([
        h.assign(rung="accuracy BEFORE", value=h.within_baseline_accuracy),
        h.assign(rung="accuracy AFTER", value=h.within_accuracy)])
    acc["rung"] = pd.Categorical(acc.rung, ["accuracy BEFORE", "accuracy AFTER"],
                                 ordered=True)
    acc["panel"] = "what it costs — accuracy"

    both = pd.concat([lad, acc], ignore_index=True)
    both["panel"] = pd.Categorical(
        both.panel, ["the ladder — share of the celltype's queries",
                     "what it costs — accuracy"], ordered=True)

    plot = (
        p9.ggplot(both, p9.aes("value", "tick"))
        + p9.geom_line(p9.aes(group="tick"), color=BASELINE, size=0.35, alpha=0.5)
        + p9.geom_point(p9.aes(color="rung", shape="rung"), size=2.1)
        + p9.facet_wrap("panel", nrow=1, scales="free_x")
        + p9.scale_x_continuous(limits=(0, 1))
        + p9.scale_color_cmap_d(cmap_name="viridis")
        + p9.labs(x=f"at dose {DOSE:g} — poison records as a fraction of the celltype",
                  y="", color="", shape="")
        + p9.ggtitle("Poison is retrieved and close enough; it is the VOTE that "
                     "stops it")
        + biorag_theme()
        + p9.theme(figure_size=(11.4, 7.0), legend_position="bottom",
                   strip_text=p9.element_text(size=8),
                   legend_text=p9.element_text(size=7),
                   axis_text_y=p9.element_text(size=6.5))
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)
    keep = ["source", "label", "tick", "panel", "rung", "value"]
    both[keep].to_csv(TAB / f"{NAME}.csv", index=False)

    m = {c: h[c].median() for c, _ in RUNGS}
    write_caption(
        FIG, NAME,
        "Poison is retrieved and close enough; it is the VOTE that stops it",
        f"{DS}; relabel / hub only, 26 celltypes at the single dose {DOSE:g}, ONE "
        f"seed; SCimilarity's shipped HNSW index with poison INSERTED and the "
        f"graph re-queried at ef=100, k=50; one row per attacked celltype, "
        f"ordered by rung 3. The middle rung is NOT a QC check -- SCimilarity "
        f"ships no reference-side filter, and qc_pass measures whether a QUERY's "
        f"nearest POISONED record falls inside the search distance bar that "
        f"query's own retrieval uses. It is a retrieval property, not a verdict "
        f"on the poison.",
        f"""v01 drew this as fig10 and called it "QC through to the answer". The
        middle rung is not QC, and naming it that was v01's labelled mistake.

        The funnel does not narrow where a defender would expect. Median over the
        26 celltypes: poison reaches the top-50 of
        {m['within_retrieval']:.3f} of queries, {m['within_qc_pass']:.3f} have
        their nearest poison inside the distance bar, and
        {m['within_attack_success']:.3f} return the attacker's label. Almost all
        the loss is at the VOTE -- the 49 genuine neighbours outvoting the
        poison -- and almost none at retrieval or at the bar.

        A filter placed at either of the first two rungs would be acting on a
        population that is already overwhelmingly harmless, and would have to be
        near-perfect to beat simply counting votes. That is an argument about
        where a defence belongs, and it is the reason CLAUDE.md §1 refuses to
        collapse retrieval failure into outcome failure.

        The per-celltype spread is why this is not pooled. At this dose the
        distance-bar rung runs {h.within_qc_pass.min():.3f} to
        {h.within_qc_pass.max():.3f} across celltypes; a single number for that
        rung describes almost none of them.

        The right panel shows the cost, and the celltypes with nothing to lose
        are visible as short segments near zero rather than hidden in a mean --
        01a_07 draws that floor directly.

        ONE DOSE, ONE SEED, ONE ATTACK TYPE. Chosen mid-ladder: high enough that
        hub selection is working, well below the parity cliff that 02b_25 and
        02b_26 both locate at 1.0. The dose response is 02b_03 and 02b_27.""")
    print(f"{NAME}: {len(both)} plotted values at dose {DOSE:g} -> pdf, png, csv, txt",
          file=sys.stderr)


if __name__ == "__main__":
    main()
