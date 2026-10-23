"""Step 02b_31 — the attack is surgical, and the denominator is what says so.

v01 asked "whose territory does the attack take" as ``fig17``. The answer depends
entirely on how a captured query is attributed, and getting that wrong is a
mistake this project has already made once and retracted: **HANDOFF §3.40,
2026-09-17**, withdrew a 25.9% spillover claim for exactly this reason. The first
version of THIS figure, drawn 2026-09-28, reproduced it.

**Attribute by the author's label and the attack looks indiscriminate.**
Grouping captured queries by ``y_true``, attacking fibroblast appears to take
more territory from myofibroblast (547) than from fibroblast (409) at dose 1.25
-- the attacked celltype supplying 42% of its own capture.

**Attribute by what the clean model ANSWERED and the attack is surgical.** Those
myofibroblast-annotated cells were ALREADY being predicted as fibroblast before
any poison existed. The attack took them from fibroblast's territory, which is
the model's, not from myofibroblast's. By ``y_pred_clean`` the attacked celltype
supplies **96.6%** at dose 1.25, **100.0%** at parity and 99.4% at dose 0.25.

The causal question is what SCimilarity answered BEFORE against what it answers
now, so both ends of the comparison have to be predictions. An author label is
neither end. §3.40 states it; this figure draws it.

**What survives is real but small.** 3.4% of fibroblast's capture at 1.25 and
4.6% of natural killer cell's at parity come from outside the attacked
celltype's clean territory -- genuine collateral, single digit, not "more than
half". The named neighbours change too: natural killer cell's collateral is
CD8-positive alpha-beta T cell and mature NK T cell by clean prediction, not the
innate lymphoid cell and cytotoxic T cell the author labels suggested.

Both panels are drawn because the gap between them is the finding. The left is
the analysis §3.40 retracted; the right is the one that answers the question.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "1_scripts" / "00_common"))
import pandas as pd  # noqa: E402
import plotnine as p9  # noqa: E402

from biorag_theme import biorag_theme, write_caption  # noqa: E402
from dataset_config import load  # noqa: E402
from ontology import Ontology  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[3]
CFG = load(BASE)
DS = CFG["dataset"]
STAGE = "02_b_attack_celltype"
NAME = "02b_31_no_access"

READS: tuple[str, ...] = ("02b_30",)
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE

SEL, MIN_SHOW = "hub", 3
#: HANDOFF §3.32: above parity the draw goes WITH REPLACEMENT, so 1.25 inserts
#: more records covering LESS of the celltype. It is a different attack and the
#: axis label says so rather than letting three columns read as one ladder.
WITH_REPLACEMENT = 1.25


def main() -> None:
    q = pd.read_parquet(TAB / "02b_30_no_access_per_query.parquet")
    assert (q.retrieval_path == "hnsw").all(), "not the graph run"
    onto = Ontology.from_obo(BASE.parent / CFG["ontology"]["obo"])
    nm = lambda c: onto.id_to_name.get(str(c), str(c))  # noqa: E731

    rows = []
    for (src, tgt, rate, sel), g in q.groupby(
            ["source", "target", "rate", "selection"], observed=True):
        if str(sel) != SEL:
            continue
        took = g[(g.y_pred_clean.astype(str) != g.y_pred_poisoned.astype(str))
                 & (g.y_pred_poisoned.astype(str) == str(tgt))]
        for col, how in (("y_true", "by the AUTHOR'S LABEL\n(retracted, HANDOFF 3.40)"),
                         ("y_pred_clean", "by what the CLEAN MODEL ANSWERED\n(the causal question)")):
            for ct, n in took[col].astype(str).value_counts().items():
                rows.append(dict(source=str(src), rate=float(rate), how=how,
                                 origin=ct, n=int(n),
                                 is_attacked=(ct == str(src)),
                                 captured=len(took)))
    g = pd.DataFrame(rows)
    g = g[g.n >= MIN_SHOW].copy()
    g["panel"] = g.source.map(nm) + " attacked"
    g["dose"] = pd.Categorical(
        g.rate.map(lambda r: f"{r:g}" + ("\n(with replacement)" if r > 1.0 else "")),
        [f"{r:g}" + ("\n(with replacement)" if r > 1.0 else "")
         for r in sorted(g.rate.unique())], ordered=True)
    g["how"] = pd.Categorical(g.how, [h for h in dict.fromkeys(g.how)], ordered=True)
    g["bucket"] = g.apply(
        lambda r: f"{nm(r.origin)[:26]}  (the attacked celltype)" if r.is_attacked
        else f"{nm(r.origin)[:26]}", axis=1)
    order = (g.groupby("bucket", observed=True).n.sum()
              .sort_values(ascending=False).index.tolist())
    g["bucket"] = pd.Categorical(g.bucket, order, ordered=True)
    g["edge"] = g.is_attacked.map({True: "the attacked celltype",
                                   False: "a different celltype"})

    plot = (
        p9.ggplot(g, p9.aes("dose", "n", fill="bucket"))
        + p9.geom_col(p9.aes(color="edge"), width=0.72, size=0.9)
        + p9.facet_grid("panel ~ how", scales="free_y")
        + p9.scale_color_manual(values={"the attacked celltype": "#000000",
                                        "a different celltype": "#ffffff"})
        + p9.scale_fill_cmap_d(cmap_name="viridis")
        + p9.labs(x="dose — poison records as a multiple of the attacked celltype",
                  y="queries the attack moved TO the target label",
                  fill="origin of the captured query",
                  color="black outline =")
        + p9.ggtitle("The attack is surgical — but only the right denominator says so")
        + biorag_theme()
        + p9.theme(figure_size=(11.6, 6.6), legend_position="right",
                   strip_text=p9.element_text(size=7.5),
                   legend_text=p9.element_text(size=7))
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)
    g.drop(columns=["bucket", "edge", "dose"]).to_csv(TAB / f"{NAME}.csv", index=False)

    def share(src, rate, how_sub):
        s = g[(g.source == src) & (g.rate == rate) & (g.how.astype(str).str.contains(how_sub))]
        return (s[s.is_attacked].n.sum() / s.captured.iloc[0] * 100) if len(s) else float("nan")
    write_caption(
        FIG, NAME,
        "The attack is surgical — but only the right denominator says so",
        f"{DS}; from 02b_30's per-query predictions, relabel / {SEL}, "
        f"SCimilarity's shipped HNSW index with poison INSERTED and the graph "
        f"re-queried at ef=100, k=50; bars count queries whose prediction MOVED "
        f"to the attacker's target label, grouped LEFT by the query's author "
        f"label and RIGHT by what the clean model predicted for it; groups under "
        f"{MIN_SHOW} omitted. Dose {WITH_REPLACEMENT:g} draws WITH REPLACEMENT "
        f"(HANDOFF §3.32) -- more records covering less of the celltype, a "
        f"different attack, labelled rather than placed silently on one ladder.",
        f"""The answer depends entirely on how a captured query is attributed,
        and getting it wrong is a mistake this project already made and
        retracted. HANDOFF §3.40 withdrew a 25.9% spillover claim on 2026-09-17
        for precisely this reason. The first version of this figure, drawn
        2026-09-28, reproduced it, and the left panel is what that looked like.

        By the AUTHOR'S LABEL the attack looks indiscriminate: attacking
        fibroblast appears to take more from myofibroblast (547) than from
        fibroblast (409) at dose 1.25, the attacked celltype supplying
        {share('CL:0000057', 1.25, 'AUTHOR'):.1f}% of its own capture.

        By what the CLEAN MODEL ANSWERED it is surgical. Those
        myofibroblast-annotated cells were already being predicted as fibroblast
        before any poison existed; the attack took them from fibroblast's
        territory, which is the model's. The attacked celltype supplies
        {share('CL:0000057', 1.25, 'CLEAN'):.1f}% at dose 1.25,
        {share('CL:0000057', 1.0, 'CLEAN'):.1f}% at parity and
        {share('CL:0000623', 1.0, 'CLEAN'):.1f}% for natural killer cell.

        The causal question is what SCimilarity answered BEFORE against what it
        answers now, so both ends must be predictions. An author label is
        neither end.

        WHAT SURVIVES IS REAL BUT SMALL. {100 - share('CL:0000057', 1.25, 'CLEAN'):.1f}%
        of fibroblast's capture at 1.25 and
        {100 - share('CL:0000623', 1.0, 'CLEAN'):.1f}% of natural killer cell's
        at parity come from outside the attacked celltype's clean territory --
        genuine collateral, single digit, not "more than half". The named
        neighbours change too: natural killer cell's collateral is CD8-positive
        alpha-beta T cell and mature NK T cell by clean prediction, not the
        innate lymphoid cell and cytotoxic T cell the author labels suggested.

        ONE REALISATION. 02b_30 re-ran conditions 02b_01 had already run, with
        identical condition_seed and identical n_poison, and the two disagree:
        within_accuracy moves by up to 0.019 across the 8 shared conditions,
        because multi-threaded add_items makes graph insertion non-deterministic
        (CLAUDE.md §1). Both numbers are in committed tables and the comparison
        is re-derivable from them.

        An earlier draft of this caption also cited a per-CELL instability, "138
        of 1,858 cells differ between two insertion orders". That figure came
        from an agent comparing two runs of its own, only one of which is
        committed, so it cannot be re-derived from anything under 2_results/ and
        has been removed. 02b_34 measures the per-cell band properly. The SHARES
        drawn here are stable to about a percentage point; how stable the
        individual cells are is not yet a published number.""")
    print(f"{NAME}: {len(g)} bars; fibroblast 1.25 attacked-share "
          f"by author {share('CL:0000057',1.25,'AUTHOR'):.1f}% vs by clean "
          f"{share('CL:0000057',1.25,'CLEAN'):.1f}% -> pdf, png, csv, txt",
          file=sys.stderr)


if __name__ == "__main__":
    main()
