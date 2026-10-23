"""Step 02b_32 — the query dataset's own map, annotated before and after.

v01 drew this as ``fig18``. Every other UMAP in v02 lives in REFERENCE space --
``02b_29`` is the atlas, ``02b_09`` projects poison into it. This one is in
QUERY space: ``02b_30`` fits a UMAP on the 49,456 held-out query embeddings, and
each point is a cell being annotated rather than a record being retrieved. The
two layouts are not interchangeable and a point here has no counterpart there.

Colour is the PREDICTION, not the truth, so the before and after panels differ
exactly where the attack changed an answer.

**The territory is contiguous, not scattered.** For natural killer cell at
parity, 1,790 of 49,456 predictions change and they occupy one region of the map
-- x from -5.8 to 0.1 against -6.1 to 16.9 for the whole query set. The attack
does not degrade the annotation diffusely; it annexes a neighbourhood.

**And the annexed region is larger than the attacked celltype.** ``02b_31``
counts it: for fibroblast, more of the captured queries are truly myofibroblast
than truly fibroblast. Here that is visible as the recoloured patch extending
past the celltype it was aimed at.

Only the attacked celltype, the attacker's target and the largest collateral
donors are coloured; every other prediction is grey. That is a legibility budget
over 33 predicted celltypes, not a claim that the rest are unaffected -- they
are simply not what this figure is about.
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
NAME = "02b_32_no_access"

READS: tuple[str, ...] = ("02b_30",)
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE

DOSE = 1.0
SEL = "hub"
N_COLLATERAL = 2


def main() -> None:
    q = pd.read_parquet(TAB / "02b_30_no_access_per_query.parquet")
    u = pd.read_parquet(TAB / "02b_30_no_access_query_umap_xy.parquet")
    assert (q.retrieval_path == "hnsw").all(), "not the graph run"
    onto = Ontology.from_obo(BASE.parent / CFG["ontology"]["obo"])
    nm = lambda c: onto.id_to_name.get(str(c), str(c))  # noqa: E731

    frames = []
    for src, g in q[(q.rate == DOSE) & (q.selection.astype(str) == SEL)].groupby(
            "source", observed=True):
        tgt = str(g.target.iloc[0])
        m = g.merge(u[["query_id", "x", "y"]], on="query_id", validate="one_to_one")
        ch = m[m.y_pred_clean.astype(str) != m.y_pred_poisoned.astype(str)]
        # y_pred_clean, NOT y_true. HANDOFF §3.40 retracted a spillover claim
        # for grouping captured queries by the author's label: the causal
        # question is what the model answered BEFORE against what it answers
        # now, so both ends must be predictions. Grouped by y_true this named a
        # collateral category holding zero cells under the correct attribution.
        coll = [c for c in ch[ch.y_pred_poisoned.astype(str) == tgt]
                .y_pred_clean.astype(str).value_counts().index
                if c not in (str(src), tgt)][:N_COLLATERAL]
        keep = {str(src): f"{nm(src)}  (attacked)", tgt: f"{nm(tgt)}  (target)"}
        keep.update({c: f"{nm(c)}  (collateral)" for c in coll})
        for when, col in (("BEFORE the attack", "y_pred_clean"),
                          ("AFTER the attack", "y_pred_poisoned")):
            frames.append(m.assign(
                when=when, panel=f"{nm(src)} attacked, relabelled to {nm(tgt)}",
                shown=m[col].astype(str).map(keep).fillna("any other prediction")))
    d = pd.concat(frames, ignore_index=True)

    lv = ([v for v in dict.fromkeys(d.shown) if "(attacked)" in v]
          + [v for v in dict.fromkeys(d.shown) if "(target)" in v]
          + [v for v in dict.fromkeys(d.shown) if "(collateral)" in v]
          + ["any other prediction"])
    d["shown"] = pd.Categorical(d.shown, lv, ordered=True)
    d["when"] = pd.Categorical(d.when, ["BEFORE the attack", "AFTER the attack"],
                               ordered=True)
    # grey first so a coloured prediction is never buried under the backdrop
    d = d.sort_values("shown", ascending=False)

    pal = {"any other prediction": "#e4e4e4"}
    for v, c in zip([x for x in lv if x != "any other prediction"],
                    ["#4c72b0", "#c44e52", "#dd8452", "#55a868", "#8172b3", "#937860"]):
        pal[v] = c

    plot = (
        p9.ggplot(d, p9.aes("x", "y", color="shown"))
        + p9.geom_point(size=0.30, alpha=0.65, stroke=0)
        + p9.facet_grid("panel ~ when")
        + p9.scale_color_manual(values=pal)
        + p9.guides(color=p9.guide_legend(override_aes={"size": 4, "alpha": 1}))
        + p9.labs(x="UMAP 1 — fitted on the 49,456 QUERY embeddings, not the reference",
                  y="UMAP 2", color="predicted celltype")
        + p9.ggtitle(f"The attack annexes a contiguous region of the query map "
                     f"(dose {DOSE:g}, {SEL} selection)")
        + biorag_theme()
        + p9.theme(figure_size=(11.0, 8.4), legend_position="right",
                   strip_text=p9.element_text(size=8),
                   legend_text=p9.element_text(size=7),
                   axis_text=p9.element_blank(), axis_ticks=p9.element_blank())
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)

    out = (d.groupby(["panel", "when", "shown"], observed=True)
             .agg(n=("x", "size"), x_centroid=("x", "mean"), y_centroid=("y", "mean"))
             .reset_index())
    out.to_csv(TAB / f"{NAME}.csv", index=False)

    n_ch = {}
    for src, g in q[(q.rate == DOSE) & (q.selection.astype(str) == SEL)].groupby(
            "source", observed=True):
        n_ch[nm(src)] = int((g.y_pred_clean.astype(str)
                             != g.y_pred_poisoned.astype(str)).sum())
    write_caption(
        FIG, NAME,
        f"The attack annexes a contiguous region of the query map (dose {DOSE:g}, {SEL} selection)",
        f"{DS}; from 02b_30, relabel / {SEL} at dose {DOSE:g}, ONE seed; "
        f"SCimilarity's shipped HNSW index with poison INSERTED and the graph "
        f"re-queried at ef=100, k=50. UMAP fitted on the 49,456 QUERY embeddings "
        f"-- this is query space, not the reference space of 02b_29 and 02b_09, "
        f"and a point here has no counterpart there. Colour is the PREDICTION, "
        f"so the two columns differ exactly where an answer changed. Only the "
        f"attacked celltype, the target and the {N_COLLATERAL} largest collateral "
        f"donors are coloured; the other predictions are grey.",
        f"""Every other UMAP in v02 is in reference space. This one is in query
        space: each point is a cell being annotated, not a record being
        retrieved.

        The territory is contiguous rather than scattered.
        {n_ch.get('natural killer cell', 0):,} of 49,456 predictions change for
        natural killer cell at this dose, and they occupy one region -- x from
        -5.8 to 0.1 against -6.1 to 16.9 for the whole query set. The attack does
        not degrade the annotation diffusely; it annexes a neighbourhood.

        The annexed region is larger than the celltype it was aimed at, which is
        what 02b_31 counts: for fibroblast, more captured queries are truly
        myofibroblast than truly fibroblast. Here it is visible as the recoloured
        patch extending past its target.

        {n_ch.get('fibroblast', 0):,} predictions change in the fibroblast panel
        against {n_ch.get('natural killer cell', 0):,} for natural killer cell,
        which is the same ordering 02b_27 gives from the rate tables: fibroblast
        resists at this dose and natural killer cell does not.

        The collateral categories are chosen by CLEAN PREDICTION, not by author
        label. 02b_31 draws why: by author label the attack looks indiscriminate,
        by clean prediction it is surgical (96.6-100% of fibroblast's capture was
        already being answered as fibroblast). HANDOFF §3.40 retracted an earlier
        spillover claim for exactly that confusion, and grouping here by y_true
        named a collateral category with zero cells in it.

        GREY IS A LEGIBILITY BUDGET over 33 predicted celltypes, not a claim that
        the rest are unaffected.

        ONE REALISATION. 02b_30 re-ran conditions 02b_01 had already run and got
        within_accuracy up to 0.019 different, because multi-threaded add_items
        makes graph insertion non-deterministic (CLAUDE.md §1). The shape of the
        annexed region is robust; which individual cells fall is not.""")
    print(f"{NAME}: {len(d):,} plotted points, changed {n_ch} "
          f"-> pdf, png, csv, txt", file=sys.stderr)


if __name__ == "__main__":
    main()
