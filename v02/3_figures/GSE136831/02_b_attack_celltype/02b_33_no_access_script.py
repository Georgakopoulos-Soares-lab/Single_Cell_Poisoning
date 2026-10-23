"""Step 02b_33 — the same annexation, cut to slide shape.

v01 drew this as ``fig19``: "a slide-shaped before/after pair, cut from the
query-only UMAP". It carries no information ``02b_32`` does not. It exists
because ``02b_32`` is a four-panel figure at 11 x 8.4 inches with a legend,
which is unreadable projected, and the single most communicative result in the
project -- a celltype's queries changing hands -- deserves to survive being put
on a slide.

Two panels, 16:9, zoomed to the annexed region rather than the whole query map,
no axis furniture, counts printed on the panels. Everything else is in
``02b_32``.

**The crop is stated because a crop can mislead.** It is the bounding box of the
attacked celltype's clean predictions, padded 15%, and the caption gives the
fraction of the whole query set that falls inside it. Queries outside the crop
are unaffected at this dose, which ``02b_32`` shows and this figure cannot.
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
NAME = "02b_33_no_access"

READS: tuple[str, ...] = ("02b_30",)
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE

SOURCE, DOSE, SEL, PAD = "CL:0000623", 1.0, "hub", 0.15


def main() -> None:
    q = pd.read_parquet(TAB / "02b_30_no_access_per_query.parquet")
    u = pd.read_parquet(TAB / "02b_30_no_access_query_umap_xy.parquet")
    assert (q.retrieval_path == "hnsw").all(), "not the graph run"
    onto = Ontology.from_obo(BASE.parent / CFG["ontology"]["obo"])
    nm = lambda c: onto.id_to_name.get(str(c), str(c))  # noqa: E731

    g = q[(q.source.astype(str) == SOURCE) & (q.rate == DOSE)
          & (q.selection.astype(str) == SEL)]
    assert len(g) == len(u), f"{len(g)} rows for {len(u)} queries"
    tgt = str(g.target.iloc[0])
    m = g.merge(u[["query_id", "x", "y"]], on="query_id", validate="one_to_one")

    # Crop to the attacked celltype's CLEAN territory, padded. Stated in the
    # caption, because a crop chosen after seeing the result can flatter it.
    own = m[m.y_pred_clean.astype(str) == SOURCE]
    dx, dy = own.x.max() - own.x.min(), own.y.max() - own.y.min()
    x0, x1 = own.x.min() - PAD * dx, own.x.max() + PAD * dx
    y0, y1 = own.y.min() - PAD * dy, own.y.max() + PAD * dy
    cut = m[(m.x.between(x0, x1)) & (m.y.between(y0, y1))]

    keep = {SOURCE: f"{nm(SOURCE)} — the attacked celltype",
            tgt: f"{nm(tgt)} — the attacker's label"}
    frames = []
    for when, col in (("BEFORE", "y_pred_clean"), ("AFTER", "y_pred_poisoned")):
        frames.append(cut.assign(when=when,
                                 shown=cut[col].astype(str).map(keep)
                                 .fillna("anything else")))
    d = pd.concat(frames, ignore_index=True)
    lv = [keep[SOURCE], keep[tgt], "anything else"]
    d["shown"] = pd.Categorical(d.shown, lv, ordered=True)
    d["when"] = pd.Categorical(d.when, ["BEFORE", "AFTER"], ordered=True)
    d = d.sort_values("shown", ascending=False)

    lab = (d.groupby(["when", "shown"], observed=True).size().reset_index(name="n")
           .query("shown != 'anything else'"))
    lab["x"], lab["y"] = x0 + 0.03 * (x1 - x0), y1 - 0.06 * (y1 - y0)
    lab.loc[lab.shown == keep[tgt], "y"] = y1 - 0.14 * (y1 - y0)
    lab["txt"] = lab.n.map("{:,}".format) + " queries"

    plot = (
        p9.ggplot(d, p9.aes("x", "y", color="shown"))
        + p9.geom_point(size=1.5, alpha=0.75, stroke=0)
        + p9.geom_text(p9.aes(x="x", y="y", label="txt", color="shown"),
                       data=lab, ha="left", size=11, fontweight="bold",
                       inherit_aes=False, show_legend=False)
        + p9.facet_wrap("when", nrow=1)
        + p9.scale_color_manual(values={keep[SOURCE]: "#c44e52",
                                        keep[tgt]: "#55a868",
                                        "anything else": "#dcdcdc"})
        + p9.guides(color=p9.guide_legend(override_aes={"size": 6, "alpha": 1}))
        + p9.labs(x="", y="", color="")
        + p9.ggtitle(f"{nm(SOURCE)}'s queries change hands at parity")
        + biorag_theme()
        + p9.theme(figure_size=(13.33, 7.5), legend_position="bottom",
                   strip_text=p9.element_text(size=13),
                   plot_title=p9.element_text(size=19),
                   legend_text=p9.element_text(size=11),
                   axis_text=p9.element_blank(), axis_ticks=p9.element_blank(),
                   panel_grid=p9.element_blank())
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)
    (d.groupby(["when", "shown"], observed=True)
       .agg(n=("x", "size")).reset_index()).to_csv(TAB / f"{NAME}.csv", index=False)

    frac = len(cut) / len(m)
    b = lab[lab.when == "BEFORE"].set_index("shown").n
    a = lab[lab.when == "AFTER"].set_index("shown").n
    write_caption(
        FIG, NAME, f"{nm(SOURCE)}'s queries change hands at parity",
        f"{DS}; from 02b_30, relabel / {SEL} at dose {DOSE:g}, ONE seed, "
        f"SCimilarity's shipped HNSW index with poison INSERTED and the graph "
        f"re-queried at ef=100, k=50. Colour is the PREDICTION. 16:9 and cropped "
        f"for projection; the crop is the bounding box of {nm(SOURCE)}'s CLEAN "
        f"predictions padded {PAD:.0%}, holding {len(cut):,} of {len(m):,} "
        f"queries ({frac:.1%} of the set).",
        f"""A presentation cut of 02b_32, carrying no information that figure
        does not. It exists because 02b_32 is four panels and a legend at
        11 x 8.4 inches, which does not survive projection, and this is the most
        communicative single result in the project.

        Inside the crop, predictions of {nm(SOURCE)} fall from
        {int(b.get(keep[SOURCE], 0)):,} to {int(a.get(keep[SOURCE], 0)):,} while
        predictions of {nm(tgt)} rise from {int(b.get(keep[tgt], 0)):,} to
        {int(a.get(keep[tgt], 0)):,}.

        THE CROP IS STATED BECAUSE A CROP CAN MISLEAD. It was chosen as the
        attacked celltype's clean territory, not after seeing which region
        changed, and it holds {frac:.1%} of the query set. Queries outside it are
        essentially unaffected at this dose -- which is a real part of the result
        and which this figure, by construction, cannot show. 02b_32 shows it.

        Not a substitute for a rate. The counts here are inside a crop; the
        scored rates are 02b_01's and the per-celltype surface is 02b_27's.

        ONE REALISATION, per CLAUDE.md §1: multi-threaded add_items makes graph
        insertion non-deterministic, so which individual cells change is one
        draw, though the shape is robust.""")
    print(f"{NAME}: crop holds {len(cut):,} of {len(m):,} queries ({frac:.1%}) "
          f"-> pdf, png, csv, txt", file=sys.stderr)


if __name__ == "__main__":
    main()
