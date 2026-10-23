"""Step 02b_29 — the reference as shipped, before anyone touches it.

v01 drew this as ``fig16``. It is the only figure in the project that shows the
thing being attacked on its own terms: SCimilarity's annotation reference, 202
celltypes, with no poison anywhere in it. Every other UMAP in v02 uses this same
layout as a grey backdrop (``02b_09``), so this is the key to those panels.

**It is a subsample and that is a choice, not a limit.** 999,902 of the
7,913,892 records, drawn proportionally across all 202 celltypes with a floor of
one each, so a celltype with 40 records still appears. ``02b_08`` fits the UMAP
once on this subsample and every poison set in the family is projected into the
fixed result.

**``UNMAPPED`` is drawn, not dropped.** 26,085 records -- 2.6% of the subsample,
the sixth largest group -- carry a label that does not resolve to a Cell Ontology
id. They are part of the reference an attacker inherits and part of what any
k-NN vote can return, so hiding them would flatter the atlas.

The twelve largest celltypes are coloured and named; the remaining 190 are grey.
That is a rendering choice about legibility, not about importance: the grey is
78% of the records, and the attacked celltypes in this project are drawn from
both groups.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "1_scripts" / "00_common"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotnine as p9  # noqa: E402

from biorag_theme import biorag_theme, write_caption  # noqa: E402
from dataset_config import load  # noqa: E402
from ontology import Ontology  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[3]
CFG = load(BASE)
DS = CFG["dataset"]
STAGE = "02_b_attack_celltype"
NAME = "02b_29_no_access"

READS: tuple[str, ...] = ("02b_08",)
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE

#: Colour is a legibility budget, not a claim about importance.
N_NAMED = 12
#: Thinned for rendering only; the fit used all 999,902.
DRAW = 260_000
SEED = 0


def main() -> None:
    d = pd.read_parquet(TAB / "02b_08_no_access_umap_global_xy.parquet",
                        columns=["kind", "celltype", "x", "y"])
    a = d[d.kind.astype(str) == "atlas"].drop(columns="kind")
    assert len(a) == a[["x", "y"]].drop_duplicates().shape[0], "duplicate points"
    n_total, n_types = len(a), a.celltype.nunique()

    onto = Ontology.from_obo(BASE.parent / CFG["ontology"]["obo"])
    counts = a.celltype.value_counts()
    named = list(counts.head(N_NAMED).index)

    def pretty(ct: str) -> str:
        if ct == "UNMAPPED":
            return "UNMAPPED (no CL id)"
        return f"{onto.id_to_name.get(ct, ct)}"

    a = a.assign(grp=np.where(a.celltype.isin(named),
                              a.celltype.map(lambda c: pretty(str(c))),
                              f"other ({n_types - N_NAMED} celltypes)"))
    order = [pretty(str(c)) for c in named] + [f"other ({n_types - N_NAMED} celltypes)"]
    a["grp"] = pd.Categorical(a.grp, order, ordered=True)

    draw = (a.sample(DRAW, random_state=SEED) if len(a) > DRAW else a)
    # grey underneath, named on top, so a small coloured celltype is not buried
    draw = draw.sort_values("grp", ascending=False)

    pal = {g: c for g, c in zip(order[:-1], [
        "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860",
        "#da8bc3", "#8c8c8c", "#ccb974", "#64b5cd", "#1f6f8b", "#b5651d"])}
    pal[order[-1]] = "#e2e2e2"

    plot = (
        p9.ggplot(draw, p9.aes("x", "y", color="grp"))
        + p9.geom_point(size=0.22, alpha=0.55, stroke=0)
        + p9.scale_color_manual(values=pal)
        + p9.guides(color=p9.guide_legend(override_aes={"size": 4, "alpha": 1}))
        + p9.labs(x="UMAP 1", y="UMAP 2", color="")
        + p9.ggtitle("SCimilarity's reference as shipped — 202 celltypes, no poison")
        + biorag_theme()
        + p9.theme(figure_size=(10.2, 7.6), legend_position="right",
                   legend_text=p9.element_text(size=7),
                   axis_text=p9.element_blank(), axis_ticks=p9.element_blank())
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)

    out = (a.groupby("grp", observed=True)
             .agg(n_records=("x", "size"), x_centroid=("x", "mean"),
                  y_centroid=("y", "mean")).reset_index())
    out["share_of_subsample"] = out.n_records / n_total
    out.to_csv(TAB / f"{NAME}.csv", index=False)

    grey = int(counts.iloc[N_NAMED:].sum())
    unm = int(counts.get("UNMAPPED", 0))
    write_caption(
        FIG, NAME,
        "SCimilarity's reference as shipped — 202 celltypes, no poison",
        f"{DS} project, but the ATLAS is SCimilarity's, not this dataset's: "
        f"{n_total:,} of the 7,913,892 annotation records, drawn proportionally "
        f"across all {n_types} celltypes with a floor of one each so no celltype "
        f"vanishes. UMAP fitted once by 02b_08 on this subsample (cosine, "
        f"n_neighbors=15, min_dist=0.30, init=pca, random_state=0); the same "
        f"fixed layout is the grey backdrop of every 02b_09 panel. Thinned to "
        f"{DRAW:,} points for rendering, seed {SEED}. The {N_NAMED} largest "
        f"celltypes are coloured and named from cl.obo; the other "
        f"{n_types - N_NAMED} are grey.",
        f"""This is the thing being attacked, on its own terms and with nothing
        added. Every UMAP elsewhere in v02 uses this layout as a backdrop, so it
        is the key to those panels.

        The subsample is a choice rather than a limit. 02b_09's caption records
        the measurement: the full 7.9M fit is about 59 min and 42 GB of the 64
        available, so it is reachable; the proportional draw with a floor was
        taken because it renders legibly and preserves every celltype, not
        because the full fit was impossible.

        UNMAPPED is drawn rather than dropped. {unm:,} records -- {unm/n_total*100:.1f}%
        of the subsample and the sixth largest group -- carry a label that does
        not resolve to a Cell Ontology id. They are in the reference an attacker
        inherits and in what any k-NN vote can return. Dropping them would
        flatter the atlas and change the denominator of every rate computed on
        it.

        The grey is {grey/n_total*100:.0f}% of the records across
        {n_types - N_NAMED} celltypes. Colour here is a legibility budget, not a
        statement about importance -- the 26 celltypes this project attacks are
        drawn from both the coloured and the grey groups.

        NOT A RESULT. Nothing here is scored; it is geometry. The scored numbers
        are in 02b_01, 02b_02 and the summaries built from them.""")
    print(f"{NAME}: {n_total:,} atlas records, {n_types} celltypes, "
          f"{DRAW:,} drawn -> pdf, png, csv, txt", file=sys.stderr)


if __name__ == "__main__":
    main()
