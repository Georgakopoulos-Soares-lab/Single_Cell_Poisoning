"""Step 02a_04 — the atlas-scale contrast: submitting versus overwriting, one
axis apart, and the retrieval/outcome dissociation at its widest.

``02a_01`` submits new records spread uniformly over the whole 7.9M-record
reference and re-queries the shipped HNSW graph. ``02a_03`` rewrites the labels
of existing records at the same six doses on the same 49,456 queries. Geometry is
``unmoved`` and the label change is ``relabel`` in both, so with selection held
at ``uniform`` the two differ in exactly one axis: whether the attacker may
overwrite or may only submit. ``02a_03`` adds ``hub`` selection as the third arm.

**POISON IN 99.6% OF TOP-50s AND THE ATLAS DOES NOT MOVE.** At a dose of 0.1 --
791,389 poisoned records, a tenth of the reference -- uniform submission puts
poison into the top-50 of 99.55% of queries and changes accuracy by -0.00137.
Uniform overwriting reaches 99.45% retrieval for -0.00003. Neither is
distinguishable from zero: across the six doses and five draws, 8 of 30
submission draws and 11 of 30 overwriting draws move accuracy the WRONG way, up.
The poison is retrieved, and then outvoted, by the 49 genuine neighbours beside
it.

This is the project's central distinction measured at its widest, on a real
7.9M-record reference: a retrieval detector fires on essentially every query, an
outcome detector on none. A benchmark reporting only one of the two cannot tell a
defence designer which to build.

**AT ATLAS SCALE, SELECTION DECIDES THE OUTCOME AND ACCESS DOES NOT.** Write
access at the same budget and the same uniform selection buys nothing -- -0.00003
against -0.00137, both noise. Pointed at hubs it takes atlas accuracy from 0.4637
to 0.3096 at a dose of 0.001 and to 0.0305 at 0.02. The three-order-of-magnitude
gap is between hub and uniform, not between overwrite and submit.

**This does not contradict ``02b_19``, and the difference is the scope.** There
overwriting beat submitting by ~13x at matched selection, because a
celltype-scale attack concentrates
its whole budget on one celltype's neighbourhood by construction -- it is already
a targeted attack, and overwriting spends the budget better than submitting does.
Spread over the atlas, neither is targeted and neither works. Concentration is
the operative variable in both figures; at celltype scale the sweep design
supplies it and at atlas scale the attacker must choose it.
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
STAGE = "02_a_attack_atlas"
NAME = "02a_04_both"

READS: tuple[str, ...] = ("02a_01", "02a_03")
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE

SUBMIT = "submit / uniform"
OVER_U = "overwrite / uniform"
OVER_H = "overwrite / hub"
ARMS = [OVER_H, SUBMIT, OVER_U]

P_RET = "poison reached the top-50"
P_ACC = "change in atlas accuracy"
P_ZOOM = "the same, uniform arms only\n— NOTE THE AXIS —"
PANELS = [P_RET, P_ACC, P_ZOOM]


def main() -> None:
    a = pd.read_parquet(TAB / "02a_01_no_access_uniform_graph.parquet")
    b = pd.read_parquet(TAB / "02a_03_with_access_relabel_inplace.parquet")

    # One axis apart only if the shared axes really are shared. 02a_01 is the
    # hnsw run -- poison inserted with add_items and the graph re-queried --
    # and 02a_03 is cached_clean, which for a LABEL-ONLY edit is exact rather
    # than approximate: no embedding moves, so labelled_kNN.bin is
    # byte-identical and the cached neighbours are the ones the graph returns.
    # Both are faithful; they are faithful by different routes.
    assert (a.retrieval_path == "hnsw").all(), "02a_01 is not the graph run"
    assert (b.retrieval_path == "cached_clean").all(), "02a_03 changed path"
    for col in ("geometry", "label_change", "dose_basis", "n_reference",
                "n_queries_global"):
        assert a[col].nunique() == 1 and b[col].nunique() == 1, col
        assert a[col].iloc[0] == b[col].iloc[0], f"{col} differs between arms"
    base = float(a.baseline_accuracy.iloc[0])
    assert abs(base - float(b.baseline_accuracy.iloc[0])) < 1e-6, "baselines differ"

    d = pd.concat([a.assign(arm=SUBMIT),
                   b.assign(arm="overwrite / " + b.selection.astype(str))])
    assert set(d.arm) == set(ARMS), sorted(set(d.arm))

    # Band is the MIN-MAX over the five independent draws, not a bootstrap
    # interval. 02a_01 publishes a donor-paired CI on ABSOLUTE accuracy -- about
    # [0.428, 0.501] -- which is fifty times the whole effect being measured,
    # because donors differ enormously in baseline accuracy. The paired delta is
    # the quantity with power here, and the draw spread is the spread that
    # belongs beside it.
    def band(metric: str, panel: str, arms: list[str]) -> pd.DataFrame:
        s = d[d.arm.isin(arms)]
        return (s.groupby(["arm", "rate"], observed=True)[metric]
                 .agg(value="mean", lo="min", hi="max")
                 .reset_index().assign(panel=panel, metric=metric,
                                       n_draws=s.draw.nunique()))

    g = pd.concat([
        band("retrieval", P_RET, ARMS),
        band("accuracy_delta", P_ACC, ARMS),
        # The zoom is where the null is legible. At the full scale the uniform
        # arms are a flat line on zero and a reader cannot see that their draws
        # fall on BOTH sides of it, which is the finding.
        band("accuracy_delta", P_ZOOM, [SUBMIT, OVER_U]),
    ], ignore_index=True)
    g["panel"] = pd.Categorical(g.panel, PANELS)
    g["arm"] = pd.Categorical(g.arm, ARMS)

    zero = pd.DataFrame({"panel": pd.Categorical([P_ACC, P_ZOOM], PANELS),
                         "y": [0.0, 0.0]})

    plot = (
        p9.ggplot(g, p9.aes("rate", "value", color="arm", fill="arm"))
        + p9.geom_hline(p9.aes(yintercept="y"), data=zero,
                        linetype="dashed", color="#666666", size=0.4)
        + p9.geom_ribbon(p9.aes(ymin="lo", ymax="hi"), alpha=0.16, color="none")
        + p9.geom_line(size=0.9)
        + p9.geom_point(size=1.5)
        + p9.facet_wrap("panel", nrow=1, scales="free_y")
        + p9.scale_x_log10(breaks=sorted(g.rate.unique()),
                           labels=[f"{r:g}" for r in sorted(g.rate.unique())])
        + p9.scale_color_cmap_d(cmap_name="viridis")
        + p9.scale_fill_cmap_d(cmap_name="viridis")
        + p9.labs(x="dose — poisoned records as a fraction of the 7.9M-record reference (log)",
                  y="mean of 5 draws, full draw range shaded",
                  color="access / selection", fill="access / selection")
        + p9.ggtitle("Poison in 99.6% of top-50s, atlas accuracy unchanged — "
                     "unless the attacker picks hubs")
        + biorag_theme()
        + p9.theme(figure_size=(11.2, 4.3), legend_position="right",
                   strip_text=p9.element_text(size=7),
                   axis_text_x=p9.element_text(angle=45, ha="right", size=6))
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)
    g.to_csv(TAB / f"{NAME}.csv", index=False)

    wrong = {arm: int((s.accuracy_delta >= 0).sum())
             for arm, s in d.groupby("arm", observed=True)}
    write_caption(
        FIG, NAME,
        "Poison in 99.6% of top-50s, atlas accuracy unchanged — unless the "
        "attacker picks hubs",
        f"{DS}; geometry unmoved and label_change relabel throughout, so with "
        f"selection held at uniform the two threat models differ in ONE axis; six "
        f"doses on a 7,913,892-record reference, 49,456 queries, five independent "
        f"draws; clean baseline accuracy {base:.4f}; lines are the mean over draws "
        f"with the full draw range shaded. 02a_01 is the hnsw run -- poison "
        f"inserted with add_items and the graph re-queried. 02a_03 is "
        f"cached_clean, which for a label-only edit is EXACT, not approximate: no "
        f"embedding moves, so labelled_kNN.bin is byte-identical. Both arms are "
        f"faithful to the graph by different routes.",
        f"""At a dose of 0.1 -- 791,389 poisoned records, a tenth of the reference
        -- uniform submission puts poison into the top-50 of 99.55% of queries and
        changes accuracy by -0.00137. Uniform overwriting reaches 99.45%
        retrieval for -0.00003.

        Neither is distinguishable from zero. Across six doses and five draws,
        {wrong[SUBMIT]} of 30 submission draws and {wrong[OVER_U]} of 30
        overwriting draws move accuracy the WRONG way, up. The zoom panel is
        drawn at the 1e-3 scale so that the bands straddling zero are visible;
        at the scale of the middle panel they are a flat line and the null
        cannot be read. Poison is retrieved and then outvoted by the 49 genuine
        neighbours beside it.

        This is the retrieval/outcome distinction at its widest, on a real
        7.9M-record reference: a detector watching retrieval fires on
        essentially every query, one watching outcomes on none. A benchmark
        reporting only one of the two cannot tell a defence designer which to
        build.

        Selection decides the outcome and access does not. Write access at the
        same budget and the same uniform selection buys nothing. Pointed at hubs
        it takes atlas accuracy from {base:.4f} to 0.3096 at a dose of 0.001 --
        7,914 records, one thousandth of the reference -- and to 0.0305 at 0.02.
        The three-order-of-magnitude gap is between hub and uniform, not between
        overwrite and submit.

        This does not contradict 02b_19, where overwriting beat submitting by
        ~13x at matched selection. That sweep is celltype-scale: it spends the whole
        budget inside
        one celltype's neighbourhood by construction, so it is already targeted,
        and overwriting spends a targeted budget better than submitting does.
        Spread over the atlas neither arm is targeted and neither works.
        Concentration is the operative variable in both; at celltype scale the
        sweep design supplies it, at atlas scale the attacker must choose it.

        WHY THE BAND IS A DRAW RANGE AND NOT A CONFIDENCE INTERVAL: 02a_01
        publishes a donor-paired interval on ABSOLUTE accuracy of about [0.428,
        0.501], roughly fifty times the entire effect, because donors differ
        enormously in baseline accuracy. The paired delta is the quantity with
        power against a uniform attack, and the spread over draws is the spread
        that belongs beside it.

        hub retrieval carries no visible band because hub selection is
        deterministic -- the busiest records are the same records in every draw
        -- so all five agree exactly. Only the labels drawn for them vary, which
        is what the accuracy band shows.

        NOT MEASURED HERE: hub selection under NO write access. 02a_01 sweeps
        uniform submission only, so the fourth cell of the access x selection
        grid is absent and this figure cannot say whether a submitting attacker
        who aims at hubs does better than one who spreads out.""")
    print(f"{NAME}: {len(g)} plotted values, baseline {base:.4f}, "
          f"draws moving the wrong way {wrong} -> pdf, png, csv, txt",
          file=sys.stderr)


if __name__ == "__main__":
    main()
