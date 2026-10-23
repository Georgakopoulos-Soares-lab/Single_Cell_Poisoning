"""Step 02b_25 — at parity the vote ties, and the answer depends on the tie-break.

CLAUDE.md §7 requires it: "Ties are reported, never silently resolved. `vote()`
returns `n_tied` and `pivotal_in_tie`, and `bounds()` turns them into best/worst
columns... Publish the interval, not the point." ``02b_01`` has computed those
bounds for every condition since it was written. **Until 2026-09-28 no figure
drew them**, so the project computed the interval and published the point --
which is the thing the rule exists to prevent. v01 drew it as ``fig21``; this is
its replacement.

**The ambiguity is a knife-edge at exactly parity.** For ``relabel`` the
best-worst spread is 0.000 at every dose through 0.5, then 0.013, 0.039, 0.087,
and **0.258 at parity** -- where the reported 0.249 could be anywhere from 0.064
to 0.324 depending on a tie-break rule. At 1.25 it collapses back to 0.021.

That shape is the mechanism, not noise. At parity the attacker holds exactly as
many records as the genuine population, so the k-NN tally ties and the outcome
is decided by ``config.vote.tie_break`` rather than by the data. One rung above,
the poison outnumbers the genuine records and there is nothing left to tie.

**Only a tie CONTAINING the pivotal label is ambiguous** -- a tie among labels
that are all wrong is already determined -- which is why
``within_ambiguous_rate_success`` (0.258 at parity) rather than
``within_tie_rate`` (0.269) is the quantity that bounds the answer.

Across the whole grid 331 of 2,184 conditions, 15.2%, carry a spread wider than
0.05. Reporting the point alone understates what a defender or a reviewer can
conclude from any of them.
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
NAME = "02b_25_no_access"

#: 02b_01 carries the bounds. It is the hnsw run -- poison inserted into the
#: shipped index and the graph re-queried -- so no shortcut is involved.
READS: tuple[str, ...] = ("02b_01",)
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE


def main() -> None:
    d = pd.read_parquet(TAB / "02b_01_no_access_celltype_sweep_graph.parquet")
    assert (d.retrieval_path == "hnsw").all(), "not the graph run"

    d = d.assign(spread=d.within_net_best - d.within_net_worst,
                 attack=d.attack.astype(str))
    g = (d.groupby(["attack", "rate"], observed=True)
           .agg(point=("within_net_attack_success", "median"),
                worst=("within_net_worst", "median"),
                best=("within_net_best", "median"),
                spread=("spread", "median"),
                ambiguous=("within_ambiguous_rate_success", "median"),
                n=("spread", "size"))
           .reset_index())

    long = pd.concat([
        g.assign(panel="net attack success\nband = best to worst tie-break",
                 value=g.point, lo=g.worst, hi=g.best),
        # The driver, plotted beside the consequence. A tie only matters when the
        # pivotal label is IN it; a tie among labels that are all wrong is
        # already decided, so tie_rate alone would overstate the ambiguity.
        g.assign(panel="share of queries whose tie\nCONTAINS the attacker's label",
                 value=g.ambiguous, lo=g.ambiguous, hi=g.ambiguous),
    ])
    order = ["net attack success\nband = best to worst tie-break",
             "share of queries whose tie\nCONTAINS the attacker's label"]
    long["panel"] = pd.Categorical(long.panel, order)

    plot = (
        p9.ggplot(long, p9.aes("rate", "value", color="attack", fill="attack"))
        + p9.geom_vline(xintercept=1.0, color="#666666", linetype="dotted", size=0.5)
        # The BAND is the finding, so it is drawn with an edge and a vertical
        # range mark rather than as a faint fill. viridis is not used here: it
        # maps `relabel` -- the only series with a band -- to yellow, which is
        # the least legible colour on white and was invisible in the first
        # render.
        + p9.geom_ribbon(p9.aes(ymin="lo", ymax="hi"), alpha=0.30, color="none")
        + p9.geom_linerange(p9.aes(ymin="lo", ymax="hi"), size=1.6, alpha=0.55)
        + p9.geom_line(size=0.9)
        + p9.geom_point(size=1.6)
        + p9.facet_wrap("panel", nrow=1)
        + p9.scale_x_log10(breaks=[0.002, 0.01, 0.05, 0.25, 0.5, 1.0, 1.25],
                           labels=["0.002", "0.01", "0.05", "0.25", "0.5",
                                   "1.0", "1.25"])
        + p9.scale_y_continuous(limits=(0, 0.46))
        + p9.scale_color_manual(values={"relabel": "#7b2d8e",
                                        "duplicate": "#1f6f8b",
                                        "interpolate": "#c8632a"})
        + p9.scale_fill_manual(values={"relabel": "#7b2d8e",
                                       "duplicate": "#1f6f8b",
                                       "interpolate": "#c8632a"})
        + p9.labs(x="dose — poison records as a multiple of the attacked celltype (log); "
                    "dotted rule is parity",
                  y="median over 26 celltypes", color="attack", fill="attack")
        + p9.ggtitle("At parity the vote ties, and the tie-break decides the answer")
        + biorag_theme()
        + p9.theme(figure_size=(10.2, 4.3), legend_position="right",
                   strip_text=p9.element_text(size=7),
                   axis_text_x=p9.element_text(angle=45, ha="right", size=6))
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)
    g.to_csv(TAB / f"{NAME}.csv", index=False)

    rel = g[g.attack == "relabel"].set_index("rate")
    wide = int((d.spread > 0.05).sum())
    write_caption(
        FIG, NAME, "At parity the vote ties, and the tie-break decides the answer",
        f"{DS}; three insertion attacks, 26 celltypes, 14 doses, ONE seed; "
        f"SCimilarity's shipped HNSW index with poison INSERTED and the graph "
        f"re-queried at ef=100, k=50; the band is the best-to-worst interval "
        f"`bounds()` computes from `vote()`'s tie counts, NOT a confidence "
        f"interval -- it is the range the answer can take under different "
        f"tie-break rules, with the plotted line at the rule "
        f"config.vote.tie_break declares. Medians over 26 celltypes.",
        f"""CLAUDE.md 7 says ties are reported, never silently resolved, and that
        the interval is published rather than the point. 02b_01 has carried these
        bounds since it was written and no figure drew them until now; v01 drew
        it as fig21. The project was computing the interval and publishing the
        point.

        The ambiguity is a knife-edge at exactly parity. For relabel the
        best-worst spread is 0.000 at every dose through 0.5, then
        {rel.loc[0.8, 'spread']:.3f} at 0.8, {rel.loc[0.85, 'spread']:.3f} at
        0.85, {rel.loc[0.9, 'spread']:.3f} at 0.9 and
        {rel.loc[1.0, 'spread']:.3f} at parity -- where the reported
        {rel.loc[1.0, 'point']:.3f} could be anywhere from
        {rel.loc[1.0, 'worst']:.3f} to {rel.loc[1.0, 'best']:.3f}. One rung
        higher it collapses to {rel.loc[1.25, 'spread']:.3f}.

        That shape is the mechanism. At parity the attacker holds exactly as many
        records as the genuine population, the k-NN tally ties, and the outcome
        is decided by the tie-break rather than by the data. Above parity the
        poison outnumbers the genuine records and there is nothing left to tie.

        The right panel is why the band is not simply the tie rate. Only a tie
        CONTAINING the pivotal label is ambiguous -- a tie among labels that are
        all wrong is already determined -- so the bound follows
        within_ambiguous_rate_success ({rel.loc[1.0, 'ambiguous']:.3f} at parity)
        rather than within_tie_rate.

        Across the whole grid {wide} of {len(d):,} conditions, {wide/len(d)*100:.1f}%,
        carry a spread wider than 0.05, and the widest anywhere is
        {d.spread.max():.4f}. Any single-number claim about those conditions is
        narrower than the evidence supports.

        ONE SEED, and the band is not a confidence interval. Seed-to-seed
        variation is not measured here; this is the range the SAME data admits
        under a different arbitration rule.""")
    print(f"{NAME}: {len(g)} plotted values, {wide} conditions with spread > 0.05 "
          f"-> pdf, png, csv, txt", file=sys.stderr)


if __name__ == "__main__":
    main()
