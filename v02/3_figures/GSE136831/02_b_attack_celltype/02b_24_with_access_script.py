"""Step 02b_24 — what knowing the hub structure buys, and why withholding your
query logs does not defend you.

``02b_05`` gives three attackers the SAME budget and scores them on the same
held-out donors. ``oracle`` ranks the busiest records using every donor,
including the ones it is scored on -- v01's assumption. ``split`` ranks on donors
A and is scored on donors B, an attacker who watched SOMEONE's queries but not
yours. ``random`` draws records at random. The gaps are the finding, not the
levels.

**HUB SELECTION BUYS 182x ITS BUDGET IN RETRIEVAL TRAFFIC.** At a dose of 0.002
-- two thousandths of the celltype rewritten -- the oracle's edited records hold
36.5% of that celltype's top-50 slots. ``random`` holds 0.1%, which is its dose.
That is the mechanism the left panel exists to show: a k-NN reference is not
uniformly consulted, and the busiest 0.2% of a celltype answers a third of its
queries. Leverage falls as the budget grows -- 182x, 110x, 69x, 35x, 20x -- and
reaches 1x at parity, where there is nothing left to choose.

**AND IT TRANSFERS ACROSS DONORS.** ``split`` reaches 0.328 traffic captured at
dose 0.002 against the oracle's 0.365, and 0.809 against 1.000 at dose 0.05. An
attacker who never saw the victim's queries keeps roughly 90% of the effect,
because hub structure is a property of the REFERENCE, not of any one query set.
Withholding query logs is therefore not a defence. v01 asserted this in a
docstring and never wrote the number; this is the number.

**``random`` HAS A THRESHOLD AND THE HUB ATTACKS DO NOT.** Random net attack
success is exactly 0.000 at every dose through 0.1, 0.001 at 0.25, and only
becomes material at 0.5 (0.233). Oracle is already at 0.133 at 0.002. The same
effect costs random at least 250x the dose.

**SCOPE, AND IT IS THE TRAP THIS FIGURE IS DRAWN TO AVOID.** ``02b_05``'s own
summary table carries GLOBAL-scope ``accuracy`` and ``attack_success`` -- median
0.4620 and 0.0155 -- because a celltype is a small slice of 49,456 queries. The
within-celltype rates are 24x larger at low dose (0.133 against 0.005). Both
scopes are plotted in the middle panel for exactly that reason: the global curve
is the honest atlas-level number and the within curve is the honest victim-level
one, and quoting either as "the attack's effect" without saying which is the
error.
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
STAGE = "02_b_attack_celltype"
NAME = "02b_24_with_access"

#: 02b_05 is read for traffic_captured ONLY -- it is a slot share, not a
#: per-query rate, so it has no pooled analogue and appears nowhere else.
#: Every RATE comes from 02b_12, which publishes 02b_05's conditions pooled
#: across celltypes with a donor-paired interval. This figure does not
#: recompute a rate from the by_donor table, and must not: 02b_05's own
#: summary columns are GLOBAL scope and the within-scope rates it would need
#: already exist, aggregated correctly, one step downstream.
READS: tuple[str, ...] = ("02b_05", "02b_12")
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE

WITHIN = "within the attacked celltype"
GLOBAL = "over all 49,456 queries"
P_CAP = "top-50 slots held by edited records\n(median over 26 celltypes, IQR)"
P_NAS = "net attack success\n(pooled, 95% donor-paired CI)"
P_ACC = "accuracy of the attacked celltype\n(pooled, 95% donor-paired CI)"
PANELS = [P_CAP, P_NAS, P_ACC]


def main() -> None:
    # --- left panel: the mechanism. A slot SHARE, from the step's own table.
    m = pd.read_parquet(TAB / "02b_05_with_access_hub_split_donor.parquet")
    cap = (m.groupby(["selection", "rate"], observed=True).traffic_captured
             .agg(value="median", lo=lambda s: s.quantile(0.25),
                  hi=lambda s: s.quantile(0.75))
             .reset_index().assign(panel=P_CAP, scope=WITHIN,
                                   n=m.source.nunique()))

    # --- middle and right: rates, pooled, as published. 02b_12 carries one row
    # per split_seed; the three are replicates of the SAME ranking procedure on
    # different donor halves, so they collapse by median. Their spread is small
    # (<= 0.02 on net attack success) and is not the uncertainty being shown --
    # the ribbon is the donor-paired interval, which is the wider of the two.
    p = pd.read_parquet(TAB / "02b_12_both_pooled_summary.parquet")
    p = p[p.source_table.str.contains("02b_05")]
    assert len(p), "02b_12 holds no 02b_05 rows"

    def pooled(metric: str, panel: str, scope: str) -> pd.DataFrame:
        pre = "within" if scope == WITHIN else "global"
        q = p[p.metric == metric]
        return (q.groupby(["selection", "rate"], observed=True)
                 .agg(value=(f"{pre}_pooled", "median"),
                      lo=(f"{pre}_ci_lo", "median"), hi=(f"{pre}_ci_hi", "median"),
                      n=("n_sources", "max"))
                 .reset_index().assign(panel=panel, scope=scope))

    g = pd.concat([
        cap,
        pooled("net_attack_success", P_NAS, WITHIN),
        # The global curve is here to be read AGAINST the within one. It is not
        # a weaker version of the same number; it is a different denominator,
        # and the 24x gap at low dose is what the caption has to name.
        pooled("net_attack_success", P_NAS, GLOBAL),
        pooled("accuracy", P_ACC, WITHIN),
    ], ignore_index=True)
    g["panel"] = pd.Categorical(g.panel, PANELS)
    g["scope"] = pd.Categorical(g.scope, [WITHIN, GLOBAL])
    g["selection"] = pd.Categorical(g.selection.astype(str),
                                    ["oracle", "split", "random"])
    # One ribbon per curve. Without this the within and global ribbons of a
    # selection are one group and the shaded band spans the gap between them --
    # which is the very quantity the figure is distinguishing.
    g["grp"] = g.selection.astype(str) + "/" + g.scope.astype(str)

    # The headline is a RATIO of two committed columns, not a new statistic:
    # the slot share traffic_captured over the dose that bought it. Computed
    # here rather than typed into the title so it cannot drift from the table.
    lo = cap[(cap.selection.astype(str) == "oracle") & (cap.rate == cap.rate.min())]
    lev = float(lo.value.iloc[0] / lo.rate.iloc[0])
    title = (f"Hub selection buys {lev:.0f}x its budget, and it transfers "
             f"across donors")

    plot = (
        p9.ggplot(g, p9.aes("rate", "value", color="selection", fill="selection"))
        + p9.geom_ribbon(p9.aes(ymin="lo", ymax="hi", group="grp"),
                         alpha=0.12, color="none")
        + p9.geom_line(p9.aes(linetype="scope"), size=0.9)
        + p9.geom_point(size=1.3)
        + p9.facet_wrap("panel", nrow=1)
        + p9.scale_x_log10(breaks=[0.002, 0.01, 0.05, 0.25, 1.0],
                           labels=["0.002", "0.01", "0.05", "0.25", "1.0"])
        + p9.scale_y_continuous(limits=(0, 1))
        + p9.scale_color_cmap_d(cmap_name="viridis")
        + p9.scale_fill_cmap_d(cmap_name="viridis")
        + p9.scale_linetype_manual(values=["solid", "dashed"])
        + p9.labs(x="dose — records rewritten, as a fraction of the celltype (log)",
                  y="", color="which records\nthe attacker\nchose",
                  fill="which records\nthe attacker\nchose",
                  linetype="scored over")
        + p9.ggtitle(title)
        + biorag_theme()
        + p9.theme(figure_size=(11.0, 4.3), legend_position="right",
                   strip_text=p9.element_text(size=7),
                   axis_text_x=p9.element_text(size=7))
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)
    g.to_csv(TAB / f"{NAME}.csv", index=False)

    write_caption(
        FIG, NAME,
        title,
        f"{DS}; modify x relabel only -- labels rewritten, no embedding moved, so "
        f"the shipped index is byte-identical and the cached clean neighbours are "
        f"exact rather than approximate; 26 celltypes x 13 doses x 3 selections x "
        f"3 donor splits; LEFT panel is a slot share from 02b_05, median over 26 "
        f"celltypes with the IQR shaded; MIDDLE and RIGHT are rates from 02b_12, "
        f"pooled across celltypes under one donor draw, 2,000-replicate "
        f"donor-paired bootstrap over 78 donors, the three donor splits collapsed "
        f"by median. The two band types differ and the strip labels say which.",
        """Three attackers, identical budget, same scored donors. oracle ranks the
        busiest records using every donor including the scored ones; split ranks
        on donors A and is scored on donors B; random draws at random.

        The left panel is the mechanism. At dose 0.002 the oracle's edited
        records hold 36.5% of the celltype's top-50 slots while random holds
        0.1% -- which is exactly its dose, 0.002. A k-NN reference is not
        consulted uniformly: the busiest two thousandths of a celltype answer a
        third of its queries. Leverage over dose runs 182x, 110x, 69x, 35x, 20x
        and reaches 1x at parity, where there is nothing left to choose.

        It transfers across donors. split reaches 0.328 traffic captured at dose
        0.002 against the oracle's 0.365, and 0.809 against 1.000 at dose 0.05 --
        roughly 90% of the effect for an attacker who never saw the victim's
        queries. Hub structure is a property of the reference, not of a query
        set, so withholding query logs is not a defence. v01 asserted this in a
        docstring and wrote no table; this is the number.

        random has a threshold and the hub attacks do not. Random net attack
        success is exactly 0.000 at every dose through 0.1, 0.001 at 0.25, and
        first becomes material at 0.5 (0.233). Oracle is at 0.133 by 0.002. The
        same effect costs random at least 250x the dose, and the right panel
        shows the matching shape: random holds the celltype at its clean 0.465
        accuracy until dose 0.25 and then collapses.

        SCOPE. The dashed curves are the same conditions scored over all 49,456
        queries instead of the attacked celltype's own. They are 24x lower at
        dose 0.002 (0.005 against 0.133) and 17x lower at parity (0.030 against
        0.494), because one celltype is a small slice of the query set. 02b_05's
        own summary table reports the GLOBAL figures -- median accuracy 0.4620,
        attack_success 0.0155 -- so a reader who takes that column as "the
        attack's effect" understates the victim-level effect by more than an
        order of magnitude. Both are correct; neither may be quoted without its
        denominator. Both are drawn here for that reason.

        NOT MEASURED HERE: seed-to-seed stability of the poison draw. The three
        replicates vary the DONOR SPLIT used for ranking, not the sampling of
        which records to edit, so this figure fixes the shape of the capability
        ladder and not its variance under re-sampling.

        FLOOR EFFECT: 4 of the 26 attacked celltypes sit below 0.05 clean
        accuracy before any attacker acts, so part of the right panel's descent
        is not the attack's doing. 01a_07 draws the clean spread.""")
    print(f"{NAME}: {len(g)} plotted values, oracle leverage {lev:.1f}x at dose "
          f"{lo.rate.iloc[0]:g} -> pdf, png, csv, txt", file=sys.stderr)


if __name__ == "__main__":
    main()
