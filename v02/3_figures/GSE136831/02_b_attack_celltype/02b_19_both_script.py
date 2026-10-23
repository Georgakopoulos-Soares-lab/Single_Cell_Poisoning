"""Step 02b_19 — every attack type on one dose axis, pooled across the atlas.

The headline. Every other figure in this module is per celltype or per knob;
``02b_12`` pools each attack type across all 26 celltypes under ONE donor draw
held fixed, which is the only form in which "what does this attack do to the
atlas" has an interval rather than a spread of 26 numbers.

**OVERWRITING BEATS SUBMITTING BY TWO ORDERS OF MAGNITUDE AT LOW DOSE.**
``relabel_inplace`` -- rewrite the labels of existing records, move nothing --
reaches 0.119 net attack success at a dose of 0.002, two thousandths of the
celltype. ``relabel`` by SUBMISSION is 0.000 at the same dose and stays there
until 0.9, then jumps to 0.393 at parity. Insertion has to outvote the genuine
records; overwriting deletes their vote first, so it needs no majority.

**Insertion attacks have a threshold; modification does not.** relabel is flat
at zero across three orders of magnitude of dose and then turns on almost
vertically between 0.85 and 1.0. flood and duplicate rise gradually because they
concentrate, but relabel spread uniformly must reach parity before any
neighbourhood tips. That shape is what ``02b_10`` estimates as D_tau.

**Read shifted_inplace's flat line as a property of 02b_11, NOT of the attack.**
That sweep fixes ``ALPHA = 0.5``, where ``02b_15`` measures median retrieval at
0.0005 against 0.8738 at alpha 0.05 -- the records are not near anyone. At
alpha 0.05 the same construction reaches 0.4055. The 0.003 here is one setting
off the end of its own ladder, and pooling cannot know that.

Donor-paired bootstrap over 78 donors, 2,000 replicates, one draw held across
every celltype at once. That pairing is why this table exists separately from
``02b_01`` rather than being a groupby on it.
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
NAME = "02b_19_both"

READS: tuple[str, ...] = ("02b_12",)
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE

#: One curve per attack NAME, pooled over that name's selections and geometries.
#: The name is not the taxonomy -- SPEC.md 5.5 -- but it is what a reader asks
#: for first, and the axes are one layer down in 02b_03 and 02b_21.
SHOW = ("net_attack_success", "retrieval")


def main() -> None:
    d = pd.read_parquet(TAB / "02b_12_both_pooled_summary.parquet")
    d = d[d.metric.isin(SHOW)].copy()

    # ONLY DOSE SWEEPS. 02b_12 pools every donor table, including the ones that
    # hold dose fixed and vary something else -- 02b_15 sweeps alpha at dose 0.25
    # alone, 02b_14 sweeps sites at 0.25. Left in, they contribute a single point
    # that the max below then merges into another table's curve: shifted_inplace
    # spiked to 0.29 at dose 0.25 and returned to zero, because that one point
    # came from 02b_15 at its best alpha while every other dose came from 02b_11
    # at alpha 0.5. One curve, two experiments, and nothing on the page saying so.
    # This is the same guard 02b_10 needs for the same reason.
    doses = d.groupby("source_table").rate.nunique()
    keep = set(doses[doses >= 2].index)
    dropped = sorted(set(doses.index) - keep)
    if dropped:
        print(f"  not dose sweeps, excluded: {', '.join(dropped)}", file=sys.stderr)
    d = d[d.source_table.isin(keep)]

    # Pool the selections within an attack name by taking the STRONGEST, not the
    # mean. The question this figure answers is "what can this attack do", and a
    # mean over hub and uniform answers a question nobody asked -- 02b_05 already
    # showed those differ by ~840x, so averaging them describes no attacker.
    # Keep the row that WON, not just its value, so the CSV records which
    # selection each attack's curve came from. The two arms do not share a
    # selection vocabulary -- submission has hub/uniform, modification has
    # oracle/split/random -- so "the strongest" silently compares hub against
    # oracle. That is a defensible pairing, both being full hub knowledge, but
    # it has to be visible rather than assumed.
    idx = (d.groupby(["metric", "attack", "rate"], observed=True)
             .within_pooled.idxmax().dropna())
    g = (d.loc[idx, ["metric", "attack", "rate", "selection",
                     "within_pooled", "within_ci_lo", "within_ci_hi"]]
           .rename(columns={"within_pooled": "value", "within_ci_lo": "lo",
                            "within_ci_hi": "hi"})
           .reset_index(drop=True).dropna(subset=["value"]))
    g["metric"] = pd.Categorical(
        g.metric.map({"net_attack_success": "net attack success",
                      "retrieval": "poison reached the top-50"}),
        ["net attack success", "poison reached the top-50"])
    g["attack"] = g.attack.astype(str)

    # THE HEADLINE, COMPUTED. An earlier revision asserted "~100x" from
    # relabel_inplace/split (0.1197) over relabel/uniform (0.0000) -- two
    # DIFFERENT selections, and a division by zero dressed as a ratio. Since
    # selection is the dominant variable in this project (02a_04, 02b_24), a
    # cross-selection ratio attributes to ACCESS what selection did.
    lo_rate = g.rate.min()
    def at(attack: str) -> tuple[float, str]:
        r = g[(g.attack == attack) & (g.rate == lo_rate)
              & (g.metric == "net attack success")]
        return float(r.value.iloc[0]), str(r.selection.iloc[0])
    ins, ins_sel = at("relabel")
    inp, inp_sel = at("relabel_inplace")
    ratio = inp / ins
    title = (f"Overwriting beats submitting by ~{ratio:.0f}x at low dose, "
             f"at matched selection")

    plot = (
        p9.ggplot(g, p9.aes("rate", "value", color="attack", fill="attack"))
        + p9.geom_ribbon(p9.aes(ymin="lo", ymax="hi"), alpha=0.13, color="none")
        + p9.geom_line(size=0.9)
        + p9.geom_point(size=1.5)
        + p9.facet_wrap("metric", nrow=1)
        # Thinned breaks: the ladder crowds six rungs between 0.7 and 1.25 and
        # they overprint on a log axis.
        + p9.scale_x_log10(breaks=[0.002, 0.005, 0.01, 0.025, 0.05, 0.1,
                                   0.25, 0.5, 1.0],
                           labels=["0.002", "0.005", "0.01", "0.025", "0.05",
                                   "0.1", "0.25", "0.5", "1.0"])
        + p9.scale_y_continuous(limits=(0, 1))
        + p9.scale_color_cmap_d(cmap_name="viridis")
        + p9.scale_fill_cmap_d(cmap_name="viridis")
        + p9.labs(x="dose — poisoned records as a fraction of the attacked celltype (log)",
                  y="pooled over 26 celltypes, 95% donor-paired CI",
                  color="attack", fill="attack")
        + p9.ggtitle(title)
        + biorag_theme()
        + p9.theme(figure_size=(9.8, 4.2), legend_position="right",
                   axis_text_x=p9.element_text(angle=45, ha="right", size=6))
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)
    g.to_csv(TAB / f"{NAME}.csv", index=False)

    write_caption(
        FIG, NAME, title,
        f"{DS}; all seven attack names, 14 doses, pooled across 26 celltypes "
        f"with ONE donor draw held across every celltype at once; donor-paired "
        f"bootstrap over 78 donors, 2,000 replicates, 95% interval shaded. Where "
        f"an attack has several selections the STRONGEST is shown and the CSV "
        f"records which one won. NOTE the two arms do not share a selection "
        f"vocabulary: submission offers hub/uniform, modification offers "
        f"oracle/split/random. At dose {lo_rate:g} the curves shown are "
        f"relabel/{ins_sel} and relabel_inplace/{inp_sel} -- both full hub "
        f"knowledge, which is what makes the ratio an ACCESS comparison rather "
        f"than a selection one.",
        f"""At dose {lo_rate:g}, rewriting the labels of existing records reaches
        {inp:.4f} net attack success where submitting new ones reaches
        {ins:.4f} -- {ratio:.1f}x, at matched hub-style selection. Insertion has
        to outvote the genuine records; overwriting deletes their vote first and
        needs no majority.

        THE RATIO IS TAKEN AT MATCHED SELECTION, AND AN EARLIER REVISION OF THIS
        CAPTION DID NOT DO THAT. It compared relabel_inplace/split (0.1197)
        against relabel/uniform (0.0000) and called the result ~100x. Those are
        different selections, and the denominator was zero. Selection is the
        dominant variable in this project -- 02a_04 shows uniform poisoning inert
        at atlas scale whether it inserts or overwrites, and 02b_24 measures hub
        selection buying 182x its budget -- so a cross-selection ratio credits
        ACCESS with what selection did. Uniform against random, both arms sit at
        0.0000 at this dose.

        Insertion attacks have a threshold and modification does not. relabel
        climbs from {ins:.4f} to about 0.39 by dose 0.25 and then plateaus until
        parity; relabel_inplace is already at {inp:.4f} at dose {lo_rate:g} and
        saturates near 0.50 by 0.25. flood and duplicate climb gradually because
        they concentrate the budget rather than spreading it. That shape is what
        02b_10 estimates as D_tau.

        CAVEAT, and it is not small: shifted_inplace's flat line is a property
        of 02b_11, not of the attack. That sweep fixes ALPHA = 0.5, which 02b_18
        measures median retrieval at 0.0005, against 0.8738 at alpha 0.05 -- the
        records are not near anyone. At alpha 0.05 the same construction reaches
        0.4055. Pooling cannot know that; only the ladder in 02b_15 can.

        synthesised_inplace reaches the highest value here but it is CEILING
        tier: it writes vectors no count matrix produces, and a defender who
        re-derives embeddings from raw counts rejects them outright.""")
    print(f"{NAME}: {len(g)} plotted values -> pdf, png, csv, txt", file=sys.stderr)


if __name__ == "__main__":
    main()
