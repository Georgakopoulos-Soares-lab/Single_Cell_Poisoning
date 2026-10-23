"""Step 02b_22 — how much poison is enough, and how often the question has no answer.

``02b_10`` estimates D_tau: the smallest dose at which an attack's response
crosses tau, fitted by isotonic regression over the dose ladder with a donor
draw held fixed across every rung. This draws it for net attack success at five
levels of tau.

**BOTH PANELS OR NEITHER, and that is the whole design.** D_tau only exists for a
condition that actually crossed tau. A condition that never gets there is
RIGHT-CENSORED and has no threshold to plot, so a median over "the ones with a
threshold" is a median over a self-selected subset -- and the selection gets
harsher as tau rises.

Read the left panel alone and ``relabel_inplace`` looks like it gets CHEAPER as
the bar gets higher: median D_tau 0.252 at tau 0.01, 0.072 at 0.10, 0.023 at
0.50. That is backwards and it is survivorship. The right panel says why: at tau
0.01 only 4% of its conditions fail to reach the bar, at tau 0.50 it is 58%. By
then the median is taken over the 96 celltype-conditions where the attack was
strongest to begin with. The number is not wrong; read without its censoring it
means the opposite of what it appears to.

**What survives that caveat.** Submission needs roughly parity and modification
does not: ``relabel`` by insertion has median D_tau 0.785 to 0.967 across every
tau, while ``relabel_inplace`` sits at 0.023 to 0.263 -- a tenfold to fortyfold
gap in the dose required, and the same one ``02b_19`` shows as a ~13x gap in
effect at fixed dose. ``flood`` is the cheapest way to get a SMALL effect, D_tau
0.042 at tau 0.01, but the most expensive to push further: 0.825 at tau 0.50
with 83% censored.

``interpolate`` and ``shifted_inplace`` are 88-100% censored at every level.
Reported here as a null rather than omitted, because an attack that never
crosses 1% is a finding about the system's robustness and belongs beside the
ones that work.
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
NAME = "02b_22_both"

READS: tuple[str, ...] = ("02b_10",)
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE


def main() -> None:
    d = pd.read_parquet(TAB / "02b_10_both_threshold_dose.parquet")
    d = d[d.response == "net_attack_success"].copy()
    d["attack"] = d.attack.astype(str)

    # Two quantities per (attack, tau), and they must be read together.
    # d_tau is conditional on having crossed tau at all; censored is what
    # fraction never did. Reporting either alone misleads in a different way.
    g = (d.groupby(["attack", "tau"], observed=True)
           .apply(lambda s: pd.Series({
               "d_tau": s.loc[s.within_censor == "none", "within_d_tau"].median(),
               "n_with": int((s.within_censor == "none").sum()),
               "n_total": len(s),
               "censored": float((s.within_censor == "right").mean()),
           }), include_groups=False)
           .reset_index())

    long = pd.concat([
        g.assign(panel="median dose to reach tau\n(only conditions that DID)",
                 value=g.d_tau),
        g.assign(panel="fraction that NEVER reached tau\n(right-censored)",
                 value=g.censored),
    ])
    long["panel"] = pd.Categorical(
        long.panel, ["median dose to reach tau\n(only conditions that DID)",
                     "fraction that NEVER reached tau\n(right-censored)"])

    plot = (
        p9.ggplot(long, p9.aes("tau", "value", color="attack"))
        + p9.geom_line(size=0.9)
        + p9.geom_point(p9.aes(size="n_with"))
        + p9.facet_wrap("panel", nrow=1, scales="free_y")
        + p9.scale_x_log10(breaks=sorted(long.tau.unique()),
                           labels=[f"{t:g}" for t in sorted(long.tau.unique())])
        + p9.scale_size_continuous(range=(0.8, 4.0))
        + p9.scale_color_cmap_d(cmap_name="viridis")
        + p9.labs(x="tau — the level of net attack success being asked for (log)",
                  y="", color="attack",
                  size="conditions\nwith a\nthreshold")
        + p9.ggtitle("D_tau is conditional: read it with the censoring beside it")
        + biorag_theme()
        + p9.theme(figure_size=(10.4, 4.3), legend_position="right",
                   strip_text=p9.element_text(size=7),
                   axis_text_x=p9.element_text(size=7))
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)
    g.to_csv(TAB / f"{NAME}.csv", index=False)

    write_caption(
        FIG, NAME, "D_tau is conditional: read it with the censoring beside it",
        f"{DS}; net attack success only, seven attack names, five levels of tau, "
        f"pooled over 26 celltypes; D_tau fitted by isotonic regression over the "
        f"dose ladder with ONE donor draw held fixed across every rung; point "
        f"size is how many celltype-conditions had a threshold at all; a "
        f"condition that never crosses tau is RIGHT-CENSORED and contributes to "
        f"the right panel only.",
        """Both panels or neither. D_tau exists only for a condition that crossed
        tau, so a median over those is a median over a self-selected subset, and
        the selection gets harsher as tau rises.

        Read the left panel alone and relabel_inplace appears to get CHEAPER as
        the bar rises: 0.252 at tau 0.01, 0.072 at 0.10, 0.023 at 0.50. That is
        survivorship, not a finding. The right panel shows the censoring going
        4% -> 15% -> 58% over the same range; by tau 0.50 the median is taken
        over the 96 conditions where the attack was strongest to begin with.

        What survives the caveat: submission needs roughly parity and
        modification does not. relabel by insertion sits at 0.785-0.967 across
        every tau; relabel_inplace at 0.023-0.263. A tenfold to fortyfold gap in
        the dose required, matching the ~13x gap in effect at fixed dose, AT
        MATCHED SELECTION, that
        02b_19 shows.

        flood is cheapest for a SMALL effect -- D_tau 0.042 at tau 0.01 -- and
        dearest to push further: 0.825 at tau 0.50 with 83% censored.

        Seven (attack, tau) points are absent from the left panel: every
        condition was censored, so there is no threshold to plot. The gap is the
        answer, not missing data, and the right panel carries those points at
        1.00.

        interpolate and shifted_inplace are 88-100% censored at every level and
        are plotted as the nulls they are. shifted_inplace's null is a property
        of 02b_11's fixed ALPHA = 0.5, where 02b_15 measures median retrieval at
        0.0005; at alpha 0.05 retrieval is 0.8738 and the same construction
        reaches 0.4055 net attack success.""")
    print(f"{NAME}: {len(g)} plotted values -> pdf, png, csv, txt", file=sys.stderr)


if __name__ == "__main__":
    main()
