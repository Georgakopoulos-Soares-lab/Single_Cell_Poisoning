"""Step 03a_05 — the donor cap is a crossover, not a defence.

A donor cap truncates any one donor's contribution to a vote. Against an
attacker who concentrates their submissions in a few donor identities it is
the most effective single layer in this module; against one who splits across
many identities it is WORSE THAN NOTHING. The diagonal in this figure is where
a defence turns into an amplifier.

WHY. Genuine donors per top-50 run at a median of 11 (the step's
`genuine_donors_median`). A cap does not know which records are poisoned; it
truncates every donor alike. While the attacker uses fewer identities than the
cap allows, the cap bites them and spares the genuine majority. Once the
attacker's identities outnumber the genuine ones, each attacker identity sits
under the cap while the genuine donors -- numerous, each holding few records --
are the ones truncated.

Redrawn 2026-10-10 from the v3 table (RUNLOG Run 55): all 26 celltypes, the
FIXED accuracy (182d5c65; the old table's within_accuracy was the vote with the
poison removed), and duplicate at the attack grid's `concentrated` selection
(the old table built it at hub, a much stronger attack). Every number in the
caption is computed from the plotted frame. NOT a titration: 2 doses of the
14-rate ladder.

Cap 50 is the control: it must reproduce no cap.
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
STAGE = "03_a_defence_vs_insert"
NAME = "03a_05_no_access"

READS: tuple[str, ...] = ("03a_02",)
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE
TITLE = "The donor cap is a crossover, not a defence"


def main() -> None:
    #: The v3 table (Run 55): 26 celltypes, fixed accuracy, duplicate at
    #: concentrated. The untagged table is superseded.
    d = pd.read_parquet(TAB / "03a_02_no_access_donor_identity_all26_v3.parquet")
    d["donor_cap"] = d.donor_cap.astype(int)
    d["splits"] = d.splits.astype(int)
    g = (d.groupby(["donor_cap", "splits"], observed=True)
           .agg(value=("within_attack_success", "mean"),
                accuracy=("within_accuracy", "mean"),
                n=("within_attack_success", "size")).reset_index())
    g["value"] *= 100
    g["accuracy"] *= 100
    # The undefended column, carried as a per-row reference so the sign of the
    # cap's effect is in the table a reader inspects, not only in the colours.
    nocap = dict(zip(g[g.donor_cap == 0].splits, g[g.donor_cap == 0].value))
    g["undefended"] = g.splits.map(nocap)
    g["delta"] = g.value - g.undefended
    g["label"] = g.value.map(lambda v: f"{v:.0f}")
    # Ordered categoricals: the axes are ladders, not magnitudes, and equal
    # spacing keeps the crossover on the diagonal where it belongs.
    g["cap_f"] = pd.Categorical(g.donor_cap.map(
        lambda c: "no cap" if c == 0 else str(c)),
        ["no cap"] + [str(c) for c in sorted(g.donor_cap.unique()) if c], ordered=True)
    g["split_f"] = pd.Categorical(g.splits.astype(str),
                                  [str(s) for s in sorted(g.splits.unique())], ordered=True)

    plot = (
        p9.ggplot(g, p9.aes("cap_f", "split_f", fill="delta"))
        + p9.geom_tile(color="white", size=0.7)
        + p9.geom_text(p9.aes(label="label"), size=6.6, color="#111111")
        # Diverging scale centred on zero: the point is the SIGN of the cap's
        # effect, and a sequential ramp would hide where it flips.
        + p9.scale_fill_gradient2(low="#2c7bb6", mid="#f7f7f7", high="#d7191c",
                                  midpoint=0, name="change vs\nno cap (pp)")
        + p9.labs(x="defender's donor cap  (records kept per donor)",
                  y="attacker's donor identities  (submissions split across)")
        + p9.ggtitle(TITLE)
        + biorag_theme()
        + p9.theme(figure_size=(7.8, 5.0), legend_position="right")
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)
    g.to_csv(TAB / f"{NAME}.csv", index=False)

    def v(cap: int, sp: int) -> float:
        s = g[(g.donor_cap == cap) & (g.splits == sp)].value
        assert len(s) == 1, f"cap {cap} splits {sp}: {len(s)} rows"
        return float(s.iloc[0])

    ctrl = g[g.donor_cap == 50]
    ctrl_max = float((ctrl.value - ctrl.undefended).abs().max())
    acc_flat = float(g[g.donor_cap == 1].accuracy.std())
    n_cell = int(g.n.iloc[0])
    n_src = int(d.source.nunique())
    sels = ", ".join(f"{a} at {s}" for a, s in
                     d[["attack", "selection"]].drop_duplicates().itertuples(index=False))
    doses = " and ".join(f"{r:g}" for r in sorted(d.rate.unique()))
    gdm = float(d.genuine_donors_median.median())
    # Per (condition, split) pair: the cap's change against no cap, so "net
    # harmful" is a measured sign, not a claim carried over from the old table.
    k = ["source", "attack", "rate", "splits"]
    pv = d.pivot_table(index=k, columns="donor_cap", values="within_attack_success",
                       observed=True)
    def eff(cap: int) -> str:
        dd = (pv[cap] - pv[0]) * 100
        return (f"cap {cap} {dd.mean():+.2f} pp (hurts {int((dd > 0).sum())}, "
                f"helps {int((dd < 0).sum())})")
    caps = [int(c) for c in sorted(pv.columns) if 0 < c < 50]
    write_caption(
        FIG, NAME, TITLE,
        f"{DS}; {n_src} celltypes x 2 attacks x 2 doses = {n_cell} conditions "
        f"per cell, {len(caps) + 2} caps x {g.splits.nunique()} split counts. "
        f"Cell text is mean attack success (%); colour is the change against "
        f"the same row's no-cap value.",
        f"At 1 donor identity cap 1 takes attack success from "
        f"{v(0,1):.2f}% to {v(1,1):.2f}%. At 40 identities the SAME cap takes "
        f"it to {v(1,40):.2f}%, against {v(0,40):.2f}% undefended. Mean change "
        f"against no cap over all {len(pv)} (condition, split) pairs: "
        + "; ".join(eff(c) for c in caps) + ". "
        f"Control: cap 50 reproduces no cap to within {ctrl_max:.3f} pp on "
        f"every row. within_accuracy at cap 1 has standard deviation "
        f"{acc_flat:.4f} pp across the split counts. Genuine donors per top-50: "
        f"median {gdm:.0f}. Attacks: {sels}; doses {doses}; ONE seed. NOT a "
        f"titration (2 of the 14-rate ladder).")
    print(f"wrote {NAME}.pdf/.png/.txt and {NAME}.csv  ({len(g)} cells)")


if __name__ == "__main__":
    main()
