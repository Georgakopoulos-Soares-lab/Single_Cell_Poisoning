"""Step 03a_10 — every layer's paired effect, with the donor interval that decides it.

`03a_04` and `03b_02` draw levels and distributions. This draws the quantity a
defender actually chooses on -- the change from no defence -- and says, per
condition, whether the 95% donor-paired interval excludes zero.

One point is one condition. Colour is the verdict:

  HELPS      the interval lies entirely below zero
  HARMS      the interval lies entirely above zero
  no effect  the interval contains zero, which includes every condition where
             the attack did nothing for the defence to remove

Reading left along each row is the effect size; reading the colour is whether
it survives the 78 donors being resampled.

WHAT THE INTERVALS ADD THAT THE POINT ESTIMATES COULD NOT.

  recompute against a LABEL-ONLY overwrite is not merely "not significant". It
  is exactly zero -- delta and both interval bounds zero -- because the layer
  admits the identical set of records and the two arms of the comparison are
  the same numbers. A degenerate interval rules out a small non-zero truth in
  a way a point estimate of 0 cannot. The caption counts these conditions.

  The verdict counts per layer and arm (helps / harms with the interval
  excluding zero) are in the caption, computed from the plotted frame.

Redrawn 2026-10-10 on the v3 interval tables (fixed accuracy, full dose
ladders). INSERT pools every v3 insert block -- relabel hub and uniform,
duplicate concentrated, interpolate uniform, and flood at ALL FIVE widths
(64-16,384 anchors), each condition a point, so no flood result is left out;
the caption also breaks stack@0.1's verdicts down by block. The keep controls
and the v3moduni repeat are not drawn. MODIFY/vector-moved pools v3modhub and
v3moduni; MODIFY/label-only is the cached_clean v3 re-run (RUNLOG Run 59).

THE PAIRING IS WHY THESE INTERVALS ARE USABLE AT ALL. Donors are resampled once
per condition and every configuration recomputed on that same draw, so the
difference keeps none of the variance the two arms share -- and they share
almost all of it, having been scored on one retrieval. Two independent
bootstraps and a subtraction would have produced intervals wide enough to call
everything here null.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "1_scripts" / "00_common"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotnine as p9  # noqa: E402
import yaml  # noqa: E402

from biorag_theme import BASELINE, biorag_theme, write_caption  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[3]
CFG = yaml.safe_load((BASE / "config" / "config.yaml").read_text())
DS = CFG["dataset"]
STAGE = "03_a_defence_vs_insert"
NAME = "03a_10_both"

READS: tuple[str, ...] = ("03a_09",)
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE
TITLE = "Every layer's paired effect, and whether 78 donors support it"
METRIC = "within_attack_success"

ARM_LABEL = {"insert": "INSERT\nattacker may only submit",
             "modify_graph": "MODIFY · vector moved",
             "modify_cached_clean": "MODIFY · label only"}
#: Layers in ladder order, bottom to top on the y axis.
LAYERS = ("stack@0.1", "label_consistency@0.1", "dedup", "donor_cap1",
          "recompute")
#: The v3 interval tables (Runs 52-54, 59). Attacks only: the keep controls
#: and the v3moduni repeat are deliberately absent.
SRC = tuple(f"03a_09_both_intervals_{t}.parquet" for t in (
    "v3relhub", "v3reluni", "v3dupcon", "v3intuni", "v3flo64", "v3flo256",
    "v3flo1024", "v3flo4096", "v3flo16384", "v3modhub", "v3moduni")) + (
    "03a_09_both_intervals_cached_clean_v3.parquet",)
VERDICT = {"helps": "#2c7bb6", "no effect": "#c8c7c2", "harms": "#d7191c"}


def main() -> None:
    d = pd.concat([pd.read_parquet(TAB / f) for f in SRC], ignore_index=True)
    # The cached_clean pair is tagged plain "v3"; name it by its block.
    d["block"] = np.where(d.arm == "modify_cached_clean", "cached_clean_v3",
                          d["tag"].astype(str))
    d = d[d.cfg.isin(LAYERS)].copy()
    lo, hi = d[f"{METRIC}_delta_lo"], d[f"{METRIC}_delta_hi"]
    d["verdict"] = np.where(hi < 0, "helps",
                            np.where(lo > 0, "harms", "no effect"))
    d["delta_pp"] = d[f"{METRIC}_delta"] * 100
    d["arm"] = pd.Categorical(d.arm.map(ARM_LABEL),
                              [ARM_LABEL[a] for a in ARM_LABEL], ordered=True)
    d["cfg"] = pd.Categorical(d.cfg, LAYERS, ordered=True)
    d["verdict"] = pd.Categorical(d.verdict, list(VERDICT), ordered=True)
    # Decided points drawn last, so a dense grey band cannot hide them.
    d = d.sort_values(["arm", "cfg", "verdict"])

    plot = (
        p9.ggplot(d, p9.aes("delta_pp", "cfg", color="verdict"))
        + p9.geom_vline(xintercept=0, color=BASELINE, size=0.45, linetype="dashed")
        + p9.geom_jitter(height=0.22, width=0, size=0.85, alpha=0.75, stroke=0)
        + p9.facet_wrap("arm", nrow=1)
        + p9.scale_color_manual(values=VERDICT, name="95% donor-paired interval")
        + p9.guides(color=p9.guide_legend(override_aes={"size": 3.4, "alpha": 1}))
        + p9.labs(x="change in attack success vs no defence, pp  (left is better)", y="")
        + p9.ggtitle(TITLE)
        + biorag_theme()
        + p9.theme(figure_size=(11.4, 4.4), legend_position="bottom",
                   strip_text=p9.element_text(size=7),
                   axis_text_y=p9.element_text(size=6.8))
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)
    d[["arm", "block", "cfg", "source", "attack", "rate", "selection", "geometry",
       "n_donors", METRIC, f"{METRIC}_delta", f"{METRIC}_delta_lo",
       f"{METRIC}_delta_hi", "verdict"]].to_csv(TAB / f"{NAME}.csv", index=False)

    def cnt(arm: str, cfg: str, v: str) -> int:
        return int(((d.arm == ARM_LABEL[arm]) & (d.cfg == cfg)
                    & (d.verdict == v)).sum())

    rc = d[(d.arm == ARM_LABEL["modify_cached_clean"]) & (d.cfg == "recompute")]
    degenerate = int(((rc[f"{METRIC}_delta"].abs() < 1e-12)
                      & (rc[f"{METRIC}_delta_lo"].abs() < 1e-12)
                      & (rc[f"{METRIC}_delta_hi"].abs() < 1e-12)).sum())

    def n(arm: str) -> int:
        return int(((d.arm == ARM_LABEL[arm]) & (d.cfg == "recompute")).sum())

    st = d[(d.arm == ARM_LABEL["insert"]) & (d.cfg == "stack@0.1")]
    by_block = "; ".join(
        f"{b} {int((x.verdict == 'helps').sum())}/{int((x.verdict == 'harms').sum())}"
        f" of {len(x)}" for b, x in st.groupby("block", observed=True))
    write_caption(
        FIG, NAME, TITLE,
        f"{DS}; one point per condition ({n('insert')} insert, "
        f"{n('modify_graph')} vector-moved, {n('modify_cached_clean')} "
        f"label-only per layer), jittered vertically only. Intervals are "
        f"{int(d.n_bootstrap.iloc[0]):,}-resample percentile bootstraps over "
        f"{int(d.n_donors.iloc[0])} donors, resampled ONCE per condition and "
        f"applied to every layer, so the difference is paired.",
        f"stack@0.1 on the insert arm: {cnt('insert','stack@0.1','helps')} of "
        f"{n('insert')} conditions helped with the interval excluding zero, "
        f"{cnt('insert','stack@0.1','harms')} harmed (helps/harms by block: "
        f"{by_block}). recompute against a LABEL-ONLY overwrite is exactly "
        f"zero on {degenerate} of {n('modify_cached_clean')} conditions -- "
        f"delta and both interval bounds all zero, because the layer admits "
        f"the identical record set -- against "
        f"{cnt('modify_graph','recompute','helps')} significant reductions of "
        f"{n('modify_graph')} on the vector-moved arm. donor_cap1 "
        f"significantly HARMS in {cnt('modify_graph','donor_cap1','harms')} "
        f"vector-moved and {cnt('modify_cached_clean','donor_cap1','harms')} "
        f"label-only conditions, against "
        f"{cnt('modify_graph','donor_cap1','helps')} and "
        f"{cnt('modify_cached_clean','donor_cap1','helps')} significant "
        f"reductions. 'no effect' includes every condition whose undefended "
        f"attack success was already zero. Point estimates are the values the "
        f"layer matrices report; 03a_09 asserts that before writing.")
    print(f"wrote {NAME}.pdf/.png/.txt and {NAME}.csv  ({len(d):,} points)")


if __name__ == "__main__":
    main()
