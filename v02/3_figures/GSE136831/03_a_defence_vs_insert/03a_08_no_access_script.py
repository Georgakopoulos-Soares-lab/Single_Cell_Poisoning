"""Step 03a_08 — reweighting the vote adds almost nothing, and the reason is measurable.

The negative that stops a reader spending effort in the wrong place. A natural
intuition about k-NN poisoning is that the vote should discount distant
neighbours: poison that squeaks into the top-50 would then carry less weight
than a genuine nearest neighbour. This measures whether that helps once the
record-property layers have already run, and it does not.

THE FIVE SCHEMES ARE WITHIN A FRACTION OF A POINT OF EACH OTHER on top of the
stack (the caption gives the span, computed from the plotted frame). softmax,
the most aggressive scheme, also costs the most accuracy. Both the full span
and softmax's cost are stated because the choice between them is the whole
question of whether weighting helps.

Redrawn 2026-10-10 from the v3 table (RUNLOG Run 55): the FIXED accuracy
(182d5c65; the old table's within_accuracy was the vote with the poison
removed) and duplicate at the attack grid's `concentrated` selection (the old
table built it at hub). Relabel, interpolate and flood stay at hub; the attack
grid's interpolate is uniform, so this interpolate is the old defence-step
attack. NOT a titration: one dose (1.0) on 6 celltypes.

THE MECHANISM IS IN THE SAME TABLE, which is why this is an explanation rather
than a shrug. `poison_rank1` -- the share of queries whose NEAREST neighbour is
poisoned -- is small on average and never large (the caption gives both). A distance weighting can only act on where the poison sits in the ranking, and
the poison is essentially never at the top. There is nothing for it to
down-weight, so it down-weights nothing.

k MATTERS FAR MORE THAN WEIGHT, and the contrast is the point of drawing the
controls on the same axis. Undefended, attack success changes by a large factor
between k=5 and k=50 (in the caption), far more than across every weighting
scheme in the stack. The y axis is logarithmic because the two
families are two orders of magnitude apart and a linear scale would flatten the
weighting family into a single line at the floor, hiding the spread this figure
exists to size.

WHY WEIGHTING IS MEASURED ONLY IN COMPOSITION. A standalone weighting arm would
score a configuration in which poison a record layer would have REJECTED is
still voting, merely with a smaller coefficient. That is not a defence anyone
would deploy, and it would flatter weighting by giving it the whole attack to
work against instead of the residue.
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
NAME = "03a_08_no_access"

READS: tuple[str, ...] = ("03a_03",)
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE
TITLE = "Reweighting the vote adds almost nothing, because the poison is never nearest"

#: Controls first, then the stack. Within the stack, uniform leads because it
#: is the "no weighting" reference every other scheme is judged against.
ORDER = ("none@k5", "none@k10", "none@k20", "none@k50",
         "stack+uniform", "stack+rank", "stack+score", "stack+cosine",
         "stack+softmax")
FAMILY = {True: "no defence, varying k  (control)",
          False: "on top of the stack, varying the weight"}


def main() -> None:
    #: The v3 table (Run 55); the untagged table is superseded.
    d = pd.read_parquet(TAB / "03a_03_no_access_vote_weighting_v3.parquet")
    sels = ", ".join(f"{a} at {s}" for a, s in
                     d[["attack", "selection"]].drop_duplicates().itertuples(index=False))
    n_src = int(d.source.nunique())
    d = d[d.config.isin(ORDER)].copy()
    d["value"] = d.within_attack_success * 100
    d["family"] = d.config.str.startswith("none@").map(FAMILY)
    d["config"] = pd.Categorical(d.config, ORDER, ordered=True)
    d["family"] = pd.Categorical(d.family, [FAMILY[True], FAMILY[False]], ordered=True)
    # log10 needs a positive floor: a condition with zero attack success is a
    # real measurement, not a gap, and dropping it would bias every box upward.
    floor = float(d[d.value > 0].value.min()) / 2
    d["plotted"] = d.value.clip(lower=floor)

    plot = (
        p9.ggplot(d, p9.aes("config", "plotted"))
        + p9.geom_boxplot(p9.aes(fill="family"), outlier_size=0.6, size=0.35)
        + p9.geom_jitter(width=0.14, size=0.7, alpha=0.4, stroke=0, color="#2b2a28")
        + p9.scale_y_log10()
        + p9.scale_fill_brewer(type="qual", palette="Set2", name="")
        + p9.labs(x="", y="attack success, % of the attacked celltype's queries  (log scale)")
        + p9.ggtitle(TITLE)
        + biorag_theme()
        + p9.theme(figure_size=(9.4, 4.8), legend_position="bottom",
                   axis_text_x=p9.element_text(angle=34, ha="right", size=7),
                   legend_title=p9.element_blank())
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)
    d[["config", "family", "source", "target", "attack", "rate", "k_used",
       "value", "plotted", "within_accuracy", "poison_rank1",
       "false_positive"]].to_csv(TAB / f"{NAME}.csv", index=False)

    def m(cfg: str, col: str = "value") -> float:
        s = d[d.config == cfg][col]
        assert len(s), f"{cfg}: no rows"
        return float(s.mean())

    stack = [c for c in ORDER if c.startswith("stack+")]
    lo, hi = min(m(c) for c in stack), max(m(c) for c in stack)
    n_cond = int((d.config == ORDER[0]).sum())
    write_caption(
        FIG, NAME, TITLE,
        f"{DS}; {n_src} celltypes x 4 attacks = {n_cond} conditions per box, boxes "
        f"span those conditions, points are the conditions themselves. "
        f"Zero-valued conditions are drawn at {floor:.4g}% so the log axis "
        f"keeps them; their true value is 0 and is in the sibling .csv.",
        f"The five weighting schemes span {hi-lo:.3f} pp -- rank "
        f"{m('stack+rank'):.3f}% to softmax {m('stack+softmax'):.3f}%, with "
        f"uniform (no weighting) at {m('stack+uniform'):.3f}%. The controls "
        f"span {m('none@k50')/m('none@k5'):.2f}x over the same conditions: "
        f"{m('none@k5'):.2f}% at k=5 to {m('none@k50'):.2f}% at k=50. "
        f"poison_rank1 -- the share of queries whose NEAREST neighbour is "
        f"poisoned -- averages {d.poison_rank1.mean():.4f} and peaks at "
        f"{d.poison_rank1.max():.4f}, which is why a distance weighting has "
        f"almost nothing to act on. softmax also costs the most accuracy: "
        f"{m('stack+softmax','within_accuracy')*100:.2f}% against "
        f"{m('stack+uniform','within_accuracy')*100:.2f}% for uniform. "
        f"Weighting is measured ONLY in composition with the stack; standalone "
        f"would score a configuration in which rejected poison still votes. "
        f"Attacks: {sels}; dose 1.0; ONE seed. NOT a titration.")
    print(f"wrote {NAME}.pdf/.png/.txt and {NAME}.csv  ({len(d)} rows)")


if __name__ == "__main__":
    main()
