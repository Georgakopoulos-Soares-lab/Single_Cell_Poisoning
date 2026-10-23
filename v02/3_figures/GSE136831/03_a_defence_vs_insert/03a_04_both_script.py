"""Step 03a_04 — a defence works when the attacker forged the thing it checks.

The cross-arm contrast, and the module's headline. Three panels, one per threat
model, each showing what six defence layers do to the attacked celltype's own
queries. Read left to right: the attacker's power grows, and the set of layers
that still work shrinks to one, then to none.

INSERT. The attacker may only submit records. The stack takes attack success
to near zero.

MODIFY, vector moved. The attacker overwrites a record and writes a new
embedding. `recompute` still works, because a moved vector is reproducible from
no count matrix -- but `label_consistency` mostly fails and `donor_cap` does
not help.

MODIFY, label only. The attacker overwrites the LABEL and leaves the embedding
untouched. `recompute` equals no defence on every condition, because the
vector is genuine and re-encoding reproduces it (the caption reports the
measured maximum difference). Nothing here removes the attack.

Every number is in the caption, computed from the plotted frame.

Redrawn 2026-10-10 on the v3 tables (fixed accuracy, 182d5c65), full dose
ladders. INSERT pools one table per attack: relabel at hub and uniform,
duplicate at concentrated, interpolate at uniform, flood at 1024 anchors (one
width, so each attack counts once; all five widths are in 03a_10). The keep
controls are not attacks and are left out. MODIFY/vector-moved pools
v3modhub and v3moduni; MODIFY/label-only is the cached_clean v3 re-run.

So the axis a defence result varies along is NOT access, which is how this was
framed in v01. It is WHAT THE ATTACKER FORGED. `recompute` checks vector
authenticity and is therefore exactly as useful as the attacker's need to forge
a vector: total against a flood, total against a moved record, and worth
precisely zero against a relabel, in both arms, at both access levels. The two
zero columns in this figure are the same fact seen twice.

SPREAD, NOT INTERVALS, AND DELIBERATELY SO. Each box is the distribution over
that arm's conditions. The layer matrices carry no bootstrap columns, and a
figure that computes a statistic instead of reading one is a violation
(CLAUDE.md 7), so this panel says nothing about precision -- only about
dispersion, which is what a box is for.

The intervals now EXIST: `03a_09` computes them from the `_by_donor` twins and
`03a_10` draws them. They are a different figure rather than an addition to
this one, because they answer a different question. This figure asks what the
attack success IS under each layer; `03a_10` asks whether the CHANGE from no
defence survives resampling the 78 donors. Overlaying a paired-difference
interval on a plot of levels would invite reading it as an interval on the
levels, which it is not.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "1_scripts" / "00_common"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotnine as p9  # noqa: E402
import yaml  # noqa: E402

from biorag_theme import biorag_theme, write_caption  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[3]
CFG = yaml.safe_load((BASE / "config" / "config.yaml").read_text())
DS = CFG["dataset"]
STAGE = "03_a_defence_vs_insert"
NAME = "03a_04_both"

#: Both layer matrices. The contrast IS the figure, so it reads one step per
#: arm and check_provenance holds it to exactly those two.
READS: tuple[str, ...] = ("03a_01", "03b_01")
FIG = BASE / "3_figures" / DS / STAGE
TAB = BASE / "2_results" / DS / STAGE
TITLE = "A defence works when the attacker forged the thing it checks"

#: The six layers every arm measures, in ladder order: nothing, then each
#: single mechanism, then the stack. label_consistency and stack are shown at
#: the 10% budget -- the most generous, and the one that makes the modify-arm
#: null hardest to dismiss as an under-tuned defence.
LAYERS = ("none", "recompute", "dedup", "donor_cap1",
          "label_consistency@0.1", "stack@0.1")
PRETTY = {"none": "no defence", "recompute": "recompute\n(vector authentic?)",
          "dedup": "dedup\n(multiplicity)", "donor_cap1": "donor cap 1\n(identity volume)",
          "label_consistency@0.1": "label consistency\n(vector–label pair)",
          "stack@0.1": "stack\n(all four)"}
_IN = "03a_01_no_access_layer_matrix_{}.parquet"
ARMS = (("03_a_defence_vs_insert",
         tuple(_IN.format(t) for t in
               ("v3relhub", "v3reluni", "v3dupcon", "v3intuni", "v3flo1024")),
         "INSERT\nattacker may only submit"),
        ("03_b_defence_vs_modify",
         ("03b_01_with_access_layer_matrix_graph_v3modhub.parquet",
          "03b_01_with_access_layer_matrix_graph_v3moduni.parquet"),
         "MODIFY · vector moved\nattacker overwrites the embedding"),
        ("03_b_defence_vs_modify",
         ("03b_01_with_access_layer_matrix_cached_clean_v3.parquet",),
         "MODIFY · label only\nembedding left untouched"))


def load(stage: str, fnames: tuple[str, ...], arm: str) -> pd.DataFrame:
    """One arm's layer matrix, reduced to the six shared layers.

    Returns: long frame, one row per (condition, layer), attack success in %.
    """
    d = pd.concat([pd.read_parquet(BASE / "2_results" / DS / stage / f)
                   for f in fnames], ignore_index=True)
    d["cfg"] = np.where(d.lc_budget.notna(),
                        d.defense.astype(str) + "@" + d.lc_budget.astype(str),
                        d.defense.astype(str))
    d = d[d.cfg.isin(LAYERS)].copy()
    d["arm"] = arm
    d["value"] = d.within_attack_success * 100
    return d[["arm", "cfg", "source", "attack", "selection", "rate", "value"]]


def main() -> None:
    g = pd.concat([load(s, f, a) for s, f, a in ARMS], ignore_index=True)
    g["cfg"] = pd.Categorical([PRETTY[c] for c in g.cfg],
                              [PRETTY[c] for c in LAYERS], ordered=True)
    g["arm"] = pd.Categorical(g.arm, [a for _, _, a in ARMS], ordered=True)

    plot = (
        p9.ggplot(g, p9.aes("cfg", "value"))
        + p9.geom_boxplot(p9.aes(fill="cfg"), outlier_size=0.5, size=0.35)
        + p9.facet_wrap("arm", nrow=1)
        + p9.scale_fill_brewer(type="qual", palette="Set2", guide=None)
        + p9.scale_y_continuous(limits=(0, 100), breaks=range(0, 101, 20))
        + p9.labs(x="", y="attack success, % of the attacked celltype's queries")
        + p9.ggtitle(TITLE)
        + biorag_theme()
        + p9.theme(figure_size=(11.2, 4.9), legend_position="none",
                   axis_text_x=p9.element_text(angle=38, ha="right", size=6.4),
                   strip_text=p9.element_text(size=7.4))
    )
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plot.save(FIG / f"{NAME}.{ext}", verbose=False)
    g.to_csv(TAB / f"{NAME}.csv", index=False)

    # Every number in the caption is looked up from the frame just plotted, so
    # a rerun on changed tables fails loudly instead of shipping a stale claim.
    def m(arm_i: int, layer: str) -> float:
        s = g[(g.arm == ARMS[arm_i][2]) & (g.cfg == PRETTY[layer])].value
        assert len(s), f"{arm_i}/{layer}: no rows"
        return float(s.mean())

    def cover(arm_i: int) -> str:
        x = g[(g.arm == ARMS[arm_i][2]) & (g.cfg == PRETTY["none"])]
        combos = x[["attack", "selection"]].drop_duplicates()
        return (f"{len(x)} conditions ({len(combos)} attack/selection pairs x "
                f"{x.rate.nunique()} doses x {x.source.nunique()} celltypes)")

    def net(arm_i: int, layer: str) -> str:
        k = ["source", "attack", "selection", "rate"]
        x = g[g.arm == ARMS[arm_i][2]]
        base = x[x.cfg == PRETTY["none"]].set_index(k).value
        lay = x[x.cfg == PRETTY[layer]].set_index(k).value
        dd = (lay - base).dropna()
        return (f"helps {int((dd < 0).sum())}, hurts {int((dd > 0).sum())} "
                f"of {len(dd)}")

    k = ["source", "attack", "selection", "rate"]
    lo = g[g.arm == ARMS[2][2]]
    rc_max = float((lo[lo.cfg == PRETTY["recompute"]].set_index(k).value
                    - lo[lo.cfg == PRETTY["none"]].set_index(k).value).abs().max())
    write_caption(
        FIG, NAME, TITLE,
        f"{DS}; boxes span each arm's conditions, whiskers 1.5 IQR. Attack "
        f"success is the share of the ATTACKED celltype's own queries whose "
        f"vote returns the attacker's label.",
        f"Means, no defence -> stack: INSERT {m(0,'none'):.2f}% -> "
        f"{m(0,'stack@0.1'):.2f}% (stack {net(0,'stack@0.1')}); "
        f"MODIFY/vector-moved {m(1,'none'):.2f}% -> {m(1,'stack@0.1'):.2f}%; "
        f"MODIFY/label-only {m(2,'none'):.2f}% -> {m(2,'stack@0.1'):.2f}%. "
        f"recompute alone leaves the label-only arm at {m(2,'recompute'):.2f}% "
        f"against {m(2,'none'):.2f}% undefended, max |difference| over its "
        f"conditions {rc_max:.6f} pp. In the vector-moved arm the same layer "
        f"reaches {m(1,'recompute'):.2f}% from {m(1,'none'):.2f}%. donor cap 1: "
        f"vector-moved {net(1,'donor_cap1')}, label-only {net(2,'donor_cap1')}. "
        f"INSERT: {cover(0)}; relabel hub+uniform, duplicate concentrated, "
        f"interpolate uniform, flood 1024 anchors. MODIFY/vector-moved: "
        f"{cover(1)}. MODIFY/label-only: {cover(2)}. ONE seed. Graph-path "
        f"absolute levels carry a single-condition spread up to ~0.1 at high "
        f"dose (RUNLOG Run 56); the comparisons between layers within a "
        f"condition do not, because every layer reads that condition's one "
        f"neighbour list.")
    print(f"wrote {NAME}.pdf/.png/.txt and {NAME}.csv  ({len(g):,} rows)")


if __name__ == "__main__":
    main()
