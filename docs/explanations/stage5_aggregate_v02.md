# Stage 5 — `aggregate`, in v02

**Status: the concept moved from one rule into the step that needs it.** There
is no single `summary.parquet` in v02. The `aggregate` rule exists in the
Snakefile and has never run.

## Why the single summary table went away

v01 swept one grid, so one aggregation served every figure. v02's steps each
bring their own grid — seven spreads × six jitters, nine alphas, thirteen doses,
six repeats — and there is no shared shape to aggregate into. Each step writes
its own table under `v02/2_results/<DATASET>/<module>/`, and `CLAUDE.md` §5's
rule ("every figure derivable from `summary.parquet` alone") is satisfied in v02
by each figure declaring what it reads in a `READS` tuple that
`check_provenance.py` verifies.

That declaration is the replacement for the single table. A figure's number has
exactly one origin and the check proves the link, rather than the convention
resting on a filename.

## The bootstrap is donor-paired, and this is not a detail

**78 donors are the unit of independence, not 49,456 cells.** Quoting an
interval as if cells were independent is a prohibited claim (SPEC.md §12).

`donor_ratio_ci` resamples donors and returns `n.sum()/d.sum()` — the ratio of
summed numerators to summed denominators over the resampled donors, not the mean
of per-donor ratios. Those differ whenever donors contribute unequal numbers of
cells, which they do.

**One donor draw is held fixed across every condition in a comparison.** This is
why `02b_12` exists as a separate table rather than a groupby on `02b_01`:
pooling 26 celltypes with a shared draw is a different quantity from pooling
them with independent draws, and only the first supports a paired statement
about the atlas.

## The tables that carry intervals

These hold donor-paired bootstrap intervals that **cannot** be derived from an
already-aggregated summary, because aggregation has passed the donor:

    global_summary   source_summary   pair_summary
    accuracy_summary   clean_baseline   clean_confusion

A figure needing a column none of them has is a spec change — escalate, do not
recompute. A figure script that recomputes a statistic instead of reading it is
a violation (`CLAUDE.md` §7), and it is the violation that breaks the "one
number, one origin" property the whole provenance chain rests on.

## Reporting rules that bind this stage

**Report negative and null results as prominently as positive ones.** v02 has
several that would be invisible under a single headline metric:

- Four of six write-access attack types report **zero** net attack success while
  driving the attacked celltype's accuracy to zero (`02b_23`). A benchmark
  reporting only attack success would call them harmless. The mechanism is
  deletion — removing records that used to vote correctly — not substitution.
- `interpolate` and `shifted_inplace` are 88–100% right-censored at every
  threshold (`02b_22`). Plotted as the nulls they are.
- Uniform poisoning is inert at atlas scale whether the attacker inserts or
  overwrites (`02a_04`).

**Censored quantities are plotted beside their censoring.** `D_tau` exists only
for conditions that crossed the threshold, so a median over those is a median
over a self-selected subset — and the selection gets harsher as the threshold
rises. Read alone, `relabel_inplace` appears to get *cheaper* as the bar rises
(0.252 → 0.072 → 0.023). That is survivorship: censoring goes 4% → 15% → 58%
over the same range. Both panels or neither.

## The precision floor

Outcome metrics aggregated from a graph or hnsw path inherit stage 4's
reproducibility band: **±0.02 within-celltype, ±0.001 global.** Aggregation does
not average it away, because the six repeats in `02b_34` each used an identical
bootstrap seed — the variation is upstream of the resampling, not inside it.
