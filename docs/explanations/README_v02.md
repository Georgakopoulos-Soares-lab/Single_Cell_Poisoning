# Stage explanations — v02

The v01 files beside this one describe a pipeline that ran. This one has to open
with a fact that changes how they should be read:

> **No v02 run has ever gone through the Snakemake DAG.** All 39 logs under
> `v02/2_results/GSE136831/logs/` were produced by a direct `python` invocation
> or by a worker script sequencing several of them. Not one names a Snakemake
> profile or a Snakemake job id. The rules still exist in
> `v02/1_scripts/00_pipeline/Snakefile` — `prepare_reference`, `embed_queries`,
> `retrieve_clean`, `make_poison`, `sweep`, `validate_shortcut`,
> `collect_validation`, `aggregate` — and `snakemake -n` shows a complete graph.
> They have simply never been the thing that ran here.

That is not a defect to hide; it is the single most useful thing for a new
reader to know, because **the stage is no longer the unit of work in v02. The
step is.**

## The two models, side by side

| | v01 | v02 |
|---|---|---|
| Unit of work | a **stage** (0–6), one Snakemake rule | a **step**, `MM_NN_<access>_<name>` |
| What runs it | `snakemake --profile …` | `python` on the step script, by hand or under a worker |
| Where it writes | `results/` | `v02/2_results/<DATASET>/<module>/` |
| How many | 7 stages | 21 analysis steps + 25 figure scripts, across 3 modules |
| Grain | one grid, swept | one **question**, with its own grid |

A v01 stage answers "do this part for every condition". A v02 step answers one
question — *does hub selection transfer across donors?*, *how far should a
record move?* — and brings whatever grid that question needs. `02b_13` sweeps
seven spreads by six jitters; `02b_34` runs four conditions six times each.
Neither is expressible as a rate in a single shared ladder, which is why the
DAG stopped being the thing that ran.

## What each stage became

| Stage | v01 file | In v02 |
|---|---|---|
| 0 `acquire` | [stage0](stage0_acquire.md) | [stage0_v02](stage0_acquire_v02.md) — same artefact, now at the **shared** cache tier |
| 1 `embed_queries` | [stage1](stage1_embed_queries.md) | [stage1_v02](stage1_embed_queries_v02.md) — now under `cache/<DATASET>/` |
| 2 `retrieve_clean` | [stage2](stage2_retrieve_clean.md) | [stage2_v02](stage2_retrieve_clean_v02.md) — depth 500, and the regime question |
| 3 `make_poison` | [stage3](stage3_make_poison.md) | [stage3_v02](stage3_make_poison_v02.md) — one entry point, many more attacks |
| 4 `sweep` | [stage4](stage4_sweep.md) | [stage4_v02](stage4_sweep_v02.md) — **four retrieval paths, and one of them is not reproducible** |
| 5 `aggregate` | [stage5](stage5_aggregate.md) | [stage5_v02](stage5_aggregate_v02.md) — donor-paired, and scope columns go wide |
| 6 `validate_shortcut` | [stage6](stage6_validate_shortcut.md) | [stage6_v02](stage6_validate_shortcut_v02.md) — **never run, and why that is defensible** |

## What is genuinely new in v02, with no v01 stage behind it

- **The retrieval path is a recorded column, not an assumption.** Every v02
  results table carries `retrieval_path`. v01 had one retriever and did not need
  the column. Read stage 4's v02 file before comparing any two tables.
- **The dataset is a parameter.** `config/config.yaml` has a `datasets:` block;
  `00_common/dataset_config.py` resolves it. Nothing hardcodes `GSE136831`.
- **The cache has two tiers.** Shared at the root (`reference.npz`, `poison/`),
  per-dataset under `cache/<DATASET>/`. Stage 1's v02 file says why the split is
  load-bearing rather than tidy.
- **Five gating checks**, not three: `check_readme`, `check_provenance`,
  `check_schema`, `check_steps`, `check_retrieval`. The last exists because a
  figure that reads the wrong table looks correct.
- **Provenance is stamped at run time.** `00_common/provenance.py` writes the
  SHA, the branch, and **whether the tree was dirty**, into every log header.
  Most v02 logs say DIRTY. That is honest rather than good, and it is why the
  log header is worth reading before trusting a table.

## What these files do not cover

**Defence.** `v02/1_scripts/03_a_defence_global/`,
`03_b_defence_celltype/` and `04_a_defence_layer/` are empty directories. No
defence work has started in v02. v01's defence results (HANDOFF §3.7–§3.10,
§3.38–§3.39) stand as v01 results and have **not** been re-measured under v02's
graph retrieval — which matters, because stage 4's v02 file shows outcome
metrics there carry a ±0.02 band that a defence claiming a small improvement
would sit inside.
