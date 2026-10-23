# CLAUDE.md — biorag-guard-sc

Retrieval-poisoning benchmark for single-cell reference atlases. Contributes the
scRNA-seq arm of the BioRAG-Guard project.

---

## 0. Start here

**Read [`HANDOFF.md`](HANDOFF.md) before anything else.** It carries the current
state, what has been measured, what remains, and the traps already hit — none of
which is derivable from this file, because this file describes the contract
rather than the work.

Quick orientation once you have:

```bash
# the environment (conda-native, arm64; conda is NOT on PATH by default)
export PATH="$HOME/miniforge3/bin:$HOME/miniforge3/condabin:$PATH"
PY="$HOME/miniforge3/envs/biorag/bin/python"

$PY -m pytest v01/1_scripts/00_pipeline/tests -q     # 160 tests, should be green
$HOME/miniforge3/envs/biorag/bin/ruff check v01/1_scripts v02/1_scripts

# the five checks that gate a commit (v02 only; v01 is frozen). This line said
# "three" while listing four, and is now five.
$PY v02/1_scripts/00_pipeline/check_readme.py     v02   # README <-> directory
$PY v02/1_scripts/00_pipeline/check_provenance.py v02   # artefact <-> producer
$PY v02/1_scripts/00_pipeline/check_schema.py     v02   # threat_model on every table
$PY v02/1_scripts/00_pipeline/check_steps.py     v02   # step id unique across all three trees
$PY v02/1_scripts/00_pipeline/check_retrieval.py v02   # no figure draws the exact-search shortcut arm

# what exists
ls v01/2_results/GSE136831/00_raw/raw/*.parquet | wc -l
```

**Do not launch a run before reading `HANDOFF.md`.** `--rerun-triggers mtime`
cascades: touching one cached file can invalidate every sweep below it. The
warning at the top of `v02/1_scripts/00_pipeline/Snakefile` explains it.

---

## 1. Overview

This repository measures how corrupted records in a single-cell reference atlas
propagate into wrong biological conclusions. Concretely: we inject poisoned cell
records into a SCimilarity-style reference, run k-NN label transfer, and report
two failure rates as a function of poisoning rate.

- **Retrieval failure** — at least one poisoned record enters the top-*k*
  neighbours of a query cell.
- **Outcome failure** — the k-NN vote assigns a label different from ground truth.

The distinction matters and must never be collapsed. A benchmark that only
reports outcome failure cannot tell a defence designer whether their filter
should act on retrieval or on aggregation.

### Behavioural constraints

These are rules, not preferences. Violating any of them invalidates the run.

**Work in embedding space, never in expression space.** The reference is a fixed
set of 128-dimensional unit vectors. Poisoning appends new vectors. Distances
from queries to the (small) poison set are computed by brute force and merged
into a clean neighbour list retrieved exactly once. Do not re-embed the
reference. Do not rebuild the index inside a sweep. This is what makes the
project laptop-feasible, and any agent that "optimises" it into a rebuild loop
has broken the project.

**Never fabricate a number.** No agent writes a figure, a caption, a summary
statistic, or a sentence of prose describing a result that is not present in a
committed file under `2_results/`. If a result is missing, say it is missing. An
invented effect size in a benchmark paper is a career problem, not a bug.

**A derived number is recomputed from the file, never from a rendered table.**
Found 2026-09-27 in three RUNLOG entries at once. Each quoted a ratio written
from the author's own markdown table rather than from the parquet, and the
three failed differently: `4,280x` divided two rounded display values *and*
named the wrong budget as the denominator, `114x` should have been 116x, and
`30x` should have been 34x — rounded DOWN, not rounded-values-divided. A
spot-check aimed at the mechanism found two of the three; enumerating every
`Nx` in the block and recomputing each from source found all three. So: any
figure that is a quotient, a percentage or a fold-change is recomputed from the
file holding its inputs, and a block quoting ratios is swept rather than
sampled.

**A number taken from one case is not a fact about the family.** Added
2026-09-28, after a caption written to CORRECT a misleading claim shipped a
misleading claim of its own: it quoted one celltype's 1.9x as characterising all
26, when 1.9x is the LOWEST of them and the median is 2.3x. The figure was about
to carry that into 26 captions. Before a figure quotes a number that stands for
a set -- a ratio, a rate, a margin -- compute it over the whole set and state
the RANGE or the median, or name the single case explicitly and say it is one.
"Measured on CL:0002062" is honest in a caption that appears once; in a caption
that appears 26 times it is not.

The question that catches this is not "is this number correct" -- it was -- but
"is this number TYPICAL". Every caption defect found on 2026-09-27/28 answered
the first question yes and the second no: ~100x compared two different
selections, 776% divided by an unsourced numerator, 28.3% used a denominator
its sentence did not name, 0.46 was a mean where the figure drew a median.

**Never quote a ratio whose denominator straddles zero.** Same finding, and it
is the more dangerous half. `02a_03`'s uniform arm moves accuracy UPWARD in 10
of its 30 draws and its budget means are non-monotone, so a multiple taken
against it changes magnitude — or sign — on a different draw. "4,280x" was not
merely arithmetically wrong, it was unreproducible in principle, and a reviewer
who checked it would have found that. State both magnitudes instead: hub moves
accuracy −0.1541, uniform never exceeds −0.000546. The honest form is also the
stronger claim.

**Never claim a run happened.** Only `docs/RUNLOG.md` records executions, and
only entries backed by a log file under `2_results/<DATASET>/logs/` are valid.

**Any executor counts, and the log file is committed.** This clause said
"Snakemake log file" until 2026-09-27. The intent was to stop the log filling
with noise, not to privilege one launcher: a direct `python` invocation, a
worker shell script, a sharded run merged afterwards and a Snakemake job are all
executions, and all of them produce evidence. Reproducibility is the reason the
log exists, so the broader rule is the correct one. Two consequences follow.
The entry names the INVOCATION -- the profile if there was one, otherwise the
command and the script that drove it -- because "which profile" is unanswerable
for a run that had none. And **the log file itself is committed**, in the same
commit as the entry citing it: an entry naming a path that a clean clone does
not contain fails the validity rule it was written to satisfy. 31 of the 39 logs
under `GSE136831/logs/` were untracked when this was found.

**THE GRAPH PATH IS NOT REPRODUCIBLE, AND THIS IS UNRESOLVED.** Found
2026-09-28. ``Bench.open_index`` calls ``hnswlib.add_items(pe, ids)`` without
``num_threads``, so it defaults to -1 and inserts on all 18 cores. Concurrent
insertion into an HNSW graph makes the resulting structure depend on thread
scheduling, so THE SAME POISON SET INSERTED TWICE GIVES A DIFFERENT GRAPH.

Measured three ways, and they agree:
- Controlled test, 20k base + 4k inserted: two identical builds disagree on
  **10.2%** of returned neighbour slots at ``num_threads=-1`` and are
  **bit-identical** at ``num_threads=1``.
- ``02b_30`` re-ran 8 of ``02b_01``'s conditions with identical
  ``condition_seed`` and identical ``n_poison``. 11 of 18 non-zero metrics moved;
  ``within_accuracy`` shifted by up to **0.019** (1.9 accuracy points).
- The CLEAN baseline in those same runs reproduced to 1e-9, because the clean
  path inserts nothing. That isolates ``add_items`` as the cause.

**Every graph-path table carries this variation**: ``02a_01``, ``02b_01``,
``02b_02``, ``02b_11_graph``, ``02b_13``, ``02b_14``, ``02b_15_graph``. Their
numbers are valid measurements of one realisation; they are not re-derivable,
which §1 otherwise requires and which is the whole point of a poisoning
benchmark.

**Do not "fix" this by setting ``num_threads=1`` without deciding what is being
measured.** Determinism costs 10.6x on insertion -- measured on the real index at
a 47,105-record poison set, ``add_items`` goes 5.2 s to 55.0 s and the condition
goes 27.5 s to 74.1 s -- so re-running ``02b_01`` alone is roughly two days and
all graph steps several. And a single-threaded insertion is not obviously the
more faithful model: a real maintainer accepting poisoned submissions would
REBUILD, which is also multi-threaded and which this project deliberately does
not do (``open_index``'s docstring says why).

**THE BAND IS NOW MEASURED, and it splits exactly along this project's central
distinction.** ``02b_34`` re-ran four conditions six times each -- identical
``condition_seed``, identical poison (one ``poison_digest`` per condition),
identical bootstrap seed -- through the same ``Bench`` path ``02b_01`` uses.

  ``within_retrieval``   range **0.00000** in all four conditions
  ``global_retrieval``   range **0.00000**
  ``within_accuracy``    range 0.0038 to **0.0168**, sd 0.0015 to 0.0069
  global-scope rates     move by at most 0.0004
  controls (baseline accuracy, clean target rate, n_poison) range 0.00000

**RETRIEVAL FAILURE REPRODUCES; OUTCOME FAILURE DOES NOT.** That is not a
coincidence of this measurement, it is the mechanism: insertion order perturbs
which neighbours come back, and 11-12% of the attacked celltype's queries do get
a different top-50 -- at one condition a query shares ZERO of its 50 with the
first repeat -- but whether ANY poison is in the list is stable, so retrieval
rates are exact. The vote absorbs most of the churn and 4-8% of predictions
flip, concentrated at parity where ``02b_01`` reports tie ambiguity above 0.5.

**So, precision.** A graph RETRIEVAL number may be quoted as measured. A graph
OUTCOME number -- accuracy, attack success, net attack success -- is good to
about ±0.02 within-celltype and ±0.001 global, and should not be written to
three decimals without saying which. Differences smaller than the band are not
findings. The retrieval arm of every graph table is safe; the outcome arm
carries this.

The options remain the human's: (a) keep the numbers and quote them to the
measured band; (b) pin ``num_threads=1`` and re-run, paying roughly two days for
``02b_01`` alone; (c) report outcome rates as bands from N repeats. (a) is now
defensible because the band is a number rather than a worry.

**Do not launch expensive work autonomously.** The `smoke` profile may be run
freely ONE RULE AT A TIME (`--until`, or a named target). A FULL smoke run is
25,744 jobs and roughly 1.5 h wall at 8 cores -- it is a small REFERENCE, not a
small grid -- so it is not free in the sense this clause originally meant, and
it needs the same approval as anything else of that size. Until 2026-09-28 §8
advertised it as "~3 min", which is how a clause meant to prevent surprise
compute came to authorise an unattended multi-hour run. The `full` profile, any download from Zenodo, and anything that loads a
reference index larger than 2 GB require explicit human approval in the
conversation before execution.

**Report negative and null results as prominently as positive ones.** If a
poisoning attack fails to move the label, that is a finding about the system's
robustness and it goes in the summary with the same weight as a success.

**Stay in your directory.** See §4. A subagent that edits a file outside its
owned paths must stop and report rather than proceeding.

### Deliberate design choices — do not "fix" these

- **`2_results/` is written only by pipeline execution, never edited by an agent.**
  It is the shared ground truth all other lanes read. Hand-editing it destroys
  the provenance chain.
- **`1_scripts/00_common/schemas.py` is owned exclusively by the orchestrator.** All four
  lanes import from it so that scripts, figures, and docs agree on column names
  and dtypes without coordinating. This is deliberately a bottleneck.
- **Seeds are explicit in every filename, never implicit.** Reproducibility of a
  poisoning benchmark is the whole point; a run you cannot re-derive is worthless.
- **Cosine distance, not Euclidean.** Reference vectors are unit-normalised onto
  a hypersphere; the SCimilarity score is `1 / (1 - c_i · c_j)`. Euclidean
  distance is monotone with cosine here but the score is not, and thresholds
  (e.g. the score < 33 outlier filter) are defined on the score.
- **Ontology distance is a first-class experimental axis, not a covariate.**
  Poisoning CD4→CD8 and macrophage→endothelial are different attacks, not the
  same attack at different strengths.

---

## 2. Tech stack — pinned

Environments are conda/mamba (miniforge, native arm64). Versions below are the
intended pins; the authoritative record is `v02/1_scripts/00_pipeline/envs/*.lock`. If a pin here
disagrees with a lockfile, the lockfile wins and the orchestrator updates this
section.

| Layer | Package | Version |
|---|---|---|
| Runtime | python | 3.12 |
| Workflow | snakemake | 8.25 |
| Arrays | numpy | 2.1 |
| Tables | pandas | 2.2 |
| Storage | pyarrow | 17.0 |
| Stats | scipy | 1.14 |
| Clustering | scikit-learn | 1.9 |
| ANN index | hnswlib | 0.8.0 |
| Single-cell | scanpy | 1.10.3 |
| Single-cell | anndata | 0.10.9 |
| Encoder | torch | 2.5 (CPU/MPS) |
| Model API | scimilarity | 0.4.0 |
| Plotting | plotnine | 0.13 |
| Plotting | matplotlib | 3.9 |
| Testing | pytest | 8.3 |
| Lint/format | ruff | 0.7 |

**Platform note (deliberate).** Local development is conda-native on Apple
silicon. Docker is **not** the local dev loop — an amd64 image under emulation is
slow and unnecessary here. The container is built for release and for HPC only,
targeting linux/amd64, from `v02/1_scripts/00_pipeline/envs/container.def`.

---

## 3. Project structure

`SPEC.md` is the authoritative build contract. Where CLAUDE.md and SPEC.md
disagree on *what the pipeline computes*, SPEC.md wins. Where they disagree on
*who may edit what*, CLAUDE.md wins.

```
biorag-guard-sc/
├── CLAUDE.md  SPEC.md  HANDOFF.md      # contracts, shared across versions
├── docs/                               # RUNLOG, DECISIONS — one narrative
├── archive/                            # NOT runnable. Historical evidence that
│                                       # belongs to no lane: workers_2026-09/
│                                       # holds the zsh drivers that sequenced
│                                       # Runs 9-27 from OUTSIDE the repo. Kept
│                                       # because one of them was already lost
│                                       # before it was copied. See its README.
├── 0_data/                             # READ-ONLY inputs, shared, checksummed
│   ├── queries/  reference/model_v1.1/  ontology/cl.obo
├── cache/                              # SHARED scratch, gitignored. A REAL
│   ├── reference.npz  poison/          # directory above both versions — no
│   └── <DATASET>/                      # symlinks. Reached via config paths.cache.
│                                       #
│                                       # Two tiers, and the split is the point.
│                                       # AT THE ROOT: what does not vary with the
│                                       # query dataset — reference.npz unpacked
│                                       # from 0_data, and poison/, because poison
│                                       # is drawn from REFERENCE cells so the same
│                                       # (attack, pair, rate, seed) gives byte-
│                                       # identical poison whatever is queried.
│                                       # Sharing it makes a cross-dataset
│                                       # comparison exact, not merely matched.
│                                       # UNDER <DATASET>/: everything derived from
│                                       # the queries — query_emb.npy,
│                                       # query_meta.parquet, clean_neighbors.npz,
│                                       # clean_neighbors_deep.npz. Fixed names at
│                                       # the root until 2026-09-28, so a second
│                                       # dataset overwrote the first's in place
│                                       # and --rerun-triggers mtime then
│                                       # invalidated every sweep beneath them.
│                                       #
│                                       # An earlier version of this entry said the
│                                       # cache holds "never what a script
│                                       # COMPUTED". That was already false when
│                                       # written: clean_neighbors_deep.npz is 47
│                                       # minutes of computation. What is true is
│                                       # that nothing here is a RESULT — results
│                                       # live in 2_results/ and are committed.
│                                       # cache/ is recomputable, and anything that
│                                       # is not recomputable does not belong.
│
├── v01/                    FROZEN. The exact-search reimplementation.
│   └── 1_scripts/ 2_results/ 3_figures/ 4_ppts/ config/
│
└── v02/                    ACTIVE. All new work lands here.
    ├── 1_scripts/          analysis scripts ONLY — never figures
    │   ├── 00_common/        shared library (schemas.py, defense.py, …)
    │   ├── 00_pipeline/      DAG, tests, and the five checks
    │   └── 0[1-4]_*/         one directory per module
    ├── 2_results/<DATASET>/  tables, arrays, p-values, coordinates — NO scripts
    │   └── 00_raw/ 0[1-4]_*/ logs/
    ├── 3_figures/<DATASET>/  figure scripts BESIDE the figures they draw
    │   └── 0[1-4]_*/         figNN_script.py + its .png/.pdf/.txt
    └── 4_ppts/ config/
```

**Two versions, one frozen.** `v01` is the record of the exact-search work and is
never edited again. `v02` carries every new script, table and figure. Duplicating
`schemas.py` across them is safe only because v01 is frozen — see
`docs/DECISIONS.md`.

**Naming: one identifier per step.** Every step in a module is
`MM_NN_<access>_<name>`, and its script, tables and figures all share
`MM_NN_<access>`:

| Part | Meaning |
|---|---|
| `MM` | module id, matching the directory: `01a`, `02a`, `02b`, `03a`, `03b` |
| `NN` | step number — execution order within the module |
| `<access>` | `no_access` cannot write to the reference, submissions only · `with_access` can overwrite existing records · `both` the contrast |

Analysis steps take the low numbers; figure steps continue the same sequence. A
script emitting a family keeps one number and varies a suffix
(`02b_11_no_access_CL0000235.png`). `grep 02b_07_with_access` returns the step and everything it
produced; `grep _with_access_` returns every result in the project that
assumed an attacker who can overwrite the reference.

**The defence modules are split by ARM, not by scope** — revised 2026-09-29.
`03_a_defence_vs_insert/` holds defences measured against an attacker who can
only submit records; `03_b_defence_vs_modify/` against one who can overwrite
existing ones. Scope (atlas or per-celltype) is a STEP inside them, not a
module of its own.

The earlier `03_a_defence_global` / `03_b_defence_celltype` / `04_a_defence_layer`
split mirrored the attack side, which is organised by scope. That was the wrong
axis: v01 measured the whole defence stack at **-63.8 pp** against insertion and
**-2.0 pp** against modification, so ACCESS is the axis a defence result varies
along and scope is secondary. `04_a_defence_layer` is gone; the per-layer
decomposition is a step in each arm, and the cross-arm contrast is a figure that
reads both. Renamed while all three directories were still empty, which is the
only moment it was free.

**Every directory carries a `README.txt`** that is a bijection with its contents:
every file has an entry, every entry names a file that exists. Adding, deleting
or renaming a file requires updating it in the same commit. An entry for a file
that is not present is a hallucination. Enforced by `check_readme.py`.


**Stages mirror the presentation.** A slide must never need a number no stage
produces, and a stage must never produce a number no slide uses. That is what
makes a figure traceable back to the script and table that made it.

**`1_scripts` is not a Python package** — an identifier cannot start with a
digit, so there is no dotted import path. Shared code is imported flat after
putting `00_common` on `sys.path`:

```python
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_common"))
from schemas import K_NEIGHBORS
from defense import DefenseConfig
```

`1_scripts/00_pipeline/tests/conftest.py` does the same once for the whole test
session. Every runnable script carries the two-line bootstrap itself.


---

## 4. File ownership rules

Non-overlapping ownership is what allows the four lanes to run in parallel
without merge conflicts. Every path in the repo has exactly one writer.

All paths below are under `v02/`. **Nothing writes to `v01/` — it is frozen.**

| Agent | May write | Must not touch |
|---|---|---|
| **Orchestrator** | `SPEC.md`, `CLAUDE.md`, `v02/config/**`, `v02/1_scripts/00_common/schemas.py`, `.claude/agents/**` | `v02/1_scripts/0[1-4]_**`, `v02/3_figures/**`, `docs/**`, `v02/2_results/**` |
| **Script agent** | `v02/1_scripts/**` except `00_common/schemas.py` | `v02/3_figures/**`, `docs/**`, `v02/config/**`, `SPEC.md`, `v02/2_results/**` (writes only by *running*) |
| **Figure agent** | `v02/3_figures/**` — scripts and images alike, they live together | `v02/1_scripts/**`, `docs/**`, `v02/2_results/**` except its own plotted-value CSVs |
| **Doc agent** | `docs/**` | everything else, without exception |
| *(nobody)* | — | `v01/**` is frozen; `v02/2_results/**` holds no scripts; `0_data/**` is read-only |

**Read access is unrestricted.** Every agent may read every file. Only writes
are partitioned.

**Escalation rule.** If a lane needs a change outside its territory — a new
column in the schema, a new config key, a corrected spec — it does not make the
change. It writes a one-paragraph request and returns control to the
orchestrator. The orchestrator makes the change in `v02/1_scripts/00_common/` or `v02/config/`,
then re-dispatches.

---

## 5. Data contracts

Defined in `1_scripts/00_common/schemas.py`. All four lanes import these; nobody hardcodes
a column name.

### `poison_set` — `cache/poison/{attack}_{pair}_{rate}_{seed}.npz`

| Field | Type | Meaning |
|---|---|---|
| `emb` | float32 (n, 128) | unit-normalised poisoned vectors |
| `label` | str (n,) | attacker-chosen Cell Ontology term |
| `provenance` | str (n,) | source reference cell ID, or `synthetic` |
| `attack` | str | `relabel` \| `duplicate` \| `interpolate` |
| `alpha` | float32 (n,) | interpolation strength; `nan` for other attacks |

### `raw_result` — `2_results/{dataset}/00_raw/raw/{attack}_{pair}_{rate}_{seed}.parquet`

One row per query cell. Long format.

| Column | Type | Meaning |
|---|---|---|
| `query_id` | str | cell barcode in the held-out dataset |
| `y_true` | category | author-provided Cell Ontology term |
| `y_pred_clean` | category | k-NN vote with no poisoning |
| `y_pred_poisoned` | category | k-NN vote after merge |
| `n_poison_in_topk` | int16 | count of poisoned records in top-*k* |
| `min_poison_score` | float32 | SCimilarity score of nearest poison |
| `poison_survives_qc` | bool | nearest poison has score ≥ 33 |
| `attack`, `pair`, `rate`, `seed` | category / float32 / int16 | condition keys |
| `ontology_dist` | int8 | tree distance between source and target term |

### `summary` — `2_results/{dataset}/02_a_attack_atlas/summary.parquet`

Aggregated over seeds. Adds `retrieval_fail_rate`, `outcome_fail_rate`, and
`ci_lo` / `ci_hi` (bootstrap, 2000 resamples over seeds).

**Rule:** every figure must be derivable from `summary.parquet` alone, except
per-cell diagnostic figures, which read `00_raw/`, and figures needing a
quantity that crosses pairs or needs an interval, which read the machine-written
tables beside it: `global_summary`, `source_summary`, `pair_summary`,
`accuracy_summary`, `clean_baseline`, `clean_confusion`. Those hold donor-paired
bootstrap intervals, which cannot be derived from `summary.parquet` because it is
already aggregated past the donor. A figure that needs a
column neither table has is a spec change — escalate.

---

## 6. Stage interfaces

The DAG. Each stage is one Snakemake rule, one script, one owner.

| Stage | Rule | Input | Output | Approx. cost (smoke / full) |
|---|---|---|---|---|
| 0 | `acquire` | Zenodo DOI, checksums | `0_data/reference/` | — / **needs approval** |
| 1 | `embed_queries` | held-out `.h5ad` | `cache/query_emb.npy` | 5 s / 60 s |
| 2 | `retrieve_clean` | query emb, reference index | `cache/clean_neighbors.parquet` | 30 s / 20 min |
| 3 | `make_poison` | reference emb, ontology graph | `cache/poison/*.npz` | 1 s / 5 s per condition |
| 4 | `sweep` | clean neighbours + poison set | `2_results/<DS>/00_raw/raw/*.parquet` | 2 s / 10 s per condition |
| 5 | `aggregate` | all raw | `2_results/<DS>/02_a_attack_atlas/summary.parquet` | 5 s / 30 s |
| 6 | `validate_shortcut` | subset of conditions | `2_results/<DS>/00_raw/shortcut_validation.parquet` | — / 40 min — **never run in v02; see below** |

**Each stage has two explanation files.** `docs/explanations/stageN_*.md` is
the v01 record, unedited and bannered; `docs/explanations/stageN_*_v02.md` says
what that stage is in v02. Start from `docs/explanations/README_v02.md`, which
opens with the fact that shapes the rest: **no v02 run has ever gone through
this DAG.** All 39 logs under `v02/2_results/GSE136831/logs/` are a direct
`python` invocation or a worker script; not one names a Snakemake profile or job
id. The rules below exist and `snakemake -n` shows a complete graph — they have
simply never been the thing that ran here, because in v02 the unit of work is
the **step** (`MM_NN_<access>`), not the stage.

**Stage 2 runs exhaustively in every profile, including `full`.** §6's table
below describes HNSW for the full profile; that is superseded. Exhaustive
retrieval makes the merge shortcut provably exact (SPEC.md §4.2) at any
poison-set size, so no published number depends on an approximation. See
`docs/DECISIONS.md` and SPEC.md §2.

**The shortcut must be validated wherever a published number rests on it.**
The merge-and-rerank shortcut is exact for exhaustive search but only
approximate under HNSW, because a poisoned node can alter graph traversal. A
result produced by the shortcut may not be published until something has
measured the shortcut against a real rebuild.

**In v02 the obligation does not bite, because no published number rests on the
shortcut** -- verified 2026-09-28 by listing every parquet each figure script
opens by name and reading its `retrieval_path`: not one carries `exact`. Stage 6
has never run here either. There is no `shortcut_validation.parquet` under `v02/2_results/` and
none of the 39 logs names `validate_shortcut`; the only stage 6 run in this
repository is v01's, Runs 3-4 of 2026-08-26/27, 96 conditions. Two things
replaced it, and both are stronger:

1. **Most v02 results do not use the shortcut at all.** `02a_01`, `02b_01`,
   `02b_02`, `02b_13`, `02b_14` (`retrieval_path = hnsw`) and `02b_11_graph`,
   `02b_15_graph` (`graph`) insert the poison into the shipped
   `labelled_kNN.bin` and re-query it. There is no approximation to validate
   because none was taken. `02a_03` and `02b_05` (`cached_clean`) are exact by
   construction: a label-only edit moves no vector, so the index is byte-
   identical and the cached clean neighbours ARE the poisoned ones.

2. **`02b_16` corroborates it, and is SUPPLEMENTARY rather than load-bearing.**
   Point 1 alone discharges the obligation; nothing published needs the
   shortcut to be accurate. `02b_16` is the methods result that measures it
   anyway -- every condition of `02b_11` and `02b_15` scored both ways, 2,964
   conditions x 2 metrics, against stage 6's 96. Read from
   `02b_16_with_access_exact_vs_graph.parquet`: the two agree exactly on 44.0%
   of `02b_11`'s accuracy conditions and 60.5% of its retrieval conditions,
   differ by more than 0.01 on 11.4% and 15.6%, and the mean signed difference
   is -0.00146 and -0.00112. The disagreement runs BOTH ways -- the graph attack
   is weaker in 33.4% of conditions and stronger in the rest -- so `remerge`
   does not systematically understate, which retracts the earlier claim that it
   did. The `exact` tables are retained only as that comparison's arm; every
   figure reads the `_graph` twin.

**What is still required.** Any NEW result resting on the shortcut needs the
same treatment before publication: measure it against the faithful path over
the conditions it covers, not a spot check. Prefer running faithfully in the
first place -- it costs about 12.8x per condition and removes the question.

---

## 7. Coding standards

Enforced by `ruff` and by review. These are refusable violations.

- **Say "celltype", never "class".** In prose, in column names, in variable
  names, in figure captions and in commit messages. `class` is ambiguous here —
  it means both a cell type and a statistical class, and this project talks
  about both. Every agent writing for this repository follows this, not only
  the orchestrator. Existing occurrences are corrected as files are touched
  rather than in a sweep, except where a column name is already committed and
  renaming it would break a reader. **All work before 2026-09-16 says "class"
  where it means celltype** — `HANDOFF.md` §3.1–§3.29, `docs/`, the scripts and
  every earlier commit message. Same concept, different word; see
  `docs/DECISIONS.md`.

- **Never say a poisoned record "passed QC", or call `score >= 33` a QC check.**
  Established 2026-09-25 by reading `scimilarity 0.4.0`. The 0.03 constant is
  genuinely theirs — `utils.pseudobulk_anndata` takes
  `qc_filters={"mito_percent": 20.0, "min_counts": 1000, "min_genes": 500,
  "max_nn_dist": 0.03}`. But **every QC it ships is QUERY-side**: it filters the
  cells *you* bring, dropping any whose nearest reference neighbour is too far
  to trust the annotation. `search_exhaustive`'s `max_dist` drops hits at query
  time. `qc_stats["query_coherence"]` — the only thing scimilarity itself calls
  `qc` — scores whether *your* query set is coherent. **Nothing it ships ever
  inspects the reference.** So poison is not "caught" or "missed" by QC; it is
  never looked at, and that absence is the finding. `qc_pass` measures the
  distance from a QUERY to its nearest POISON — a retrieval property, not a
  verdict on the poison. Say "within SCimilarity's search distance bar", or name
  the quantity. v01's `fig03_qc_evasion` got the NUMBER right and the LABEL
  wrong; do not reproduce the label. The `qc_` columns keep their names because
  the tables are committed — the fix is the language, not the schema. Full
  provenance in `schemas.SCORE_QC`.

- **Python only** in `v02/1_scripts/`. No shell scripts beyond `envs/` bootstrap.
  Figure scripts live in `v02/3_figures/` — see the figure-lane rule below.
- **No notebooks anywhere in the repo.** Exploration happens in a scratch
  directory that is gitignored.
- **Every script takes `snakemake.input` / `snakemake.output`** — no `argparse`,
  no hardcoded paths, no `os.getcwd()`.
- **Seeds are passed in, never generated.** `np.random.default_rng(seed)` at the
  top of the function; no module-level RNG, no `np.random.seed`.
- **float32 throughout.** float64 doubles memory on 23.4M × 128 arrays for no
  benefit. Cast explicitly at every boundary.
- **Type hints on every public function**, checked with `ruff`'s pyflakes rules.
- **Long-format data only.** Wide tables are for display, never for storage.
  **One documented exception: SCOPE columns.** A condition scored over two
  populations -- `within_*` (the attacked celltype's own queries) and `global_*`
  (all queries) -- puts both on ONE row rather than becoming two rows with a
  `scope` column. They are two views of a single condition, not two
  observations: the same poison set, the same retrieval, the same seed. Splitting
  them would invite joining one scope's rows to another's and would make a
  paired difference between scopes a self-join. Any further axis -- attack, rate,
  celltype, seed -- stays long.
- **One vote, imported, never reimplemented.** `1_scripts/00_common/vote.py` is
  the only k-NN vote in the repository and is orchestrator-owned like
  `schemas.py`. v01 grew SIX copies that disagreed on tie-breaking, so a clean
  baseline and the attacked results it was compared against could be scored by
  different rules -- a 0.05 pp discrepancy that went unexplained for weeks.
  `config.vote.tie_break` declares the rule; the module implements it; every lane
  imports it. A caller that needs different behaviour changes the module, not its
  own copy.

- **One retrieval regime per comparison.** A clean baseline must be retrieved the
  same way as the attacked measurement it is subtracted from. This is the same
  defect as the six votes above, one layer down: there it was two scoring rules,
  here it is two retrievers. Found 2026-09-26 in `02b_11` and `02b_15`, which
  take the attacked side from `modify.graph_topk` (HNSW) under
  `BIORAG_RETRIEVAL=graph` but the clean side from `deep_idx[:, :K_NEIGHBORS]`
  (exhaustive), so `accuracy_delta` and `net_attack_success` straddle two
  regimes. Under `exact` both sides came from the deep list and the gap
  cancelled — **making the attacked side faithful is what exposed it.**
  `01a_02` bounds the cost: median |Δaccuracy| 0.000000 and 32 of 34 celltypes
  under 0.01, but **0.110 on CL:0002062**, and top-50 overlap falls to 0.327 on
  the worst celltype. That is the same order as the graph-vs-exact divergence the
  graph run exists to measure, so it is not roundable. `02a_01` does it right —
  it queries the clean index through HNSW *before* inserting (lines 94-96), so
  both sides share a regime at the cost of one extra query per run, not per
  condition. Any new step that reports a delta, a paired difference or a net rate
  states in its docstring which retriever produced each side, and they match.
- **Ties are reported, never silently resolved.** `vote()` returns `n_tied` and
  `pivotal_in_tie`, and `bounds()` turns them into best/worst columns. Only a tie
  CONTAINING the pivotal label is ambiguous -- a tie among labels that are all
  wrong is already determined. Publish the interval, not the point.
- **plotnine, not raw matplotlib**, in the Python figure lane, for anything
  with a data mapping. Raw matplotlib is permitted only for schematics.
- **One figure lane, one direction of flow.** A figure script lives in
  `v02/3_figures/<dataset>/<stage>/` beside the images it draws. It reads
  `2_results/`, and writes the plotted values back to
  `v02/2_results/<dataset>/<stage>/<step>.csv` plus its `.png` and `.pdf`. A
  number therefore has exactly one origin — `2_results/`. A figure script that
  recomputes a statistic instead of reading it is a violation.
- **The R publication lane is DEFERRED.** v01 ran a second lane
  (`3_figures/<dataset>/publication/`, ggplot2) that redrew 5 of 26 figures and
  covered nothing the Python lane did not. It is preserved intact under `v01/`
  and can be restored by copying it forward when publication is near. Until
  then no rule about lane ordering or `render_all.R` applies.
- **A figure script must produce a figure.** One that computes but draws
  nothing is an analysis step and belongs in `1_scripts/`; its output is a
  quantitative result and belongs in `2_results/`. v01's `fig15a` — a UMAP fit
  written to gitignored `cache/` — is the case this rule exists to prevent, and
  it left two figures unreproducible from a clean clone.
- **Figure identity is the step id, not a figure number.** `MM_NN_<access>`
  (§3). The v01 scheme of numbering figures by the data they show (`00` bulk,
  `01` dose response, …) does not carry over; the module and step now say where
  a figure sits and what produced it.
- **Every figure is saved as BOTH `.pdf` and `.png`** wherever the format
  allows it. The PDF is the one that survives being put in a document.
- **Every figure script writes its plotted values** to a sibling `.csv` in
  `2_results/<dataset>/<stage>/`, NOT beside the image — `3_figures/` holds
  scripts and images only. A figure whose numbers cannot be inspected is not
  reviewable. This binds the Python lane, which produces the CSVs; the R lane
  satisfies it by construction, because a CSV is all it reads.
- **Docstrings state units and shapes**: `emb: (n_cells, 128) float32, unit norm`.
- **No `try/except` around numerical code.** Let it fail loudly. Silent
  fallbacks in a benchmark produce wrong numbers that look fine.
- **Tests before sweeps.** `v02/1_scripts/00_pipeline/tests/` must pass before any `sweep` rule
  runs. Includes golden-value tests: a fixed tiny input with a hand-checked
  expected output.

---

## 8. Run commands

```bash
# one-time setup (miniforge, arm64 native)
mamba env create -f v02/1_scripts/00_pipeline/envs/base.yaml -n biorag
mamba activate biorag

# lint + tests — run before anything else
ruff check v02/1_scripts
pytest v02/1_scripts/00_pipeline/tests -q

# smoke run: a 500k fixed-seed subsample of the REAL reference (not synthetic),
# no downloads. NOT a 3-minute job: it builds 25,744 jobs -- 12,870 make_poison
# + 12,870 sweep + 4 -- because smoke differs from full in reference SIZE, not
# in grid size. At §6's own per-condition costs that is ~10.7 core-hours, about
# 1.5 h wall at 8 cores. Derived from §6, not measured. This line read
# "synthetic ... ~3 min" until 2026-09-28 and was wrong on both counts.
snakemake -s v02/1_scripts/00_pipeline/Snakefile --profile v02/config/profiles/smoke -c8 --sdm conda

# inspect the DAG without executing
snakemake --profile config/profiles/smoke -n -p
snakemake --profile config/profiles/smoke --dag | dot -Tpdf > docs/dag.pdf

# figures from whatever is currently in 2_results/
python v02/3_figures/<DS>/02_b_attack_celltype/02b_11_no_access_script.py

# FULL RUN — requires explicit approval, loads ~10 GB reference index
snakemake --profile config/profiles/full -c10 --sdm conda

# HPC (TACC), container built for linux/amd64
snakemake --profile config/profiles/tacc --sdm apptainer
```

**Hardware note.** Development target is an M5 Max MacBook Pro, 64 GB. The
annotation reference (7.9M × 128 float32 ≈ 4 GB vectors, ≈ 5 GB HNSW graph at
M = 80) fits comfortably. The 23.4M-cell query index (≈ 21 GB) is **not loaded by
default** — no stage in this pipeline requires it. If a stage appears to need
it, that is a design error; escalate rather than raising the memory ceiling.

---

## 9. Agent briefs

**Full definitions are the `*_v02.md` files at the repository root** —
`orchestrator_v02.md`, `script-agent_v02.md`, `figure-agent_v02.md`,
`doc-agent_v02.md`. The un-suffixed `orchestrator.md`, `script-agent.md`,
`figure-agent.md` and `doc-agent.md` are the **v01 briefs**, kept as the record
and superseded: every path in them (`pipeline/`, `contracts/`, `figures/`,
`results/`) refers to the pre-split tree and no longer exists. They carry a
banner saying so. `.claude/agents/` does not currently exist; if it is created,
it is populated from the `_v02` files, not from the originals.

Summary of mandate and definition of done:

**Orchestrator** — owns the contract, not the code. Reads the spec, decomposes
work into lane-sized tasks, dispatches, integrates, resolves escalations. Makes
every change to `1_scripts/00_common/schemas.py` and `config/`. *Done when:* all lanes
report green, **the five checks in §0** pass, `2_results/` holds the run's
tables, and `docs/RUNLOG.md` has an entry backed by a log file **committed in
the same commit as the entry**.

**Script agent** — owns `v02/1_scripts/`. Writes Snakemake rules and stage scripts
against the schema contract. Never plots, never writes prose beyond docstrings.
*Done when:* `pytest` passes, the five checks in §0 pass, `snakemake -n` shows
a complete DAG, and the step's tables are under `2_results/<DS>/<module>/` with
its log committed. **Note the v01 wording — "the smoke profile runs to
`summary.parquet`" — no longer applies**: v02 has no single `summary.parquet`,
each step writes its own table, and a full smoke run needs approval (§1).

**Figure agent** — owns `figures/`, both lanes. The Python lane reads
`2_results/` only; if the needed column does not exist, it escalates rather than
recomputing it. The R lane reads `2_results/<dataset>/<stage>/*.csv` only. Never touches the
pipeline. *Done when:* every figure has a matching `.csv` of plotted values, the
figure lane renders from a clean checkout of `2_results/`.

**Doc agent** — owns `docs/`. Writes plain-language explanations of what each
stage does and why, maintains the append-only run log, and records decisions
including rejected alternatives. Writes for a reader who knows security but not
single-cell biology, and vice versa. Never states a result not present in
`2_results/`. *Done when:* every stage in §6 has a **`_v02` explanation file**, **every log
file under `2_results/<DS>/logs/` is named by a RUNLOG entry** (the inverse
check — silence must never imply a run was logged), and every entry names a
committed log, a git SHA with its dirty state, and the invocation that produced
it (profile, or the command where there was none).

---

## 10. Scope boundary

This repository covers the **scRNA-seq retrieval arm only** — SCimilarity-style
cell search and scArches-style label transfer. It does **not** cover ESM-2
protein retrieval or perturbation-response prediction, which are other arms of
BioRAG-Guard. Requests to add them are scope changes for the human, not tasks
for the orchestrator.

No pathogen-related sequences, no hazardous biological content, no wet-lab
protocols enter this repository. All poisoning scenarios use public or synthetic
transcriptomic data and target retrieval mechanics, not biological payloads.
