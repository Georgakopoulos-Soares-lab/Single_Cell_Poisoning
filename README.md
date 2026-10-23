# Single_Cell_Poisoning: retrieval poisoning of a single-cell reference atlas

This repository measures how corrupted records in a single-cell reference
atlas (SCimilarity, model_v1.1) spread into wrong cell-type calls. Poisoned
cell records are added to, or edited in, the reference. Query cells
(GSE136831, lung tissue) are labelled by k-nearest-neighbour vote, and two
failure rates are reported against the poisoning rate:

- **retrieval failure:** at least one poisoned record enters a query's
  top-k neighbours;
- **outcome failure:** the vote returns a label other than the ground truth.

The two are always reported separately. `CLAUDE.md` and `SPEC.md` are the
project's full contract.

## Two iterations: why there is a `v02/` folder

The work ran in two iterations. The first (v01) used exact search, and its
re-implementation is kept in the private working repository. The second
(**v02**), which this repository contains, measures most attacks by
inserting the poison into SCimilarity's shipped HNSW index and querying it
again. This repository is a byte-for-byte copy of
v02 at commit `52078cb3` of the private repository; nothing was renamed or
tidied. The run logs are the one exception: the home-folder path in them is
replaced by `<REPO>` or `<HOME>` (see `PROVENANCE.md`). So the code, the
docstrings and the explanation files still mention v01, `docs/RUNLOG.md`,
`docs/DECISIONS.md` and `HANDOFF.md`. **Those references point to the
private record and are not in this repository.**

## What is here

| Path | What it holds |
|---|---|
| `v02/1_scripts/` | the analysis steps, one script per step `MM_NN_<access>_<name>.py`; shared code in `00_common/`; tests and checks in `00_pipeline/` |
| `v02/config/` | `config.yaml` (paths, parameters) and profiles |
| `v02/2_results/GSE136831/` | every result table, and the run log of each run in `logs/` |
| `v02/3_figures/GSE136831/` | the figure scripts only. **The images are not included**: run a script to draw its figure from `2_results/` |
| `docs/explanations/*_v02.md` | plain-language explanation of each pipeline stage |
| `0_data/` | checksums and provenance of the inputs only; **the inputs themselves are not included** (see `DATA.md`) |

## Getting started

1. Build the environment (`ENVIRONMENT.md`). **`base.yaml` omits three
   packages the code imports directly: `pyyaml`, `requests` and
   `umap-learn`.** Install them too.
2. Check the install, with no data needed:

       python -m pytest v02/1_scripts/00_pipeline/tests -q

   On this copy as shipped, with no data and no cache present, this gave
   **400 passed, 0 failed, 0 skipped** (macOS arm64, 2026-10-23). These are
   unit tests only: no test loads the real data. (`CLAUDE.md` says "160
   tests"; that is out of date, and the suite now has 400.)
3. Fetch the inputs and check their md5 (`DATA.md`).
4. Run the steps **directly with Python, in module order**:

       python v02/1_scripts/<module>/<step>.py

   **Do not use `snakemake`.** A Snakefile exists, but no v02 result was
   ever produced through it: every table here came from a direct Python
   run or a shell driver, as its log in `v02/2_results/GSE136831/logs/`
   records. The order is: shared preparation in `00_pipeline/`
   (`prepare_reference.py`, `embed_queries.py`, `retrieve_clean.py`,
   `retrieve_clean_deep.py`, `build_provenance.py`,
   `build_atlas_umap.py`), then `01_a`, `02_a`, `02_b`, `03_a` and `03_b`.
   Several 03a/03b steps make one table per `BIORAG_TAG`, set by
   environment variables; `BIORAG_SETTINGS.md` lists the settings each
   shipped tag was made with.

**Hardware.** Everything ran on one Apple M5 Max laptop with 64 GB. Some
steps are long there: from the logs, 02b_01 took 23.4 h in one process,
02b_13 16.2 h, and 02b_02 about 16-17 h per shard (3 shards). The cache
step `retrieve_clean_deep.py` took 47.7 min. Graph-path steps load the
shipped 9.3 GB HNSW index afresh for each condition (02b_01's docstring).

## How close a re-run should be

- **Exact steps** (exhaustive retrieval, and the label-only `cached_clean`
  steps such as 02a_03) should match the shipped tables exactly, given the
  same inputs and package versions.
- **Graph-path steps** (02a_01, 02b_01, 02b_02, 02b_11_graph, 02b_13,
  02b_14, 02b_15_graph) insert poison into the HNSW index on all CPU
  cores, so the graph depends on thread timing. Step 02b_34 measured this
  over six repeats:
  - **retrieval rates reproduce exactly** (range 0.00000);
  - **outcome rates do not:** within-celltype accuracy varied by up to
    0.0168, and global rates by at most 0.0004.

  Treat an outcome difference of about ±0.02 within-celltype, or ±0.001
  global, as agreement.
- **Across operating systems** (macOS against Linux): untested.

## Reproduce test: what was run, and its limits

Before sharing, five fast steps were re-run from commit `52078cb3`: 02a_03,
02b_12, 02b_16, 03a_09 (all 29 of its shipped tags) and 03b_04. That is 33
runs, and **all 33 exited 0**. They ran on the same laptop, against its
existing cache. Their outputs were compared with the shipped tables: same
columns and rows, and every cell equal, **floats included, with no
tolerance** (NaN equal to NaN). The only columns left out were 03b_04's
three wall-clock timing columns, a choice declared before the run.

**Result: 34 of 34 tables matched exactly.**

What this does **not** show:
- **02b_12, 02b_16 and 03a_09 only re-aggregate earlier tables.** They
  test the aggregation and interval code, not the retrieval and attack code
  that made those tables. Only 02a_03 and 03b_04 recompute from the
  reference and the cache.
- **The long retrieval and attack steps were not re-run**: 02b_01, 02b_02,
  02b_11, 02b_13, 02b_14, 02b_15, 03a_01, 03b_01 and the others, hours
  each.
- **The cache builders were not re-run**: the existing cache was reused.
- **No other machine has been tested.**
- Most tables were first produced by earlier, sometimes uncommitted, code;
  each log records which (`PROVENANCE.md`).

## Known gaps

- **The data download URLs in `DATA.md` have not been verified.**
- **`cl.obo` is missing from `v02/config/checksums.txt`.** Its md5 is in
  `0_data/CHECKSUMS.txt`.
- **`check_readme` fails on five counts in this copy, and that is
  expected.** This copy shares code, config and result tables, not the
  rendered figures. So the `README.txt` of each of the five
  `v02/3_figures/GSE136831/<module>/` folders was left out along with the
  figure images, and `python v02/1_scripts/00_pipeline/check_readme.py v02`
  reports each missing README. If you see exactly these five, nothing is
  broken. The other four checks pass on this copy:
  - `check_provenance`: 263 artefacts, 0 problems;
  - `check_schema`: 99 tables, 0 problems;
  - `check_steps`: 60 step ids, 0 problems;
  - `check_retrieval`: 51 figure table reads, 0 problems.

  `ruff check v02/1_scripts` passes.
- The conda lock `base.lock` is for macOS arm64 only. There is no Linux
  lock yet.

## Licence

MIT; see `LICENSE`.
