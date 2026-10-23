# Provenance of this repository

- **Source:** every file except `LICENSE` and the five top-level `.md`
  files added with it was taken byte for byte from commit `52078cb3` of a
  private working repository (sub-folder `biorag_project/`), with
  `git archive`, on 2026-10-22.
- **Contents:**
  - `v02/1_scripts`, `v02/config`, `v02/2_results`;
  - the figure scripts of `v02/3_figures` (not the images);
  - `CLAUDE.md`, `SPEC.md`, `.gitignore`;
  - the `0_data` provenance files;
  - the 8 `docs/explanations/*_v02.md`.

  That is 664 files, 497,140,184 bytes.
- **The only change: the run logs.** In 131 files under
  `v02/2_results/GSE136831/logs/`, 179 occurrences of the author's
  home-folder path were replaced:
  - the repository's path became `<REPO>`;
  - any other home path became `<HOME>`.

  No other byte was changed. A per-file manifest (sha256 before and after)
  is kept with the private record.

## Which code made each table

The tables were **not** all produced by commit `52078cb3`. Each run log
records the commit that ran it and whether the working tree had
uncommitted changes. For example: "sha 65205abb (DIRTY, diff 0742503ccadc,
…) branch restructure-v01-v02". Those commit ids, diff hashes and the branch
name refer to the private repository.

The reproduce test in `README.md` is the evidence that this commit's code
reproduces the tables it covers: 34 of 34 tables matched exactly, for five
steps. It does not cover the other steps.

## References that cannot be followed here

Text in the code, the logs and `docs/explanations/` mentions:
- v01;
- `docs/RUNLOG.md`, `docs/DECISIONS.md` and `HANDOFF.md`;
- paths such as `archive/workers_2026-09/`, the shell drivers that ran
  some steps.

These belong to the private repository and are not shared.
