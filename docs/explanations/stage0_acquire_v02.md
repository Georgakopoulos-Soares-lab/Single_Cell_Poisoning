# Stage 0 — `acquire`, in v02

**Status: unchanged in substance, moved in location. Never re-run in v02.**

The reference was downloaded, checksummed and unpacked once, in August 2026,
under v01. v02 reads the product of that work and has never repeated it. The
`prepare_reference` rule still exists in `v02/1_scripts/00_pipeline/Snakefile`
and has not executed there.

## What changed

**`cache/reference.npz` is now explicitly at the SHARED cache tier**, at the
repository root rather than under a dataset directory. This is a decision, not
a leftover: the reference does not vary with the query dataset, so a second
dataset must read the *same bytes*, not an equivalent copy. `CLAUDE.md` §3
carries the reasoning; `00_common/dataset_config.py` exposes it as
`cache_root()`.

The same tier holds `cache/poison/` — 13,142 entries — for the reason stage 3's
v02 file gives: poison is drawn from *reference* cells, so it too is independent
of what is being queried.

## What a reader should take from this in v02

**The reference is an artefact with its own provenance, and it is imperfect.**
Two facts from v01 carry forward unchanged and are load-bearing for v02 results:

- Reference provenance is **95.5%, not 100%** — any donor-level analysis
  operates on a subset (HANDOFF §5).
- The shipped reference already contains **6.5% duplicate records**. That is a
  curation fact about SCimilarity, measured before any attacker exists, and it
  is why donor voting raises clean accuracy (0.816 → 0.931) while providing no
  defence. Say curation, never defence — SPEC.md §12.

Neither is something v02 introduced, and neither has been re-measured.

## Why there is no v02 run to describe

Re-running stage 0 would require re-downloading a 30.31 GB tarball, which
`CLAUDE.md` §1 places behind explicit human approval. There has been no reason
to: the checksums in `config/checksums.txt` verify the artefact on disk, and
nothing in v02 modifies it. `0_data/` is read-only to every lane.
