1_scripts/00_common  --  shared library imported by every lane; schemas.py and vote.py are orchestrator-owned
===========================================

NAMING. Every step in a module carries MM_NN_<access>_<name>:
  MM  module id, matching this directory (01a, 02a, 02b, 03a, 03b, 04a)
  NN  step number; execution order within the module
  <access>  no_access    attacker cannot write to the reference; they can
                         only get records ACCEPTED into it. Insertion must
                         change the graph, or the poison is never retrieved.
            with_access  attacker can OVERWRITE existing records. Rewriting
                         labels leaves every embedding in place, so the graph
                         is byte-identical and a checksum on it still passes.
            both         this step runs the contrast between the two
A step's script, its tables and its figures all share MM_NN_<access>, so one
grep finds everything the step produced. Analysis steps take the low numbers;
figure steps continue the same sequence. A script emitting a family of images
keeps one number and varies a suffix (e.g. 02b_11_no_access_CL0000235.png).

README. This file is a bijection with the directory: every file has an entry,
every entry names a file that exists. Adding, deleting or renaming a file
requires updating this file in the same commit. An entry for a file that is not
present is a hallucination. Enforced by 1_scripts/00_pipeline/check_readme.py.

CONTENTS
  atlas_attack.py
      Choosing records and forging labels at ATLAS scope.
  attacks.py
      One entry point that builds any attack's poison set.
  biorag_theme.py
      Shared plotnine theme and palette for every biorag-guard-sc figure.
  celltype_sweep.py
      Machinery shared by the per-celltype graph sweeps.
  dataset_config.py
      Resolve ``config/config.yaml`` for the dataset a run is about.
  defended_sweep.py
      Stage 7 — run a defence over a neighbourhood and measure what it costs.
  defense.py
      Defence layers: filter the retrieved neighbourhood, then vote.
  flood_common.py
      Helpers shared by the cluster-flood attack and the defences built against it.
  label_consistency.py
      Score each record against its own neighbourhood — the defence's input.
  modify.py
      Overwriting reference records, embeddings and all. SPEC.md §5.4, ``build 3``.
  ontology.py
      Cell Ontology parsing and tree distance.
  provenance.py
      Which code actually ran — the stamp a log header must carry.
  schemas.py
      Column names, dtypes and enums shared by every lane of biorag-guard-sc.
  retrieval_store.py
      Save and reload one condition's graph retrieval (neighbour ids, cosines,
      and the poison's SHA-1), one compressed npz per condition under
      <cache>/retrievals/, written atomically. Opt-in via
      BIORAG_SAVE_RETRIEVAL=1. Added 2026-10-04 so a defence or metric can be
      re-scored without re-running the non-reproducible graph path.
  selection.py
      How an attacker chooses which reference records to draw from.
  threshold.py
      The threshold dose and its donor-paired interval. SPEC.md §5.4c.
  vote.py
      The k-NN vote. One implementation, imported everywhere.
