1_scripts/00_pipeline  --  workflow DAG, the five checks, and the test suite
=============================================

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
  Snakefile
      TODO: describe
  aggregate.py
      Stage 5 — collapse per-cell raw results into per-condition rates.
  attack_table.py
      Regenerate SPEC.md §5.4's attack-type table from one source.
  build_atlas_umap.py
      Fit UMAP on the whole annotation index. Produces the layout every map reads.
  build_provenance.py
      Recover per-record provenance for the annotation index. THE MISSING PRODUCER.
  check_provenance.py
      Every artefact traces to the script that produced it, by its step id.
  check_readme.py
      Every directory carries a README.txt that matches its contents exactly.
  check_retrieval.py
      No figure draws the shortcut arm of a step that was also run on the graph.
  check_schema.py
      Every analysis table declares which attacker produced it.
  check_steps.py
      Every step id is unique across scripts, results and figures.
  embed_queries.py
      Stage 1 — turn held-out query cells into unit-norm embeddings.
  fetch_cellxgene.py
      Download the CELLxGENE query datasets named in the manifest.
  make_poison.py
      Stage 3 — build a poison set for one experimental condition.
  merge_shards.py
      Concatenate a sharded step's tables back into its canonical one.
  prepare_reference.py
      Stage 0 — prepare the reference: map labels to ontology ids, subsample, verify.
  retrieve_clean.py
      Stage 2 — retrieve each query's clean top-k neighbours, exactly once.
  retrieve_clean_deep.py
      A DEEPER clean neighbour list, for the attacks that remove records.
  sweep.py
      Stage 4 — merge a poison set into precomputed clean neighbours and vote.
  update_readme.py
      Refresh a directory's README.txt from what is actually in it.
  validate_shortcut.py
      Stage 6 — check the merge shortcut against an honestly rebuilt index.
  verify_cellxgene.py
      Verify every downloaded query dataset and inventory its cell types.
