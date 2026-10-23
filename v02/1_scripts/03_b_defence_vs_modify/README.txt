1_scripts/03_b_defence_vs_modify  --  analysis scripts -- write tables into 2_results, never figures
========================================================

NAMING. Every step in a module carries MM_NN_<access>_<name>:
  MM  module id, matching this directory (01a, 02a, 02b, 03a, 03b)
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

THIS MODULE. Defences measured against an attacker who can OVERWRITE existing
records. The edited record keeps a genuine embedding and a genuine donor id,
so re-encoding reproduces it and donor voting has nothing to catch -- v01
measured the whole stack at about -2 pp here, with 45 of 130 (celltype,
defence) combinations NET HARMFUL. That null is this module's headline and it
is reported with the same weight as any positive result (CLAUDE.md 1).
Steps here are 03b_NN_with_access_*.

Scope is a STEP, not a module: an atlas-scope and a per-celltype step both
live here if they share this arm.

README. This file is a bijection with the directory: every file has an entry,
every entry names a file that exists. Adding, deleting or renaming a file
requires updating this file in the same commit. An entry for a file that is not
present is a hallucination. Enforced by 1_scripts/00_pipeline/check_readme.py.

CONTENTS
  03b_01_with_access_layer_matrix.py
      Every defence layer against an attacker who can OVERWRITE reference
      records. Six configurations x four attack types x 26 celltypes x four
      doses, graph-faithful, with each configuration scored off the SAME
      retrieval so the defended-minus-undefended difference carries no
      insertion variance. Reports the null as the headline. The label-
      consistency threshold is set by a BUDGET (the bottom fraction of genuine
      records sacrificed), calibrated globally, not by the arbitrary 0.5.

      SCHEMA FIX 2026-09-30, forward-looking only. The donor frame now
      carries `selection` and the rows carry `n_conditions`; both were
      omitted. Without `selection` the cached_clean donor table cannot
      distinguish its hub and uniform rows -- all 194,688 are duplicates on
      the natural key -- and 03a_09 has to recover the pairing from row
      order. Without `n_conditions`, merge_shards cannot run its
      completeness check and Run 37 supplied --expect by hand. The
      COMMITTED tables predate the fix and are deliberately NOT regenerated:
      cached_clean is deterministic so a re-run would change nothing, and
      the graph arm is not, so a re-run would move published numbers within
      the +/-0.02 band and orphan every figure drawn from them.
      OVERRIDES 2026-10-01, all opt-in and all tagged. BIORAG_DOSES moves the
      ladder (the attack grid ran 02b_11 at 13 doses against this step's 4);
      BIORAG_TYPES names "geometry:selection" pairs and CONSTRUCTS them, since
      the default list never carried shifted/synthesised at uniform; a request
      mixing unmoved with moved geometries is refused because they take
      different retrieval paths. BIORAG_TAG names the run in the output file.
      With none of them set the step reproduces the committed tables.
      2026-10-04: rows carry `flood_spread`/`flood_jitter` for synthesised
      edits (0.99 / 1.0); within_accuracy is the vote actually returned, not
      the vote with the edited records removed. BIORAG_SAVE_RETRIEVAL=1 keeps
      each graph condition's neighbour ids, cosines, edited rows and new
      labels under <cache>/retrievals/03b_01_graph_<tag>/.
      BIORAG_TYPES takes an optional third field, geometry:selection:keep
      (2026-10-04), for 02b_11's shifted_inplace/keep attack.
  03b_03_with_access_repeatability.py
      The add_items band measured over 03b_01's 208 graph conditions x 12
      configurations instead of 02b_34's four. Reads two realisations of the
      graph arm side by side. HAS NEVER RUN: its second realisation was the
      graph rep2 run, which was dropped on 2026-10-01 before it started, so
      its input does not exist.
  03b_04_both_integrity.py
      Which integrity check -- hash of the index, hash of the label file,
      re-embedding -- catches which attack, and what each costs. Re-embedding
      wall time is reported unavailable, not estimated: the reference count
      matrices are not in this repository. HAS NEVER COMPLETED. Its one run
      (logged as 03b_02_integrity_20261001_092614.log, under the step id it
      carried before being renumbered) raised ValueError at the third attack:
      it routed `unmoved` through build_modify, which refuses it by design.
      FIXED 2026-10-01: edit_in_place builds unmoved itself, drawing rows with
      `choose` exactly as 03b_01's cached_clean arm does, so the record set it
      hashes is the one 03b_01 scored. Covered by tests/test_integrity.py.
      Not yet re-run.
      RENUMBERED 2026-10-01 from 03b_02, which is the id of the figure step
      03b_02_with_access_script.py; check_steps.py flagged the collision.
