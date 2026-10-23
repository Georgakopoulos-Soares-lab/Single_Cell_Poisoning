1_scripts/02_b_attack_celltype  --  analysis scripts -- write tables into 2_results, never figures
======================================================

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
  02b_01_no_access_celltype_sweep_graph.py
      Step 02b_01 — every attackable celltype, dosed, on SCimilarity's own graph.
  02b_02_no_access_flood_anchors.py
      Step 02b_02 — how many places should a flood spread its poison across?
  02b_05_with_access_hub_split_donor.py
      Step 02b_05 — does hub targeting still work if the attacker never saw your queries?
  02b_06_no_access_umap_xy.py
      Step 02b_06 — where the poison actually lands, in two dimensions.
  02b_08_no_access_umap_global_xy.py
      Step 02b_08 — one map of the whole atlas, with every attack projected into it.
  02b_10_both_threshold_dose.py
      Step 02b_10 — the dose at which each attack starts working, with an interval.
  02b_11_with_access_inplace.py
      Step 02b_11 — the attacker overwrites records, embeddings and all.
  02b_12_both_pooled_summary.py
      Step 02b_12 — one curve per attack type, pooled across all 26 celltypes.
  02b_13_no_access_flood_geometry.py
      Step 02b_13 — flood's two geometry knobs, titrated at last.
  02b_14_no_access_flood_sites.py
      Step 02b_14 — does it matter WHERE the flood puts its clusters?
  02b_15_with_access_alpha_sweep.py
      Step 02b_15 — how far should a shifted record actually move?
  02b_16_with_access_exact_vs_graph.py
      Step 02b_16 — how wrong is the merge shortcut, measured rather than assumed?
  02b_30_no_access_query_view.py
      Step 02b_30 — the same attacks, seen from the QUERY side: which cell was taken.
  02b_34_no_access_graph_repeatability.py
      Step 02b_34 — how far does the SAME condition move when you run it again?
