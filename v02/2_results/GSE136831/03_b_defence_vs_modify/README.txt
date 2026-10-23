2_results/GSE136831/03_b_defence_vs_modify  --  quantitative results only -- tables, arrays, p-values, coordinates
=================================================================

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
  03b_01_with_access_layer_matrix_cached_clean.parquet
      Written by 03b_01_with_access_layer_matrix.py under BIORAG_PHASE=
      cached_clean. 2,496 rows: 26 celltypes x 2 selections x 4 doses x 12
      configuration-rows. The MODIFY-arm null -- recompute is identical to no
      defence on all 208 conditions, donor_cap is worse than nothing in 103 of
      them, and the best layer is label_consistency at a 10% budget, -2.78 pp.
      Merged from 5 shards by merge_shards.py with --expect, because this run
      predates the n_conditions column its completeness check normally reads.
  03b_01_with_access_layer_matrix_by_donor_cached_clean.parquet
      The same conditions unreduced: 194,688 rows = 208 x 12 x 78 donors, for
      the donor-paired bootstrap. The unit of independence is the donor.
  03b_01_with_access_layer_matrix_cached_clean_v2.parquet
      The cached_clean arm REGENERATED under the 2026-09-30 schema fix, so it
      carries `n_conditions` (2,496 on every row). Same 2,496 rows as the
      table above, and BIT-IDENTICAL to it: joined 1:1 on (selection,
      defense, lc_budget, source, rate), max |diff| 0 on all ten metric
      columns. The only column difference is the added n_conditions. Merged
      from 5 shards (RUNLOG Run 41).
      SUPERSEDED for accuracy by cached_clean_v3 (RUNLOG Run 59): this table
      and the one above predate the 2026-10-04 accuracy fix, so their
      accuracy columns are the vote with the poison removed. Attack success
      is unaffected and matches v3 exactly.
  03b_01_with_access_layer_matrix_by_donor_cached_clean_v2.parquet
      The same conditions unreduced: 194,688 rows. This is the version whose
      donor frame carries `selection`, so hub and uniform rows are
      distinguishable on the natural key -- the defect 03a_09 had to work
      around by recovering the pairing from row order.
  03b_01_with_access_layer_matrix_graph.parquet
      Written by 03b_01_with_access_layer_matrix.py under BIORAG_PHASE=graph.
      2,496 rows: 26 celltypes x 2 geometries x 4 doses = 208 conditions x 12
      configuration-rows. The arm where the attacker MOVES the vector, and the
      one place in this module where a defence works: recompute takes
      shifted_inplace from 30.61% to -29.98 pp and synthesised_inplace from
      11.23% to -10.80 pp, because a moved vector is reproducible from no count
      matrix. label_consistency alone manages -5.14 and -1.06 pp, and every
      donor_cap is NET HARMFUL (+1.12 and +1.66 pp, hurting roughly twice as
      many conditions as it helps). Read with the cached_clean twin beside it:
      together they show the defence turns on whether the attacker forged the
      VECTOR, not on whether they had write access. Merged from 2 shards with
      --expect 2496 (Run 37).
  03b_01_with_access_layer_matrix_by_donor_graph.parquet
      The same conditions unreduced: 194,688 rows = 208 x 12 x 78 donors, for
      the donor-paired bootstrap. Merged from 2 shards with --expect 194688.
  03b_01_with_access_layer_matrix_graph_modhub.parquet
  03b_01_with_access_layer_matrix_graph_moduni.parquet
      Written by 03b_01_with_access_layer_matrix.py under BIORAG_PHASE=graph
      in the full-parity grid (RUNLOG Run 48), shifted and synthesised, 26
      celltypes x 12 configuration-rows per condition. modhub: hub selection
      at the 9 rates 0.005, 0.025, 0.1, 0.5, 0.7, 0.8, 0.85, 0.9, 1, which
      with the committed graph table's 4 make a 13-rate ladder; 468
      conditions, 5,616 rows. moduni: uniform selection at all 13 rates 0.002
      to 1; 676 conditions, 8,112 rows. Graph path, so outcome rates carry the
      add_items band. Merged from 2 shards each. Not yet analysed or drawn.
  03b_01_with_access_layer_matrix_by_donor_graph_modhub.parquet
  03b_01_with_access_layer_matrix_by_donor_graph_moduni.parquet
      The same conditions unreduced: 438,048 rows (modhub) and 632,736
      (moduni), conditions x 12 x 78 donors, each merged with --expect. Written
      after the 2026-09-30 schema fix, so the donor frame carries `selection`.
  03b_04_both_integrity.parquet
      Written by 03b_04_both_integrity.py (RUNLOG Run 49). 24 rows: 6
      celltypes x 4 attacks (relabel, flood, shifted_inplace,
      unmoved_inplace), with whether a hash of the index and a hash of the
      label file each detect the edit, and how many labels and embeddings it
      changed. unmoved_inplace is the one attack an index hash misses.
  03b_01_with_access_layer_matrix_graph_v3modhub.parquet
  03b_01_with_access_layer_matrix_graph_v3moduni.parquet
      Written by 03b_01_with_access_layer_matrix.py under BIORAG_PHASE=graph
      in DEFENCE RE-RUN v3 (RUNLOG Run 52), commit 1eafac83, after the
      2026-10-04 accuracy fix; supersede every earlier 03b_01 graph table.
      shifted and synthesised at hub (v3modhub) and uniform (v3moduni), the
      full 13-rate ladder 0.002-1, label_change relabel: 676 conditions x 12
      = 8,112 rows each. Synthesised edits carry flood_spread 0.99. Every
      retrieval saved under cache/retrievals/03b_01_graph_<tag>/ (676 each).
  03b_01_with_access_layer_matrix_by_donor_graph_v3modhub.parquet
  03b_01_with_access_layer_matrix_by_donor_graph_v3moduni.parquet
      The same conditions unreduced: 632,736 rows each (676 x 12 x 78).
  03b_01_with_access_layer_matrix_graph_v3modkeep.parquet
      Block 15 of DEFENCE RE-RUN v3 (RUNLOG Run 52, closing entry), same code
      and accuracy: shifted_inplace with label_change `keep`, at hub and
      uniform, 13 rates, 26 celltypes = 676 conditions x 12 = 8,112 rows.
      Unlike the insert keep tables this is an ATTACK, not a control: the
      vector moves and the label stays. Synthesised has no keep arm in the
      attack grid. Retrievals under cache/retrievals/03b_01_graph_v3modkeep/
      (676).
  03b_01_with_access_layer_matrix_by_donor_graph_v3modkeep.parquet
      The same conditions unreduced: 632,736 rows.
  03b_01_with_access_layer_matrix_graph_v3moduni_rep2.parquet
  03b_01_with_access_layer_matrix_by_donor_graph_v3moduni_rep2.parquet
      Second realisation of v3moduni (RUNLOG Run 56): same script, types,
      13 doses, 2 shards and seeds, at cae08c4c; only hnswlib's multi-threaded
      add_items differs. 8,112 and 632,736 rows. Retrievals under
      cache/retrievals/03b_01_graph_v3moduni_rep2/ (676). v3moduni ONLY --
      NOT the full grid: v3modhub, v3modkeep and the insert blocks have one
      realisation each.
  03b_01_with_access_layer_matrix_cached_clean_v3.parquet
      The label-only arm (unmoved_inplace, BIORAG_PHASE=cached_clean) on the
      fixed accuracy (RUNLOG Run 59), at 6129f7db/384db651: hub and uniform,
      the full 13-rate ladder 0.002-1, 26 celltypes = 676 conditions x 12 =
      8,112 rows. Supersedes cached_clean and cached_clean_v2 for accuracy.
      On their 2,496 shared rows attack success, n_poison and false_positive
      are identical; only within/global accuracy changed (the vote returned,
      not the vote with the poison removed). recompute equals none exactly on
      all 676 conditions. cached_clean is deterministic: no graph noise.
  03b_01_with_access_layer_matrix_by_donor_cached_clean_v3.parquet
      The same conditions unreduced: 632,736 rows (676 x 12 x 78), merged
      from 8 shards with --expect.
  03b_03_with_access_repeatability_v3moduni.parquet
      Written by 03b_03_with_access_repeatability.py (BIORAG_TAG=v3moduni,
      RUNLOG Run 56): v3moduni against v3moduni_rep2, 97,344 rows (8,112
      conditions x 12 metrics) with rep1, rep2 and the differences. Controls
      exact everywhere; within attack success differs by median 0, p95
      0.008, max 0.116 (classical monocyte, synthesised, dose 1.0).
  03b_02_with_access.csv
      Plotted values for 3_figures/.../03b_02_with_access.{pdf,png}. 36 rows =
      3 geometries x 12 configurations, each with mean attack success, the
      share of GENUINE records the layer rejects, accuracy, and that panel's
      undefended level. Redrawn 2026-10-10 on the v3 tables (graph_v3modhub,
      graph_v3moduni, cached_clean_v3): 676 conditions per geometry, 13
      doses, accuracy under attack. 8 of the 9 donor_cap points sit worse
      than undefended.
