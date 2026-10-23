2_results/GSE136831/02_b_attack_celltype  --  quantitative results only -- tables, arrays, p-values, coordinates
================================================================

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
  02b_01_no_access_celltype_sweep_by_donor.parquet
      Written by 02b_01_no_access_celltype_sweep_graph.py -- Step 02b_01 — every attackable celltype, dosed, on SCimilarity's own graph.
  02b_01_no_access_celltype_sweep_graph.parquet
      Written by 02b_01_no_access_celltype_sweep_graph.py -- Step 02b_01 — every attackable celltype, dosed, on SCimilarity's own graph.
  02b_02_no_access_flood_anchors.parquet
      Written by 02b_02_no_access_flood_anchors.py -- Step 02b_02 — how many places should a flood spread its poison across?
  02b_02_no_access_flood_by_donor.parquet
      Written by 02b_02_no_access_flood_anchors.py -- Step 02b_02 — how many places should a flood spread its poison across?
  02b_03_no_access.csv
      Written by 02b_03_no_access_script.py -- Step 02b_03 — retrieval plateaus at 80%, and the vote waits for parity.
  02b_04_no_access.csv
      Written by 02b_04_no_access_script.py -- Step 02b_04 — coverage keeps rising; wins stop. Flood's anchor curve.
  02b_05_with_access_hub_split_by_donor.parquet
      Written by 02b_05_with_access_hub_split_donor.py -- Step 02b_05 — does hub targeting still work if the attacker never saw your queries?
  02b_05_with_access_hub_split_donor.parquet
      Written by 02b_05_with_access_hub_split_donor.py -- Step 02b_05 — does hub targeting still work if the attacker never saw your queries?
  02b_06_no_access_umap_counts.parquet
      Written by 02b_06_no_access_umap_xy.py -- Step 02b_06 — where the poison actually lands, in two dimensions.
  02b_06_no_access_umap_xy.parquet
      Written by 02b_06_no_access_umap_xy.py -- Step 02b_06 — where the poison actually lands, in two dimensions.
  02b_07_no_access_CL0000057.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_07_no_access_CL0000084.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_07_no_access_CL0000097.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_07_no_access_CL0000158.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_07_no_access_CL0000160.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_07_no_access_CL0000186.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_07_no_access_CL0000192.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_07_no_access_CL0000235.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_07_no_access_CL0000236.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_07_no_access_CL0000451.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_07_no_access_CL0000583.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_07_no_access_CL0000623.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_07_no_access_CL0000646.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_07_no_access_CL0000784.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_07_no_access_CL0000786.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_07_no_access_CL0000815.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_07_no_access_CL0000860.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_07_no_access_CL0000875.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_07_no_access_CL0001065.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_07_no_access_CL0002062.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_07_no_access_CL0002063.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_07_no_access_CL0002144.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_07_no_access_CL0002543.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_07_no_access_CL1000271.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_07_no_access_CL1000413.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_07_no_access_CL1001567.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_07_no_access_counts.csv
      Written by 02b_07_no_access_script.py -- Step 02b_07 — the four insertion attacks, drawn where they land.
  02b_08_no_access_umap_global_xy.parquet
      Written by 02b_08_no_access_umap_global_xy.py -- Step 02b_08 — one map of the whole atlas, with every attack projected into it.
  02b_09_no_access_CL0000057.csv
      Written by 02b_09_no_access_script.py -- Step 02b_09 — how the reference changes as an attack is dosed up.
  02b_09_no_access_CL0000084.csv
      Written by 02b_09_no_access_script.py -- Step 02b_09 — how the reference changes as an attack is dosed up.
  02b_09_no_access_CL0000097.csv
      Written by 02b_09_no_access_script.py -- Step 02b_09 — how the reference changes as an attack is dosed up.
  02b_09_no_access_CL0000158.csv
      Written by 02b_09_no_access_script.py -- Step 02b_09 — how the reference changes as an attack is dosed up.
  02b_09_no_access_CL0000160.csv
      Written by 02b_09_no_access_script.py -- Step 02b_09 — how the reference changes as an attack is dosed up.
  02b_09_no_access_CL0000186.csv
      Written by 02b_09_no_access_script.py -- Step 02b_09 — how the reference changes as an attack is dosed up.
  02b_09_no_access_CL0000192.csv
      Written by 02b_09_no_access_script.py -- Step 02b_09 — how the reference changes as an attack is dosed up.
  02b_09_no_access_CL0000235.csv
      Written by 02b_09_no_access_script.py -- Step 02b_09 — how the reference changes as an attack is dosed up.
  02b_09_no_access_CL0000236.csv
      Written by 02b_09_no_access_script.py -- Step 02b_09 — how the reference changes as an attack is dosed up.
  02b_09_no_access_CL0000451.csv
      Written by 02b_09_no_access_script.py -- Step 02b_09 — how the reference changes as an attack is dosed up.
  02b_09_no_access_CL0000583.csv
      Written by 02b_09_no_access_script.py -- Step 02b_09 — how the reference changes as an attack is dosed up.
  02b_09_no_access_CL0000623.csv
      Written by 02b_09_no_access_script.py -- Step 02b_09 — how the reference changes as an attack is dosed up.
  02b_09_no_access_CL0000646.csv
      Written by 02b_09_no_access_script.py -- Step 02b_09 — how the reference changes as an attack is dosed up.
  02b_09_no_access_CL0000784.csv
      Written by 02b_09_no_access_script.py -- Step 02b_09 — how the reference changes as an attack is dosed up.
  02b_09_no_access_CL0000786.csv
      Written by 02b_09_no_access_script.py -- Step 02b_09 — how the reference changes as an attack is dosed up.
  02b_09_no_access_CL0000815.csv
      Written by 02b_09_no_access_script.py -- Step 02b_09 — how the reference changes as an attack is dosed up.
  02b_09_no_access_CL0000860.csv
      Written by 02b_09_no_access_script.py -- Step 02b_09 — how the reference changes as an attack is dosed up.
  02b_09_no_access_CL0000875.csv
      Written by 02b_09_no_access_script.py -- Step 02b_09 — how the reference changes as an attack is dosed up.
  02b_09_no_access_CL0001065.csv
      Written by 02b_09_no_access_script.py -- Step 02b_09 — how the reference changes as an attack is dosed up.
  02b_09_no_access_CL0002062.csv
      Written by 02b_09_no_access_script.py -- Step 02b_09 — how the reference changes as an attack is dosed up.
  02b_09_no_access_CL0002063.csv
      Written by 02b_09_no_access_script.py -- Step 02b_09 — how the reference changes as an attack is dosed up.
  02b_09_no_access_CL0002144.csv
      Written by 02b_09_no_access_script.py -- Step 02b_09 — how the reference changes as an attack is dosed up.
  02b_09_no_access_CL0002543.csv
      Written by 02b_09_no_access_script.py -- Step 02b_09 — how the reference changes as an attack is dosed up.
  02b_09_no_access_CL1000271.csv
      Written by 02b_09_no_access_script.py -- Step 02b_09 — how the reference changes as an attack is dosed up.
  02b_09_no_access_CL1000413.csv
      Written by 02b_09_no_access_script.py -- Step 02b_09 — how the reference changes as an attack is dosed up.
  02b_09_no_access_CL1001567.csv
      Written by 02b_09_no_access_script.py -- Step 02b_09 — how the reference changes as an attack is dosed up.
  02b_10_both_threshold_dose.parquet
      Written by 02b_10_both_threshold_dose.py -- Step 02b_10 — the dose at which each attack starts working, with an interval.
  02b_11_with_access_inplace.parquet
      Written by 02b_11_with_access_inplace.py -- Step 02b_11 — the attacker overwrites records, embeddings and all.
  02b_11_with_access_inplace_by_donor.parquet
      Written by 02b_11_with_access_inplace.py -- Step 02b_11 — the attacker overwrites records, embeddings and all.
  02b_11_with_access_inplace_by_donor_graph.parquet
      Written by 02b_11_with_access_inplace.py -- Step 02b_11 — the attacker overwrites records, embeddings and all.
  02b_11_with_access_inplace_graph.parquet
      Written by 02b_11_with_access_inplace.py -- Step 02b_11 — the attacker overwrites records, embeddings and all.
  02b_12_both_pooled_summary.parquet
      Written by 02b_12_both_pooled_summary.py -- Step 02b_12 — one curve per attack type, pooled across all 26 celltypes.
  02b_13_no_access_flood_geometry.parquet
      Written by 02b_13_no_access_flood_geometry.py -- Step 02b_13 — flood's two geometry knobs, titrated at last.
  02b_13_no_access_flood_geometry_by_donor.parquet
      Written by 02b_13_no_access_flood_geometry.py -- Step 02b_13 — flood's two geometry knobs, titrated at last.
  02b_14_no_access_flood_sites.parquet
      Written by 02b_14_no_access_flood_sites.py -- Step 02b_14 — does it matter WHERE the flood puts its clusters?
  02b_14_no_access_flood_sites_by_donor.parquet
      Written by 02b_14_no_access_flood_sites.py -- Step 02b_14 — does it matter WHERE the flood puts its clusters?
  02b_15_with_access_alpha_sweep.parquet
      Written by 02b_15_with_access_alpha_sweep.py -- Step 02b_15 — how far should a shifted record actually move?
  02b_15_with_access_alpha_sweep_by_donor.parquet
      Written by 02b_15_with_access_alpha_sweep.py -- Step 02b_15 — how far should a shifted record actually move?
  02b_15_with_access_alpha_sweep_by_donor_graph.parquet
      Written by 02b_15_with_access_alpha_sweep.py -- Step 02b_15 — how far should a shifted record actually move?
  02b_15_with_access_alpha_sweep_graph.parquet
      Written by 02b_15_with_access_alpha_sweep.py -- Step 02b_15 — how far should a shifted record actually move?
  02b_16_with_access_exact_vs_graph.parquet
      Written by 02b_16_with_access_exact_vs_graph.py -- Step 02b_16 — how wrong is the merge shortcut, measured rather than assumed?
  02b_17_no_access.csv
      Written by 02b_17_no_access_script.py -- Step 02b_17 — flood's geometry, and the setting that was costing 39%.
  02b_18_with_access.csv
      Written by 02b_18_with_access_script.py -- Step 02b_18 — how far to move a record, and two mechanisms that disagree.
  02b_19_both.csv
      Written by 02b_19_both_script.py -- Step 02b_19 — every attack type on one dose axis, pooled across the atlas.
  02b_20_no_access.csv
      Written by 02b_20_no_access_script.py -- Step 02b_20 — flood does not need real cells to stand on.
  02b_21_with_access.csv
      Written by 02b_21_with_access_script.py -- Step 02b_21 — the merge shortcut is close, two-sided, and concentrated.
  02b_22_both.csv
      Written by 02b_22_both_script.py -- Step 02b_22 — how much poison is enough, and how often the question has no answer.
  02b_23_with_access.csv
      Written by 02b_23_with_access_script.py -- Step 02b_23 — the write-access grid, where net attack success reports a null
  02b_24_with_access.csv
      Written by 02b_24_with_access_script.py -- Step 02b_24 — what knowing the hub structure buys, and why withholding your
  02b_25_no_access.csv
      Written by 02b_25_no_access_script.py -- Step 02b_25 — at parity the vote ties, and the answer depends on the tie-break.
  02b_26_no_access.csv
      Written by 02b_26_no_access_script.py -- Step 02b_26 — wrong, or differently wrong? It depends on the dose.
  02b_27_no_access.csv
      Written by 02b_27_no_access_script.py -- Step 02b_27 — does the celltype matter more than the dose? Only if you aim.
  02b_28_no_access.csv
      Written by 02b_28_no_access_script.py -- Step 02b_28 — the full ladder per celltype: reached, close enough, believed.
  02b_29_no_access.csv
      Written by 02b_29_no_access_script.py -- Step 02b_29 — the reference as shipped, before anyone touches it.
  02b_30_no_access_per_query.parquet
      Written by 02b_30_no_access_query_view.py -- Step 02b_30 — the same attacks, seen from the QUERY side: which cell was taken.
  02b_30_no_access_query_umap_xy.parquet
      Written by 02b_30_no_access_query_view.py -- Step 02b_30 — the same attacks, seen from the QUERY side: which cell was taken.
  02b_31_no_access.csv
      Written by 02b_31_no_access_script.py -- Step 02b_31 — the attack is surgical, and the denominator is what says so.
  02b_32_no_access.csv
      Written by 02b_32_no_access_script.py -- Step 02b_32 — the query dataset's own map, annotated before and after.
  02b_33_no_access.csv
      Written by 02b_33_no_access_script.py -- Step 02b_33 — the same annexation, cut to slide shape.
  02b_34_no_access_repeats.parquet
      Written by 02b_34_no_access_graph_repeatability.py -- Step 02b_34 — how far does the SAME condition move when you run it again?
  02b_34_no_access_spread.parquet
      Written by 02b_34_no_access_graph_repeatability.py -- Step 02b_34 — how far does the SAME condition move when you run it again?
