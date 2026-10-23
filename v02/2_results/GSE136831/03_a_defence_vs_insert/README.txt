2_results/GSE136831/03_a_defence_vs_insert  --  quantitative results only -- tables, arrays, p-values, coordinates
===============================================================

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

THIS MODULE. Defences measured against an attacker who can only INSERT --
submit records for acceptance, never touch what is already in the reference.
Poison is therefore always NEW, which is what quarantine exploits and what
makes label consistency scoreable before ingestion. Steps here are
03a_NN_no_access_*; the arm is in the directory name, so the <access> token
is redundant here and kept only because the contract requires it.

Scope is a STEP, not a module: an atlas-scope and a per-celltype step both
live here if they share this arm.

README. This file is a bijection with the directory: every file has an entry,
every entry names a file that exists. Adding, deleting or renaming a file
requires updating this file in the same commit. An entry for a file that is not
present is a hallucination. Enforced by 1_scripts/00_pipeline/check_readme.py.

CONTENTS
  03a_01_no_access_layer_matrix.parquet
      Written by 03a_01_no_access_layer_matrix.py. 2,496 rows: 26 celltypes x
      4 attacks x 2 doses = 208 conditions, each scored under 12 defence
      configurations. The INSERT-arm positive result, and the mirror of the
      03b module's null. Undefended within_attack_success averages 23.87%;
      stack at a 10% label-consistency budget moves it -23.77 pp, helping 173
      of 208 conditions and worsening none. The layers separate by MECHANISM,
      not by strength: recompute is EXACTLY 0.00 on duplicate and relabel (all
      52 conditions each) because those submit a genuine reference vector that
      re-encoding reproduces, and only catches flood, which synthesises one.
      label_consistency catches the other two, because the vector-label
      PAIRING is forged even when the vector is not. Merged from 3 shards.
  03a_01_no_access_layer_matrix_by_donor.parquet
      The same conditions unreduced: 194,688 rows = 208 x 12 x 78 donors, for
      the donor-paired bootstrap. The unit of independence is the donor.
      Merged from 3 shards with --expect 194688 (Run 37); a by_donor table
      cannot carry n_conditions, so its completeness check is supplied by hand.
  03a_01_no_access_layer_matrix_dose.parquet
      Written by 03a_01_no_access_layer_matrix.py under BIORAG_DOSES=
      0.002,0.01,0.05,0.1,1.25 -- the rungs the default table never covered,
      so together they make a 7-point ladder. 6,240 rows: 26 celltypes x 4
      attacks x 5 doses = 520 conditions x 12 configuration-rows, hub
      selection. Merged from 4 shards (RUNLOG Run 42). Predates the
      2026-10-01 `n_anchors` column, so it does not carry it; flood here is at
      the default 16,384 anchors. Not yet analysed or drawn.
  03a_01_no_access_layer_matrix_by_donor_dose.parquet
      The same conditions unreduced: 486,720 rows = 520 x 12 x 78 donors.
      Merged from 4 shards with --expect 486720.
  03a_01_no_access_layer_matrix_relhub.parquet
  03a_01_no_access_layer_matrix_reluni.parquet
  03a_01_no_access_layer_matrix_dupcon.parquet
  03a_01_no_access_layer_matrix_intuni.parquet
  03a_01_no_access_layer_matrix_flo16k.parquet
  03a_01_no_access_layer_matrix_flo64.parquet
  03a_01_no_access_layer_matrix_flo256.parquet
  03a_01_no_access_layer_matrix_flo1024.parquet
  03a_01_no_access_layer_matrix_flo4096.parquet
      Written by 03a_01_no_access_layer_matrix.py in the full-parity grid,
      one table per BIORAG_TAG (RUNLOG Run 48), 26 celltypes x 12
      configuration-rows per condition. relhub: relabel / hub at the 7 rates
      0.005, 0.025, 0.5, 0.7, 0.8, 0.85, 0.9, 182 conditions, 2,184 rows.
      flo16k: flood / hub at 16,384 anchors on the same 7 rates, 182
      conditions, 2,184 rows; with the default table and `_dose` these two
      complete a 14-rate hub ladder. reluni (relabel / uniform), dupcon
      (duplicate / concentrated), intuni (interpolate / uniform) and flo64,
      flo256, flo1024, flo4096 (flood / hub at that many anchors) each run all
      14 rates 0.002 to 1.25: 364 conditions, 4,368 rows. All carry
      `n_anchors` (flood only) and `n_conditions`. Merged from 3 shards each.
      Not yet analysed or drawn.
  03a_01_no_access_layer_matrix_by_donor_relhub.parquet
  03a_01_no_access_layer_matrix_by_donor_reluni.parquet
  03a_01_no_access_layer_matrix_by_donor_dupcon.parquet
  03a_01_no_access_layer_matrix_by_donor_intuni.parquet
  03a_01_no_access_layer_matrix_by_donor_flo16k.parquet
  03a_01_no_access_layer_matrix_by_donor_flo64.parquet
  03a_01_no_access_layer_matrix_by_donor_flo256.parquet
  03a_01_no_access_layer_matrix_by_donor_flo1024.parquet
  03a_01_no_access_layer_matrix_by_donor_flo4096.parquet
      The same conditions unreduced, conditions x 12 x 78 donors: 170,352 rows
      for relhub and flo16k, 340,704 for the other seven. Each merged with
      --expect at that count.
  03a_02_no_access_donor_identity.parquet
      Written by 03a_02_no_access_donor_identity.py. 1,536 rows: 6 celltypes x
      2 attacks x 2 doses x 8 defender caps x 8 attacker split counts. The
      donor cap measured against an attacker who knows about it. It is a
      CROSSOVER, not a defence: at 1 donor identity the cap cuts attack
      success from 35.86% to 0.77%, and at 40 identities the same cap raises
      it to 75.46% -- more than double the undefended rate. cap 50 reproduces
      cap 0 exactly (0 helped, 0 hurt of 192), which is the internal control.
      Accuracy is flat in splits at every cap, so the cap's cost does not
      depend on the attack, only its benefit does.
  03a_02_no_access_donor_identity_by_donor.parquet
      The same conditions unreduced: 119,808 rows, for the donor-paired
      bootstrap.
  03a_02_no_access_donor_identity_all26.parquet
      Written by 03a_02_no_access_donor_identity.py under
      BIORAG_ALL_CELLTYPES=1. 6,656 rows: all 26 attackable celltypes x 2
      attacks x 2 doses x 8 caps x 8 splits. The same crossover grid as the
      six-celltype table, lifted to the whole set so a headline does not rest
      on six of twenty-six. Merged from 2 shards (RUNLOG Run 43). Not yet
      analysed or drawn.
  03a_02_no_access_donor_identity_by_donor_all26.parquet
      The same conditions unreduced: 519,168 rows, for the donor-paired
      bootstrap. Merged from 2 shards.
  03a_03_no_access_vote_weighting.parquet
      Written by 03a_03_no_access_vote_weighting.py. 216 rows: 6 celltypes x
      4 attacks x 5 weighting schemes plus 4 k-sweep controls. Weighting is a
      MARGINAL layer and this table is the evidence for saying so: on top of
      the stack, rank weighting reaches 0.122% attack success against uniform
      weighting's 0.1803%, a 0.0583 pp gap. Across ALL FIVE schemes the span is
      0.1815 pp, because softmax is an outlier at 0.3034% -- the most
      aggressive scheme and the worst of them. The
      reason is in the same table -- poison_rank1 averages 0.0048 and peaks at
      0.0314, so the poisoned record is almost never the nearest neighbour
      and there is little for a distance weighting to down-weight. k matters
      more than weight: undefended attack success runs 15.21% at k=5 to
      32.26% at k=50.
  03a_03_no_access_vote_weighting_by_donor.parquet
      The same conditions unreduced: 16,848 rows, for the donor-paired
      bootstrap.
  03a_04_both.csv
      Plotted values for 3_figures/.../03a_04_both.{pdf,png}. One row per
      (arm, defence layer, condition): the three-arm contrast, 23,088 rows
      (insert 10,920, vector-moved 8,112, label-only 4,056). Redrawn
      2026-10-10 on the v3 tables: insert = relabel hub+uniform, duplicate
      concentrated, interpolate uniform, flood 1,024 anchors.
  03a_05_no_access.csv
      Plotted values for 3_figures/.../03a_05_no_access.{pdf,png}. The 64
      cap-by-split cells, each with its mean attack success, the matching
      no-cap value and the signed difference the colour encodes.
  03a_06_no_access_umap.parquet
      Written by 03a_06_no_access_umap.py. 14,240 points: UMAP coordinates for
      two contrasting conditions, with a per-RECORD verdict from each defence
      layer. CL:0002062 relabel -- recompute admits 869 of 869 poison records,
      label consistency admits 0. CL:0000583 flood -- recompute admits 0 of
      3,000. Held here rather than in cache/ because a fit is a computed
      quantity and coordinates are a result; v01 put its atlas fit in
      gitignored cache/ and left two figures unreproducible from a clean clone.
  03a_07_no_access.csv
      Plotted values for 3_figures/.../03a_07_no_access.{pdf,png}. One row per
      (point, panel it survives into), 46,222 rows across the four panels.
  03a_08_no_access.csv
      Plotted values for 3_figures/.../03a_08_no_access.{pdf,png}. 216 rows,
      one per (condition, configuration), carrying both the true attack
      success and the log-axis floor the zero-valued conditions are drawn at.
  03a_09_both_intervals.parquet
      Written by 03a_09_both_intervals.py. 7,488 rows = 3 arms x 208
      conditions x 12 configurations. Per metric: the level, its 95% interval,
      and the PAIRED change against that condition's `none` row with its own
      interval. The headline the point estimates could not state: recompute
      against a label-only overwrite is delta 0.0000 with interval [0, 0] on
      all 208 conditions -- degenerate, not merely insignificant -- while
      donor_cap1 significantly HARMS in 71 of 208 on the vector-moved arm
      against 38 significant reductions.
      SUPERSEDED (pre-v3, pre accuracy fix, 4 doses): the v3 tag tables and
      03a_09_both_intervals_cached_clean_v3.parquet replace it.
  03a_09_both_intervals_relhub.parquet
  03a_09_both_intervals_reluni.parquet
  03a_09_both_intervals_dupcon.parquet
  03a_09_both_intervals_intuni.parquet
  03a_09_both_intervals_flo16k.parquet
  03a_09_both_intervals_flo64.parquet
  03a_09_both_intervals_flo256.parquet
  03a_09_both_intervals_flo1024.parquet
  03a_09_both_intervals_flo4096.parquet
  03a_09_both_intervals_modhub.parquet
  03a_09_both_intervals_moduni.parquet
      Written by 03a_09_both_intervals.py under BIORAG_TAG=<tag> (RUNLOG Run
      50): donor-paired 95% intervals, level and paired change against
      `none`, for every layer over that tag's layer matrix from Run 48. One
      row per (condition, configuration), so each table has the same row
      count as its layer matrix: 2,184 for relhub and flo16k, 4,368 for the
      other seven insert tags, 5,616 for modhub, 8,112 for moduni. Every row
      carries `tag`; flood tags also carry `n_anchors`. Insert tags are arm
      `insert`, modhub and moduni arm `modify_graph`. Not yet analysed or
      drawn.
  03a_01_no_access_layer_matrix_v3relhub.parquet
  03a_01_no_access_layer_matrix_v3reluni.parquet
  03a_01_no_access_layer_matrix_v3dupcon.parquet
  03a_01_no_access_layer_matrix_v3intuni.parquet
  03a_01_no_access_layer_matrix_v3flo64.parquet
  03a_01_no_access_layer_matrix_v3flo256.parquet
  03a_01_no_access_layer_matrix_v3flo1024.parquet
  03a_01_no_access_layer_matrix_v3flo4096.parquet
  03a_01_no_access_layer_matrix_v3flo16384.parquet
      Written by 03a_01_no_access_layer_matrix.py in DEFENCE RE-RUN v3 (RUNLOG
      Run 52), at commit 1eafac83, AFTER the 2026-10-04 accuracy fix: these
      supersede every earlier 03a_01 table. within_accuracy is the vote
      actually returned (earlier tables recorded the vote with the poison
      removed). Each attack runs the full 14-rate ladder 0.002-1.25 in ONE
      table at the attack grid's selection: relabel at hub and at uniform,
      duplicate at concentrated, interpolate at uniform, flood at 64, 256,
      1,024, 4,096 and 16,384 anchors. Flood is built at spread 0.999 /
      jitter 1.0 and says so in flood_spread / flood_jitter: a stronger flood
      than 02b_02's 0.99, by the user's decision to keep both and label them.
      label_change is `relabel` on every row. 364 conditions x 12
      configuration-rows = 4,368 rows each. Every retrieval is saved under
      cache/retrievals/03a_01_<tag>/ (364 files per tag).
  03a_01_no_access_layer_matrix_by_donor_v3relhub.parquet
  03a_01_no_access_layer_matrix_by_donor_v3reluni.parquet
  03a_01_no_access_layer_matrix_by_donor_v3dupcon.parquet
  03a_01_no_access_layer_matrix_by_donor_v3intuni.parquet
  03a_01_no_access_layer_matrix_by_donor_v3flo64.parquet
  03a_01_no_access_layer_matrix_by_donor_v3flo256.parquet
  03a_01_no_access_layer_matrix_by_donor_v3flo1024.parquet
  03a_01_no_access_layer_matrix_by_donor_v3flo4096.parquet
  03a_01_no_access_layer_matrix_by_donor_v3flo16384.parquet
      The same conditions unreduced: 340,704 rows each (364 x 12 x 78
      donors), merged with --expect.
  03a_01_no_access_layer_matrix_v3relunikeep.parquet
  03a_01_no_access_layer_matrix_v3dupconkeep.parquet
  03a_01_no_access_layer_matrix_v3flo1024keep.parquet
      The keep variants of DEFENCE RE-RUN v3 (RUNLOG Run 52, closing entry),
      same code and accuracy as the tables above: label_change `keep` on every
      row, so the poison carries the source's own label. These are CONTROLS
      (SPEC.md 5.4), not attacks: relabel at uniform, duplicate at
      concentrated, and flood at 1,024 anchors only (spread 0.999 / jitter
      1.0). 364 conditions x 12 = 4,368 rows each; retrievals under
      cache/retrievals/03a_01_<tag>/ (364 each).
  03a_01_no_access_layer_matrix_by_donor_v3relunikeep.parquet
  03a_01_no_access_layer_matrix_by_donor_v3dupconkeep.parquet
  03a_01_no_access_layer_matrix_by_donor_v3flo1024keep.parquet
      The same conditions unreduced: 340,704 rows each, merged with --expect.
  03a_09_both_intervals_v3relhub.parquet
  03a_09_both_intervals_v3reluni.parquet
  03a_09_both_intervals_v3dupcon.parquet
  03a_09_both_intervals_v3intuni.parquet
  03a_09_both_intervals_v3flo64.parquet
  03a_09_both_intervals_v3flo256.parquet
  03a_09_both_intervals_v3flo1024.parquet
  03a_09_both_intervals_v3flo4096.parquet
  03a_09_both_intervals_v3flo16384.parquet
  03a_09_both_intervals_v3modhub.parquet
  03a_09_both_intervals_v3moduni.parquet
      Written by 03a_09_both_intervals.py under BIORAG_TAG (RUNLOG Run 53):
      donor-paired 95% intervals over the v3 layer matrices. Same row count as
      each matrix: 4,368 per insert tag, 8,112 for v3modhub and v3moduni.
      The intervals on accuracy are, for the first time, intervals on
      accuracy under attack. Supersede every earlier 03a_09 table. Re-run at
      the end of Run 52 (RUNLOG Run 54) with byte-identical output.
  03a_09_both_intervals_v3relunikeep.parquet
  03a_09_both_intervals_v3dupconkeep.parquet
  03a_09_both_intervals_v3flo1024keep.parquet
  03a_09_both_intervals_v3modkeep.parquet
      The same intervals for the four keep tags (RUNLOG Run 54): 4,368 rows
      per insert tag, 8,112 for v3modkeep.
  03a_09_both_intervals_v3moduni_rep2.parquet
      The same intervals for the second realisation of v3moduni (RUNLOG Run
      57), 8,112 rows. For comparison with v3moduni only; v3moduni's table is
      the one to report. 1,821 rows with an interval excluding zero against
      v3moduni's 1,828.
  03a_09_both_intervals_cached_clean_v3.parquet
      The same intervals for the label-only arm on the fixed accuracy
      (RUNLOG Run 59): BIORAG_TAG=v3 over the cached_clean_v3 pair, arm
      `modify_cached_clean`, 8,112 rows. recompute's delta and both bounds
      are zero on 676 of 676 conditions. Supersedes the label-only rows of
      03a_09_both_intervals.parquet.
  03a_10_both.csv
      Plotted values for 3_figures/.../03a_10_both.{pdf,png}. 26,520 rows, one
      per (arm, layer, condition), carrying the paired delta, its interval and
      the helps/harms/no-effect verdict the colour encodes. Redrawn 2026-10-10
      on the v3 intervals; `block` names the source table (the nine insert
      tags incl. all five flood widths, v3modhub, v3moduni, cached_clean_v3).
  03a_02_no_access_donor_identity_all26_v3.parquet
  03a_02_no_access_donor_identity_by_donor_all26_v3.parquet
  03a_03_no_access_vote_weighting_v3.parquet
  03a_03_no_access_vote_weighting_by_donor_v3.parquet
      Item 4 part 1 (RUNLOG Run 55), at dac565c0: 03a_02 at all 26 celltypes
      and 03a_03 re-run with accuracy under attack, duplicate at the grid's
      concentrated selection (the earlier tables built it at hub). Supersede
      the untagged and _all26 tables. 6,656 / 519,168 rows (03a_02) and
      216 / 16,848 (03a_03). NOT TITRATIONS: 03a_02 has 2 doses (0.25, 1.0)
      of the 14-rate ladder; 03a_03 has one dose (1.0) on 6 celltypes.
