2_results/GSE136831/logs  --  execution logs -- proof a run happened (CLAUDE.md s1)
================================================

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
  02a_01_no_access_uniform_graph_.log
      Written by 02a_01_no_access_uniform_graph.py -- Step 02a_01 — scattered poisoning, measured on SCimilarity's actual graph.
  02a_03_with_access_relabel_inplace_20260925_002748.log
      Written by 02a_03_with_access_relabel_inplace.py -- Step 02a_03 — scattered poisoning by an attacker who can OVERWRITE.
  02b_01_no_access_celltype_sweep_graph_20260924_235809.log
      Written by 02b_01_no_access_celltype_sweep_graph.py -- Step 02b_01 — every attackable celltype, dosed, on SCimilarity's own graph.
  02b_02_bydonor_merge_.log
      TODO: describe
  02b_02_bydonor_merge_20260927_091902.log
      TODO: describe
  02b_02_merge_.log
      TODO: describe
  02b_02_no_access_flood_anchors_shard0_20260926_022205.log
      Written by 02b_02_no_access_flood_anchors.py -- Step 02b_02 — how many places should a flood spread its poison across?
  02b_02_no_access_flood_anchors_shard1_20260926_022205.log
      Written by 02b_02_no_access_flood_anchors.py -- Step 02b_02 — how many places should a flood spread its poison across?
  02b_02_no_access_flood_anchors_shard2_20260926_022205.log
      Written by 02b_02_no_access_flood_anchors.py -- Step 02b_02 — how many places should a flood spread its poison across?
  02b_05_with_access_hub_split_donor_20260924_234009.log
      Written by 02b_05_with_access_hub_split_donor.py -- Step 02b_05 — does hub targeting still work if the attacker never saw your queries?
  02b_10_both_threshold_dose_20260927_205921.log
      Written by 02b_10_both_threshold_dose.py -- Step 02b_10 — the dose at which each attack starts working, with an interval.
  02b_10_both_threshold_dose_20260927_210210.log
      Written by 02b_10_both_threshold_dose.py -- Step 02b_10 — the dose at which each attack starts working, with an interval.
  02b_10_both_threshold_dose_20260927_210315.log
      Written by 02b_10_both_threshold_dose.py -- Step 02b_10 — the dose at which each attack starts working, with an interval.
  02b_10_both_threshold_dose_20260927_210347.log
      Written by 02b_10_both_threshold_dose.py -- Step 02b_10 — the dose at which each attack starts working, with an interval.
  02b_10_both_threshold_dose_20260927_211503.log
      Written by 02b_10_both_threshold_dose.py -- Step 02b_10 — the dose at which each attack starts working, with an interval.
  02b_11_graph_bydonor_merge_20260927_205920.log
      TODO: describe
  02b_11_graph_merge_20260927_205920.log
      TODO: describe
  02b_11_graph_shard0_20260927_033956.log
      TODO: describe
  02b_11_graph_shard1_20260927_033956.log
      TODO: describe
  02b_11_graph_shard2_20260927_033956.log
      TODO: describe
  02b_11_with_access_inplace_20260925_002804.log
      Written by 02b_11_with_access_inplace.py -- Step 02b_11 — the attacker overwrites records, embeddings and all.
  02b_12_both_pooled_summary_20260927_205921.log
      Written by 02b_12_both_pooled_summary.py -- Step 02b_12 — one curve per attack type, pooled across all 26 celltypes.
  02b_12_both_pooled_summary_20260927_211541.log
      Written by 02b_12_both_pooled_summary.py -- Step 02b_12 — one curve per attack type, pooled across all 26 celltypes.
  02b_13_no_access_flood_geometry_20260925_022913.log
      Written by 02b_13_no_access_flood_geometry.py -- Step 02b_13 — flood's two geometry knobs, titrated at last.
  02b_14_no_access_flood_sites_20260924_232945.log
      Written by 02b_14_no_access_flood_sites.py -- Step 02b_14 — does it matter WHERE the flood puts its clusters?
  02b_15_graph_bydonor_merge_.log
      TODO: describe
  02b_15_graph_bydonor_merge_20260927_091903.log
      TODO: describe
  02b_15_graph_merge_.log
      TODO: describe
  02b_15_graph_merge_20260925_232413.log
      TODO: describe
  02b_15_graph_shard0_20260925_232413.log
      TODO: describe
  02b_15_graph_shard0_20260926_194702.log
      TODO: describe
  02b_15_graph_shard1_20260925_232413.log
      TODO: describe
  02b_15_graph_shard1_20260926_194702.log
      TODO: describe
  02b_15_graph_shard2_20260925_232413.log
      TODO: describe
  02b_15_graph_shard2_20260926_194702.log
      TODO: describe
  02b_15_with_access_alpha_sweep_20260925_103739.log
      Written by 02b_15_with_access_alpha_sweep.py -- Step 02b_15 — how far should a shifted record actually move?
  02b_16_with_access_exact_vs_graph_20260927_210439.log
      Written by 02b_16_with_access_exact_vs_graph.py -- Step 02b_16 — how wrong is the merge shortcut, measured rather than assumed?
  02b_30_no_access_query_view_20260928_080629.log
      Written by 02b_30_no_access_query_view.py -- Step 02b_30 — the same attacks, seen from the QUERY side: which cell was taken.
  02b_30_no_access_query_view_20260928_081352.log
      Written by 02b_30_no_access_query_view.py -- Step 02b_30 — the same attacks, seen from the QUERY side: which cell was taken.
  02b_34_no_access_graph_repeatability_20260928_085820.log
      Written by 02b_34_no_access_graph_repeatability.py -- Step 02b_34 — how far does the SAME condition move when you run it again?
  retrieve_clean_deep_20260924_234009.log
      TODO: describe
  shard_cleanup_merge_20260927_211250.log
      TODO: describe
  03b_01_prepare_20260929_135619.log
      Written by 03b_01_with_access_layer_matrix.py under BIORAG_PREPARE=1 -- the
      once-only work the sweep shards must not each repeat: the clean graph
      retrieval (19 s) and the reference-kNN top-up (27,583 rows, 19 s). Both
      open the 9.3 GB index, so five shards doing it concurrently would need
      ~100 GB and would race to write the same cache file.
  03b_01_cached_clean_shard0_20260929_135619.log
      Written by 03b_01_with_access_layer_matrix.py -- shard 0 of 5 of the
      cached_clean arm: relabel_inplace against six defence
      configurations, 26 celltypes across the shards, four doses.
  03b_01_cached_clean_shard1_20260929_135619.log
      Written by 03b_01_with_access_layer_matrix.py -- shard 1 of 5 of the
      cached_clean arm: relabel_inplace against six defence
      configurations, 26 celltypes across the shards, four doses.
  03b_01_cached_clean_shard2_20260929_135619.log
      Written by 03b_01_with_access_layer_matrix.py -- shard 2 of 5 of the
      cached_clean arm: relabel_inplace against six defence
      configurations, 26 celltypes across the shards, four doses.
  03b_01_cached_clean_shard3_20260929_135619.log
      Written by 03b_01_with_access_layer_matrix.py -- shard 3 of 5 of the
      cached_clean arm: relabel_inplace against six defence
      configurations, 26 celltypes across the shards, four doses.
  03b_01_cached_clean_shard4_20260929_135619.log
      Written by 03b_01_with_access_layer_matrix.py -- shard 4 of 5 of the
      cached_clean arm: relabel_inplace against six defence
      configurations, 26 celltypes across the shards, four doses.
  03a_01_shard0_20260929_153311.log
      Written by 03a_01_no_access_layer_matrix.py -- shard 0 of 2 of the INSERT
      arm: four attacks against eight defence configurations, hub selection,
      two doses.
  03a_01_shard1of4_20260929_153311.log
      Written by 03a_01_no_access_layer_matrix.py -- shard 1 of 4, run beside
      shard 0 of 2. [1::4] u [3::4] == [1::2], so 0of2, 1of4 and 3of4
      partition the 26 celltypes exactly. Two concurrent is the memory
      ceiling: each loads the 9.3 GB index and peaks near 20 GB.
  03b_01_graph_shard0_20260929_152850.log
      ABANDONED. Written by 03b_01_with_access_layer_matrix.py under
      BIORAG_PHASE=graph; died after 4 of 104 conditions on the first
      synthesised condition, which needs n_anchors/spread/jitter that the
      step did not pass. Kept because RUNLOG cites it: a run that failed is
      still a run, and the four conditions it did complete were discarded.
  03b_01_graph_shard1_20260929_152850.log
      ABANDONED, same cause as shard 0 above.
  03a_01_shard3of4_20260929_153311.log
      Written by 03a_01_no_access_layer_matrix.py -- shard 3 of 4, launched by
      hand after both earlier shards finished inside the scheduler's 60 s poll
      window. With 0of2 and 1of4 it partitions the 26 celltypes exactly. Its
      file was renamed shard3 -> shard2 before merging, because merge_shards.py
      requires contiguous indices; the celltypes still partition, and all four
      of its checks run on content rather than filenames.
  03b_01_graph_shard0_20260930_084557.log
      Written by 03b_01_with_access_layer_matrix.py under BIORAG_PHASE=graph --
      shard 0 of 2, 104 conditions, all 26 celltypes. The completed re-run of
      the arm abandoned on 2026-09-29 (see the _152850 logs above); this one
      passes n_anchors/spread/jitter, which is what the earlier attempt lacked.
      Driven by archive/workers_2026-09/drive_defence.sh.
  03b_01_graph_shard1_20260930_084557.log
      As shard 0 above, the complementary 13 celltypes.
  03a_02_20260930_084557.log
      Written by 03a_02_no_access_donor_identity.py -- single shard, 24
      conditions x 64 cap-by-split cells. Records the clean-graph baseline and
      "genuine donors per top-50: median 11, IQR 7-17", which is the number the
      cap crossover turns on. Driven by drive_defence.sh.
  03a_03_20260930_084557.log
      Written by 03a_03_no_access_vote_weighting.py -- single shard, 24
      conditions x 9 configurations. Logs poison@rank1 per condition; it never
      exceeds 0.026 in any of the 24. Driven by drive_defence.sh, concurrently
      with 03a_02 above.
  drive_defence_20260930_084557.done
      The completion stamp written by drive_defence.sh, holding one line:
      "DRIVER DONE 2026-09-30 14:16". It marks the driver exiting, NOT the work
      being complete -- three of its four merges had silently refused, and
      Run 37 finished them. Kept because it is what the waiting tasks watched.
  merge_shards_defence_20260930_142306.log
      Written by merge_shards.py (Run 37). The three merges the driver left:
      03a_01's by_donor table and both of 03b_01's graph tables, each with the
      --expect the driver omitted. Its header records where each expectation
      came from, because a completeness check supplied by hand is only as good
      as its provenance.
  03a_06_umap_20260930_161040.log
      Written by 03a_06_no_access_umap.py -- single process, no sharding. Fits
      the two conditions the defence triptych draws and records the per-record
      defence verdicts in its closing lines, which are what the figure's claim
      rests on: recompute admits every relabel record and no flood record.
  03a_09_intervals_20260930_164928.log
      Written by 03a_09_both_intervals.py -- the donor-paired bootstrap over
      all three arms, 17 s, reading only committed tables. Records the two
      checks the step's validity rests on: the cached_clean `selection`
      recovery verified at 2.98e-08 over 1,248 comparisons, and the point
      estimates agreeing with the committed layer matrices on 14,976.
      Overwritten twice before this run succeeded -- see RUNLOG Run 39 for the
      two float32 and null-key defects that produced silently wrong joins.
  03b_01_cached_clean_shard0_20260930_171044.log
      Written by 03b_01_with_access_layer_matrix.py -- a VERIFICATION run of one
      celltype of 26, proving the 2026-09-30 schema fix emits `selection` in the
      donor frame and `n_conditions` in the rows, and that the cached_clean arm
      is deterministic (it reproduced the committed table exactly, max |diff|
      0.000e+00 over 96 rows). Its two `_shard0` parquets were deleted after the
      check: they duplicate committed data and a stray shard file beside a
      canonical table would be double-counted by anything globbing for shards.
      See RUNLOG Run 40.
  03b_01_cached_clean_shard0_20260930_203037.log
  03b_01_cached_clean_shard1_20260930_203037.log
  03b_01_cached_clean_shard2_20260930_203037.log
  03b_01_cached_clean_shard3_20260930_203037.log
  03b_01_cached_clean_shard4_20260930_203037.log
      Written by 03b_01_with_access_layer_matrix.py under BIORAG_PHASE=
      cached_clean, 5 shards concurrent, driven by
      archive/workers_2026-10/drive_rerun.sh (phase 1). The full regeneration
      of the cached_clean arm under the schema fix; it produced the _v2
      tables, bit-identical to the committed ones. See RUNLOG Run 41.
  finish_phase1_20260930_210551.log
      Written by archive/workers_2026-10/finish_phase1.sh: the two
      merge_shards calls that turned Run 41's shards into the _v2 tables:
      2,496 rows self-verified via n_conditions, 194,688 donor rows.
  finish_phase1.done
      Empty marker finish_phase1.sh touched on exit. Kept because it is what
      the next driver waited on.
  03a_01_dose_shard0of4_20260930_211732.log
  03a_01_dose_shard1of4_20260930_211732.log
  03a_01_dose_shard2of4_20260930_211732.log
  03a_01_dose_shard3of4_20260930_211732.log
      Written by 03a_01_no_access_layer_matrix.py under BIORAG_DOSES=
      0.002,0.01,0.05,0.1,1.25, driven by archive/workers_2026-10/
      drive_dose.sh, finished by finish_dose.sh. Meant to run two at a
      time; the driver's slot check over-counted (it matched watcher shells)
      and would never have launched shards 2-3, so they were started
      separately and all four then ran concurrently. Shards 2-3
      started 49 min after 0-1 and under a different working-tree diff (see
      each log's [provenance] line). See RUNLOG Run 42.
  drive_dose_20261001_063301.progress
      Merge record, written by archive/workers_2026-10/finish_dose.sh (which
      took over drive_dose.sh's wait and merge): 4 shards -> 6,240 and
      486,720 rows.
  drive_dose_20261001_063302.done
      finish_dose.sh's completion marker.
  watchdog_20260930_231702.tsv
      First start of archive/workers_2026-10/watchdog.sh; one sample, then
      superseded 40 s later by the file below. Kept because it was written.
  watchdog_20260930_231742.tsv
      Resource samples every 5 min, 2026-09-30 23:17 to 2026-10-01 09:59:
      python count, total and max RSS, CPU idle, swap, step. This is the
      evidence for Run 47's failure -- swap 10.0 GB at 09:49 with one process,
      32.1 GB at 09:54 with eight -- and for the 3-shard ceiling chosen after
      it: four 03a_01 processes held at <= 25.2 GB RSS, <= 20.2 GB swap for
      86 samples.
  03a_02_all26_shard0_20261001_063513.log
  03a_02_all26_shard1_20261001_063513.log
      Written by 03a_02_no_access_donor_identity.py under
      BIORAG_ALL_CELLTYPES=1, 2 shards concurrent, driven by
      archive/workers_2026-10/drive_03a02.sh. See RUNLOG Run 43.
  drive_03a02_20261001_063513.progress
      drive_03a02.sh's merge record: 2 shards -> 6,656 and 519,168 rows.
  drive_03a02_20261001_063513.done
      drive_03a02.sh's completion marker.
  03b_02_integrity_20261001_092614.log
      Written by v02/1_scripts/03_b_defence_vs_modify/03b_02_both_integrity.py,
      since RENUMBERED 03b_04_both_integrity.py (the id collided with figure
      step 03b_02). A FAILED run: ValueError from build_modify at the third
      attack, after two attacks' hash checks. No table was written. See
      RUNLOG Run 44.
  drive_03b02_20261001_092614.done
      drive_03b02.sh's completion marker. MISLEADING ON ITS FACE: the driver
      wrote it without checking the exit status, so it sits beside a
      traceback. It marks that the process ended, not that it succeeded.
  03b_01_graph_memprobe_20261001_092717.log
      Written by 03b_01_with_access_layer_matrix.py under BIORAG_PHASE=graph,
      shard 0 of 26 -- one celltype, a MEMORY PROBE, not a result. Its last
      line, PEAK_RSS_GB 17.37, is what set the modify-arm concurrency to 2.
      Its shard parquets were deleted as planned. See RUNLOG Run 45.
  drive_graphprobe_20261001_092717.done
      archive/workers_2026-10/drive_graphprobe.sh's completion marker.
  smoke_parity_20261001_094200.log
      Written by archive/workers_2026-10/smoke_parity.sh: one dose of each
      new override combination (reluni, dupcon, floanc, moduni) checked for
      the column values its overrides promise. Ends SMOKE PASS. Its smoke
      parquets were deleted. See RUNLOG Run 46.
  smoke_parity.ok
      Empty marker the smoke test wrote on pass; the parity driver's gate.
  03a_01_relhub_shard0of8_20261001_092820.log
  03a_01_relhub_shard1of8_20261001_092820.log
  03a_01_relhub_shard2of8_20261001_092820.log
  03a_01_relhub_shard3of8_20261001_092820.log
  03a_01_relhub_shard4of8_20261001_092820.log
  03a_01_relhub_shard5of8_20261001_092820.log
  03a_01_relhub_shard6of8_20261001_092820.log
  03a_01_relhub_shard7of8_20261001_092820.log
      Written by 03a_01_no_access_layer_matrix.py under BIORAG_ATTACKS=
      relabel BIORAG_SELECTIONS=hub BIORAG_DOSES=0.005,0.025,0.5,0.7,0.8,
      0.85,0.9 BIORAG_TAG=relhub, 8 shards launched at once by
      archive/workers_2026-10/drive_parity.sh. A FAILED run: every log stops
      at "[27s] ... conditions" with zero conditions complete. The machine
      kernel-panicked (watchdog timeout under memory exhaustion) at about
      10:04 local. Truncated, not cleaned up -- the truncation is the
      evidence. See RUNLOG Run 47.
  drive_parity_20261001_092820.progress
      drive_parity.sh's progress, ending at "[09:52] insert relhub" -- the
      last thing it did before the panic.
  03a_01_relhub_shard0of3_20261001_102923.log
  03a_01_relhub_shard1of3_20261001_102923.log
  03a_01_relhub_shard2of3_20261001_102923.log
  03a_01_reluni_shard0of3_20261001_102923.log
  03a_01_reluni_shard1of3_20261001_102923.log
  03a_01_reluni_shard2of3_20261001_102923.log
  03a_01_dupcon_shard0of3_20261001_102923.log
  03a_01_dupcon_shard1of3_20261001_102923.log
  03a_01_dupcon_shard2of3_20261001_102923.log
  03a_01_intuni_shard0of3_20261001_102923.log
  03a_01_intuni_shard1of3_20261001_102923.log
  03a_01_intuni_shard2of3_20261001_102923.log
  03a_01_flo16k_shard0of3_20261001_102923.log
  03a_01_flo16k_shard1of3_20261001_102923.log
  03a_01_flo16k_shard2of3_20261001_102923.log
  03a_01_flo64_shard0of3_20261001_102923.log
  03a_01_flo64_shard1of3_20261001_102923.log
  03a_01_flo64_shard2of3_20261001_102923.log
  03a_01_flo256_shard0of3_20261001_102923.log
  03a_01_flo256_shard1of3_20261001_102923.log
  03a_01_flo256_shard2of3_20261001_102923.log
  03a_01_flo1024_shard0of3_20261001_102923.log
  03a_01_flo1024_shard1of3_20261001_102923.log
  03a_01_flo1024_shard2of3_20261001_102923.log
  03a_01_flo4096_shard0of3_20261001_102923.log
  03a_01_flo4096_shard1of3_20261001_102923.log
  03a_01_flo4096_shard2of3_20261001_102923.log
      Written by 03a_01_no_access_layer_matrix.py, 3 shards per block, launched
      by archive/workers_2026-10/drive_parity_v2.sh. One block per tag, run in
      the order listed: relhub, reluni, dupcon, intuni, then flood at 16,384,
      64, 256, 1,024 and 4,096 anchors. Each log's header names its exact
      overrides (BIORAG_ATTACKS, BIORAG_SELECTIONS, BIORAG_DOSES, BIORAG_TAG,
      BIORAG_FLOOD_ANCHORS). Every shard completed; its last line reports the
      by_donor rows it wrote. See RUNLOG Run 48.
  03b_01_modhub_shard0of2_20261001_102923.log
  03b_01_modhub_shard1of2_20261001_102923.log
  03b_01_moduni_shard0of2_20261001_102923.log
  03b_01_moduni_shard1of2_20261001_102923.log
      Written by 03b_01_with_access_layer_matrix.py under BIORAG_PHASE=graph,
      2 shards per block, the last two blocks of the same driver. modhub runs
      shifted and synthesised at hub over 9 rates, moduni the same at uniform
      over 13. Every shard completed. See RUNLOG Run 48.
  drive_parity_v2_20261001_102923.progress
      drive_parity_v2.sh's progress: the start banner (3 insert shards, modify
      concurrency 2, swap kill at 28,000 MB), then each block's launch time,
      its merge_shards output with row counts, and its merge time. Ends
      "[10:57] modify moduni merged". See RUNLOG Run 48.
  drive_parity_v2_20261001_102923.done
      drive_parity_v2.sh's completion marker, "PARITY V2 DONE 2026-10-04
      10:57". Written only after all 11 blocks merged; the driver exits before
      this line on any shard failure or swap-sentry kill.
  watchdog_20261001_102923.tsv
      Written by archive/workers_2026-10/watchdog.sh every 5 minutes through
      Run 48: biorag python count, total and maximum RSS in GB, CPU idle
      percentage, swap used in MB, and the running step. 846 samples, 827 with a
      step running; stopped by hand at 12:28 on 2026-10-04. Total RSS
      peaked at 36.06 GB and swap at 1.50 MB.
  03b_04_integrity_20261004_110027.log
      Written by 03b_04_both_integrity.py, single process, launched by
      archive/workers_2026-10/drive_integrity_after_parity.sh once Run 48's
      .done existed. 24 conditions in 58 s; ends "exit: 0". See RUNLOG Run 49.
  drive_integrity_20261004_110027.done
      drive_integrity_after_parity.sh's marker, "DRIVE_INTEGRITY DONE
      2026-10-04 11:01". Written only after the step's exit status was checked
      as 0, unlike drive_03b02.sh's marker for Run 44.
  03a_09_intervals_relhub_20261004_122937.log
  drive_03a09_20261004_122937.done
      A ONE-TAG TEST of archive/workers_2026-10/drive_03a09_tags.sh on
      relhub, run on the BIORAG_TAG change to 03a_09 before it was committed:
      provenance reads DIRTY (code 1 changed). 5 s, exit 0; point estimates
      agreed with the relhub layer matrix on 4,368 comparisons. Its output was
      overwritten by the committed-code run of all 11 tags. See RUNLOG Run 50.
  03a_09_intervals_relhub_20261004_123047.log
  03a_09_intervals_reluni_20261004_123047.log
  03a_09_intervals_dupcon_20261004_123047.log
  03a_09_intervals_intuni_20261004_123047.log
  03a_09_intervals_flo16k_20261004_123047.log
  03a_09_intervals_flo64_20261004_123047.log
  03a_09_intervals_flo256_20261004_123047.log
  03a_09_intervals_flo1024_20261004_123047.log
  03a_09_intervals_flo4096_20261004_123047.log
  03a_09_intervals_modhub_20261004_123047.log
  03a_09_intervals_moduni_20261004_123047.log
  drive_03a09_20261004_123047.done
      Written by 03a_09_both_intervals.py under BIORAG_TAG=<tag>, one log per
      tag, driven by archive/workers_2026-10/drive_03a09_tags.sh at the
      committed code 2a19cf1a (provenance clean in code). Every tag exited 0
      and its point estimates agreed with its own layer matrix (tol 1e-06);
      1 min 49 s for all 11. The .done reads "DRIVE_03A09 DONE 2026-10-04
      12:32" and lists the tags. See RUNLOG Run 50.
  03a_01_v3relhub_shard0of3_20261004_131537.log
  03a_01_v3relhub_shard1of3_20261004_131537.log
  03a_01_v3relhub_shard2of3_20261004_131537.log
  drive_rerun_v3_20261004_131537.progress
  watchdog_20261004_131537.tsv
      An ABORTED launch of archive/workers_2026-10/drive_rerun_v3.sh (commit
      f8158b1e), stopped by hand about a minute in at the user's request so
      the keep variants could be added. Each shard log ends at the clean
      baseline, before any condition finished; no table or retrieval was
      written. Kept as they are. See RUNLOG Run 51.
  03a_01_v3relhub_shard0of3_20261004_132108.log
  03a_01_v3relhub_shard1of3_20261004_132108.log
  03a_01_v3relhub_shard2of3_20261004_132108.log
  03a_01_v3reluni_shard0of3_20261004_132108.log
  03a_01_v3reluni_shard1of3_20261004_132108.log
  03a_01_v3reluni_shard2of3_20261004_132108.log
  03a_01_v3dupcon_shard0of3_20261004_132108.log
  03a_01_v3dupcon_shard1of3_20261004_132108.log
  03a_01_v3dupcon_shard2of3_20261004_132108.log
  03a_01_v3intuni_shard0of3_20261004_132108.log
  03a_01_v3intuni_shard1of3_20261004_132108.log
  03a_01_v3intuni_shard2of3_20261004_132108.log
  03a_01_v3flo64_shard0of3_20261004_132108.log
  03a_01_v3flo64_shard1of3_20261004_132108.log
  03a_01_v3flo64_shard2of3_20261004_132108.log
  03a_01_v3flo256_shard0of3_20261004_132108.log
  03a_01_v3flo256_shard1of3_20261004_132108.log
  03a_01_v3flo256_shard2of3_20261004_132108.log
  03a_01_v3flo1024_shard0of3_20261004_132108.log
  03a_01_v3flo1024_shard1of3_20261004_132108.log
  03a_01_v3flo1024_shard2of3_20261004_132108.log
  03a_01_v3flo4096_shard0of3_20261004_132108.log
  03a_01_v3flo4096_shard1of3_20261004_132108.log
  03a_01_v3flo4096_shard2of3_20261004_132108.log
  03a_01_v3flo16384_shard0of3_20261004_132108.log
  03a_01_v3flo16384_shard1of3_20261004_132108.log
  03a_01_v3flo16384_shard2of3_20261004_132108.log
  03b_01_v3modhub_shard0of2_20261004_132108.log
  03b_01_v3modhub_shard1of2_20261004_132108.log
  03b_01_v3moduni_shard0of2_20261004_132108.log
  03b_01_v3moduni_shard1of2_20261004_132108.log
      Written by DEFENCE RE-RUN v3, archive/workers_2026-10/drive_rerun_v3.sh
      at commit 1eafac83 (every shard's provenance: clean in code). The 11
      blocks finished by 2026-10-08 11:12; every shard completed and every
      by_donor merge passed --expect. See RUNLOG Run 52.
  03a_01_v3relunikeep_shard0of3_20261004_132108.log
  03a_01_v3relunikeep_shard1of3_20261004_132108.log
  03a_01_v3relunikeep_shard2of3_20261004_132108.log
  03a_01_v3dupconkeep_shard0of3_20261004_132108.log
  03a_01_v3dupconkeep_shard1of3_20261004_132108.log
  03a_01_v3dupconkeep_shard2of3_20261004_132108.log
  03a_01_v3flo1024keep_shard0of3_20261004_132108.log
  03a_01_v3flo1024keep_shard1of3_20261004_132108.log
  03a_01_v3flo1024keep_shard2of3_20261004_132108.log
  03b_01_v3modkeep_shard0of2_20261004_132108.log
  03b_01_v3modkeep_shard1of2_20261004_132108.log
      Blocks 12-15 of DEFENCE RE-RUN v3, the keep variants. v3relunikeep ran
      at 1eafac83; the driver re-reads provenance per block, so the other
      three report ae9e1879, which changes no code (git diff 1eafac83 ae9e1879
      -- v02/1_scripts is empty). Every shard: clean in code, completed, and
      every by_donor merge passed --expect. See RUNLOG Run 52 (closing entry).
  drive_rerun_v3_20261004_132108.progress
  drive_rerun_v3_20261004_132108.done
  watchdog_20261004_132108.tsv
      The driver's progress file for all 15 blocks and the final 03a_09 pass;
      the .done reads "RERUN V3 DONE 2026-10-09 17:50". The watchdog has 1,304
      samples from 2026-10-04 13:21 to 2026-10-09 17:50 and was stopped by
      hand after the driver exited. See RUNLOG Run 52 (closing entry).
  03a_09_intervals_v3relhub_20261008_143434.log
  03a_09_intervals_v3reluni_20261008_143434.log
  03a_09_intervals_v3dupcon_20261008_143434.log
  03a_09_intervals_v3intuni_20261008_143434.log
  03a_09_intervals_v3flo64_20261008_143434.log
  03a_09_intervals_v3flo256_20261008_143434.log
  03a_09_intervals_v3flo1024_20261008_143434.log
  03a_09_intervals_v3flo4096_20261008_143434.log
  03a_09_intervals_v3flo16384_20261008_143434.log
  03a_09_intervals_v3modhub_20261008_143434.log
  03a_09_intervals_v3moduni_20261008_143434.log
  drive_03a09_20261008_143434.done
      03a_09 under BIORAG_TAG for the 11 finished v3 tags, driven by
      archive/workers_2026-10/drive_03a09_tags.sh at 1eafac83; every tag
      exit 0, point estimates agreeing with each v3 matrix to 1e-06. 2 min
      22 s. See RUNLOG Run 53.
  03a_09_intervals_v3relhub_20261009_175031.log
  03a_09_intervals_v3reluni_20261009_175031.log
  03a_09_intervals_v3dupcon_20261009_175031.log
  03a_09_intervals_v3intuni_20261009_175031.log
  03a_09_intervals_v3flo64_20261009_175031.log
  03a_09_intervals_v3flo256_20261009_175031.log
  03a_09_intervals_v3flo1024_20261009_175031.log
  03a_09_intervals_v3flo4096_20261009_175031.log
  03a_09_intervals_v3flo16384_20261009_175031.log
  03a_09_intervals_v3modhub_20261009_175031.log
  03a_09_intervals_v3moduni_20261009_175031.log
  03a_09_intervals_v3relunikeep_20261009_175031.log
  03a_09_intervals_v3dupconkeep_20261009_175031.log
  03a_09_intervals_v3flo1024keep_20261009_175031.log
  03a_09_intervals_v3modkeep_20261009_175031.log
  drive_03a09_20261009_175031.done
      03a_09 under BIORAG_TAG for all 15 v3 tags, run by drive_rerun_v3.sh
      after its last block. Every tag exit 0, point estimates agreeing with
      each v3 matrix to 1e-06. The 11 tables Run 53 wrote came out
      byte-identical. 3 min. See RUNLOG Run 54.
  03a_02_all26_v3_shard0of2_20261009_193430.log
  03a_02_all26_v3_shard1of2_20261009_193430.log
  03a_03_v3_shard0of2_20261009_193430.log
  03a_03_v3_shard1of2_20261009_193430.log
  drive_item4_v3_20261009_193430.progress
  drive_item4_v3_20261009_193430.done
      Item 4 part 1, archive/workers_2026-10/drive_item4_v3.sh at dac565c0 --
      03a_02 (all 26 celltypes) and 03a_03 re-run on the fixed accuracy,
      duplicate at concentrated, tag v3. Every shard exit 0, every merge
      checked; .done reads "DRIVE_ITEM4_V3 DONE 2026-10-09 22:23". NOT
      titrations: 03a_02 runs 2 doses, 03a_03 one dose on 6 celltypes. See
      RUNLOG Run 55.
  03b_01_v3moduni_rep2_shard0of2_20261009_220741.log
  03b_01_v3moduni_rep2_shard1of2_20261009_220741.log
  drive_rep2_v3moduni_20261009_220741.progress
  03b_03_v3moduni_20261009_220741.log
  drive_rep2_v3moduni_20261009_220741.done
      Item 4 part 2, archive/workers_2026-10/drive_rep2_v3moduni.sh at
      cae08c4c -- after drive_item4_v3, a second realisation of v3moduni
      (tag v3moduni_rep2, 2 shards, 11 h 37 m / 11 h 07 m), both merges
      checked, then 03b_03 on the pair (1 s). .done reads "DRIVE_REP2_V3MODUNI DONE
      2026-10-10 10:01". v3moduni ONLY, NOT the full grid (v3modhub,
      v3modkeep and the insert blocks are not repeated). See RUNLOG Run 56.
  03a_09_intervals_v3moduni_rep2_20261010_132328.log
  drive_03a09_20261010_132328.done
      03a_09 with BIORAG_TAG=v3moduni_rep2 via drive_03a09_tags.sh at
      463fd21c. Exit 0, point estimates agree with the rep2 matrix to 1e-06
      (16,224 comparisons). 19 s. See RUNLOG Run 57.
  03b_01_cached_clean_v3_shard0of8_20261010_134304.log
  drive_cached_clean_v3_20261010_134304.progress
  drive_cached_clean_v3_20261010_134304.fail
      ABORTED first launch of drive_cached_clean_v3.sh (label-only arm on
      the fixed accuracy) at 869175cd. A driver bug -- zsh stored the shard
      pid as the literal "$!" -- made the liveness check see shard 0 as dead
      after one condition; the driver aborted and shard 0 was stopped by
      hand. No table written. See RUNLOG Run 58.
  03b_01_cached_clean_v3_shard0of8_20261010_134602.log
  03b_01_cached_clean_v3_shard1of8_20261010_134602.log
  03b_01_cached_clean_v3_shard2of8_20261010_134602.log
  03b_01_cached_clean_v3_shard3of8_20261010_134602.log
  03b_01_cached_clean_v3_shard4of8_20261010_134602.log
  03b_01_cached_clean_v3_shard5of8_20261010_134602.log
  03b_01_cached_clean_v3_shard6of8_20261010_134602.log
  03b_01_cached_clean_v3_shard7of8_20261010_134602.log
  drive_cached_clean_v3_20261010_134602.progress
  drive_cached_clean_v3_20261010_134602.done
  03a_09_intervals_cached_clean_v3_20261010_134602.log
      Run 59, the label-only arm on the fixed accuracy (03b_01 cached_clean,
      tag v3, 13 doses, hub+uniform, 8 shards by celltype), 13:46-17:25.
      Shards 0-2 started by drive_cached_clean_v3.sh at 6129f7db, whose
      driver was stopped by hand at 13:50 (its free-% memory gate
      over-admitted); shards 3-7, both merges (8,112 and 632,736 rows) and
      03a_09 (BIORAG_TAG=v3, 18 s, 16,224 point estimates agree to 1e-06) by
      drive_cached_clean_v3_resume.sh at 384db651, at most 5 shards at once.
      No Traceback in any shard log. .done reads "DRIVE_CACHED_CLEAN_V3 DONE
      2026-10-10 17:25". See RUNLOG Run 59.
