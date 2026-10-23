1_scripts/03_a_defence_vs_insert  --  analysis scripts -- write tables into 2_results, never figures
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
  03a_01_no_access_layer_matrix.py
      Every defence layer against an attacker who can only SUBMIT. Eight
      configurations x four attacks x 26 celltypes x two doses, graph-faithful,
      each configuration scored off the SAME retrieval. This is the arm where
      defences work: v01 put the stack under 5% on all four attacks, but on
      four celltypes where label consistency is nearly free. Every row carries
      the benefit and the genuine-record cost together, and the threshold is
      set by a BUDGET rather than by the arbitrary 0.5. splits is fixed at 1
      here so the matrix has one donor regime; 03a_02 sweeps it.

      OVERRIDES, all opt-in and all tagged so a partial-grid run can never
      overwrite the committed default table. BIORAG_DOSES (2026-09-30) moves
      the ladder off 0.25/1.0; a DOSES-only run keeps the tag `_dose`.
      BIORAG_ATTACKS, BIORAG_SELECTIONS and BIORAG_FLOOD_ANCHORS (2026-10-01)
      drive the attack grid's own combinations one per run, because they are
      not a cross product; BIORAG_TAG names the run. Rows now carry
      `n_anchors` (flood only), since flood's axis is n_anchors, not selection
      -- build_flood takes no selection.
      2026-10-04: rows carry `flood_spread` and `flood_jitter` (flood only;
      this step builds flood at 0.999 / 1.0, 02b_02 at 0.99), and
      within_accuracy is now the vote actually returned (it was the vote with
      poison removed; see diagnosis/2026-10-04_attack_vs_defence_none). With
      BIORAG_SAVE_RETRIEVAL=1 each condition's neighbour ids, cosines and
      poison digest are kept under <cache>/retrievals/03a_01_<tag>/ via
      00_common/retrieval_store.py, so it can be re-scored without the index.
      BIORAG_LABEL_CHANGE=keep (2026-10-04) submits the poison under its own
      source label -- the attack grid's volume/geometry controls; the vote and
      label consistency read each poison record's own label, never `target`.
  03a_02_no_access_donor_identity.py
      Where a per-donor vote cap stops helping and starts helping the attacker.
      8 caps x 8 fabricated-identity counts, 6 celltypes, relabel and duplicate,
      two doses. The whole grid scores off ONE retrieval per condition, because
      splits rewrites donor METADATA only. Retires v01's `donor vote` column,
      which assigned every poisoned record the string "attacker" and so measured
      splits=1 without saying so.

      BIORAG_ALL_CELLTYPES=1 (2026-09-30) runs every attackable celltype
      instead of the six, and writes `_all26`-tagged tables beside the
      committed six-celltype ones.
  03a_03_no_access_vote_weighting.py
      Does reweighting add anything the stack has not already taken? Five schemes
      measured ONLY in composition -- stack+w against stack+uniform -- because
      standalone is the configuration where poison a record layer would have
      rejected is still voting. Carries the k-sweep control v01 demanded: rank
      decays fiftyfold across the list, so without it "weighting defends" and
      "fewer neighbours is better" are indistinguishable.
  03a_06_no_access_umap.py
      An ANALYSIS step that draws nothing: it fits UMAP coordinates for two
      contrasting conditions and writes them, with a per-RECORD verdict from
      each defence layer, to 2_results/. A fit is a computed quantity, so the
      coordinates are a result; v01 wrote its atlas fit to gitignored cache/
      and left two figures unreproducible from a clean clone. Two conditions
      chosen off 03a_01's committed matrix as the largest rescue in their
      attack family: CL:0002062 relabel, where recompute admits all 869
      retrieved poison records, and CL:0000583 flood, where it admits none of
      3,000. That pair is the whole argument -- same access, same dose, same
      layer, opposite outcome -- and drawing only the flood would suggest
      poison is something you can see. Drawn by 3_figures/.../03a_07.
  03a_09_both_intervals.py
      Donor-paired 95% intervals for every layer in BOTH arms, read off the
      three committed _by_donor tables; re-runs no attack and loads no index.
      Donors are resampled ONCE per condition and every configuration is
      recomputed on that same draw, so a defended-minus-undefended difference
      keeps none of the variance the two arms share -- and sharing one
      retrieval, they share nearly all of it. Two independent bootstraps and a
      subtraction would call every result here null. Asserts its own point
      estimates against the committed layer matrices on 14,976 comparisons
      before writing. Recovers 03b_01's cached_clean `selection`, which that
      step failed to emit in its donor frame, from row order and VERIFIES the
      recovery against the main table every run (worst |diff| 2.98e-08 over
      1,248 comparisons, against 0.978 for the swapped assignment); exits
      rather than guessing if it ever fails. 03b_01 should emit the column.
      Drawn by 3_figures/.../03a_10.
      OVERRIDE 2026-10-04, opt-in: BIORAG_TAG=<tag> reads ONE tagged pair
      from the full-parity grid (RUNLOG Run 48) instead of the three default
      tables -- 03a_01_..._<tag> for an insert tag, 03b_01_..._graph_<tag> for
      modhub/moduni, chosen by which pair exists -- and writes
      03a_09_both_intervals_<tag>.parquet with `tag` (and `n_anchors` for a
      flood tag) on every row. The point-estimate check runs against that
      tag's own main table. Unset, the step reproduces the committed default
      output; the default code path is unchanged.
