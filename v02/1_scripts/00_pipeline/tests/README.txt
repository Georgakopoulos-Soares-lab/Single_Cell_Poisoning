1_scripts/00_pipeline/tests  --  analysis scripts -- write tables into 2_results, never figures
===================================================

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
  __init__.py
      TODO: describe
  conftest.py
      Put the shared library and the pipeline stage on the path for tests.
  fixtures.py
      Hand-built tiny fixtures for golden-value tests.
  test_aggregate.py
      Golden-value tests for stage 5, aggregate.
  test_atlas_attack.py
      Atlas-scope record choice and label forging, shared by both arms.
  test_by_donor.py
      ``by_donor`` must re-pool to exactly what ``scope`` already publishes.
  test_check_reads.py
      A figure must declare the analysis step it draws, and the claim must be true.
  test_check_retrieval.py
      A figure must not draw the shortcut arm of a step that was also run on graph.
  test_check_steps.py
      The step-id check must fail on the collision it was written for.
  test_combo_poison.py
      Tests for simultaneous multi-type attacks.
  test_condition_seed.py
      The condition seed, and the float width a stored rate has to keep.
  test_dataset_config.py
      The config a run resolves must not move when the config FILE is restructured.
  test_dedup.py
      Near-duplicate collapse inside a retrieved neighbourhood.
  test_defended_sweep.py
      Tests for the defended sweep — what produces the defence table.
  test_defense.py
      Tests for the defence layers.
  test_donor_cap.py
      A cap of c votes per donor, generalising one-vote-per-donor.
  test_embed_queries.py
      Tests for stage 1, embed_queries.
  test_flood_sites.py
      Where a flood puts its clusters — v01's ``mode``, isolated as one axis.
  test_graph_repeatability.py
      ``02b_34`` exists to size a band, so its two comparisons must not lie.
  test_integrity.py
      03b_04's in-place edit on a toy reference: unmoved leaves every
      embedding byte-identical and changes exactly the drawn labels (the case
      Run 44 crashed on), shifted moves exactly the records it relabels, and
      neither mutates the reference it was given.
  test_label_consistency.py
      Tests for label-consistency scoring — the input the defence needs.
  test_make_poison.py
      Golden-value tests for stage 3, make_poison.
  test_modify.py
      Overwriting records, and the exactness the deeper neighbour list buys.
  test_ontology.py
      Tests for Cell Ontology parsing and tree distance.
  test_provenance.py
      A run must say which code produced it, and a writer must name what it wrote.
  test_quarantine.py
      Quarantine holds recently added records out of the vote.
  test_query_cache.py
      A second dataset must not overwrite the first one's cache in place.
  test_query_view.py
      ``02b_30``'s two tables must be joinable on ``query_id``, and complete.
  test_retrieve_clean.py
      Golden-value tests for stage 2, retrieve_clean.
  test_selection.py
      Selection and label as AXES, not as things baked into an attack name.
  test_shard.py
      Sharding a sweep must not change a single number.
  test_sweep.py
      Golden-value tests for stage 4, sweep.
  test_threshold.py
      The threshold estimator, on curves whose answer is known by hand.
  test_retrieval_store.py
      retrieval_store: round trip, deterministic paths, atomic write, and the
      poison-digest check that refuses a mismatched rebuild.
  test_validate_shortcut.py
      Tests for stage 6, validate_shortcut.
  test_weighted_vote.py
      Distance-weighted voting: weights, and the vote they produce.
