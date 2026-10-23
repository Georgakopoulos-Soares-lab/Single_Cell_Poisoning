1_scripts/01_a_baseline  --  analysis scripts -- write tables into 2_results, never figures
===============================================

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
  01a_01_no_access_retrieve_hnsw.py
      Step 01a_01 — what SCimilarity's OWN graph returns for each query, unpoisoned.
  01a_02_no_access_exact_vs_hnsw.py
      Step 01a_02 — does exact retrieval match what SCimilarity actually returns?
  01a_03_no_access_clean_baseline.py
      Step 01a_03 — how accurate is the deployed system before any attacker exists?
  01a_04_no_access_accuracy_ci.py
      Step 01a_04 — donor-paired intervals on the unpoisoned accuracies and the granularity gap.
  01a_05_no_access_genuine_lc_scores.py
      Step 01a_05 — how often does a GENUINE reference record disagree with its own neighbourhood?
