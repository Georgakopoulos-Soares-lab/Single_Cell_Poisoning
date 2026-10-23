1_scripts/00_pipeline/tests/fixtures  --  committed snapshots a test compares against
===================================================

WHAT LIVES HERE. Golden files: a value that was correct on a known date,
written down so a later change has to justify moving it. A fixture is EVIDENCE,
not scratch. Regenerating one to make a failing test pass destroys the only
thing it was for -- if a fixture and the code disagree, one of them is wrong and
that has to be decided, not overwritten.

Hand-built tiny INPUTS for golden-value tests live in ../fixtures.py instead;
this directory holds committed OUTPUTS.

README. This file is a bijection with the directory: every file has an entry,
every entry names a file that exists. Adding, deleting or renaming a file
requires updating this file in the same commit. An entry for a file that is not
present is a hallucination. Enforced by 1_scripts/00_pipeline/check_readme.py.

CONTENTS
  resolved_config_GSE136831.json
      TODO: describe
