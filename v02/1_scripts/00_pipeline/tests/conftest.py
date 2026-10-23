"""Put the shared library and the pipeline stage on the path for tests.

``1_scripts`` cannot be a Python package — an identifier may not start with a
digit — so there is no dotted path to import through. The directories that hold
importable modules are added directly instead, and tests import flat:

    from schemas import K_NEIGHBORS
    from defense import DefenseConfig

Every runnable script does the same thing for itself; this file does it once for
the whole test session.
"""

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2]      # .../1_scripts
# the tests directory itself holds fixtures.py, which the poison tests import
for d in (SCRIPTS / "00_common", SCRIPTS / "00_pipeline",
          Path(__file__).resolve().parent):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
