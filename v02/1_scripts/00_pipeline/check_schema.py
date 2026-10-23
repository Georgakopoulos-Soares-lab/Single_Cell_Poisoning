"""Every analysis table declares which attacker produced it.

``schemas.ThreatModel`` says the column is "Present on every analysis table",
and v01 shows what an unenforced contract is worth: 35 of its 39 tables omit it,
so a result there cannot be attributed to an attacker model without opening the
script that wrote it. That is the question the whole project turns on -- an
append-only attacker and one with write access face different ceilings, and the
CONTRAST is the finding.

Checked before v02 has tables rather than after, because this is a one-line fix
per script today and a migration across dozens of files later.

Reads parquet FOOTERS only, never the data, so it stays fast on a full tree.

Scope: ``.parquet`` under ``2_results``, except :data:`SKIP_SUFFIXES`.
Plotted-value ``.csv`` files are
exempt -- they are display projections of a table that already carries the
column, and a figure may legitimately pool both arms.

Exits non-zero, so it can gate a commit.
"""
from __future__ import annotations

import pathlib
import sys

#: Machine-generated condition families; declared by PATTERN in their README and
#: too numerous to open individually.
SKIP_DIRS = {"00_raw", "logs", "__pycache__"}

#: Tables exempt from the threat_model requirement.
#:
#: Any ``*_by_donor*.parquet`` is SUBSTRATE, not an analysis table. It holds the
#: unreduced per-donor numerator and denominator sums that 02b_10 and 02b_12
#: bootstrap from (``schemas.BY_DONOR_DOC``); nothing reads a rate off it and no
#: figure plots it. Its condition is identified by the axes it does carry --
#: attack, geometry, label_change, selection -- and the analysis table written
#: beside it by the same step carries threat_model for the same conditions.
#:
#: Requiring the column here would be reasonable too, and arguably tidier. It is
#: not done because adding it would mean editing sweep drivers while three of
#: them are queued or running, producing one table without the column and three
#: with it. A uniform absence is honest; a mixture is a trap for whoever
#: concatenates them. Revisit when nothing is in flight.
#: MATCHED ANYWHERE IN THE NAME, not as a suffix. A graph run appends its own
#: suffix AFTER this one -- 02b_11_with_access_inplace_by_donor_graph.parquet --
#: so endswith("_by_donor.parquet") missed exactly the tables the graph reruns
#: produce and demanded threat_model of substrate that has never carried it.
#: The same suffix-ordering trap caught flood_common.donor_tables and worker_d's
#: merge paths; this is its third instance. 2026-09-27.
SKIP_CONTAINS = ("_by_donor",)
#: Column required on every analysis table.
THREAT_MODEL = "threat_model"
#: The only values it may take. Mirrors :class:`schemas.ThreatModel`. v01's
#: tables say ``insert``/``modify`` for the same two arms and are read through
#: v01's own schema; a v01 value appearing in a v02 table means a script was
#: copied forward without being updated, so it is rejected here rather than
#: silently producing a table nobody can join on.
VALID = {"no_access", "with_access"}
#: Recognised, but wrong HERE -- reported with the translation rather than as an
#: opaque failure.
V01_VALUES = {"insert": "no_access", "modify": "with_access"}


def tables(root: pathlib.Path):
    """Analysis tables under ``root``, skipping raw condition families."""
    for p in sorted((root / "2_results").rglob("*.parquet")):
        if not SKIP_DIRS & set(p.parts) and not any(x in p.name for x in SKIP_CONTAINS):
            yield p


def check(root: pathlib.Path) -> tuple[int, list[str]]:
    """(tables checked, problems)."""
    import pyarrow.parquet as pq

    bad, n = [], 0
    for p in tables(root):
        n += 1
        try:
            names = set(pq.read_schema(p).names)
        except Exception as e:                      # a table we cannot open is a problem
            bad.append(f"UNREADABLE {p}: {e}")
            continue
        if THREAT_MODEL not in names:
            bad.append(f"MISSING    {p.relative_to(root)}: no {THREAT_MODEL!r} column")
            continue
        # VALID was declared and never used in the first draft -- an inert
        # constant reads as a decision that was made. Reading one column of an
        # aggregate table is cheap, so check the values too.
        seen = set(pq.read_table(p, columns=[THREAT_MODEL])[THREAT_MODEL].to_pylist())
        for v in sorted(seen - VALID):
            hint = f" (v01 spelling of {V01_VALUES[v]!r})" if v in V01_VALUES else ""
            bad.append(f"BAD VALUE  {p.relative_to(root)}: {v!r}{hint}")
    return n, bad


def main(roots: list[str]) -> int:
    total, problems = 0, []
    for r in roots:
        n, bad = check(pathlib.Path(r))
        total += n
        problems += bad
    for p in problems:
        print(p, file=sys.stderr)
    print(f"{total} analysis tables checked; {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["v02"]))
