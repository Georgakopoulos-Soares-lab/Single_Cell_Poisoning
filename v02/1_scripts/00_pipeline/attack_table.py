"""Regenerate SPEC.md §5.4's attack-type table from one source.

The table enumerates every attack type -- ``scope`` x ``access`` x ``geometry``
x ``label`` x ``selection``, 72 of them -- with what has been run, what has been
titrated, and whether it should be run. It lives here rather than being typed
into SPEC.md because a hand-maintained 72-row table drifts the moment one
verdict changes, and drift in a design document is worse than no document.

**Blanks are filled with a verdict, never left empty.** An unexplained gap in a
grid reads as an oversight and invites someone to fill it. Four verdicts say why
an attack type is not run, and they are not interchangeable:

``baseline``      nothing changes. Overwriting a record without moving it or
                  relabelling it leaves the reference identical -- that is
                  ``01a_03``, not an attack.
``degenerate``    real, but it reproduces an attack type already measured.
                  Concentrated selection draws WITH replacement and a record
                  cannot be overwritten twice, so concentrated x modify reduces
                  to uniform modify at the pool size. Checkable: run one and it
                  should match uniform at rate 0.01 exactly.
``no question``   coherent but answers nothing worth asking. ``flood`` rotates
                  anchors toward the SOURCE celltype's centroid; at atlas scope
                  there is no manifold centre to aim at.
``not run``       coherent, and nobody has stated what question it would answer.
                  The honest default -- inventing a justification for each would
                  make the column look complete and mean nothing.

The ``titration`` column names EVERY axis varied, not just dose. ``9 dose,
1 alpha`` is a healthy dose ladder beside a construction parameter that was
never moved, which is the state of ``shifted``/``keep`` and was invisible while
the column reported dose alone.

Run it with no arguments; it prints the table to stdout.
"""
from __future__ import annotations

# scope, access, geometry, label, selection -> status, titration, run?, why.
# Only attack types that deviate from the four default verdicts appear here.
MEASURED: dict[tuple[str, ...], tuple[str, str, str, str]] = {
    ("atlas", "insert", "unmoved", "relabel", "uniform"):
        ("v02 30", "6 pts 0.1%-10%", "YES", "single-target variant unbuilt"),
    ("atlas", "modify", "unmoved", "relabel", "uniform"):
        ("not run", "-", "YES", "coordinated modify, untested"),
    ("atlas", "modify", "unmoved", "relabel", "hub"):
        ("not run", "-", "YES", "coordinated modify, untested"),
    ("atlas", "modify", "shifted", "keep", "uniform"):
        ("not run", "-", "NO", "pending celltype alpha sweep"),
    ("atlas", "modify", "shifted", "keep", "hub"):
        ("not run", "-", "NO", "pending celltype alpha sweep"),
    ("celltype", "insert", "unmoved", "relabel", "uniform"):
        ("v02 208", "8 dose", "YES", "extend ladder to 0.002"),
    ("celltype", "insert", "unmoved", "relabel", "concentrated"):
        ("v02 208", "8 dose, 1 pool", "YES", "extend ladder + pool sweep"),
    ("celltype", "insert", "unmoved", "relabel", "hub"):
        ("build 2", "-", "YES", "selection symmetry w/ modify"),
    ("celltype", "insert", "unmoved", "keep", "uniform"):
        ("build 4", "-", "YES", "volume control"),
    ("celltype", "insert", "unmoved", "keep", "concentrated"):
        ("build 4", "-", "YES", "volume control"),
    ("celltype", "insert", "shifted", "relabel", "uniform"):
        ("v02 208", "8 dose, 1 alpha", "YES", "extend ladder + alpha"),
    ("celltype", "insert", "synthesised", "relabel", "uniform"):
        ("v01", "1 pt 0.25 only", "NO", "construction variant, low value"),
    ("celltype", "insert", "synthesised", "relabel", "concentrated"):
        ("v02 520", "4 dose, 5 anchor", "YES", "ladder + 02b_13 geometry"),
    ("celltype", "insert", "synthesised", "keep", "concentrated"):
        ("not run", "-", "YES", "geometry control for flood"),
    ("celltype", "modify", "unmoved", "relabel", "uniform"):
        ("v02 936", "4 dose", "YES", "extend ladder to 1.25"),
    ("celltype", "modify", "unmoved", "relabel", "hub"):
        ("v02 936", "4 dose", "YES", "extend ladder to 1.25"),
    ("celltype", "modify", "shifted", "relabel", "uniform"):
        ("v01", "9 dose, 2 alpha", "YES", "reproduce + alpha sweep"),
    ("celltype", "modify", "shifted", "relabel", "hub"):
        ("v01", "9 dose, 2 alpha", "YES", "reproduce + alpha sweep"),
    ("celltype", "modify", "shifted", "keep", "uniform"):
        ("v01", "9 dose, 1 alpha", "YES", "step 5: alpha 11 pts"),
    ("celltype", "modify", "shifted", "keep", "hub"):
        ("v01", "9 dose, 1 alpha", "YES", "step 5: alpha 11 pts"),
    ("celltype", "modify", "synthesised", "relabel", "uniform"):
        ("build 3", "-", "YES", "geometry symmetry w/ insert"),
    ("celltype", "modify", "synthesised", "relabel", "hub"):
        ("build 3", "-", "YES", "geometry symmetry w/ insert"),
    ("celltype", "modify", "synthesised", "keep", "hub"):
        ("not run", "-", "NO", "pending celltype alpha sweep"),
}

GEOMETRY_LABEL = (("unmoved", "relabel"), ("unmoved", "keep"),
                  ("shifted", "relabel"), ("shifted", "keep"),
                  ("synthesised", "relabel"), ("synthesised", "keep"))
HEADER = ("scope", "access", "geometry", "label", "selection",
          "status", "titration", "run?", "why")


def verdict(scope: str, access: str, geometry: str, label: str,
            selection: str) -> tuple[str, str, str, str]:
    """Status, titration, run recommendation and reason for one attack type.

    Args:
        scope: ``celltype`` or ``atlas`` -- the pool the budget is drawn from.
        access: ``insert`` (append) or ``modify`` (overwrite).
        geometry: ``unmoved``, ``shifted`` or ``synthesised``.
        label: ``relabel`` if the record's label lies, else ``keep``.
        selection: ``uniform``, ``concentrated`` or ``hub``.

    Returns:
        ``(status, titration, run?, why)``, all human-readable.
    """
    key = (scope, access, geometry, label, selection)
    if key in MEASURED:
        return MEASURED[key]
    if geometry == "unmoved" and label == "keep" and access == "modify":
        return "baseline", "-", "NO", "nothing changes"
    if selection == "concentrated" and access == "modify":
        return "degenerate", "-", "NO", "= uniform modify at 1%"
    if geometry == "synthesised" and scope == "atlas":
        return "no question", "-", "NO", "no atlas manifold to aim at"
    return "not run", "-", "NO", "no argument made for it"


def build() -> list[tuple[str, ...]]:
    """Every attack type, in reading order.

    Returns:
        72 rows matching :data:`HEADER`.
    """
    return [(sc, ac, ge, la, se, *verdict(sc, ac, ge, la, se))
            for sc in ("atlas", "celltype")
            for ac in ("insert", "modify")
            for ge, la in GEOMETRY_LABEL
            for se in ("uniform", "concentrated", "hub")]


def render(rows: list[tuple[str, ...]]) -> str:
    """The table as fixed-width text, ruled between scope/access blocks.

    Fixed width rather than markdown because the column contents vary from two
    characters to thirty, and a renderer left to guess makes the grid unreadable
    at exactly the width a reviewer is likely to use.

    Args:
        rows: output of :func:`build`.

    Returns:
        The rendered table, ending with a one-line tally.
    """
    width = [max(len(r[i]) for r in [*rows, HEADER]) for i in range(len(HEADER))]
    rule = "+" + "+".join("-" * (w + 2) for w in width) + "+"

    def row(cells: tuple[str, ...]) -> str:
        return "| " + " | ".join(c.ljust(width[i]) for i, c in enumerate(cells)) + " |"

    out = [rule, row(HEADER), rule]
    previous: tuple[str, ...] | None = None
    for r in rows:
        if previous is not None and r[:2] != previous[:2]:
            out.append(rule)
        out.append(row(r))
        previous = r
    out += [rule, "",
            f"{len(rows)} attack types: "
            f"{sum(1 for r in rows if r[7] == 'YES')} to run, "
            f"{sum(1 for r in rows if r[7] == 'NO')} not."]
    return "\n".join(out)


if __name__ == "__main__":
    print(render(build()))
