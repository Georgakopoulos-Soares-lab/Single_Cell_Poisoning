"""Every artefact traces to the script that produced it, by its step id.

The reproducibility guarantee is that a colleague given the same input can
regenerate every table and figure. Under the v02 naming that becomes mechanical
rather than a search, because a step's script and all of its outputs share one
identifier::

    1_scripts/02_b_attack_celltype/02b_07_with_access_sweep.py
    2_results/<DS>/02_b_attack_celltype/02b_07_with_access_sweep.parquet
    3_figures/<DS>/02_b_attack_celltype/02b_11_with_access_script.py
    3_figures/<DS>/02_b_attack_celltype/02b_11_with_access.png

So the check is: strip the ``MM_NN_access`` prefix off an artefact, and require a
script carrying that prefix in the place the rules say it must live.

This replaces v01's longest-static-prefix matching, which had to guess where a
runtime-built filename stopped being static and got it wrong twice. A naming
convention the producer obeys is cheaper to verify than a heuristic that infers
it afterwards.

Two placement rules, because the trees differ:

``2_results``   an analysis table's producer is in ``1_scripts/<stage>/``; a
                figure's plotted-value CSV may instead be produced by the
                figure script in ``3_figures/<DS>/<stage>/``.
``3_figures``   a figure's producer is ADJACENT -- ``<step>_script.*`` in the
                same directory. Nothing else is accepted, because the whole
                point of co-locating them is that no search is needed.

Exits non-zero, so it can gate a commit.
"""
from __future__ import annotations

import pathlib
import re
import sys

#: ``MM_NN_access`` -- module, step, and whether the attacker can WRITE to
#: the reference. no_access: submissions only, existing records untouchable.
#: with_access: can overwrite them. both: the step runs the contrast.
STEP = re.compile(r"^(\d{2}[a-z]_\d{2}_(?:no_access|with_access|both))")
#: Directories holding machine-generated families, checked by check_readme.py
#: against a declared PATTERN instead of per-file.
SKIP_DIRS = {"logs", "00_raw", "__pycache__"}
#: Not artefacts: documentation and caption sidecars.
SKIP_SUFFIX = {".txt", ".md"}


def step_of(name: str) -> str | None:
    """The ``MM_NN_access`` id an artefact belongs to, or None if unnamed."""
    m = STEP.match(name)
    return m.group(1) if m else None


def scripts_under(d: pathlib.Path) -> set[str]:
    """Step ids of every script directly in ``d``."""
    out = set()
    if d.is_dir():
        for p in d.iterdir():
            if p.is_file() and p.suffix in (".py", ".R") and (s := step_of(p.name)):
                out.add(s)
    return out


def check(root: pathlib.Path) -> list[str]:
    """Problems under one version root; empty list means clean."""
    bad: list[str] = []
    for tree in ("2_results", "3_figures"):
        base = root / tree
        if not base.is_dir():
            continue
        for d in sorted(base.rglob("*")):
            if not d.is_dir() or SKIP_DIRS & set(d.parts):
                continue
            stage = d.name
            for f in sorted(d.iterdir()):
                if (not f.is_file() or f.name == "README.txt"
                        or f.suffix in SKIP_SUFFIX or "_script." in f.name):
                    continue
                s = step_of(f.name)
                if s is None:
                    bad.append(f"UNNAMED   {f}: does not start with MM_NN_access")
                    continue
                if tree == "3_figures":
                    if s not in scripts_under(d):
                        bad.append(f"ORPHAN    {f}: no {s}_script.* beside it")
                else:
                    analysis = scripts_under(root / "1_scripts" / stage)
                    figure = scripts_under(base.parent / "3_figures" / d.parent.name / stage)
                    if s not in analysis | figure:
                        bad.append(f"ORPHAN    {f}: no {s}_* script in "
                                   f"1_scripts/{stage}/ or the figure tree")
    return bad


def check_reads(root: pathlib.Path) -> list[str]:
    """Every figure declares the analysis step it draws, and the claim is true.

    ``check`` above verifies the FORWARD link -- an artefact traces to the script
    whose step id it shares. This verifies the BACKWARD one: a figure script
    traces to the analysis step whose table it opened. Nothing enforced that,
    and the step number cannot carry it -- ``02b_03`` reads ``02b_01`` while
    ``02b_04`` reads ``02b_02``, so the offset is not even constant.

    Two directions, because a declaration that is merely present proves nothing:

    1. every step named in ``READS`` exists and produced a table, so a figure
       cannot claim an input that is not there;
    2. every table the script actually OPENS belongs to a step named in
       ``READS``, so a silently changed input fails instead of passing with a
       stale declaration.

    Read with :mod:`ast`, never imported -- importing a figure script would draw
    the figure.

    Args:
        root: a version directory, e.g. ``v02``.

    Returns:
        One string per problem; empty when the tree is sound.
    """
    import ast  # noqa: PLC0415
    import re  # noqa: PLC0415

    out: list[str] = []
    fig_root = root / "3_figures"
    if not fig_root.exists():
        return out
    for f in sorted(fig_root.rglob("*_script.py")):
        tree = ast.parse(f.read_text())
        declared: tuple[str, ...] | None = None
        for node in tree.body:
            if (isinstance(node, ast.AnnAssign | ast.Assign)
                    and getattr(node, "value", None) is not None):
                names = ([node.target] if isinstance(node, ast.AnnAssign)
                         else node.targets)
                if any(getattr(t, "id", None) == "READS" for t in names):
                    declared = tuple(ast.literal_eval(node.value))
        if declared is None:
            out.append(f"{f}: no READS -- the analysis step it draws is "
                       f"undeclared, so the link cannot be checked")
            continue
        for sid in declared:
            if not list((root / "1_scripts").rglob(f"{sid}_*.py")):
                out.append(f"{f}: READS names {sid}, which has no script")
            if not list((root / "2_results").rglob(f"{sid}_*.parquet")):
                out.append(f"{f}: READS names {sid}, which produced no table")
        # every step whose artefact this script opens must be declared
        opened = set(re.findall(r"\b(0[1-4][ab]_\d{2})_\w*\.(?:parquet|npz)",
                                f.read_text()))
        for sid in sorted(opened - set(declared)):
            out.append(f"{f}: opens a {sid} artefact that READS does not "
                       f"declare -- the declaration has drifted from the code")
    return out


def main(roots: list[str]) -> int:
    problems, n = [], 0
    for r in roots:
        root = pathlib.Path(r)
        problems += check(root)
        problems += check_reads(root)
        n += sum(1 for tree in ("2_results", "3_figures")
                 for f in (root / tree).rglob("*")
                 if f.is_file() and f.name != "README.txt"
                 and f.suffix not in SKIP_SUFFIX and not SKIP_DIRS & set(f.parts))
    for p in problems:
        print(p, file=sys.stderr)
    print(f"{n} artefacts checked; {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["v02"]))
