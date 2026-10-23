"""Refresh a directory's README.txt from what is actually in it.

``check_readme.py`` makes a stale README fail the commit; this makes fixing one
a command rather than a chore, which is the difference between a rule that holds
and a rule that gets bypassed with --no-verify.

It cannot invent. Each entry's description comes from one of two places:

* a ``.py`` file -- its own module docstring, first line, copied verbatim.
* a data file -- the docstring of the SCRIPT THAT PRODUCED IT, found by the
  ``MM_NN_access`` prefix the two share. So a table is described by the code
  that wrote it, and the description cannot drift from the producer without the
  producer itself changing.

Anything it cannot source is written as ``TODO: describe`` rather than guessed.
That line is meant to be replaced by a human with the two further sentences the
convention asks for; it is deliberately conspicuous.

The preamble above ``CONTENTS`` is preserved untouched, so hand-written notes --
known defects, deferrals, dependencies on cache/ -- survive a refresh.

Usage:
    python update_readme.py v02/2_results/GSE136831/01_a_baseline
    python update_readme.py v02            # every directory beneath it
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys

STEP = re.compile(r"^(\d{2}[a-z]_\d{2}_(?:no_access|with_access|both))")
SKIP = {"README.txt", ".DS_Store"}


def first_line(py: pathlib.Path) -> str:
    """First line of a python file's module docstring, or '' if it has none."""
    try:
        doc = ast.get_docstring(ast.parse(py.read_text())) or ""
    except SyntaxError:
        return ""
    return doc.strip().split("\n")[0] if doc else ""


def producers(root: pathlib.Path) -> dict[str, pathlib.Path]:
    """Map every step id in the version tree to the script that owns it."""
    out: dict[str, pathlib.Path] = {}
    for p in root.rglob("*.py"):
        if "__pycache__" in str(p):
            continue
        if m := STEP.match(p.name):
            out.setdefault(m.group(1), p)
    return out


def describe(f: pathlib.Path, prod: dict[str, pathlib.Path]) -> str:
    """One line for ``f``: its own docstring, or its producer's."""
    if f.suffix == ".py":
        return first_line(f) or "TODO: describe"
    if (m := STEP.match(f.name)) and (src := prod.get(m.group(1))):
        d = first_line(src)
        if d:
            return f"Written by {src.name} -- {d}"
    return "TODO: describe"


def refresh(d: pathlib.Path, prod: dict[str, pathlib.Path]) -> bool:
    """Rewrite ``d``'s README contents section. True if it changed."""
    r = d / "README.txt"
    if not r.exists():
        return False
    old = r.read_text()
    if "PATTERN:" in old:          # pattern directory: declares a generator, not files
        return False
    head = old.partition("CONTENTS")[0].rstrip()
    files = sorted(x.name for x in d.iterdir()
                   if x.is_file() and x.name not in SKIP and not x.name.startswith("."))
    lines = [head, "", "CONTENTS"]
    if not files:
        lines.append("  (none yet)")
    for f in files:
        lines.append(f"  {f}")
        lines.append(f"      {describe(d / f, prod)}")
    lines.append("")
    new = "\n".join(lines)
    if new != old:
        r.write_text(new)
        return True
    return False


def main(args: list[str]) -> int:
    targets = [pathlib.Path(a) for a in args] or [pathlib.Path("v02")]
    root = targets[0]
    while root.name not in ("v01", "v02") and root.parent != root:
        root = root.parent
    prod = producers(root)
    n = 0
    for t in targets:
        for d in ([t] if (t / "README.txt").exists() else []) + [
                x for x in sorted(t.rglob("*"))
                if x.is_dir() and not x.is_symlink() and "__pycache__" not in str(x)]:
            n += refresh(d, prod)
    print(f"{n} README.txt updated")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
