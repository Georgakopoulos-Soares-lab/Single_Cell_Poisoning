"""Every step id is unique across scripts, results and figures.

CLAUDE.md §3 promises that a step's script, its tables and its figures all share
``MM_NN_<access>``, so that one grep finds everything that step produced. Nothing
enforced it. ``check_readme.py`` compares each README against its own directory
and is blind by construction to a number reused in a DIFFERENT directory -- which
is exactly the failure that occurred: ``02a_02`` was already ``02a_01``'s figure
when a new analysis script was written under the same id, and both READMEs were
correct. It was caught by reading a directory listing.

Four properties, each a way the grep promise can break:

1. **One script per step.** Two scripts sharing an id make ``grep`` ambiguous
   about which produced what.
2. **One access per step.** ``02b_05_with_access`` and ``02b_05_no_access``
   would be two experiments under one number.
3. **Convention.** Every file under an ``MM_`` analysis or figure directory
   parses as ``MM_NN_<access>``, or it cannot be found by step at all.
4. **Module ordering.** A step's ``MM`` matches the directory holding it, so
   ``02a`` really is the atlas module and ``02b`` really is per-celltype.

Run as ``python check_steps.py v02``; exits non-zero on any problem.
"""
from __future__ import annotations

import collections
import pathlib
import re
import sys

STEP = re.compile(r"^(0[1-4][a-b])_(\d{2})_(no_access|with_access|both)(?:_|\.|$)")
MODULE_DIR = re.compile(r"^(0[1-4])_([a-b])_")
SKIP = {"README.txt"}


def step_id(name: str) -> tuple[str, str, str] | None:
    """``(module, number, access)`` for a filename, or None if it is not a step."""
    m = STEP.match(name)
    return (m.group(1), m.group(2), m.group(3)) if m else None


def main(version: str) -> int:
    base = pathlib.Path(version)
    owners: dict[str, dict[str, list[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    problems: list[str] = []

    for root, kind in ((base / "1_scripts", "script"),
                       (base / "2_results", "result"),
                       (base / "3_figures", "figure")):
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if (not p.is_file() or p.name in SKIP or "__pycache__" in str(p)
                    or p.name.startswith(".")):
                continue
            sid = step_id(p.name)
            mod_dir = MODULE_DIR.match(p.parent.name)
            if sid is None:
                # only files inside a module directory are required to be steps
                if mod_dir and kind != "result":
                    problems.append(
                        f"{p}: inside module {p.parent.name} but does not parse "
                        f"as MM_NN_<access>; nothing can find it by step")
                continue
            mm, nn, access = sid
            owners[f"{mm}_{nn}"][kind].append(p.name)
            owners[f"{mm}_{nn}"]["_access"].append(access)
            if mod_dir and f"{mod_dir.group(1)}{mod_dir.group(2)}" != mm:
                problems.append(
                    f"{p}: step id says module {mm} but it sits in "
                    f"{p.parent.name}")

    for sid, kinds in sorted(owners.items()):
        scripts = kinds["script"] + [f for f in kinds["figure"]
                                     if f.endswith(".py")]
        if len(scripts) > 1:
            problems.append(f"{sid}: {len(scripts)} scripts share this id -- "
                            f"{', '.join(scripts)}")
        acc = set(kinds["_access"])
        if len(acc) > 1:
            problems.append(f"{sid}: mixed access {sorted(acc)} under one id")

    for p in problems:
        print(p, file=sys.stderr)
    print(f"{len(owners)} step ids checked; {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "v02"))
