"""Every directory carries a README.txt that matches its contents exactly.

The reproducibility guarantee is that a colleague can open any directory and
learn what each file is without reading code. That only holds if the README
cannot drift, so this enforces a BIJECTION: every file has an entry, and every
entry names a file that exists. An entry for a file that is not there is a
hallucination and fails the check.

Two directory kinds, because one size does not fit both:

``enumerated``  one entry per file. The normal case.
``pattern``     machine-generated families too numerous to list. The README
                declares ``PATTERN:`` and ``COUNT:`` instead, and this verifies
                every file matches the pattern and the count is truthful.

Deliberately independent of whatever produced the files: it compares the README
against the filesystem, never against a mapping a generator built. A
transformation cannot validate itself -- that is exactly how three stems were
missed in v01.

Exits non-zero on any mismatch, so it can gate a commit.
"""
from __future__ import annotations

import pathlib
import re
import sys

#: Files that are never listed: the README itself and editor/OS noise.
IGNORE = {"README.txt", ".DS_Store", "__pycache__"}
#: Directory names that are not part of the version tree. ``cache`` is a symlink
#: to the SHARED cache above v01/ and v02/: regenerable scratch, gitignored, and
#: not this version's to describe. Symlinks are skipped generally, so a future
#: one cannot silently pull an unrelated tree into the check.
NOT_OURS = {"cache"}
#: A README line that names a file: two-space indent, then the bare filename.
ENTRY = re.compile(r"^ {2}(\S+)\s*$")
PATTERN_LINE = re.compile(r"^PATTERN:\s*(\S+)")
COUNT_LINE = re.compile(r"^COUNT:\s*(\d+)")


def files_in(d: pathlib.Path) -> set[str]:
    """Names of the files a README for ``d`` must account for."""
    return {f.name for f in d.iterdir()
            if f.is_file() and f.name not in IGNORE and not f.name.startswith(".")}


def glob_to_re(pat: str) -> re.Pattern[str]:
    """Translate a README PATTERN (``{field}`` placeholders) to a regex."""
    return re.compile("^" + re.sub(r"\\\{[^}]*\\\}", r"[^_/]+", re.escape(pat)) + "$")


def check_dir(d: pathlib.Path) -> list[str]:
    """Problems in one directory; empty list means clean."""
    readme = d / "README.txt"
    actual = files_in(d)
    if not readme.exists():
        return [] if not actual else [f"{d}: no README.txt ({len(actual)} files)"]
    text = readme.read_text()
    pm, cm = PATTERN_LINE.search(text), COUNT_LINE.search(text)
    if pm:                                   # pattern directory
        rx, bad = glob_to_re(pm.group(1)), []
        off = sorted(f for f in actual if not rx.match(f))
        if off:
            bad.append(f"{d}: {len(off)} files do not match PATTERN "
                       f"{pm.group(1)!r}, e.g. {off[:3]}")
        if cm and int(cm.group(1)) != len(actual):
            bad.append(f"{d}: COUNT says {cm.group(1)}, found {len(actual)}")
        return bad
    listed = {m.group(1) for line in text.splitlines() if (m := ENTRY.match(line))}
    bad = []
    for missing in sorted(actual - listed):
        bad.append(f"{d}: file not in README -- {missing}")
    for ghost in sorted(listed - actual):
        bad.append(f"{d}: README names a file that does not exist -- {ghost}")
    return bad


def main(roots: list[str]) -> int:
    problems, n = [], 0
    for root in roots:
        for d in sorted(pathlib.Path(root).rglob("*")):
            if (d.is_dir() and not d.is_symlink()
                    and d.name not in NOT_OURS
                    and "__pycache__" not in str(d)):
                n += 1
                problems += check_dir(d)
    for p in problems:
        print(p, file=sys.stderr)
    print(f"{n} directories checked; {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["v02"]))
