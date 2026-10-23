"""No figure draws the shortcut arm of a step that was also run on the graph.

Two steps were run TWICE, under different retrievers, and both tables were kept::

    02b_11_with_access_inplace.parquet         retrieval_path = exact
    02b_11_with_access_inplace_graph.parquet   retrieval_path = graph
    02b_15_with_access_alpha_sweep.parquet     retrieval_path = exact
    02b_15_with_access_alpha_sweep_graph.parquet  retrieval_path = graph

The ``_graph`` twin is the publishable one -- ``modify.graph_topk`` writes the
replacements into the shipped index and re-queries it, which is what SCimilarity
actually returns. The plain one is ``modify.remerge`` over the depth-500
exhaustive list: the merge shortcut, kept so that ``02b_16`` can measure the two
against each other. ``02b_16`` found them disagreeing by more than 0.01 on 9.1%
of 936 conditions and by more than 0.10 on 3.2%, in BOTH directions, with no
advance predictor of which celltype is which. HANDOFF.md §5 carries the table.

The failure mode is silent. Two filenames differ by one suffix, both load, both
have every column a figure wants, and the wrong one publishes a number from a
retriever the paper does not claim. CLAUDE.md §6 says no published number rests
on the shortcut; today that is true only because every figure happens to open
the right name, which nothing enforced.

So: **if a step has both variants, a figure may not name the plain one** --
unless the same script asserts on ``retrieval_path``, which turns an accident
into a visible decision. That is the escape hatch, not an exception: ``02b_18``
and ``02b_23`` already assert, and an assert is what HANDOFF.md §5 asks for,
because a filename is not evidence and the column is.

Always fine, and not flagged:

- opening the ``_graph`` variant;
- opening a DERIVED table -- ``02b_10``, ``02b_12``, ``02b_16`` -- whose
  producer already chose the arm, upstream, where it can be reviewed once;
- a step with only one variant, which is most of them.

Pairs are collected across every dataset and applied to every figure, because
the stem carries the step id and ``check_steps.py`` makes that id unique for the
whole tree. A step that needed a graph rerun for one dataset needed it for all
of them, so the rule must not go quiet on a dataset whose rerun has not landed.

Read with :mod:`ast`, never imported -- importing a figure script would draw the
figure. Exits non-zero, so it can gate a commit.
"""
from __future__ import annotations

import ast
import fnmatch
import pathlib
import re
import sys

#: The column that carries the regime. Asserting on it is the escape hatch.
RETRIEVAL_PATH = "retrieval_path"
#: The suffix ``worker_d`` appends to the graph rerun of a step.
GRAPH = "_graph"
#: A name anchored to a step id -- ``MM_NN_<access>``. An f-string whose leading
#: literal does not start with one is too loose to attribute to a step, so it is
#: not guessed at.
STEP = re.compile(r"^0[1-4][a-b]_\d{2}_(?:no_access|with_access|both)")
#: Never part of the version tree; see ``check_readme.NOT_OURS``.
SKIP_PARTS = {"__pycache__"}


def paired(root: pathlib.Path) -> dict[str, str]:
    """``{plain basename: graph basename}`` for every step run under both.

    Args:
        root: a version directory, e.g. ``v02``.

    Returns:
        One entry per step whose table exists in BOTH regimes. A step with only
        one table is absent, so it is never flagged.
    """
    out: dict[str, str] = {}
    base = root / "2_results"
    if not base.is_dir():
        return out
    for g in sorted(base.rglob(f"*{GRAPH}.parquet")):
        if SKIP_PARTS & set(g.parts):
            continue
        plain = g.with_name(g.name[: -len(f"{GRAPH}.parquet")] + ".parquet")
        if plain.exists():
            out[plain.name] = g.name
    return out


def names_read(tree: ast.Module) -> list[tuple[str, bool]]:
    """Every parquet filename a script names, as ``(name, is_pattern)``.

    A plain string literal yields itself. An f-string yields its literal parts
    joined by ``*``, flagged as a pattern, so that a name assembled at runtime is
    matched against the pair list rather than silently escaping it. Only the
    basename is kept -- ``TAB / "x.parquet"`` and ``"…/x.parquet"`` are one read.

    Args:
        tree: the parsed figure script.

    Returns:
        ``(basename or glob pattern, whether it is a pattern)``, in source order.
    """
    out: list[tuple[str, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.endswith(".parquet"):
                out.append((pathlib.PurePosixPath(node.value).name, False))
        elif isinstance(node, ast.JoinedStr):
            parts = ["*" if isinstance(v, ast.FormattedValue)
                     else str(getattr(v, "value", "")) for v in node.values]
            joined = "".join(parts)
            head = parts[0] if parts else ""
            if joined.endswith(".parquet") and STEP.match(head):
                out.append((pathlib.PurePosixPath(joined).name, True))
    return out


def asserts_on_path(tree: ast.Module) -> bool:
    """Whether the script asserts on ``retrieval_path`` anywhere.

    Both spellings count -- ``d.retrieval_path`` and ``d["retrieval_path"]`` --
    and the VALUE asserted is deliberately not inspected: ``02a_04`` asserts
    ``hnsw`` on one table and ``cached_clean`` on another, and both are right.
    What is being checked is that a human picked an arm on purpose.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        for sub in ast.walk(node.test):
            if isinstance(sub, ast.Attribute) and sub.attr == RETRIEVAL_PATH:
                return True
            if isinstance(sub, ast.Constant) and sub.value == RETRIEVAL_PATH:
                return True
    return False


def check(root: pathlib.Path) -> tuple[int, list[str]]:
    """(parquet reads examined, problems)."""
    pairs = paired(root)
    bad: list[str] = []
    n = 0
    fig_root = root / "3_figures"
    if not fig_root.is_dir():
        return 0, bad
    for f in sorted(fig_root.rglob("*_script.py")):
        if SKIP_PARTS & set(f.parts):
            continue
        tree = ast.parse(f.read_text())
        reads = names_read(tree)
        n += len(reads)
        if asserts_on_path(tree):
            continue
        for name, is_pattern in reads:
            if not is_pattern:
                hits = [name] if name in pairs else []
            else:
                hits = sorted(p for p in pairs if fnmatch.fnmatch(p, name))
            for plain in hits:
                what = ("names" if not is_pattern
                        else f"builds {name!r}, which can resolve to")
                bad.append(
                    f"SHORTCUT  {f}: {what} {plain}, the exact-search "
                    f"comparison arm of a step that was also run on the graph "
                    f"-- read {pairs[plain]} instead, or assert on "
                    f"{RETRIEVAL_PATH} to show the arm was chosen on purpose")
    return n, bad


def main(roots: list[str]) -> int:
    problems, n = [], 0
    for r in roots:
        reads, bad = check(pathlib.Path(r))
        n += reads
        problems += bad
    for p in problems:
        print(p, file=sys.stderr)
    print(f"{n} figure table reads checked; {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["v02"]))
