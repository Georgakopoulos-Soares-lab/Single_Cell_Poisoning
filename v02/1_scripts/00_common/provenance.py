"""Which code actually ran — the stamp a log header must carry.

**The defect this exists to fix.** Log headers recorded
``git rev-parse --short HEAD``, written by unversioned zsh wrappers. That is
HEAD, not the working tree, and we have proof it lies: ``02b_10`` and ``02b_12``
wrote tables carrying a per-row ``threat_model`` column while their logs stamp
``ce75aab4``, whose version of those scripts cannot produce that column -- it
arrives in ``a1bd6bf7``, committed one minute later. The runs were launched from
a tree that already held the newer code. Anyone reproducing from the stamped SHA
gets a different table and no signal that it differs.

So the stamp is emitted by the PYTHON step, not by whatever launched it. It is
then correct under a Snakemake job, a bare ``python`` invocation, a worker shell
script, or a wrapper that no longer exists.

**A dirty tree is announced, never rounded to HEAD.** ``sha abc1234`` when the
tree has uncommitted edits is the defect; ``sha abc1234 (DIRTY, diff <digest>)``
is honest. The digest makes two runs from the SAME dirty tree recognisably the
same code and a different dirty tree visibly different, which is the most a
stamp can offer short of archiving the diff itself.

**DIRTY is scoped to the code a step executes**: :data:`CODE_PATHS`, which is
``v02/1_scripts`` and ``v02/config``. Those are what a run actually reads -- the
step script, the shared library under ``00_common``, and the config the step
loads. ``2_results`` is output, ``3_figures`` cannot affect a sweep, ``docs``
cannot affect anything.

The scope is not cosmetic, in either direction. Scoped too WIDE it says nothing:
this repository's git toplevel sits one level ABOVE the project directory, so
untracked scratch anywhere under it -- ``tacc/`` alone is 20-odd files today --
would mark every run DIRTY forever and move the digest for reasons unrelated to
the code, and a flag that is always on is not a flag. Scoped to ``git diff HEAD``
ALONE it lies: a new, untracked module under ``00_common/`` changes what ran and
leaves ``git diff HEAD`` empty, so the stamp would say clean and be wrong -- the
same failure as SHA-is-HEAD-not-tree, one step sideways. So inside the scope both
tracked modifications AND untracked files set DIRTY and enter the digest, while
untracked files outside it are counted on the line and set nothing.

**The residual gap, named rather than removed.** Two remain.

1. A step's behaviour also depends on what it READS -- ``cache/`` arrays,
   ``0_data``, the conda environment. None of those is under version control,
   so no git-based stamp can cover them; ``cache/`` is gitignored by design. A
   clean stamp means "this source tree", not "this input".
2. The digest hashes untracked in-scope files by path AND by content, so editing
   one moves it, but a file larger than :data:`UNTRACKED_BYTE_CAP` contributes
   its path and size only. Nothing in ``1_scripts`` or ``config`` is remotely
   near that cap; a file that is has no business there.

**Three git calls, once per process.** No single git command yields the SHA, the
branch, the scoped diff and the untracked list together, so the probe makes three
fixed calls and :func:`probe` caches the result for the lifetime of the process.
The rule being honoured is "not once per condition" -- these steps run for hours
and a subprocess in the condition loop would be a smell. Nothing here runs in a
loop. The untracked list is fetched repository-wide ONCE and partitioned in
Python rather than asked for twice at two scopes.
"""
from __future__ import annotations

import dataclasses
import hashlib
import pathlib
import subprocess
import sys
from typing import Final, TextIO

#: This module lives at ``v02/1_scripts/00_common/provenance.py``.
_SCRIPTS: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parents[1]

#: The paths whose state decides DIRTY: everything a step executes or reads as
#: configuration. See the module docstring for why this and nothing wider.
CODE_PATHS: Final[tuple[pathlib.Path, ...]] = (_SCRIPTS,
                                               _SCRIPTS.parent / "config")

#: What every field reports when git cannot be consulted at all.
UNKNOWN: Final[str] = "unknown"
#: Hex characters kept from the diff digest. 12 gives 48 bits -- ample to tell
#: two dirty trees apart in a project whose whole history is a few hundred runs,
#: and short enough to read off a log header.
DIGEST_CHARS: Final[int] = 12
#: Characters kept for the short SHA. Truncated in Python rather than taken from
#: ``git rev-parse --short``, whose length varies with repository size: a stamp
#: that changes width as the history grows is a stamp two runs cannot be
#: compared on.
SHORT_CHARS: Final[int] = 8
#: Bytes of an untracked in-scope file that enter the digest. Past this the file
#: contributes its path and size only -- see residual gap 2.
UNTRACKED_BYTE_CAP: Final[int] = 4 << 20
#: Seconds any one git call may take before it is treated as unavailable.
GIT_TIMEOUT: Final[float] = 30.0


@dataclasses.dataclass(frozen=True)
class Provenance:
    """The identity of the code that ran. Dimensionless; all fields are text.

    Attributes:
        sha: full 40-hex commit id of HEAD, or :data:`UNKNOWN`.
        short_sha: first :data:`SHORT_CHARS` of ``sha``, or :data:`UNKNOWN`.
        branch: current branch name, ``HEAD`` when detached, or :data:`UNKNOWN`.
        dirty: True when anything under ``code_paths`` differs from HEAD, by
            tracked modification OR by an untracked file. False on a clean scope,
            and False when git is unavailable -- see ``available``, which is the
            field that distinguishes "clean" from "not asked".
        diff_digest: :data:`DIGEST_CHARS` hex over the scoped diff and the
            untracked in-scope files, or None when the scope is clean or git is
            unavailable.
        n_changed: tracked files changed WITHIN ``code_paths``.
        n_untracked_code: untracked non-ignored files WITHIN ``code_paths``.
            These DO set ``dirty``.
        n_untracked_elsewhere: untracked non-ignored files anywhere else in the
            repository. Reported for visibility; these set nothing.
        code_paths: the scope actually measured, repository-relative, so the
            record states what it looked at. Empty when the running code is not
            under this repository at all, which ``available`` cannot express.
        available: whether git answered. False in a container, on a node with no
            git, or outside a repository.
    """

    sha: str
    short_sha: str
    branch: str
    dirty: bool
    diff_digest: str | None
    n_changed: int
    n_untracked_code: int
    n_untracked_elsewhere: int
    code_paths: tuple[str, ...]
    available: bool

    def state(self) -> str:
        """The parenthesised state clause of :meth:`line`.

        Every count names its scope. Two scopes on one line without saying which
        is which is its own trap: ``+22 untracked`` beside a clean flag reads as
        a contradiction until you know the 22 are outside the code.
        """
        if not self.available:
            return "no git"
        if not self.code_paths:
            return f"sha only, code not under this repository ({UNKNOWN} tree)"
        far = (f", elsewhere +{self.n_untracked_elsewhere} untracked"
               if self.n_untracked_elsewhere else "")
        if not self.dirty:
            return f"clean in code{far}"
        return (f"DIRTY, diff {self.diff_digest}, code {self.n_changed} changed "
                f"+{self.n_untracked_code} untracked{far}")

    def line(self, script: str | pathlib.Path | None = None) -> str:
        """One grep-friendly line naming the step and the code it ran as.

        Args:
            script: the step's ``__file__``; its basename is quoted. None omits
                it, for a caller that has already named itself.
        """
        who = f" {pathlib.Path(script).name}" if script is not None else ""
        return (f"[provenance]{who} sha {self.short_sha} ({self.state()}) "
                f"branch {self.branch} full {self.sha}")


def _git(args: list[str], cwd: pathlib.Path) -> bytes | None:
    """Run one git command, or return None if git cannot answer.

    The try/except is deliberate and is NOT the pattern CLAUDE.md §7 forbids:
    that rule is about numerical code, where a silent fallback turns a wrong
    number into a plausible one. Nothing here is a measurement. A missing git
    binary (``FileNotFoundError``), a non-repository (exit 128) or a hung call
    must degrade to an honest ``unknown`` stamp rather than kill a 16-hour
    sweep, because the run is the valuable thing and the stamp is metadata
    about it.
    """
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                           check=False, timeout=GIT_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def _digest(diff: bytes, untracked: list[pathlib.Path]) -> str:
    """Hex digest over the scoped diff and the untracked in-scope files.

    Untracked files are hashed by path AND content, because an untracked module
    under ``00_common/`` is code that ran and a digest that ignored its contents
    would call two different trees the same. Past
    :data:`UNTRACKED_BYTE_CAP` only the size enters, so a growing file still
    moves the digest without the stamp reading a gigabyte.

    Args:
        diff: raw bytes of ``git diff HEAD -- <scope>``.
        untracked: absolute paths of untracked non-ignored files in scope.

    Returns:
        :data:`DIGEST_CHARS` hex characters.
    """
    h = hashlib.sha256(diff)
    for p in untracked:
        size = p.stat().st_size if p.is_file() else -1
        h.update(f"\n--untracked {p} {size}--\n".encode())
        if 0 <= size <= UNTRACKED_BYTE_CAP:
            h.update(p.read_bytes())
    return h.hexdigest()[:DIGEST_CHARS]


_CACHE: dict[tuple, Provenance] = {}


def probe(near: str | pathlib.Path | None = None, *,
          code_paths: tuple[pathlib.Path, ...] | None = None) -> Provenance:
    """Ask git what code this process is running, once per process.

    Args:
        near: any path inside the repository -- normally a caller's
            ``__file__``. git is run with its directory as cwd, so the answer
            does not depend on where the process was launched from. None uses
            this module's own directory, which is inside the repository by
            construction.
        code_paths: the scope DIRTY is judged over; :data:`CODE_PATHS` when None.
            Injectable so a test can build a throwaway repository and dirty it.

    Returns:
        The :class:`Provenance` record, memoised per (directory, scope).
    """
    here = pathlib.Path(near or __file__).resolve()
    cwd = here.parent if here.is_file() else here
    scope = tuple(p.resolve() for p in (code_paths or CODE_PATHS))
    key = (cwd, scope)
    if key in _CACHE:
        return _CACHE[key]

    def _done(prov: Provenance) -> Provenance:
        _CACHE[key] = prov
        return prov

    # One call for three answers. Argument ORDER matters: `--abbrev-ref` applies
    # to every rev after it, so `rev-parse --abbrev-ref HEAD HEAD` returns the
    # branch name TWICE and the SHA never. The SHA must come first.
    head = _git(["rev-parse", "--show-toplevel", "HEAD", "--abbrev-ref", "HEAD"],
                cwd)
    if head is None:
        return _done(Provenance(UNKNOWN, UNKNOWN, UNKNOWN, False, None,
                                0, 0, 0, (), False))

    top, sha, branch = head.decode(errors="replace").split()
    # The remaining calls run from the toplevel, so `ls-files --others` sees the
    # whole repository and reports repository-relative paths; called from
    # 00_common/ it would miss everything above it and print `../../../...`.
    root = pathlib.Path(top)
    prov = dict(sha=sha, short_sha=sha[:SHORT_CHARS], branch=branch)

    # An absolute pathspec outside the repository is a FATAL error to git, not an
    # empty result, so the scope is made repository-relative and anything not
    # under this root is dropped. Losing every scope path means the running code
    # is not in this repository -- reported, never read as clean.
    rel = tuple(str(p.relative_to(root)) for p in scope
                if p == root or root in p.parents)
    if not rel:
        return _done(Provenance(**prov, dirty=False, diff_digest=None,
                                n_changed=0, n_untracked_code=0,
                                n_untracked_elsewhere=0, code_paths=(),
                                available=True))

    diff = _git(["diff", "HEAD", "--", *rel], root) or b""
    others = _git(["ls-files", "--others", "--exclude-standard", "-z"], root) or b""

    # `ls-files --others` enumerates FILES; `git status` would collapse a wholly
    # untracked directory to one entry and hide what is in it.
    untracked = [pathlib.PurePosixPath(p.decode()) for p in others.split(b"\0") if p]
    in_code = sorted(p for p in untracked
                     if any(p == pathlib.PurePosixPath(r)
                            or pathlib.PurePosixPath(r) in p.parents for r in rel))

    n_changed = diff.count(b"\ndiff --git ") + diff.startswith(b"diff --git ")
    dirty = bool(diff) or bool(in_code)
    return _done(Provenance(
        **prov, dirty=dirty,
        diff_digest=_digest(diff, [root / p for p in in_code]) if dirty else None,
        n_changed=n_changed, n_untracked_code=len(in_code),
        n_untracked_elsewhere=len(untracked) - len(in_code),
        code_paths=rel, available=True))


def emit(script: str | pathlib.Path | None = None, *,
         stream: TextIO | None = None,
         code_paths: tuple[pathlib.Path, ...] | None = None) -> Provenance:
    """Print the provenance line and return the record.

    Called at the top of a step's ``main()``. Reports to stderr, as every other
    progress line in this project does, so the wrapper's ``2>&1`` capture puts
    it in the log beside the step's own output.

    Args:
        script: the step's ``__file__``.
        stream: destination; ``sys.stderr`` when None.
        code_paths: passed through to :func:`probe`.
    """
    prov = probe(script, code_paths=code_paths)
    print(prov.line(script), file=stream if stream is not None else sys.stderr,
          flush=True)
    return prov
