"""A run must say which code produced it, and a writer must name what it wrote.

Two defects, both found by auditing the RUNLOG on 2026-09-27, and both about the
record rather than the numbers.

``write_by_donor`` wrote donor tables that no log line anywhere named as output.
``02b_13`` and ``02b_14`` printed nothing after calling it, so two tables that
``02b_12`` reads downstream had their provenance resting on an mtime. The fix
moved the report into the writer, and :func:`test_by_donor_reports_what_it_wrote`
checks the reported count against the file on disk rather than against the
frames handed in -- reporting the input would re-introduce the same class of lie
one level down.

Log headers stamped ``git rev-parse --short HEAD``, which is HEAD and not the
working tree. ``02b_10`` and ``02b_12`` wrote a ``threat_model`` column while
stamping ``ce75aab4``, a commit whose code cannot produce it. The tests below
build throwaway repositories, because the only way to check that a dirty tree is
announced is to dirty one.

The scope tests are the sharp ones. DIRTY is judged over ``CODE_PATHS`` --
``v02/1_scripts`` and ``v02/config`` -- and the two failure modes sit on either
side of that line: too wide and the flag is always on and says nothing, too
narrow (``git diff HEAD`` alone) and a new untracked module reads as clean.
:func:`test_untracked_code_is_dirty` and
:func:`test_untracked_outside_code_is_not_dirty` pin both edges.
"""
import io
import pathlib
import shutil
import subprocess
import sys

import pandas as pd
import pytest

import provenance
from celltype_sweep import write_by_donor

#: These tests drive a real git binary; there is nothing to assert without one.
needs_git = pytest.mark.skipif(shutil.which("git") is None,
                               reason="git not on PATH")

#: Identity is passed with ``-c`` rather than written into a config, so a test
#: cannot depend on -- or disturb -- the developer's own git identity.
GIT = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    """Drop the per-process memo so each test probes its own repository."""
    provenance._CACHE.clear()


class Repo:
    """A throwaway repository shaped like this project's scope.

    Holds ``code/`` (standing in for ``v02/1_scripts`` + ``v02/config``) and
    ``output/`` outside it, so a test can dirty either side of the line.

    Attributes:
        root: repository toplevel.
        step: the committed file under ``code/`` that plays the step script.
        scope: what to pass as ``code_paths``.
    """

    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        (root / "code").mkdir(parents=True)
        (root / "output").mkdir()
        self.step = root / "code" / "step.py"
        self.step.write_text("x = 1\n")
        (root / "output" / "table.csv").write_text("a\n1\n")
        self.scope = (root / "code",)
        subprocess.run([*GIT, "init", "-q", "-b", "trunk"], cwd=root, check=True)
        subprocess.run([*GIT, "add", "-A"], cwd=root, check=True)
        subprocess.run([*GIT, "commit", "-qm", "one"], cwd=root, check=True)

    def probe(self) -> provenance.Provenance:
        """Probe this repository, always through a cleared memo."""
        provenance._CACHE.clear()
        return provenance.probe(self.step, code_paths=self.scope)


@pytest.fixture
def repo(tmp_path) -> Repo:
    return Repo(tmp_path / "repo")


# --------------------------------------------------------------------------
# FIX A -- the writer names what it wrote
# --------------------------------------------------------------------------

def test_by_donor_reports_what_it_wrote(tmp_path, capsys) -> None:
    """The report names the resolved path and the row count now ON DISK.

    The count is read back out of the parquet rather than summed over the input
    frames: a report derived from the input would agree with itself even if the
    write dropped rows, which is exactly the failure the log is meant to catch.
    """
    frames = [pd.DataFrame({"donor": [0, 1], "within_retrieval_num": [1.0, 2.0]}),
              pd.DataFrame({"donor": [2], "within_retrieval_num": [3.0]})]
    out = tmp_path / "nested" / "02b_13_no_access_flood_geometry_by_donor.parquet"

    df = write_by_donor(frames, out)
    err = capsys.readouterr().err

    on_disk = len(pd.read_parquet(out))
    assert on_disk == len(df) == 3
    assert str(on_disk) in err
    assert str(out.resolve()) in err


def test_by_donor_report_tracks_a_growing_table(tmp_path, capsys) -> None:
    """Called once per source, each report carries that call's count.

    The callers accumulate ``donor_frames`` and re-write the whole table after
    every source, so a truncated run is still readable. The report has to move
    with it; a fixed total would be wrong on every line but the last.
    """
    out = tmp_path / "t_by_donor.parquet"
    frames: list[pd.DataFrame] = []
    counts = []
    for i in range(3):
        frames.append(pd.DataFrame({"donor": range(i + 1)}))
        write_by_donor(frames, out)
        counts.append(len(pd.read_parquet(out)))

    reported = [ln for ln in capsys.readouterr().err.splitlines()
                if "donor rows" in ln]
    assert len(reported) == 3
    assert counts == [1, 3, 6]
    for n, line in zip(counts, reported):
        assert f"{n:,} donor rows" in line


# --------------------------------------------------------------------------
# FIX B -- the stamp tells the truth about the code
# --------------------------------------------------------------------------

@needs_git
def test_clean_tree_is_clean(repo) -> None:
    """A committed, untouched scope stamps its SHA and says clean."""
    p = repo.probe()

    assert p.available
    assert not p.dirty
    assert p.diff_digest is None
    assert len(p.sha) == 40
    assert p.short_sha == p.sha[:provenance.SHORT_CHARS]
    assert p.branch == "trunk"
    assert p.code_paths == ("code",)
    assert "clean in code" in p.line(repo.step)
    assert "DIRTY" not in p.line(repo.step)


@needs_git
def test_modified_code_is_dirty(repo) -> None:
    """An uncommitted edit makes the stamp say DIRTY and carry a digest.

    The SHA is unchanged -- that is the point. Stamping it alone, as the zsh
    wrappers did, is indistinguishable from the clean run at that commit.
    """
    clean = repo.probe()
    repo.step.write_text("x = 2  # uncommitted\n")
    dirty = repo.probe()

    assert dirty.sha == clean.sha
    assert dirty.dirty and not clean.dirty
    assert dirty.diff_digest is not None
    assert dirty.n_changed == 1
    line = dirty.line(repo.step)
    assert "DIRTY" in line
    assert dirty.diff_digest in line
    assert dirty.short_sha in line


@needs_git
def test_untracked_code_is_dirty(repo) -> None:
    """A NEW UNTRACKED module in scope sets DIRTY. This is the closed gap.

    ``git diff HEAD`` is empty for a file git has never seen, so a stamp built
    on the diff alone would call this tree clean while a step imports the new
    module and produces a table that HEAD's code cannot produce. That is the
    ``02b_10`` defect wearing a different hat.
    """
    (repo.root / "code" / "helper.py").write_text("def h(): return 1\n")
    p = repo.probe()

    assert p.dirty
    assert p.n_changed == 0          # nothing TRACKED changed
    assert p.n_untracked_code == 1
    assert p.diff_digest is not None
    assert "+1 untracked" in p.line(repo.step)


@needs_git
def test_untracked_outside_code_is_not_dirty(repo) -> None:
    """Untracked scratch outside the scope is counted and sets nothing.

    The git toplevel sits above the project directory, so scratch like ``tacc/``
    would otherwise mark every run DIRTY forever. A flag that is always on
    carries no information, which is the failure on the other side of the line
    from :func:`test_untracked_code_is_dirty`.
    """
    (repo.root / "output" / "scratch.txt").write_text("notes\n")
    (repo.root / "loose.md").write_text("notes\n")
    p = repo.probe()

    assert not p.dirty
    assert p.diff_digest is None
    assert p.n_untracked_code == 0
    assert p.n_untracked_elsewhere == 2
    assert "clean in code" in p.line(repo.step)
    assert "elsewhere +2 untracked" in p.line(repo.step)


@needs_git
def test_every_count_on_the_line_names_its_scope(repo) -> None:
    """Both scopes appear, each labelled. Two bare counts would be a trap.

    ``+22 untracked`` beside a clean flag reads as a contradiction until the
    reader knows the 22 are outside the code.
    """
    (repo.root / "code" / "helper.py").write_text("h = 1\n")
    (repo.root / "output" / "scratch.txt").write_text("notes\n")
    repo.step.write_text("x = 9\n")
    state = repo.probe().state()

    assert state.startswith("DIRTY")
    assert "code 1 changed +1 untracked" in state
    assert "elsewhere +1 untracked" in state


@needs_git
def test_digest_identifies_the_tree_not_the_run(repo) -> None:
    """Same dirty scope, same digest; a different dirty scope, a different one.

    This is the whole value of the digest: two runs launched from one uncommitted
    state are recognisably the same code, and a run from a different uncommitted
    state cannot be mistaken for them.
    """
    repo.step.write_text("x = 2\n")
    first = repo.probe()
    assert repo.probe().diff_digest == first.diff_digest

    repo.step.write_text("x = 3\n")
    assert repo.probe().diff_digest != first.diff_digest


@needs_git
def test_digest_follows_untracked_content_not_just_its_name(repo) -> None:
    """Editing an untracked in-scope file moves the digest.

    A digest over paths alone would call two trees identical whenever the only
    difference is inside a file git has never seen -- and an untracked module is
    precisely the case this scope exists to catch.
    """
    new = repo.root / "code" / "helper.py"
    new.write_text("def h(): return 1\n")
    first = repo.probe()
    new.write_text("def h(): return 2\n")
    second = repo.probe()

    assert first.dirty and second.dirty
    assert second.diff_digest != first.diff_digest


@needs_git
def test_elsewhere_churn_leaves_the_digest_alone(repo) -> None:
    """Scratch outside the scope changes the count and not the digest.

    Otherwise the digest could not answer "same code?" across two runs that
    merely differed in what was lying around the working directory.
    """
    repo.step.write_text("x = 2\n")
    before = repo.probe()
    (repo.root / "output" / "junk.txt").write_text("junk\n")
    after = repo.probe()

    assert after.diff_digest == before.diff_digest
    assert after.n_untracked_elsewhere == before.n_untracked_elsewhere + 1


@needs_git
def test_code_outside_the_repository_is_not_read_as_clean(repo, tmp_path) -> None:
    """A scope git cannot see reports itself, rather than passing as clean.

    An absolute pathspec outside the repository is FATAL to git, not empty, so
    without this the scoped diff would come back None, be coerced to empty, and
    stamp a confident "clean in code" over a tree it never looked at.
    """
    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()
    provenance._CACHE.clear()
    p = provenance.probe(repo.step, code_paths=(elsewhere,))

    assert p.available
    assert p.code_paths == ()
    assert not p.dirty
    assert "clean" not in p.state()
    assert "not under this repository" in p.state()


def test_outside_a_repository_is_unknown_not_a_crash(tmp_path) -> None:
    """No repository: every field degrades to ``unknown`` and the run goes on."""
    bare = tmp_path / "not_a_repo"
    bare.mkdir()
    (bare / "step.py").write_text("x = 1\n")
    p = provenance.probe(bare / "step.py")

    assert not p.available
    assert p.sha == p.short_sha == p.branch == provenance.UNKNOWN
    assert p.diff_digest is None
    assert "no git" in p.line(bare / "step.py")


def test_missing_git_binary_is_unknown_not_a_crash(tmp_path, monkeypatch) -> None:
    """git absent entirely -- a container, or a TACC node -- still stamps.

    A 16-hour sweep must not die because its metadata could not be collected, so
    ``FileNotFoundError`` from the missing binary is swallowed deliberately. The
    stamp then says ``unknown``, which is honest; silence would not be.
    """
    def _no_git(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(provenance.subprocess, "run", _no_git)
    p = provenance.probe(tmp_path)

    assert not p.available
    assert p.sha == provenance.UNKNOWN


@needs_git
def test_probe_shells_out_once_per_process(repo, monkeypatch) -> None:
    """Repeated calls consult git once. These steps run for hours in a loop."""
    provenance.probe(repo.step, code_paths=repo.scope)

    calls: list[list[str]] = []
    real = provenance.subprocess.run

    def _count(args, *a, **k):
        calls.append(args)
        return real(args, *a, **k)

    monkeypatch.setattr(provenance.subprocess, "run", _count)
    for _ in range(50):
        provenance.probe(repo.step, code_paths=repo.scope)
    assert calls == []


@needs_git
def test_the_real_scope_is_the_code_a_step_executes(repo) -> None:
    """``CODE_PATHS`` is ``v02/1_scripts`` and ``v02/config``, and nothing else.

    Named here so a later widening to ``2_results`` or ``3_figures`` -- neither
    of which can change a sweep -- fails a test rather than quietly making every
    run DIRTY again.
    """
    assert [p.name for p in provenance.CODE_PATHS] == ["1_scripts", "config"]
    assert all(p.parent.name == "v02" for p in provenance.CODE_PATHS)


@needs_git
def test_emit_writes_the_line_and_returns_the_record(repo, capsys) -> None:
    """``emit`` is what a step calls; it prints to stderr and hands back facts."""
    p = provenance.emit(repo.step, code_paths=repo.scope)
    out = capsys.readouterr()

    assert out.out == ""
    assert out.err.strip() == p.line(repo.step)
    assert "step.py" in out.err


@needs_git
def test_emit_honours_an_explicit_stream(repo) -> None:
    """The destination is injectable, so a caller can log it where it likes."""
    buf = io.StringIO()
    p = provenance.emit(repo.step, stream=buf, code_paths=repo.scope)
    assert buf.getvalue().strip() == p.line(repo.step)
    assert sys.stderr is not buf
