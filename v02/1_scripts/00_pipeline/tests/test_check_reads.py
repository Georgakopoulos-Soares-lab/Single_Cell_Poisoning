"""A figure must declare the analysis step it draws, and the claim must be true.

``check_provenance`` already verified the forward link -- an artefact traces to
the script sharing its step id. This covers the backward one, which nothing
enforced: a figure script traces to the step whose table it opened.

The number cannot carry that link. ``02b_03`` reads ``02b_01`` and ``02b_04``
reads ``02b_02``, so the offset is not even constant, and a docstring naming the
input drifts silently when the code changes.
"""
import pathlib

import pytest
from check_provenance import check_reads

FIG = '''"""Step {sid} — a figure."""
NAME = "{sid}_no_access"

READS: tuple[str, ...] = {reads!r}

d = pd.read_parquet(TAB / "{opens}")
'''


@pytest.fixture
def tree(tmp_path: pathlib.Path) -> pathlib.Path:
    """A version dir with one analysis step and one figure that draws it."""
    v = tmp_path / "v02"
    (v / "1_scripts/02_b_x").mkdir(parents=True)
    (v / "2_results/DS/02_b_x").mkdir(parents=True)
    (v / "3_figures/DS/02_b_x").mkdir(parents=True)
    (v / "1_scripts/02_b_x/02b_01_no_access_sweep.py").touch()
    (v / "2_results/DS/02_b_x/02b_01_no_access_sweep.parquet").touch()
    (v / "3_figures/DS/02_b_x/02b_03_no_access_script.py").write_text(
        FIG.format(sid="02b_03", reads=("02b_01",),
                   opens="02b_01_no_access_sweep.parquet"))
    return v


def test_a_correct_declaration_passes(tree) -> None:
    assert check_reads(tree) == []


def test_a_missing_declaration_is_caught(tree) -> None:
    f = tree / "3_figures/DS/02_b_x/02b_03_no_access_script.py"
    f.write_text(f.read_text().replace(
        "READS: tuple[str, ...] = ('02b_01',)\n", ""))
    out = check_reads(tree)
    assert len(out) == 1 and "no READS" in out[0]


def test_declaring_a_step_that_does_not_exist_is_caught(tree) -> None:
    f = tree / "3_figures/DS/02_b_x/02b_03_no_access_script.py"
    f.write_text(f.read_text().replace("'02b_01'", "'02b_99'"))
    out = check_reads(tree)
    assert any("has no script" in o for o in out)


def test_declaring_a_step_with_no_table_is_caught(tree) -> None:
    (tree / "2_results/DS/02_b_x/02b_01_no_access_sweep.parquet").unlink()
    out = check_reads(tree)
    assert any("produced no table" in o for o in out)


def test_a_drifted_declaration_is_caught(tree) -> None:
    """The case this exists for: the code's input changed, the declaration did not.

    A figure silently repointed at another step would otherwise keep passing,
    which is exactly how a stale docstring stops meaning anything.
    """
    (tree / "1_scripts/02_b_x/02b_02_no_access_other.py").touch()
    (tree / "2_results/DS/02_b_x/02b_02_no_access_other.parquet").touch()
    f = tree / "3_figures/DS/02_b_x/02b_03_no_access_script.py"
    f.write_text(f.read_text().replace("02b_01_no_access_sweep.parquet",
                                       "02b_02_no_access_other.parquet"))
    out = check_reads(tree)
    assert any("READS does not declare" in o for o in out)


def test_npz_inputs_count_too(tree) -> None:
    f = tree / "3_figures/DS/02_b_x/02b_03_no_access_script.py"
    f.write_text(f.read_text() + '\nz = np.load("02b_07_no_access_xy.npz")\n')
    out = check_reads(tree)
    assert any("02b_07" in o for o in out)


def test_a_figure_script_is_never_imported(tree) -> None:
    """Importing one would DRAW the figure; the check parses with ast instead."""
    f = tree / "3_figures/DS/02_b_x/02b_03_no_access_script.py"
    f.write_text(f.read_text() + "\nraise SystemExit('must never execute')\n")
    assert check_reads(tree) == []
