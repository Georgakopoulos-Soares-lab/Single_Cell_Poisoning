"""The step-id check must fail on the collision it was written for.

A checker that has never returned non-zero is a checker nobody has tested. The
first case here reconstructs the actual defect: ``02a_02`` existed as
``02a_01``'s figure and a new analysis script was written under the same id.
Both READMEs were correct, ``check_readme`` passed, and one grep would have
returned two unrelated experiments.
"""
import pathlib

import pytest
from check_steps import main, step_id


@pytest.fixture
def tree(tmp_path: pathlib.Path) -> pathlib.Path:
    """A minimal well-formed version directory."""
    v = tmp_path / "v02"
    for d in ("1_scripts/02_a_attack_atlas", "2_results/DS/02_a_attack_atlas",
              "3_figures/DS/02_a_attack_atlas"):
        (v / d).mkdir(parents=True)
    (v / "1_scripts/02_a_attack_atlas/02a_01_no_access_uniform.py").touch()
    (v / "3_figures/DS/02_a_attack_atlas/02a_02_no_access_script.py").touch()
    (v / "3_figures/DS/02_a_attack_atlas/02a_02_no_access.png").touch()
    (v / "2_results/DS/02_a_attack_atlas/02a_01_no_access_uniform.parquet").touch()
    return v


def test_a_clean_tree_passes(tree) -> None:
    assert main(str(tree)) == 0


def test_the_real_collision_is_caught(tree, capsys) -> None:
    """A new analysis script reusing a figure's id -- the defect that happened."""
    (tree / "1_scripts/02_a_attack_atlas/02a_02_with_access_inplace.py").touch()
    assert main(str(tree)) == 1
    err = capsys.readouterr().err
    assert "02a_02" in err
    assert "scripts share this id" in err or "mixed access" in err


def test_two_scripts_under_one_id_are_caught(tree) -> None:
    (tree / "1_scripts/02_a_attack_atlas/02a_01_no_access_other.py").touch()
    assert main(str(tree)) == 1


def test_mixed_access_under_one_id_is_caught(tree) -> None:
    """02b_05_with_access and 02b_05_no_access would be two experiments, one id."""
    (tree / "2_results/DS/02_a_attack_atlas/02a_01_with_access_x.parquet").touch()
    assert main(str(tree)) == 1


def test_a_module_mismatch_is_caught(tree) -> None:
    """A 02b step filed in the 02a directory breaks the module ordering."""
    (tree / "1_scripts/02_a_attack_atlas/02b_09_no_access_stray.py").touch()
    assert main(str(tree)) == 1


def test_an_unparseable_script_in_a_module_dir_is_caught(tree) -> None:
    (tree / "1_scripts/02_a_attack_atlas/helper_utils.py").touch()
    assert main(str(tree)) == 1


def test_readmes_and_caches_are_ignored(tree) -> None:
    (tree / "1_scripts/02_a_attack_atlas/README.txt").touch()
    (tree / "1_scripts/02_a_attack_atlas/__pycache__").mkdir()
    (tree / "1_scripts/02_a_attack_atlas/__pycache__/x.pyc").touch()
    assert main(str(tree)) == 0


def test_step_id_parses_every_access_value() -> None:
    assert step_id("02b_10_both_threshold_dose.py") == ("02b", "10", "both")
    assert step_id("02b_09_no_access_CL0000057.pdf") == ("02b", "09", "no_access")
    assert step_id("02a_03_with_access_relabel.parquet") == ("02a", "03", "with_access")
    assert step_id("schemas.py") is None
