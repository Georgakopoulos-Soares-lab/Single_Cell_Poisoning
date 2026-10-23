"""A figure must not draw the shortcut arm of a step that was also run on graph.

``02b_11`` and ``02b_15`` each wrote two tables whose names differ by one
suffix: ``_graph`` is ``modify.graph_topk`` on the shipped index, the plain name
is ``modify.remerge`` over the exhaustive list. They disagree by more than 0.10
on 3.2% of conditions in both directions (``02b_16``), so opening the wrong one
publishes a number from a retriever the paper does not claim -- and both files
load, with every column a figure wants. Nothing enforced the choice; this does.

The escape hatch is asserting on ``retrieval_path``, which is what HANDOFF.md §5
asks of a figure anyway: the filename is not evidence, the column is.
"""
import pathlib

import pytest
from check_retrieval import check, main, paired

#: A figure that opens one table by name, with an optional assert.
FIG = '''"""Step 02b_98 — a figure."""
NAME = "02b_98_with_access"

READS: tuple[str, ...] = ("02b_11",)

def main() -> None:
    d = pd.read_parquet(TAB / "{opens}")
{assertion}
    return d
'''

ASSERT = '    assert (d.retrieval_path == "graph").all(), "not the graph run"'
ASSERT_SUBSCRIPT = '    assert (d["retrieval_path"] == "exact").all(), "on purpose"'


def write_fig(tree: pathlib.Path, opens: str, assertion: str = "") -> pathlib.Path:
    f = tree / "3_figures/DS/02_b_x/02b_98_with_access_script.py"
    f.write_text(FIG.format(opens=opens, assertion=assertion))
    return f


@pytest.fixture
def tree(tmp_path: pathlib.Path) -> pathlib.Path:
    """A version dir with one doubly-run step, one single-run step, one figure.

    ``02b_11_with_access_inplace`` exists in both regimes, exactly as the real
    tree has it. ``02b_12_both_pooled_summary`` is derived and single. The
    figure opens the graph variant, which is the state that must pass.
    """
    v = tmp_path / "v02"
    (v / "2_results/DS/02_b_x").mkdir(parents=True)
    (v / "3_figures/DS/02_b_x").mkdir(parents=True)
    for name in ("02b_11_with_access_inplace.parquet",
                 "02b_11_with_access_inplace_graph.parquet",
                 "02b_12_both_pooled_summary.parquet"):
        (v / "2_results/DS/02_b_x" / name).touch()
    write_fig(v, "02b_11_with_access_inplace_graph.parquet")
    return v


def test_the_pair_is_found(tree) -> None:
    assert paired(tree) == {
        "02b_11_with_access_inplace.parquet":
            "02b_11_with_access_inplace_graph.parquet"}


def test_reading_the_graph_variant_passes(tree) -> None:
    assert check(tree)[1] == []
    assert main([str(tree)]) == 0


def test_reading_the_plain_variant_is_caught(tree, capsys) -> None:
    """The defect the check exists for: a new figure opens the shortcut arm."""
    write_fig(tree, "02b_11_with_access_inplace.parquet")
    assert main([str(tree)]) == 1
    err = capsys.readouterr().err
    assert "02b_11_with_access_inplace.parquet" in err
    assert "02b_11_with_access_inplace_graph.parquet" in err
    assert "retrieval_path" in err


def test_the_plain_variant_with_an_assert_passes(tree) -> None:
    """Asserting makes the choice deliberate, which is all the rule asks."""
    write_fig(tree, "02b_11_with_access_inplace.parquet", ASSERT)
    assert check(tree)[1] == []


def test_a_subscript_assert_also_counts(tree) -> None:
    """``d["retrieval_path"]`` is the same decision as ``d.retrieval_path``."""
    write_fig(tree, "02b_11_with_access_inplace.parquet", ASSERT_SUBSCRIPT)
    assert check(tree)[1] == []


def test_a_step_with_one_variant_is_unaffected(tree) -> None:
    """Most steps ran once; reading them must stay silent."""
    write_fig(tree, "02b_12_both_pooled_summary.parquet")
    assert check(tree)[1] == []


def test_the_pair_must_exist_for_the_rule_to_bite(tree) -> None:
    """Delete the graph twin and the plain name is the only table there is."""
    (tree / "2_results/DS/02_b_x/02b_11_with_access_inplace_graph.parquet").unlink()
    write_fig(tree, "02b_11_with_access_inplace.parquet")
    assert check(tree)[1] == []


def test_a_name_built_at_runtime_cannot_escape(tree) -> None:
    """An f-string that CAN resolve to the shortcut arm is flagged like a literal."""
    f = tree / "3_figures/DS/02_b_x/02b_98_with_access_script.py"
    f.write_text('SUF = ""\n'
                 'd = pd.read_parquet(TAB / f"02b_11_with_access_inplace{SUF}.parquet")\n')
    out = check(tree)[1]
    assert len(out) == 1 and "can resolve to" in out[0]


def test_a_pair_in_one_dataset_binds_every_dataset(tree) -> None:
    """The stem carries the step id, and check_steps makes that id tree-wide.

    A dataset whose graph rerun has not landed still must not publish the
    shortcut arm, so the rule is not keyed on the dataset directory.
    """
    (tree / "2_results/DS2/02_b_x").mkdir(parents=True)
    (tree / "2_results/DS2/02_b_x/02b_11_with_access_inplace.parquet").touch()
    (tree / "3_figures/DS2/02_b_x").mkdir(parents=True)
    (tree / "3_figures/DS2/02_b_x/02b_99_with_access_script.py").write_text(
        'd = pd.read_parquet(TAB / "02b_11_with_access_inplace.parquet")\n')
    out = check(tree)[1]
    assert len(out) == 1 and "DS2" in out[0]


def test_the_real_tree_passes() -> None:
    """The repository as it stands: every figure already opens the right arm."""
    root = pathlib.Path(__file__).resolve().parents[3]
    if not (root / "3_figures").is_dir():
        pytest.skip("no figure tree in this checkout")
    assert check(root)[1] == []
