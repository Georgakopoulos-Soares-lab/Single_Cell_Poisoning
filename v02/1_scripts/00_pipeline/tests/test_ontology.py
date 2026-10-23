"""Tests for Cell Ontology parsing and tree distance."""

from __future__ import annotations
# 1_scripts cannot be a package (leading digit), so the shared library in
# 00_common is put on the path and imported flat.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_common"))


import pytest

from ontology import Ontology

# A miniature .obo. CL:0000002 carries the trailing edge modifier that
# cl.obo really uses and that a naive parser turns into a phantom node.
MINI = """format-version: 1.2

[Term]
id: CL:0000000
name: cell

[Term]
id: CL:0000001
name: lymphocyte
is_a: CL:0000000 ! cell

[Term]
id: CL:0000002
name: T cell
is_a: CL:0000001 {is_inferred="true"} ! lymphocyte

[Term]
id: CL:0000003
name: B cell
is_a: CL:0000001 ! lymphocyte
synonym: "B lymphocyte" EXACT []

[Term]
id: CL:0000009
name: ancient cell
is_a: CL:0000000 ! cell
is_obsolete: true
"""


@pytest.fixture
def onto(tmp_path):
    p = tmp_path / "mini.obo"
    p.write_text(MINI)
    return Ontology.from_obo(p)


def test_primary_names_resolve(onto) -> None:
    assert onto.to_id("T cell") == "CL:0000002"


def test_lookup_is_case_insensitive(onto) -> None:
    assert onto.to_id("t CELL") == "CL:0000002"


def test_exact_synonyms_resolve(onto) -> None:
    assert onto.to_id("B lymphocyte") == "CL:0000003"


def test_obsolete_terms_are_excluded(onto) -> None:
    assert onto.to_id("ancient cell") is None


def test_unknown_names_return_none(onto) -> None:
    assert onto.to_id("not a cell type") is None


def test_edge_modifiers_do_not_create_phantom_nodes(onto) -> None:
    # REGRESSION. cl.obo writes `is_a: CL:0000001 {is_inferred="true"} ! ...`.
    # Splitting only on "!" leaves the modifier attached, so CL:0000001 and
    # 'CL:0000001 {is_inferred="true"}' become different nodes. Every distance
    # through such an edge is then inflated, and paths detour through nonsense.
    # This was real: natural killer -> CD4 T cell measured 11 instead of 7.
    assert onto.distance("CL:0000002", "CL:0000001") == 1
    assert onto.distance("CL:0000002", "CL:0000003") == 2


def test_distance_to_self_is_zero(onto) -> None:
    assert onto.distance("CL:0000002", "CL:0000002") == 0


def test_distance_is_symmetric(onto) -> None:
    assert onto.distance("CL:0000002", "CL:0000000") == onto.distance(
        "CL:0000000", "CL:0000002"
    )


def test_unreachable_terms_return_none(onto) -> None:
    assert onto.distance("CL:0000002", "CL:9999999") is None


def test_ancestor_relationship_is_detected(onto) -> None:
    # Calling a T cell a "lymphocyte" is a vaguer truth, not a lie, so such
    # pairs must be excludable.
    assert onto.is_ancestor_or_descendant("CL:0000002", "CL:0000001")
    assert not onto.is_ancestor_or_descendant("CL:0000002", "CL:0000003")
