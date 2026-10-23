"""The config a run resolves must not move when the config FILE is restructured.

The golden fixture is today's resolved config, serialised the one way this
module serialises. It is the proof that dataset 1's numbers are unchanged by
the second-dataset work: restructure ``config.yaml`` into per-dataset blocks
and this test still passes, or the restructure moved something it should not
have. Do not regenerate the fixture to make it pass.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import yaml

from dataset_config import load, resolve

BASE = pathlib.Path(__file__).resolve().parents[3]          # .../v02
FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures" / \
    "resolved_config_GSE136831.json"


def dump(cfg: dict) -> str:
    """Serialise a resolved config the one way the fixture was written."""
    return json.dumps(cfg, sort_keys=True, indent=2, default=str)


def test_resolved_config_matches_golden_fixture() -> None:
    """``load(v02)`` reproduces the committed snapshot byte for byte."""
    assert dump(load(BASE)) == FIXTURE.read_text().rstrip("\n")


def test_golden_fixture_is_the_active_dataset() -> None:
    """The fixture pins GSE136831, so a dataset switch fails loudly here."""
    assert json.loads(FIXTURE.read_text())["dataset"] == "GSE136831"


# ── the two shapes ───────────────────────────────────────────────────────────

FLAT = {
    "dataset": "DS1",
    "paths": {"reference_dir": "0_data/reference/model_v1.1",
              "queries_h5ad": "0_data/queries/DS1/sub.h5ad",
              "results": "v02/2_results/DS1"},
    "query": {"label_column": "celltype_id", "donor_column": "sample"},
    "pairs": [{"name": "a_to_b", "source": "CL:0000235"}],
    "vote": {"tie_break": "lowest_cl_id"},
}

BLOCK = {
    "dataset": "DS1",
    "paths": {"reference_dir": "0_data/reference/model_v1.1"},
    "vote": {"tie_break": "lowest_cl_id"},
    "datasets": {
        "DS1": {
            "paths": {"queries_h5ad": "0_data/queries/DS1/sub.h5ad",
                      "results": "v02/2_results/DS1"},
            "query": {"label_column": "celltype_id", "donor_column": "sample"},
            "pairs": [{"name": "a_to_b", "source": "CL:0000235"}],
        },
        "DS2": {
            "paths": {"queries_h5ad": "0_data/queries/DS2/sub.h5ad",
                      "results": "v02/2_results/DS2"},
            "query": {"label_column": "cell_type_ontology_term_id",
                      "donor_column": "donor_id"},
            "pairs": [{"name": "c_to_d", "source": "CL:0000236"}],
        },
    },
}


def write(tmp_path: pathlib.Path, cfg: dict) -> pathlib.Path:
    """Write ``cfg`` where :func:`load` looks for it; return the version root."""
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "config.yaml").write_text(yaml.safe_dump(cfg))
    return tmp_path


def test_block_shape_resolves_to_the_flat_shape(tmp_path: pathlib.Path) -> None:
    """The restructure is a no-op on what a reader sees."""
    flat = load(write(tmp_path / "flat", FLAT))
    block = load(write(tmp_path / "block", BLOCK))
    assert dump(block) == dump(flat)


def test_reference_dir_survives_a_paths_block() -> None:
    """A block declaring some of ``paths`` must not drop the rest of it.

    The tool is fixed, so ``reference_dir`` is stated once at the top level and
    no dataset block repeats it. A wholesale replacement of ``paths`` would
    lose it and stage 0 would fail on a path that is in the file.
    """
    out = resolve(BLOCK)
    assert out["paths"]["reference_dir"] == "0_data/reference/model_v1.1"
    assert out["paths"]["results"] == "v02/2_results/DS1"


def test_the_other_datasets_block_does_not_leak() -> None:
    """Only the active dataset's block is merged, and ``datasets`` is dropped."""
    out = resolve(BLOCK)
    assert "datasets" not in out
    assert out["paths"]["queries_h5ad"] == "0_data/queries/DS1/sub.h5ad"
    assert out["query"]["donor_column"] == "sample"
    assert [p["name"] for p in out["pairs"]] == ["a_to_b"]


def test_selecting_the_second_dataset() -> None:
    """Same file, different ``dataset:``, entirely different queries."""
    out = resolve({**BLOCK, "dataset": "DS2"})
    assert out["paths"]["results"] == "v02/2_results/DS2"
    assert out["query"]["label_column"] == "cell_type_ontology_term_id"
    assert [p["name"] for p in out["pairs"]] == ["c_to_d"]
    assert out["paths"]["reference_dir"] == "0_data/reference/model_v1.1"


def test_dataset_block_beats_the_top_level() -> None:
    """Where both state a key, the block is the more specific statement."""
    cfg = {"dataset": "DS1",
           "query": {"label_column": "celltype_id", "donor_column": "sample"},
           "datasets": {"DS1": {"query": {"donor_column": "donor_id"}}}}
    out = resolve(cfg)
    assert out["query"]["donor_column"] == "donor_id"
    assert out["query"]["label_column"] == "celltype_id"


# ── the failure that this exists to prevent ──────────────────────────────────

def test_missing_block_raises_and_names_what_is_defined() -> None:
    """No silent fallback to another dataset's queries."""
    with pytest.raises(ValueError) as e:
        resolve({**BLOCK, "dataset": "DS3"})
    msg = str(e.value)
    assert "DS3" in msg and "DS1" in msg and "DS2" in msg


def test_datasets_without_a_dataset_key_raises() -> None:
    cfg = {k: v for k, v in BLOCK.items() if k != "dataset"}
    with pytest.raises(ValueError) as e:
        resolve(cfg)
    assert "DS1" in str(e.value) and "DS2" in str(e.value)


def test_a_non_mapping_datasets_raises() -> None:
    with pytest.raises(ValueError):
        resolve({"dataset": "DS1", "datasets": ["DS1", "DS2"]})


def test_a_non_mapping_block_raises() -> None:
    with pytest.raises(ValueError):
        resolve({"dataset": "DS1", "datasets": {"DS1": "v02/2_results/DS1"}})


def test_resolve_does_not_mutate_its_input() -> None:
    """Callers keep the parsed file; snakemake shares one config dict."""
    before = dump(BLOCK)
    resolve(BLOCK)
    assert dump(BLOCK) == before


def test_flat_shape_is_copied_not_aliased() -> None:
    out = resolve(FLAT)
    out["paths"]["results"] = "elsewhere"
    assert FLAT["paths"]["results"] == "v02/2_results/DS1"


# ── the DAG is a reader too ──────────────────────────────────────────────────

SNAKEFILE = BASE / "1_scripts" / "00_pipeline" / "Snakefile"


def test_snakefile_resolves_its_config() -> None:
    """``configfile:`` is a plain YAML load and knows nothing about blocks.

    Every stage receives this one dict as ``snakemake.config``, so if the DAG
    stops resolving it, stage 0 through stage 6 read dataset 1's queries and
    write into dataset 1's results directory whatever ``dataset:`` says.
    """
    text = SNAKEFILE.read_text()
    assert "from dataset_config import" in text
    assert "resolve(dict(config))" in text


def test_the_snakefile_recipe_updates_a_config_in_place() -> None:
    """The three statements the Snakefile runs, on a dict snakemake would hand it."""
    config = {k: v for k, v in BLOCK.items()}
    resolved = resolve(dict(config))
    config.pop("datasets", None)
    config.update(resolved)
    assert "datasets" not in config
    assert config["paths"]["results"] == "v02/2_results/DS1"
    assert config["paths"]["reference_dir"] == "0_data/reference/model_v1.1"
    assert [p["name"] for p in config["pairs"]] == ["a_to_b"]
