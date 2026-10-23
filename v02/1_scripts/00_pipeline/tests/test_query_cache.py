"""A second dataset must not overwrite the first one's cache in place.

``cache/`` used to hold ``query_emb.npy``, ``query_meta.parquet`` and the two
neighbour lists under fixed names with no dataset anywhere in the path, so
pointing ``config.dataset`` at a second dataset rewrote dataset 1's files --
and under ``--rerun-triggers mtime`` that invalidates every sweep beneath them.
The repository now carries 47 query datasets, so this was about to be exercised.

Two claims are tested and they pull in opposite directions, which is the point:
what is derived from the QUERIES must differ between datasets, and what is
derived from the REFERENCE must not. Sharing the poison sets is not a
concession -- it is what makes two datasets attacked under the same key
comparable on byte-identical poison.
"""

from __future__ import annotations

import pathlib

import pytest

from dataset_config import (QUERY_CACHE_DIRS, QUERY_CACHE_FILES,
                            SHARED_CACHE_ENTRIES, cache_root, query_cache,
                            query_cache_name)

BASE = pathlib.Path(__file__).resolve().parents[3]          # .../v02

DS1 = {"dataset": "GSE136831", "paths": {"cache": "cache"}}
DS2 = {"dataset": "CELLxGENE_abc123", "paths": {"cache": "cache"}}


def v02(tmp_path: pathlib.Path) -> pathlib.Path:
    """A version root under ``tmp_path``.

    ``cache_root`` resolves a relative ``paths.cache`` against ``base.parent``,
    exactly as the project root sits above ``v02/``. Passing ``tmp_path``
    itself would put the cache in pytest's shared parent directory, where two
    tests would then collide.
    """
    return tmp_path / "v02"


# ── query-derived: one directory per dataset ─────────────────────────────────

def test_query_files_sit_under_the_dataset_directory(tmp_path: pathlib.Path) -> None:
    """Every query-derived name resolves below ``<cache root>/<dataset>``."""
    root = cache_root(v02(tmp_path), DS1)
    qc = query_cache(v02(tmp_path), DS1)
    assert qc == root / "GSE136831"
    for name in QUERY_CACHE_FILES + QUERY_CACHE_DIRS:
        assert (qc / name).parent == qc
        assert (qc / name).is_relative_to(root)
        assert (qc / name) != (root / name)


def test_two_datasets_get_different_query_paths(tmp_path: pathlib.Path) -> None:
    """The defect, stated as an assertion: these paths must not collide."""
    one = query_cache(v02(tmp_path), DS1)
    two = query_cache(v02(tmp_path), DS2)
    assert one != two
    for name in QUERY_CACHE_FILES + QUERY_CACHE_DIRS:
        assert one / name != two / name


def test_query_cache_creates_the_directory(tmp_path: pathlib.Path) -> None:
    """A missing parent should surface as a missing FILE, not a missing dir."""
    out = query_cache(v02(tmp_path), DS1)
    assert out.is_dir()


def test_query_cache_can_be_asked_not_to_create(tmp_path: pathlib.Path) -> None:
    out = query_cache(v02(tmp_path), DS1, create=False)
    assert not out.exists()


# ── reference-derived: shared, deliberately ──────────────────────────────────

def test_shared_entries_are_the_same_path_for_two_datasets(
    tmp_path: pathlib.Path,
) -> None:
    """``reference.npz`` and ``poison/`` do not move when the dataset changes.

    Poison is drawn from REFERENCE cells for an (attack, pair, rate, seed), so
    the same key yields the same records whatever is being queried. Two
    datasets attacked identically therefore face byte-identical poison, which
    makes a cross-dataset comparison exact rather than approximately matched.
    """
    for name in SHARED_CACHE_ENTRIES:
        assert cache_root(v02(tmp_path), DS1) / name == cache_root(v02(tmp_path), DS2) / name


def test_shared_entries_are_not_under_any_dataset_directory(
    tmp_path: pathlib.Path,
) -> None:
    root = cache_root(v02(tmp_path), DS1)
    for name in SHARED_CACHE_ENTRIES:
        assert (root / name).parent == root
        assert query_cache_name(DS1) not in (root / name).parts


# ── paths.cache is configuration, and was being ignored ──────────────────────

def test_cache_root_honours_a_relative_paths_cache(tmp_path: pathlib.Path) -> None:
    """Relative is taken against ``base.parent`` -- the project root."""
    assert cache_root(v02(tmp_path), DS1) == tmp_path / "cache"


def test_cache_root_honours_a_non_default_value(tmp_path: pathlib.Path) -> None:
    """The key was declared in config and read by nothing; now it decides."""
    cfg = {"dataset": "GSE136831", "paths": {"cache": "scratch/biorag"}}
    assert cache_root(v02(tmp_path), cfg) == tmp_path / "scratch" / "biorag"
    assert query_cache(v02(tmp_path), cfg) == \
        tmp_path / "scratch" / "biorag" / "GSE136831"


def test_cache_root_honours_an_absolute_paths_cache(tmp_path: pathlib.Path) -> None:
    """An external scratch disk is the reason the key exists."""
    elsewhere = tmp_path / "mnt" / "fast"
    cfg = {"dataset": "GSE136831", "paths": {"cache": str(elsewhere)}}
    assert cache_root(v02(tmp_path), cfg) == elsewhere


def test_a_missing_paths_cache_raises(tmp_path: pathlib.Path) -> None:
    """No default: a silent fallback would scatter scratch across the tree."""
    with pytest.raises(ValueError) as e:
        cache_root(v02(tmp_path), {"dataset": "GSE136831", "paths": {}})
    assert "cache" in str(e.value)


def test_the_live_config_declares_a_cache_root() -> None:
    """The committed config must satisfy the rule this module now enforces."""
    from dataset_config import load
    assert cache_root(BASE, load(BASE)).name == "cache"


# ── a dataset name has to be one directory component ─────────────────────────

@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "../escape"])
def test_an_unusable_dataset_name_raises(name: str) -> None:
    """A separator or a ``..`` would put a dataset's cache outside the root."""
    with pytest.raises(ValueError):
        query_cache_name({"dataset": name})


def test_a_missing_dataset_key_raises() -> None:
    with pytest.raises(ValueError):
        query_cache_name({"paths": {"cache": "cache"}})


# ── the DAG composes the same path, as a relative string ─────────────────────

SNAKEFILE = BASE / "1_scripts" / "00_pipeline" / "Snakefile"


def test_the_snakefile_namespaces_every_query_derived_input() -> None:
    """Snakemake keys its metadata on the path, so the DAG must agree.

    The Snakefile builds cache paths as relative strings rather than as
    absolute paths, so it cannot call :func:`query_cache` directly. It calls
    :func:`query_cache_name` instead and composes one f-string, which this
    pins: no query-derived name may appear directly under ``{CACHE}``.
    """
    text = SNAKEFILE.read_text()
    assert 'QCACHE = f"{CACHE}/{query_cache_name(config)}"' in text
    for name in QUERY_CACHE_FILES + QUERY_CACHE_DIRS:
        assert f"{{CACHE}}/{name}" not in text
    for name in SHARED_CACHE_ENTRIES:
        assert f"{{QCACHE}}/{name}" not in text


def test_the_snakefile_still_shares_reference_and_poison() -> None:
    text = SNAKEFILE.read_text()
    assert 'f"{CACHE}/reference.npz"' in text
    assert 'f"{CACHE}/poison/{CONDITION}.npz"' in text


# ── no script may reach round the helper ─────────────────────────────────────

SCRIPTS = BASE / "1_scripts"


def test_no_script_hardcodes_the_cache_root() -> None:
    """``BASE.parent / "cache"`` was the spelling at all twelve call sites.

    It ignored ``paths.cache`` and it had no dataset in it. Both are fixed by
    going through :func:`cache_root`, so the literal must not come back.
    """
    offenders = [
        p.relative_to(SCRIPTS)
        for p in SCRIPTS.rglob("*.py")
        if "__pycache__" not in p.parts
        and p.name not in ("dataset_config.py", "test_query_cache.py")
        and ('BASE.parent / "cache"' in (t := p.read_text())
             or 'BASE.parent / "cache/' in t
             or 'base.parent / "cache"' in t)
    ]
    assert offenders == []


def test_no_script_reads_a_query_file_from_the_shared_root() -> None:
    """``CACHE / "query_emb.npy"`` is the bug; ``QCACHE / ...`` is the fix."""
    offenders = []
    for p in SCRIPTS.rglob("*.py"):
        if "__pycache__" in p.parts or p.name == "test_query_cache.py":
            continue
        text = p.read_text()
        for name in QUERY_CACHE_FILES:
            if f'CACHE / "{name}"' in text.replace(f'QCACHE / "{name}"', ""):
                offenders.append((p.relative_to(SCRIPTS), name))
    assert offenders == []
