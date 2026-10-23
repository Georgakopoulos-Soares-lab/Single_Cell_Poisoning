"""Resolve ``config/config.yaml`` for the dataset a run is about.

THE DATASET IS THE ONLY DYNAMIC AXIS. The retrieval tool is SCimilarity and
stays SCimilarity, so ``paths.reference_dir``, ``hnsw``, ``ontology``,
``poison``, ``vote``, ``rates`` and ``seeds`` are the same for every dataset and
are read straight off the top level. What a second dataset actually changes is
small and was measured, not guessed: where its queries live, where its results
go, which obs columns carry the label and the donor, and which Cell Ontology
pairs exist in its tissue.

Two file shapes are accepted, because the config is restructured in a separate
commit from this module and the tree must be green in between:

FLAT -- the shape as first written. Dataset-specific values sit at their
top-level positions and :func:`resolve` returns them untouched.

BLOCK -- the shape a second dataset needs::

    dataset: GSE136831
    datasets:
      GSE136831:
        paths: {queries_dir: ..., queries_h5ad: ..., results: ...}
        query: {label_column: ..., donor_column: ...}
        pairs: [...]

:func:`resolve` selects ``datasets[cfg["dataset"]]`` and merges it INTO the flat
positions, per key and recursively into sub-dicts -- so
``CFG["paths"]["queries_h5ad"]``, ``CFG["query"]["donor_column"]`` and
``CFG["pairs"]`` keep
resolving exactly as they do today, and ``paths.reference_dir``, which no
dataset block mentions, survives. The two shapes resolve to byte-identical
config; a golden fixture in ``00_pipeline/tests/fixtures/`` pins that.

Naming a dataset with no block is an ERROR, never a fallback. Falling back to
the top level would point a second dataset's run at the first dataset's queries
and write its tables into the first dataset's results directory -- silently,
because every path still resolves.

This module also owns the CACHE LAYOUT, for the same reason: ``cache/`` is the
other place a second dataset could overwrite the first one in place. See
:func:`cache_root` and :func:`query_cache`, and the block comment above them
for which artefacts are shared and which are not.
"""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import yaml

#: Top-level key naming the dataset under analysis.
DATASET_KEY = "dataset"
#: Top-level key holding one block per dataset. Absent in the flat shape.
DATASETS_KEY = "datasets"


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Merge ``over`` into a copy of ``base``, recursing into sub-dicts.

    A key present in both wins for ``over`` -- the dataset block is the more
    specific statement, so it overrides the top level. Recursion is what keeps
    ``paths.reference_dir`` alive when a block declares only ``paths.results``:
    a wholesale replacement of ``paths`` would drop it.

    Args:
        base: the less specific mapping (the config's top level).
        over: the more specific mapping (the dataset block).

    Returns:
        A new dict; neither argument is mutated.
    """
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve(cfg: dict[str, Any]) -> dict[str, Any]:
    """Collapse a parsed config down to the active dataset's view of it.

    Args:
        cfg: the parsed ``config.yaml``, in either the flat or the block shape.

    Returns:
        A DEEP COPY with every dataset-specific value at its flat top-level
        position and no ``datasets`` key. The flat shape is returned as a copy
        of itself. Nothing in the result aliases ``cfg``, so a caller that
        edits its config cannot reach back into another caller's -- snakemake
        hands the same dict to every job.

    Raises:
        ValueError: if ``datasets`` is present but is not a mapping, if
            ``dataset`` is missing, or if no block is defined for the named
            dataset -- with the defined ones listed.
    """
    if DATASETS_KEY not in cfg:
        return copy.deepcopy(cfg)

    blocks = cfg[DATASETS_KEY]
    if not isinstance(blocks, dict):
        raise ValueError(
            f"config {DATASETS_KEY!r} must be a mapping of dataset name -> "
            f"block, got {type(blocks).__name__}"
        )
    if DATASET_KEY not in cfg:
        raise ValueError(
            f"config declares {DATASETS_KEY!r} but no {DATASET_KEY!r}; "
            f"defined datasets: {sorted(blocks)}"
        )

    name = cfg[DATASET_KEY]
    if name not in blocks:
        raise ValueError(
            f"{DATASET_KEY}: {name!r} has no block under {DATASETS_KEY!r}; "
            f"defined datasets: {sorted(blocks)}"
        )

    block = blocks[name]
    if not isinstance(block, dict):
        raise ValueError(
            f"{DATASETS_KEY}[{name!r}] must be a mapping, got "
            f"{type(block).__name__}"
        )

    top = {k: v for k, v in cfg.items() if k != DATASETS_KEY}
    return copy.deepcopy(_merge(top, block))


def load(base: pathlib.Path) -> dict[str, Any]:
    """Read ``<base>/config/config.yaml`` and resolve it for the active dataset.

    This is the ONE way a script in ``v02/1_scripts/`` obtains its config.
    Reading the YAML directly bypasses :func:`resolve` and so reads a second
    dataset's run as if it were the first.

    Args:
        base: the version root, i.e. the directory holding ``config/`` --
            normally ``pathlib.Path(__file__).resolve().parents[2]``.

    Returns:
        The fully resolved config. See :func:`resolve` for the shape.

    Raises:
        FileNotFoundError: if no ``config/config.yaml`` exists under ``base``.
        ValueError: as :func:`resolve`.
    """
    path = base / "config" / "config.yaml"
    return resolve(yaml.safe_load(path.read_text()))


# ── the cache root, and the part of it that belongs to one dataset ───────────
#
# ``cache/`` holds what is UNPACKED from ``0_data`` or computed once and reused
# by every condition (CLAUDE.md §3). Its contents split cleanly in two, and the
# split is by WHAT THE ARTEFACT IS DERIVED FROM, not by which stage wrote it.
#
# SHARED, at the cache root. ``reference.npz`` is the shipped SCimilarity atlas
# unpacked into arrays, and ``poison/`` holds one set per
# (attack, pair, rate, seed) drawn from those reference cells. Neither looks at
# a query, so neither varies with the query dataset. Sharing ``poison/`` is not
# merely safe, it is a FEATURE: two datasets attacked under the same key get
# BYTE-IDENTICAL poison, which makes a cross-dataset comparison exact rather
# than approximately matched. Do not "fix" this into a per-dataset copy.
#
# PER DATASET, under ``<cache root>/<dataset>/``. Everything below is a function
# of the query cells: their embeddings, their obs, and their retrieval against
# the reference. Before this namespacing these sat at the cache root under fixed
# names, so pointing ``config.dataset`` at a second dataset OVERWROTE the first
# dataset's files in place -- and under ``--rerun-triggers mtime`` that
# invalidates every sweep beneath them, the cascade the Snakefile's CAUTION
# block warns about, reached by a config edit rather than a touch.

#: Query-derived files, relative to :func:`query_cache`.
QUERY_CACHE_FILES = (
    "query_emb.npy",            # stage 1 -- this dataset's cells, encoded
    "query_meta.parquet",       # stage 1 -- this dataset's obs
    "clean_neighbors.npz",      # stage 2 -- these queries x the reference
    "clean_neighbors_deep.npz", # depth-DEEP_K exhaustive, same queries
)

#: Query-derived directories, relative to :func:`query_cache`. Stage 6 scores
#: one condition's merge shortcut against a rebuilt index FOR THESE QUERIES, so
#: its per-condition parquets are as dataset-specific as the neighbours they
#: are compared against.
QUERY_CACHE_DIRS = ("validation",)

#: Entries that stay at the cache root, shared by every dataset. See above.
SHARED_CACHE_ENTRIES = ("reference.npz", "poison")

#: Config key under ``paths`` naming the cache root.
CACHE_KEY = "cache"


def cache_root(base: pathlib.Path, cfg: dict[str, Any]) -> pathlib.Path:
    """Locate the shared cache directory named by ``paths.cache``.

    Every call site used to hardcode ``base.parent / "cache"`` while the config
    key sat unread, so a run could not be pointed at a different scratch disk
    without editing a dozen scripts. The key is authoritative here.

    Args:
        base: the version root, i.e. the directory holding ``config/`` --
            normally ``pathlib.Path(__file__).resolve().parents[2]``.
        cfg: a resolved config, as returned by :func:`resolve` or :func:`load`.

    Returns:
        An absolute path. A relative ``paths.cache`` is taken relative to
        ``base.parent`` -- the project root, which is the working directory
        every run is launched from -- so the default ``cache`` resolves to the
        real directory sitting above ``v01/`` and ``v02/``.

    Raises:
        ValueError: if ``paths.cache`` is missing. There is no default: a
            silent fallback would write a second disk's worth of scratch into
            whichever directory happened to be nearby.
    """
    paths = cfg.get("paths")
    if not isinstance(paths, dict) or CACHE_KEY not in paths:
        raise ValueError(
            f"config has no paths.{CACHE_KEY}; the cache root is configuration, "
            f"not a constant, and there is no default"
        )
    raw = pathlib.Path(str(paths[CACHE_KEY]))
    return raw if raw.is_absolute() else base.parent / raw


def query_cache_name(cfg: dict[str, Any]) -> str:
    """Return the ONE path component that namespaces query-derived cache.

    Split out from :func:`query_cache` so that snakemake, which composes cache
    paths as relative STRINGS in the DAG rather than as absolute paths, states
    the layout by calling this rather than by restating ``f"{CACHE}/{name}"``
    itself. One rule, two spellings of the path.

    Args:
        cfg: a resolved config.

    Returns:
        The dataset name.

    Raises:
        ValueError: if ``dataset`` is missing, empty, or is not usable as a
            single directory name -- a name containing a separator or a
            ``..`` would place a dataset's cache outside the cache root.
    """
    if DATASET_KEY not in cfg:
        raise ValueError(f"config has no {DATASET_KEY!r}")
    name = str(cfg[DATASET_KEY])
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(
            f"{DATASET_KEY}: {name!r} is not usable as a directory name; the "
            f"dataset names one component under the cache root"
        )
    return name


def query_cache(
    base: pathlib.Path, cfg: dict[str, Any], *, create: bool = True
) -> pathlib.Path:
    """Locate this dataset's private corner of the shared cache.

    Args:
        base: the version root, as :func:`cache_root`.
        cfg: a resolved config.
        create: create the directory (and the cache root) if absent. On by
            default because every caller either writes into it or reads a file
            a previous stage wrote into it, and a missing directory should
            surface as a missing FILE with a name in it, not as a missing
            parent.

    Returns:
        ``<cache root>/<dataset>``, absolute.

    Raises:
        ValueError: as :func:`cache_root` and :func:`query_cache_name`.
    """
    out = cache_root(base, cfg) / query_cache_name(cfg)
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out
