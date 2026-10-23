"""Stage 0 — prepare the reference: map labels to ontology ids, subsample, verify.

The shipped reference labels cells by NAME; queries and attack pairs use Cell
Ontology identifiers. This stage builds that bridge once and asserts the
assumptions every later stage relies on, rather than letting them fail quietly
1,260 conditions later.
"""
# 1_scripts cannot be a package (leading digit), so the shared library in
# 00_common is put on the path and imported flat.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_common"))

# No `from __future__ import annotations` here: snakemake's script directive
# prepends a preamble, which would push it off line 1 and make it a SyntaxError.
# Python 3.12 evaluates the annotations used below natively.

import numpy as np

from schemas import EMB_DTYPE
from ontology import Ontology
from retrieve_clean import load_reference, subsample_reference

UNMAPPED = "UNMAPPED"
"""Placeholder for reference labels with no ontology id. Kept in the reference
rather than dropped: those cells are still legitimate neighbours that can
out-vote poison. They simply can never be an attack source or target."""


def map_labels(names: np.ndarray, onto: Ontology) -> tuple[np.ndarray, dict[str, int]]:
    """Convert cell-type names to CL ids.

    Args:
        names: (n,) str, reference labels as names.
        onto: parsed Cell Ontology.

    Returns:
        ids: (n,) str, CL id or :data:`UNMAPPED`.
        report: unmapped name -> cell count, for the caller to check.
    """
    cache = {nm: (onto.to_id(nm) or UNMAPPED) for nm in set(names.tolist())}
    ids = np.array([cache[nm] for nm in names])
    unmapped: dict[str, int] = {}
    for nm in set(names.tolist()):
        if cache[nm] == UNMAPPED:
            unmapped[nm] = int((names == nm).sum())
    return ids, unmapped


def check_pairs_viable(
    labels: np.ndarray,
    pairs: list[dict],
    max_rate: float,
    min_target: int = 200,
    min_source: int = 1000,
) -> None:
    """Assert every configured pair can actually be built from this reference.

    Args:
        labels: (n,) str, reference labels as CL ids.
        pairs: config entries with ``source``, ``target``, ``name``.
        max_rate: largest rate in the grid, as a fraction of the source class.
        min_target: minimum target cells for a stable interpolation centroid.
        min_source: minimum source cells, so the top rate yields a poison set
            large enough to mean anything.

    Raises:
        ValueError: any source or target is too thin.

    Note:
        Under the source-class rate basis a source can always supply
        ``rate * n_source`` distinct cells, so the old "can it reach the top
        rate" check is satisfied by construction. What can still go wrong is a
        class being small in absolute terms: 25% of 40 cells is 10 poisoned
        records, which is a rounding error rather than an attack.
    """
    # Combos declare `members` naming other pairs rather than a source/target of
    # their own; their constituent terms are validated as those singles.
    singles = [p for p in pairs if "members" not in p]
    known = {p["name"] for p in singles}
    problems = []
    for p in pairs:
        if "members" in p:
            missing = [m for m in p["members"] if m not in known]
            if missing:
                problems.append(f"{p['name']}: unknown member pair(s) {missing}")
            elif len(p["members"]) < 2:
                problems.append(f"{p['name']}: a combo needs at least two members")

    counts = {
        t: int((labels == t).sum()) for p in singles for t in (p["source"], p["target"])
    }
    for p in singles:
        n_src = counts[p["source"]]
        if n_src < min_source:
            problems.append(
                f"{p['name']}: source {p['source']} has only {n_src:,} cells; "
                f"the top rate would build {int(round(max_rate * n_src)):,} poison records"
            )
        if counts[p["target"]] < min_target:
            problems.append(
                f"{p['name']}: target {p['target']} has only {counts[p['target']]:,} cells"
            )
    if problems:
        raise ValueError("unviable pairs:\n  " + "\n  ".join(problems))


# ── snakemake entry point ────────────────────────────────────────────────────
if "snakemake" in globals():  # pragma: no cover - exercised by the workflow
    import sys

    cfg = snakemake.config  # noqa: F821
    onto = Ontology.from_obo(snakemake.input.obo)  # noqa: F821

    # Snakemake stringifies values passed through a profile's `config:` block,
    # so YAML `null` arrives as the string "None", which is truthy. Treat every
    # spelling of "no subsample" the same.
    rows = cfg.get("reference_rows")
    if str(rows).lower() in ("none", "null", "", "0"):
        rows = None
    if rows:
        # Deterministic subsample; the seed is explicit so it is re-derivable.
        import hnswlib

        from schemas import EMB_DIM

        probe = hnswlib.Index(space="cosine", dim=EMB_DIM)
        probe.load_index(f"{snakemake.input.model}/annotation/labelled_kNN.bin")  # noqa: F821
        n_total = probe.get_current_count()
        del probe
        idx = subsample_reference(n_total, int(rows), int(cfg["reference_subsample_seed"]))
    else:
        idx = None

    emb, names = load_reference(snakemake.input.model, indices=idx)  # noqa: F821
    labels, unmapped = map_labels(names, onto)

    n_mapped = int((labels != UNMAPPED).sum())
    print(f"reference rows      : {len(emb):,}", file=sys.stderr)
    print(f"labels mapped to CL : {n_mapped:,} ({n_mapped/len(labels)*100:.1f}%)", file=sys.stderr)
    for nm, n in sorted(unmapped.items(), key=lambda x: -x[1]):
        print(f"  UNMAPPED {n:>9,}  {nm}", file=sys.stderr)

    check_pairs_viable(
        labels,
        cfg["pairs"],
        max_rate=max(cfg["rates"]),
        min_target=int(cfg["poison"]["min_target_cells"]),
    )

    cell_id = (idx if idx is not None else np.arange(len(emb))).astype(np.int64)
    np.savez(
        snakemake.output[0],  # noqa: F821
        emb=emb.astype(EMB_DTYPE),
        label=labels,
        cell_id=np.array([f"ref{i}" for i in cell_id]),
    )
