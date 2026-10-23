"""Verify every downloaded query dataset and inventory its cell types.

Scope is all of ``0_data/queries``, not only the CELLxGENE collection: one index
covering every query dataset is what lets GSE136831 be compared against the rest
on equal terms, rather than being a special case described nowhere.

The downloader checks byte counts, which proves a transfer completed but not that
the HDF5 opens or that the fields this project needs are present. A dataset that
passes the size check and then fails to open would otherwise surface much later,
inside a sweep, as an opaque crash.

Reads headers only -- ``anndata.read_h5ad(backed="r")`` does not pull the
expression matrix into memory, so this stays cheap enough to run on all 47
datasets beside a live analysis job.

Layout: one directory per dataset, named by its id, holding its ``.h5ad``. This
mirrors ``2_results/<DATASET>/`` so a dataset is a self-contained unit at every
level of the tree, and gives each one somewhere to carry its own provenance.

Writes two collection-level tables beside the dataset directories:

``INVENTORY.tsv``
    One row per dataset: cell count, donor count, and how many cells carry a
    usable Cell Ontology annotation. That last column decides whether a dataset
    can score a label-transfer attack at all -- free-text or absent labels
    cannot, no matter how many cells there are.

``COMPOSITION.tsv``
    Long format, one row per (dataset, cell type). Which classes a dataset holds
    -- not merely how many -- is what decides whether a cell type has enough
    cross-dataset support to carry a claim.

Both are single files rather than one per dataset because every question asked of
them crosses datasets, and because ``dataset_id`` as a column is precisely the
long format the project requires for storage.
"""

import csv
import pathlib
import sys
import traceback

import anndata as ad
import pandas as pd

BASE = pathlib.Path(__file__).resolve().parents[2]
QUERIES = BASE / "0_data/queries"
OUT_INV = QUERIES / "INVENTORY.tsv"
OUT_COMP = QUERIES / "COMPOSITION.tsv"

# Datasets do not share an obs schema. CELLxGENE's curated names come first;
# GSE136831 predates that curation and uses its own. Tried in order, and the
# column actually used is recorded per dataset in INVENTORY.tsv so the choice is
# visible rather than guessed at.
CT_KEYS = ("cell_type_ontology_term_id", "celltype_id", "cell_type")
DONOR_KEYS = ("donor_id", "sample", "Subject_Identity")
UNSPECIFIC = {"CL:0000003"}                  # 'native cell' -- true but useless


def first_present(obs, keys: tuple[str, ...]) -> str | None:
    """Return the first of ``keys`` that is a column of ``obs``, else None."""
    return next((k for k in keys if k in obs.columns), None)


def discover() -> list[tuple[str, str, pathlib.Path]]:
    """Find every dataset directory under ``0_data/queries``.

    Returns:
        (source, dataset_id, path) per dataset, where source is the collection
        the dataset came from (e.g. ``cellxgene``) or ``local`` for one sitting
        directly under queries/.
    """
    found = []
    for h5 in sorted(QUERIES.glob("*/*.h5ad")):
        found.append(("local", h5.parent.name, h5))
    for h5 in sorted(QUERIES.glob("*/*/*.h5ad")):
        found.append((h5.parent.parent.name, h5.parent.name, h5))
    return found


def manifest_sizes() -> dict[str, int]:
    """Expected byte size per dataset id, for collections that ship a manifest."""
    sizes: dict[str, int] = {}
    for man in QUERIES.glob("*/MANIFEST.tsv"):
        for r in csv.DictReader(open(man), delimiter="\t"):
            sizes[r["dataset_id"]] = int(r["bytes"])
    return sizes


def inspect(source: str, ds_id: str, path: pathlib.Path) -> dict:
    """Open one .h5ad backed and summarise the fields the sweeps require."""
    row: dict = {"source": source, "dataset_id": ds_id,
                 "gb": round(path.stat().st_size / 1e9, 3)}
    try:
        a = ad.read_h5ad(path, backed="r")
    except Exception as e:                      # noqa: BLE001 - recorded per file
        row["status"] = f"UNREADABLE {type(e).__name__}: {e}"
        return row
    obs = a.obs
    row["n_cells"] = int(a.n_obs)
    row["n_genes"] = int(a.n_vars)

    ct_key = first_present(obs, CT_KEYS)
    donor_key = first_present(obs, DONOR_KEYS)
    row["ct_column"] = ct_key or ""
    row["donor_column"] = donor_key or ""

    if ct_key:
        ct = obs[ct_key].astype(str)
        # Only ontology ids can be joined to cl.obo; free text cannot.
        good = ct.str.startswith("CL:") & ~ct.isin(UNSPECIFIC)
        row["n_ct_labelled"] = int(good.sum())
        row["n_ct_classes"] = int(ct[good].nunique())
        row["_composition"] = [
            {"source": source, "dataset_id": ds_id,
             "cell_type_ontology_term_id": k, "n_cells": int(v),
             "n_donors": int(obs.loc[ct.eq(k), donor_key].nunique())
                         if donor_key else 0}
            for k, v in ct[good].value_counts().items()
        ]
    else:
        row["n_ct_labelled"] = 0
        row["n_ct_classes"] = 0
    row["n_donors"] = int(obs[donor_key].nunique()) if donor_key else 0
    row["status"] = "ok"
    a.file.close()
    return row


def main() -> None:
    """Inspect every complete dataset and write the collection tables."""
    sizes = manifest_sizes()
    datasets = discover()
    # A download still in flight is a valid partial file, not a corrupt one;
    # skipping it until it is whole is what makes this safe to run beside a fetch.
    ready = [d for d in datasets
             if d[1] not in sizes or d[2].stat().st_size == sizes[d[1]]]
    print(f"inspecting {len(ready)} complete of {len(datasets)} datasets",
          flush=True)

    rows = []
    for i, (source, ds_id, path) in enumerate(ready, 1):
        try:
            row = inspect(source, ds_id, path)
        except Exception:                       # noqa: BLE001
            traceback.print_exc()
            row = {"source": source, "dataset_id": ds_id, "status": "CRASH"}
        rows.append(row)
        print(f"  [{i}/{len(ready)}] {source[:9]:<9} {ds_id[:8]}  "
              f"{row.get('n_cells', 0):>9,} cells  "
              f"{row.get('n_ct_labelled', 0):>9,} CL  "
              f"{row.get('n_ct_classes', 0):>3} classes  "
              f"{row.get('n_donors', 0):>4} donors  {row['status']}", flush=True)

    comp = [c for r in rows for c in r.pop("_composition", [])]
    pd.DataFrame(rows).to_csv(OUT_INV, sep="\t", index=False)
    cdf = pd.DataFrame(comp)
    cdf.to_csv(OUT_COMP, sep="\t", index=False)

    df = pd.DataFrame(rows)
    print(f"\nwrote {OUT_INV.relative_to(BASE)}  ({len(df)} rows)")
    print(f"wrote {OUT_COMP.relative_to(BASE)}  ({len(cdf)} rows)")
    if len(cdf):
        shared = (cdf.groupby("cell_type_ontology_term_id")
                     .agg(n_datasets=("dataset_id", "nunique"),
                          n_cells=("n_cells", "sum"),
                          n_donors=("n_donors", "sum"))
                     .sort_values(["n_datasets", "n_cells"], ascending=False))
        print(f"{len(shared)} distinct cell types; "
              f"{(shared.n_datasets >= 3).sum()} in >=3 datasets")
    bad = df[df["status"] != "ok"]
    if len(bad):
        print(bad[["dataset_id", "status"]].to_string(index=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
