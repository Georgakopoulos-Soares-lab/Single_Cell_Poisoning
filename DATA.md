# Inputs

The inputs are not in this repository. Put each file at the path below,
relative to the repository root (the paths are from `v02/config/config.yaml`
lines 25-28, 135 and 212), then check its md5.

| File | Put it at | md5 | Bytes | Source |
|---|---|---|---|---|
| `model_v1.1.tar.gz` (SCimilarity model and reference) | `0_data/reference/model_v1.1.tar.gz`, unpacked into `0_data/reference/model_v1.1/` | `546251b7c435f3b1dbe38e2e420ad57f` | 30,310,810,843 | SCimilarity's Zenodo record, recorded as 10685499 in the project notes |
| `GSE136831_subsample.h5ad` (query cells) | `0_data/queries/GSE136831/GSE136831_subsample.h5ad` | `9589d3368c2dbb287b94e2915ad7de31` | 2,201,750,174 | the same Zenodo record (`v02/config/checksums.txt` calls both files "Zenodo source artifacts") |
| `cl.obo` (Cell Ontology) | `0_data/ontology/cl.obo` | `79fcc8bc4dfa70e5de6d3912bcba1f95` | 17,353,259 | Cell Ontology release 2026-06-08 (`0_data/CHECKSUMS.txt`) |

**Not verified:** the Zenodo record number and the Cell Ontology release
download address have not been checked against the live sites. Confirm that
the md5 values match before you run anything.

**Checksums:** `0_data/CHECKSUMS.txt` lists all three files.
`v02/config/checksums.txt` lists only the first two: **`cl.obo` is missing
from it.**

On macOS:

    md5 0_data/reference/model_v1.1.tar.gz

On Linux:

    md5sum 0_data/reference/model_v1.1.tar.gz

**Disk:** about 30.3 GB for the tarball plus its unpacked contents, 2.2 GB
for the queries, and several GB for `cache/`. The steps build `cache/` the
first time they run (see README, "Getting started").
