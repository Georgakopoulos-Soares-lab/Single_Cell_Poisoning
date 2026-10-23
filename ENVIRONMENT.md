# Environment

The environment files are in `v02/1_scripts/00_pipeline/envs/`:
- `base.yaml`: conda-forge and bioconda, pinned to minor versions:
  - Python 3.12, numpy 2.1, pandas 2.2, pyarrow 17.0, scipy 1.14,
    hnswlib 0.8.0, scanpy 1.10.3, anndata 0.10.9, pytorch 2.5, plotnine
    0.13, matplotlib 3.9, snakemake 8.25, pytest 8.3, ruff 0.7, `zarr<3`;
  - then pip: `scimilarity==0.4.0`.
- `base.lock`: an exact conda lock for **macOS arm64 only**. It has no pip
  packages.

## The three packages `base.yaml` omits

The code imports these directly:
- `pyyaml` (`00_common/dataset_config.py` and the figure scripts);
- `requests` (`00_pipeline/fetch_cellxgene.py`);
- `umap-learn` (`build_atlas_umap.py`, 02b_06, 02b_08, 02b_30, 03a_06).

On the original laptop they arrived as dependencies of other packages.
Without them, any script that imports one stops with an import error.
`dataset_config.py` imports `yaml`, so this includes every script that
reads the config through it.
The laptop's versions were pyyaml 6.0.3, requests 2.34.2 and umap-learn
0.5.12.

## Commands (not yet tested on another machine)

macOS arm64, exact versions:

    conda create -n biorag --file v02/1_scripts/00_pipeline/envs/base.lock
    conda activate biorag
    pip install scimilarity==0.4.0
    conda install -c conda-forge pyyaml=6.0.3 requests=2.34.2 umap-learn=0.5.12

Linux, or any other platform (minor-version pins only):

    conda env create -f v02/1_scripts/00_pipeline/envs/base.yaml
    conda activate biorag
    conda install -c conda-forge pyyaml=6.0.3 requests=2.34.2 umap-learn=0.5.12

**No Linux lock exists yet.** On Linux, the solver may pick newer patch
releases than the laptop used, and float results may differ in the last
digits. This is untested.

## Check the install

    python -m pytest v02/1_scripts/00_pipeline/tests -q

On the laptop this gave 400 passed (no data needed).
