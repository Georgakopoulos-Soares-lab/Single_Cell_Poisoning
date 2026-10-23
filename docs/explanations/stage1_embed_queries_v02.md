# Stage 1 — `embed_queries`, in v02

**Status: product reused, never re-run in v02.** `cache/GSE136831/query_emb.npy`
was computed under v01 in August 2026 and every v02 step reads it.

## What changed: the cache moved under the dataset, and this was a bug fix

Query artefacts now live at `cache/<DATASET>/`:

    cache/GSE136831/query_emb.npy           49,456 x 128 float32
    cache/GSE136831/query_meta.parquet
    cache/GSE136831/clean_neighbors.npz
    cache/GSE136831/clean_neighbors_deep.npz

Until 2026-09-28 these sat at fixed names in `cache/` with no dataset in the
path. **A second query dataset would have overwritten the first's in place** —
and because the project runs with `--rerun-triggers mtime`, that overwrite would
then have invalidated every sweep beneath them (HANDOFF §5, first trap). The
files would have been silently *replaced*, not detected as different.

So the two-tier cache is not tidiness. It is the difference between a second
dataset being a config entry and a second dataset destroying the first's
results.

## What goes in and comes out

In: the held-out `.h5ad` named by `config.datasets.<id>.paths.queries_h5ad`, and
the SCimilarity encoder from `0_data/reference/model_v1.1/`.

Out: `query_emb.npy` — 49,456 × 128 float32, unit-normalised — and
`query_meta.parquet` carrying `y_true` and the donor column.

**49,456, not 50,000.** The dataset ships 50,000 cells; the unlabelled ones are
dropped. The count is asserted, not assumed:
`config.datasets.<id>.query.expected_n_unlabeled` declares it and the load
fails if the number moves. A silently changing denominator is how a rate drifts
without anyone noticing.

## The columns are named in config, not in code

`label_column` and `donor_column` are config keys. In GSE136831 the donor column
is `obs.sample`, giving **78 donors**. Another dataset will name them
differently, and that is the whole point of stage 1 being parameterised: the
encoder is fixed, the schema of what it is fed is not.

## The number that everything is read against

**78 donors is the unit of independence, not 49,456 cells.** Stage 5's v02 file
covers what follows from that. It is recorded here because it is a property of
the query artefact this stage produces, not of the statistics applied later —
and quoting a cell-level interval is a prohibited claim (SPEC.md §12).
