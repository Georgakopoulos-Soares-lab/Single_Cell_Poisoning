# Stage 2 — `retrieve_clean`, in v02

**Status: run once, under v01's rule, and the product is load-bearing for every
v02 comparison.** `retrieve_clean_deep_20260924_234009.log` (RUNLOG Run 9) is
the record.

## What changed: depth 500, not depth 50

    cache/GSE136831/clean_neighbors.npz        idx, cos  (49,456 x  50)
    cache/GSE136831/clean_neighbors_deep.npz   idx, cos  (49,456 x 500)

The deep list is 283 MB and took **47 minutes**. It exists because the merge
shortcut needs more than the final answer: when poison displaces genuine records
out of the top 50, the ranks below 50 are what the clean side falls back to. A
depth-50 list cannot answer "what would have been 51st".

This is also the one piece of `cache/` that is genuinely expensive. An earlier
version of `CLAUDE.md` §3 claimed the cache holds "never what a script
COMPUTED"; that was already false when written, and the entry now says the true
thing instead — nothing in `cache/` is a *result*, and everything in it is
recomputable, but "recomputable" here means 47 minutes.

## The rule this stage exists to enforce in v02

**One retrieval regime per comparison.** A clean baseline must be retrieved the
same way as the attacked measurement it is subtracted from.

v02 found this broken. `02b_11` and `02b_15` took the attacked side from
`modify.graph_topk` (HNSW) but the clean side from `deep_idx[:, :K_NEIGHBORS]`
(exhaustive), so `accuracy_delta` and `net_attack_success` straddled two
retrievers. Under the old `exact` setting both sides came from the deep list and
the discrepancy cancelled — **making the attacked side faithful is what exposed
it** (`CLAUDE.md` §7).

The cost of getting it wrong is bounded by `01a_02` and it is not roundable:
median |Δaccuracy| is 0.000000 and 32 of 34 celltypes are under 0.01, but
**CL:0002062 is 0.110**, and top-50 overlap falls to 0.327 on the worst
celltype. That is the same order as the graph-vs-exact divergence the graph runs
exist to measure.

`02a_01` shows the correct pattern: query the clean index through HNSW *before*
inserting (lines 94–96), so both sides share a regime at the cost of one extra
query per run rather than per condition.

## Exhaustive in every profile

Stage 2 runs exhaustively in `full` as well as `smoke`. `CLAUDE.md` §6 supersedes
the older table that described HNSW for the full profile. Exhaustive retrieval
is what makes the merge shortcut provably exact (SPEC.md §4.2) at any poison-set
size — which is the argument stage 6's v02 file leans on.

## What to check before using the cached list

It is cached, and `--rerun-triggers mtime` reasons backwards from the target, so
an existing file means "done" no matter what changed upstream. Before reusing
it: recompute two or three stored sweeps against current inputs, compare, then
`touch`. Deleting it to "regenerate metadata" once invalidated 12,874 sweeps.
