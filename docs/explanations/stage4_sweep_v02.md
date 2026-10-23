# Stage 4 — `sweep`, in v02

**This is still the load-bearing stage, and it is where v02 diverges from v01
most.** v01 had one retriever. v02 has four paths, records which one ran in
every table, and has measured that one of them is **not reproducible**.

The `sweep` rule in `00_pipeline/Snakefile` has never executed in v02. The
machinery that actually ran is `00_common/celltype_sweep.py` — "machinery shared
by the per-celltype graph sweeps" — driven by a step script.

## The four retrieval paths

Every v02 results table carries a `retrieval_path` column. v01 did not need one.

| path | what it does | exact? | where it ran |
|---|---|---|---|
| `hnsw` | insert poison into the shipped `labelled_kNN.bin` with `add_items`, re-query | faithful | `02a_01`, `02b_01`, `02b_02`, `02b_13`, `02b_14`, `02b_30`, `02b_34` |
| `graph` | `modify.graph_topk` — in-place update of existing records | faithful | `02b_11_graph`, `02b_15_graph` |
| `cached_clean` | label-only edit; index is byte-identical so cached clean neighbours **are** the poisoned ones | **exact by construction** | `02a_03`, `02b_05` |
| `exact` | the merge-and-rerank shortcut over the depth-500 list | approximate under HNSW | `02b_11`, `02b_15` — retained only as `02b_16`'s comparison arm |

**No figure reads an `exact` table.** Verified 2026-09-28 by listing every
parquet each figure script opens by name. `check_retrieval.py` now enforces it:
a figure may not name the plain variant of a step that also has a `_graph` twin
unless it asserts on `retrieval_path`. The two filenames differ by one suffix,
so the wrong one looks right — which is why this is a gate rather than a
convention.

## The finding this stage must carry: the graph path is not reproducible

`hnswlib.add_items` defaults to `num_threads=-1`. Concurrent insertion into an
HNSW graph makes the resulting structure depend on thread scheduling, so **the
same poison set inserted twice gives a different graph.** Every `graph` and
`hnsw` result in this project was produced that way.

`02b_34` (RUNLOG Run 29) measured the band — four conditions, six repeats each,
identical `condition_seed`, identical `poison_digest`, identical bootstrap seed:

| metric | range over 6 repeats |
|---|---|
| `within_retrieval` | **0.000000** |
| `global_retrieval` | **0.000000** |
| `within_baseline_accuracy` | **0.000000** |
| `within_accuracy` | 0.003767 – 0.016827 |
| `within_net_attack_success` | 0.003767 – 0.016827 |

**Retrieval failure reproduces exactly; outcome failure does not.** That split
falls along the distinction this whole benchmark exists to draw (`CLAUDE.md`
§1), and it was not designed that way — it is what the measurement returned.

The mechanism is visible in the same table. The graph genuinely differs: 8–11%
of the attacked celltype's queries get a different top-50, and at CL:0000623
dose 1.0 one query shares **none** of its 50 neighbours with the first repeat
(`within_topk_overlap_min` = 0.000000). But whether *any* poisoned record is in
the list is stable, so retrieval rates do not move. The vote absorbs the rest;
`within_pred_changed` is 0.038–0.055, concentrated where the vote is tied. The
baseline column at exactly 0.000000 localises all of it to insertion rather than
to the machine or the bootstrap.

**Precision, therefore.** A graph retrieval number may be quoted as measured. A
graph outcome number — accuracy, attack success, net attack success — is good to
about **±0.02 within-celltype and ±0.001 global**. A difference smaller than
that is not a finding. This is unresolved, not fixed: `num_threads=1` is
bit-identical and 10.6× slower on insertion, and re-running `02b_01` alone is
roughly two days. `docs/DECISIONS.md` records it as open.

## What is recorded per condition, and why the shape changed

v01 wrote one row per query cell. v02's celltype sweeps write **one row per
condition**, with the per-query detail reduced inside `Bench.measure`:

```
Retrieved:
    is_poison   (n_q, k) bool       — which neighbours are poison
    topk_cos    (n_q, k) float32
    acc         VoteResult          — pivoted on truth
    hit         VoteResult          — pivoted on target
    ids         (n_q, k)            — added by 02b_34 for the overlap metrics
```

One retrieval, one vote, two readers. `acc` and `hit` are the *same* vote read
against two pivots, not two votes — which is what stops a clean baseline and
its attacked measurement being scored by different rules. v01 grew six
divergent vote implementations and a 0.05 pp discrepancy went unexplained for
weeks; `00_common/vote.py` is now the only vote in the repository.

`02b_30` is the exception that writes per-query rows — 395,648 of them, 8
conditions × 49,456 queries — because the query-side figures ask *which* queries
changed, not how many.

## Scope columns go wide, deliberately

A condition scored over two populations puts `within_*` and `global_*` on **one
row**, not two rows with a `scope` column. They are two views of one condition —
same poison, same retrieval, same seed — not two observations. Splitting them
would make a paired difference between scopes a self-join. This is the single
documented exception to long-format storage (`CLAUDE.md` §7); every other axis
stays long.

## Ties are reported, not resolved

`vote()` returns `n_tied` and `pivotal_in_tie`; `bounds()` turns them into
best/worst columns. Only a tie *containing* the pivotal label is ambiguous — a
tie among labels that are all wrong is already determined.

This matters more in v02 than v01 because of where the ties are. At dose 1.0 the
vote ties constantly: `02b_25` finds the reported 0.249 could be anywhere from
0.064 to 0.324, and **331 of 2,184 conditions carry a spread wider than 0.05**.
Publish the interval, not the point.

## One trap that is pure arithmetic

**At dose 1.0, `n_poison == n_celltype` exactly**, so hub selection and uniform
selection take *every* record and become the same attack. The gap between them
collapses from 0.844 to 0.001 at that rung and reopens to 0.061 at 1.25
(HANDOFF §3.43). Any figure whose selection panels meet at parity is showing the
degeneracy, not agreement. A dose ladder that stops at 1.0 cannot separate
selection from budget at its last rung.
