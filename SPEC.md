# SPEC.md — biorag-guard-sc build contract

Authoritative description of **what the pipeline computes**. Where this file and
`CLAUDE.md` disagree on computation, this file wins; where they disagree on who
may edit what, `CLAUDE.md` wins (`CLAUDE.md` §3).

Status: **draft for review.** Nothing here has been executed. No number in this
document is a result; every quantity below is a parameter or a definition.

---

## 1. Question

Given a reference atlas used for k-NN label transfer, how does the fraction of
corrupted reference records translate into wrong cell-type calls?

Two failure rates, measured separately and never collapsed:

- **Retrieval failure** — at least one poisoned record enters the top-*k*
  neighbours of a query cell. A property of the retriever.
- **Outcome failure** — the k-NN vote returns a label other than `y_true`.
  A property of the aggregator.

A defence that filters retrieved neighbours acts on the first; a defence that
changes the vote rule acts on the second. Reporting only the second tells a
defence designer nothing about where to intervene.

**Baseline obligation.** `y_pred_clean` is recorded per query in every raw
result, so clean misclassification is separable from poisoning-induced
misclassification. Outcome failure rate is reported alongside the clean error
rate; the difference is the attack effect. Reporting outcome failure without the
clean baseline would attribute the classifier's own errors to the attack.

---

## 2. Data — real, public, checksummed

No synthetic data. Both inputs are public Zenodo artifacts, md5s recorded in
`config/checksums.txt` and verified on download.

| Role | Source | Size | md5 |
|---|---|---|---|
| Reference + encoder | [Zenodo 10685499](https://zenodo.org/records/10685499) `model_v1.1.tar.gz` | 30.31 GB | `546251b7…` |
| Held-out queries | [Zenodo 13685881](https://zenodo.org/records/13685881) `GSE136831_subsample.h5ad` | 2.20 GB | `9589d336…` |

**Queries** (verified, on disk): 50,000 cells × 44,942 genes from Adams et al.
2020 (GSE136831), human lung — IPF 23,803, healthy 15,149, COPD 11,048.
`obs.celltype_id` carries Cell Ontology identifiers directly (33 distinct
terms); this is `y_true`. `obsm` is empty, so query embeddings must be computed
by the SCimilarity encoder in stage 1.

**Reference** (download in progress): SCimilarity's precomputed reference
embeddings, kNN graph, and per-cell metadata, from the same tarball as the
encoder weights.

> **Open — resolve before stage 3 is implemented.** Whether the tarball ships a
> Cell Ontology graph. If not, fetch `cl.obo` (~30 MB) and add it as a stage-0
> input. Ontology distance is a first-class axis (`CLAUDE.md` §1) and cannot be
> faked.

### The dataset is a parameter — added 2026-09-27

The **tool** is fixed to SCimilarity. The **dataset** is not. `config/config.yaml`
carries a `datasets:` block keyed by dataset id — query paths, the label and
donor column names, the expected count of unlabelled cells, and that dataset's
pair list — and `1_scripts/00_common/dataset_config.py` resolves it. `dataset:`
at the top level selects the active one.

Nothing in `1_scripts/` hardcodes `GSE136831`. Adding a second query dataset is
a config entry plus a query `.h5ad`; it is not a code change. Adding a second
*tool* is, and has not been designed — see `docs/DECISIONS.md`.

Two cache tiers follow from this: `cache/` at the repository root holds what is
unpacked from `0_data/` and is shared across datasets, while
`cache/<DATASET>/` holds anything derived per dataset. `dataset_config` exposes
both as `cache_root()` and `query_cache()`.

### Profiles

Both profiles draw from the same real reference; they differ only in how many
rows are used and how neighbours are retrieved.

| | `smoke` | `full` |
|---|---|---|
| Reference rows | 500,000, fixed-seed subsample | all (~7.9M) |
| Retrieval | exhaustive cosine (numpy) | exhaustive cosine (numpy) |
| Merge shortcut | **exact** (§4.2) | **exact** (§4.2) |

**Why `full` is exhaustive, not HNSW** (decided 2026-08-26, ratified 2026-09-16).
This table previously specified an HNSW index and an approximate shortcut for
`full`, matching CLAUDE.md §6. The run was executed exhaustively instead, and
three sources recorded that at the time — `config/profiles/full/config.yaml`,
whose header calls it a deliberate deviation; `1_scripts/00_pipeline/retrieve_clean.py`,
which *raises* if retrieval is anything but exhaustive; and `docs/RUNLOG.md`'s
Run 2 entry, which logs the profile as "exhaustive cosine (not HNSW)". Only this
table was left stale, and it is the authoritative contract, so it was telling
readers the headline numbers are approximate when they are exact.

The trade: an hour of extra compute removes an approximation from every published
figure. Stage 6 then measures the HNSW gap as its own question, rather than
gating the main result on it. See `docs/DECISIONS.md`.
| Approval needed | no | **yes, human, per run** |

> **Deviation from `CLAUDE.md` §2/§8 — orchestrator edit required.** CLAUDE.md
> defines `smoke` as a "synthetic 500k reference, no downloads". This spec
> replaces synthetic generation with a subsample of the real reference: same
> size, same runtime, real labels, and no separate generator to maintain. The
> phrase "synthetic 500k reference" in CLAUDE.md §8 and the `smoke.yaml`
> description in §3 must change to "subsampled 500k reference". The "no
> downloads" property no longer holds for a cold checkout; stage 0 must run once.

---

## 3. Fixed constants

| Symbol | Value | Meaning |
|---|---|---|
| `D` | 128 | embedding dimension, unit-normalised |
| `k` | 50 | neighbours in the vote |
| `SCORE_QC` | 33 | SCimilarity score below which a hit is an outlier |
| `N_BOOT` | 2000 | bootstrap resamples over seeds |

SCimilarity score from cosine similarity `c`:

```
score(c) = 1 / (1 - c)
```

Monotone in `c`, so ranking by cosine and ranking by score agree. The QC
threshold is defined on the score: `score >= 33  <=>  c >= 1 - 1/33 = 0.969697`.
Scores are computed only where reported, never used for ranking.

> `c = 1` gives `score = inf`. The `duplicate` attack produces exact copies, so
> this is reachable, not hypothetical. `min_poison_score` is float32 and `inf`
> is a legal value that compares correctly against `SCORE_QC`. Do not clip it,
> and do not guard it with try/except (`CLAUDE.md` §7).

---

## 4. Stages

### Stage 0 — `acquire`

```
acquire() -> data/reference/, data/queries/
  download both Zenodo artifacts, resumable
  verify md5 against data/CHECKSUMS.txt   # mismatch is a hard failure
  extract model_v1.1.tar.gz -> data/reference/
```
Needs human approval. Runs once.

### Stage 1 — `embed_queries`

```
embed_queries(h5ad, encoder) -> cache/query_emb.npy, cache/query_meta.parquet
  A = read_h5ad(h5ad)                    # (50000, 44942) counts
  align A.var to the encoder's 28k gene symbols   # missing genes -> 0
  E = encoder.get_embeddings(A.X)        # (50000, 128) float32
  E = E / norm(E, axis=1, keepdims=True) # unit norm; assert no zero rows
  meta = A.obs[['celltype_id']] -> y_true
  write E, meta
```
Runs once; cached. The encoder is used **here only** — never inside a sweep.

### Stage 2 — `retrieve_clean`

```
retrieve_clean(query_emb, reference_emb) -> cache/clean_neighbors.parquet
  smoke: exhaustive  argtopk(Q @ R_sub.T, k)      # R_sub = 500k subsample
  full : hnswlib cosine index, ef >= 4k, top-k
  emit per query: k neighbour ids, k cosines, k labels
```
Runs **exactly once per profile**. Every condition in the sweep reuses it.

### Stage 3 — `make_poison`

```
make_poison(reference_emb, ontology, attack, pair, rate, seed) -> cache/poison/*.npz
  rng = np.random.default_rng(seed)
  n_p = round(rate * n_ref)
  src, tgt = pair                       # CL terms
  S = reference rows labelled src

  relabel:     emb = S[rng.choice(len(S), n_p)]          # untouched vectors
               alpha = nan
  duplicate:   emb = exact copies of chosen S rows       # c = 1 by construction
               alpha = nan
  interpolate: a ~ rng.uniform(0.3, 0.9, n_p)
               emb = normalize((1-a) * S_i + a * T_centroid)
               alpha = a        # LARGER alpha = CLOSER TO TARGET.
                                # "interpolation strength" = how far the lie
                                # travelled. An earlier draft of this line had
                                # it inverted; the implementation and its
                                # monotonicity test define the direction.

  label = tgt  (all rows)               # attacker-chosen, always the target
  provenance = source cell id for ALL three attacks
  assert allclose(norm(emb, axis=1), 1)
```

### Stage 4 — `sweep`  *(the load-bearing stage)*

```
sweep(query_emb, clean_neighbors, poison_set) -> results/raw/*.parquet
  for each block of queries (chunked; C is discarded, never materialised whole):
      C = Q_block @ P.T                      # (b, n_p) cosine, float32, brute force
      take top-k of C per row                # poison contributes at most k

      merge k clean + k poison candidates    # 2k per query
      sort by cosine desc, keep k            # merged top-k

      n_poison_in_topk   = count of poison in merged top-k
      min_poison_score   = 1 / (1 - max cosine to ANY poison)   # nearest poison,
                                                                # in top-k or not
      poison_survives_qc = min_poison_score >= 33
      y_pred_poisoned    = majority label over merged top-k
                           ties -> lowest CL id, deterministic
  write one row per query, long format
```

**4.2 Why the shortcut is exact under exhaustive search.** Every member of the
true merged top-*k* is either clean or poisoned. If clean, it outranks at most
`k-1` other clean records, so it already appears in the clean top-*k*. If
poisoned, it appears in the poison top-*k* by the same argument. So the union of
the two top-*k* lists contains the true merged top-*k*. Retrieving exactly `k`
clean neighbours in stage 2 is therefore **sufficient, not approximate** — which
is what makes one retrieval serve all 1,260 conditions.

Under HNSW this argument fails: an inserted poisoned node changes graph
traversal, so the clean top-*k* retrieved without poison need not contain the
clean members of the poisoned top-*k*. Hence stage 6.

### Stage 5 — `aggregate`

```
aggregate(results/raw/*) -> results/summary.parquet
  group by (attack, pair, rate):
    retrieval_fail_rate = mean over cells of (n_poison_in_topk > 0)
    outcome_fail_rate   = mean over cells of (y_pred_poisoned != y_true)
    clean_error_rate    = mean over cells of (y_pred_clean    != y_true)
    qc_evasion_rate     = mean over cells of poison_survives_qc
    ci_lo, ci_hi        = bootstrap over SEEDS, 2000 resamples, percentile 2.5/97.5
```
Resampling is over seeds, not cells: seeds are the independent replicates.

### Stage 6 — `validate_shortcut`  *(not optional)*

```
validate_shortcut(subset of conditions) -> results/shortcut_validation.parquet
  for a small set of conditions:
      rebuild HNSW index over reference + poison   # the expensive honest path
      query it directly
      compare per-cell against the stage-4 shortcut
      report agreement + signed disagreement
```
`full`-profile results may not be published without this.

---

## 5. Sweep grid

**Rewritten 2026-09-23.** The plan below the rule is what ran; the original
§5 is kept at the bottom of this section, superseded, because it specified a
different design and a reader who finds only the new one cannot tell what
changed.

### 5.1 Which celltypes are attacked

Every celltype meeting both thresholds in `1_scripts/00_common/celltype_sweep.py`:

| Constant | Value | Why |
|---|---|---|
| `MIN_CELLTYPE` | 2,000 reference records | below this there is too little to poison |
| `MIN_QUERY_CELLS` | 100 query cells | below this there is nothing to measure the effect on |

That yields **26 celltypes**. The set is derived, never listed, so a change to
the reference or the query set changes it without anyone editing a constant.

### 5.2 How each source's target is chosen — and what is arbitrary about it

`pick_targets` gives each source ONE target: the eligible celltype minimising
`|ontology_distance - TARGET_DIST|` with `TARGET_DIST = 4`, **ties broken on the
CL id string**. Deterministic: the same reference and ontology produce the same
26 pairs on every run, and no seed varies them.

**Why one target and not a near/mid/far ladder.** A ladder per source would
triple the cost, and HANDOFF §3.17 found ontology distance is confounded with
WHICH celltype is attacked. Holding distance roughly fixed while the source
varies is what isolates the source. The ladder in the superseded §5 below was
dropped for exactly this reason.

**Two things are arbitrary, and they are recorded here because a reader cannot
infer them from the rule.**

*The tie-break decides most of the grid.* 21 of the 26 pairs had more than one
candidate at the minimum `|d - 4|`, and the CL accession number picked among
them. `myofibroblast cell` had SEVEN equally eligible targets -- club cell,
smooth muscle cell, macrophage, dendritic cell, alveolar macrophage, basal cell,
capillary endothelial cell -- and got club cell because `CL:0000158` is the
lowest number. Across all 26 sources the same rule admits roughly 3 x 10^11
equally valid pairings; the committed one is a single draw from that set.

*Consequence for reading a per-celltype result.* "Myofibroblast resists
insertion" means "myofibroblast attacked toward club cell resists insertion".
Nothing measured says the other six behave the same, and because the pair is
fixed across seeds, repeating the sweep cannot reveal the sensitivity -- the
target axis has n = 1 with no interval. Per-celltype claims must carry their
target.

*`TARGET_DIST = 4` is itself a choice*, not a derived quantity. Realised
distances are 4 for 21 pairs and 5 for 5, because nothing closer to 4 exists for
those sources.

**A latent trap, for anyone changing `TARGET_DIST`.** `pick_targets` does NOT
call `Ontology.is_ancestor_or_descendant`, whose own docstring says such a pair
"is not a viable attack: calling a T cell a 'lymphocyte' is a vaguer truth, not
a lie". Eight ordered pairs among the 26 are ancestor/descendant --
dendritic/plasmacytoid dendritic at distance 1, T cell/regulatory T cell,
macrophage/alveolar macrophage and NK/innate lymphoid at 2. All 26 pairs came
out as cousins only because aiming at 4 pushes away from distances 1-2. Lower
`TARGET_DIST` for a near rung and the selector will take them, and "success"
there means the label became vaguer while staying true. Wire the guard in before
running any distance ladder.

### 5.3 The grid that ran

| Step | Grid | n |
|---|---|---|
| `02a_01` uniform | 6 atlas budgets x 5 draws | 30 |
| `02b_01` per celltype | 3 attacks x 26 celltypes x 8 rates x 1 seed | 624 |
| `02b_02` flood anchors | 26 celltypes x 5 anchor counts x 4 rates | 520 |
| `02b_05` hub, split donor | 3 selections x 4 rates x 26 sources x 3 split seeds | 936 |

Rates in `02b_01` are `0.25, 0.5, 0.7, 0.8, 0.85, 0.9, 1.0, 1.25` as a fraction
of the SOURCE CELLTYPE (§10), narrowed from v01's 16 rates spanning 0.01-1.5
because v01 established where the cliff is and the graph arm costs ~57 s per
condition against v01's ~0.3 s merge shortcut. The low end is therefore
unmeasured in v02 and available in v01.

**One seed, not ten.** The superseded plan called for 10 seeds so the bootstrap
could resample them. The graph arm instead reports donor-paired intervals over
78 donors, which is the unit of independence for query cells; seed variance is
not estimated in `02b_01` and no interval there should be read as covering it.

### 5.4 The symmetric grid — what must exist for insert vs modify to be a comparison

**The finding that forces this section.** The two arms currently share exactly ONE
budget (rate 0.25) and one selection rule (uniform). Insert was swept at
0.25-1.25 of the celltype, modify at 0.01-0.25. Every insert-vs-modify statement
other than at that single point crosses a budget gap, a selection gap, or both,
and is therefore not a controlled comparison.

At the one comparable point -- rate 0.25, uniform selection, unmoved geometry,
relabel -- insert scores 0.0002 and modify scores 0.0318, a factor of ~160. That
is the only fair arm-to-arm number the project currently owns.

**The principle.** For the contrast to mean anything, a cell present in one arm
must be present in the other at the same budget and the same selection. Cells
that CANNOT exist are listed anyway and marked, because a blank left unexplained
reads as an oversight and invites someone to fill it.

**The grid.** All 72 attack types -- scope x access x geometry x label x
selection. ``run?`` is a recommendation, and ``why`` carries its reason so any
line can be overruled on its own. Twelve carry a titration today; eleven of those
are curves and one is a single point.

+----------+--------+-------------+---------+--------------+-------------+------------------+------+---------------------------------+
| scope    | access | geometry    | label   | selection    | status      | titration        | run? | why                             |
+----------+--------+-------------+---------+--------------+-------------+------------------+------+---------------------------------+
| atlas    | insert | unmoved     | relabel | uniform      | v02 30      | 6 pts 0.1%-10%   | YES  | single-target variant unbuilt   |
| atlas    | insert | unmoved     | relabel | concentrated | not run     | -                | NO   | no argument made for it         |
| atlas    | insert | unmoved     | relabel | hub          | not run     | -                | NO   | no argument made for it         |
| atlas    | insert | unmoved     | keep    | uniform      | not run     | -                | NO   | no argument made for it         |
| atlas    | insert | unmoved     | keep    | concentrated | not run     | -                | NO   | no argument made for it         |
| atlas    | insert | unmoved     | keep    | hub          | not run     | -                | NO   | no argument made for it         |
| atlas    | insert | shifted     | relabel | uniform      | not run     | -                | NO   | no argument made for it         |
| atlas    | insert | shifted     | relabel | concentrated | not run     | -                | NO   | no argument made for it         |
| atlas    | insert | shifted     | relabel | hub          | not run     | -                | NO   | no argument made for it         |
| atlas    | insert | shifted     | keep    | uniform      | not run     | -                | NO   | no argument made for it         |
| atlas    | insert | shifted     | keep    | concentrated | not run     | -                | NO   | no argument made for it         |
| atlas    | insert | shifted     | keep    | hub          | not run     | -                | NO   | no argument made for it         |
| atlas    | insert | synthesised | relabel | uniform      | no question | -                | NO   | no atlas manifold to aim at     |
| atlas    | insert | synthesised | relabel | concentrated | no question | -                | NO   | no atlas manifold to aim at     |
| atlas    | insert | synthesised | relabel | hub          | no question | -                | NO   | no atlas manifold to aim at     |
| atlas    | insert | synthesised | keep    | uniform      | no question | -                | NO   | no atlas manifold to aim at     |
| atlas    | insert | synthesised | keep    | concentrated | no question | -                | NO   | no atlas manifold to aim at     |
| atlas    | insert | synthesised | keep    | hub          | no question | -                | NO   | no atlas manifold to aim at     |
+----------+--------+-------------+---------+--------------+-------------+------------------+------+---------------------------------+
| atlas    | modify | unmoved     | relabel | uniform      | not run     | -                | YES  | coordinated modify, untested    |
| atlas    | modify | unmoved     | relabel | concentrated | degenerate  | -                | NO   | = uniform modify at 1%          |
| atlas    | modify | unmoved     | relabel | hub          | not run     | -                | YES  | coordinated modify, untested    |
| atlas    | modify | unmoved     | keep    | uniform      | baseline    | -                | NO   | nothing changes                 |
| atlas    | modify | unmoved     | keep    | concentrated | baseline    | -                | NO   | nothing changes                 |
| atlas    | modify | unmoved     | keep    | hub          | baseline    | -                | NO   | nothing changes                 |
| atlas    | modify | shifted     | relabel | uniform      | not run     | -                | NO   | no argument made for it         |
| atlas    | modify | shifted     | relabel | concentrated | degenerate  | -                | NO   | = uniform modify at 1%          |
| atlas    | modify | shifted     | relabel | hub          | not run     | -                | NO   | no argument made for it         |
| atlas    | modify | shifted     | keep    | uniform      | not run     | -                | NO   | pending celltype alpha sweep    |
| atlas    | modify | shifted     | keep    | concentrated | degenerate  | -                | NO   | = uniform modify at 1%          |
| atlas    | modify | shifted     | keep    | hub          | not run     | -                | NO   | pending celltype alpha sweep    |
| atlas    | modify | synthesised | relabel | uniform      | no question | -                | NO   | no atlas manifold to aim at     |
| atlas    | modify | synthesised | relabel | concentrated | degenerate  | -                | NO   | = uniform modify at 1%          |
| atlas    | modify | synthesised | relabel | hub          | no question | -                | NO   | no atlas manifold to aim at     |
| atlas    | modify | synthesised | keep    | uniform      | no question | -                | NO   | no atlas manifold to aim at     |
| atlas    | modify | synthesised | keep    | concentrated | degenerate  | -                | NO   | = uniform modify at 1%          |
| atlas    | modify | synthesised | keep    | hub          | no question | -                | NO   | no atlas manifold to aim at     |
+----------+--------+-------------+---------+--------------+-------------+------------------+------+---------------------------------+
| celltype | insert | unmoved     | relabel | uniform      | v02 208     | 8 dose           | YES  | extend ladder to 0.002          |
| celltype | insert | unmoved     | relabel | concentrated | v02 208     | 8 dose, 1 pool   | YES  | extend ladder + pool sweep      |
| celltype | insert | unmoved     | relabel | hub          | build 2     | -                | YES  | selection symmetry w/ modify    |
| celltype | insert | unmoved     | keep    | uniform      | build 4     | -                | YES  | volume control                  |
| celltype | insert | unmoved     | keep    | concentrated | build 4     | -                | YES  | volume control                  |
| celltype | insert | unmoved     | keep    | hub          | not run     | -                | NO   | no argument made for it         |
| celltype | insert | shifted     | relabel | uniform      | v02 208     | 8 dose, 1 alpha  | YES  | extend ladder + alpha           |
| celltype | insert | shifted     | relabel | concentrated | not run     | -                | NO   | no argument made for it         |
| celltype | insert | shifted     | relabel | hub          | not run     | -                | NO   | no argument made for it         |
| celltype | insert | shifted     | keep    | uniform      | not run     | -                | NO   | no argument made for it         |
| celltype | insert | shifted     | keep    | concentrated | not run     | -                | NO   | no argument made for it         |
| celltype | insert | shifted     | keep    | hub          | not run     | -                | NO   | no argument made for it         |
| celltype | insert | synthesised | relabel | uniform      | v01         | 1 pt 0.25 only   | NO   | construction variant, low value |
| celltype | insert | synthesised | relabel | concentrated | v02 520     | 4 dose, 5 anchor | YES  | extend ladder + spread/jitter   |
| celltype | insert | synthesised | relabel | hub          | not run     | -                | NO   | no argument made for it         |
| celltype | insert | synthesised | keep    | uniform      | not run     | -                | NO   | no argument made for it         |
| celltype | insert | synthesised | keep    | concentrated | not run     | -                | YES  | geometry control for flood      |
| celltype | insert | synthesised | keep    | hub          | not run     | -                | NO   | no argument made for it         |
+----------+--------+-------------+---------+--------------+-------------+------------------+------+---------------------------------+
| celltype | modify | unmoved     | relabel | uniform      | v02 936     | 4 dose           | YES  | extend ladder to 1.25           |
| celltype | modify | unmoved     | relabel | concentrated | degenerate  | -                | NO   | = uniform modify at 1%          |
| celltype | modify | unmoved     | relabel | hub          | v02 936     | 4 dose           | YES  | extend ladder to 1.25           |
| celltype | modify | unmoved     | keep    | uniform      | baseline    | -                | NO   | nothing changes                 |
| celltype | modify | unmoved     | keep    | concentrated | baseline    | -                | NO   | nothing changes                 |
| celltype | modify | unmoved     | keep    | hub          | baseline    | -                | NO   | nothing changes                 |
| celltype | modify | shifted     | relabel | uniform      | v01         | 9 dose, 2 alpha  | YES  | reproduce + alpha sweep         |
| celltype | modify | shifted     | relabel | concentrated | degenerate  | -                | NO   | = uniform modify at 1%          |
| celltype | modify | shifted     | relabel | hub          | v01         | 9 dose, 2 alpha  | YES  | reproduce + alpha sweep         |
| celltype | modify | shifted     | keep    | uniform      | v01         | 9 dose, 1 alpha  | YES  | step 5: alpha 11 pts            |
| celltype | modify | shifted     | keep    | concentrated | degenerate  | -                | NO   | = uniform modify at 1%          |
| celltype | modify | shifted     | keep    | hub          | v01         | 9 dose, 1 alpha  | YES  | step 5: alpha 11 pts            |
| celltype | modify | synthesised | relabel | uniform      | build 3     | -                | YES  | geometry symmetry w/ insert     |
| celltype | modify | synthesised | relabel | concentrated | degenerate  | -                | NO   | = uniform modify at 1%          |
| celltype | modify | synthesised | relabel | hub          | build 3     | -                | YES  | geometry symmetry w/ insert     |
| celltype | modify | synthesised | keep    | uniform      | not run     | -                | NO   | no argument made for it         |
| celltype | modify | synthesised | keep    | concentrated | degenerate  | -                | NO   | = uniform modify at 1%          |
| celltype | modify | synthesised | keep    | hub          | not run     | -                | NO   | pending celltype alpha sweep    |
+----------+--------+-------------+---------+--------------+-------------+------------------+------+---------------------------------+

72 attack types: 19 to run, 53 not.

**The titration column names EVERY axis varied, not just dose.** A row reading
``9 dose, 1 alpha`` has a healthy-looking dose ladder and a construction
parameter that was never moved -- which is the state of ``shifted / keep`` today,
and was invisible while the column reported dose alone. Three rows are overstated
no longer: ``shifted`` ran at one or two alphas, and concentrated ``unmoved``
has never varied its pool fraction at all.

**Statuses.** ``v02 n`` measured on SCimilarity's graph under the unified vote ·
``v01`` measured on exact search before the vote was unified · ``build N``
agreed, numbered per the work order below · ``baseline`` nothing changes ·
``degenerate`` real but reduces to an attack type already measured ·
``no question`` coherent but answers nothing worth asking · ``not run`` coherent
and nobody has stated what question it would answer.

**DEGENERATE -- concentrated x modify.** Not incoherent; redundant. Concentrated
selection draws from a restricted pool WITH replacement, so the same record is
chosen repeatedly. Under insert, drawing a cell five times CREATES five records
-- that amplification IS the attack. Under modify, a record can only be
overwritten once, and selection is a set membership test, so choosing it five
times still overwrites it once.

A concentrated modify at any nominal budget therefore reduces to a UNIFORM modify
at the pool size (1% of the celltype), which the uniform column already measures.
The cell reproduces an existing point rather than holding a new condition. Six
cells, and the claim is checkable: run one and it should match uniform modify at
rate 0.01 exactly.

**BASELINE -- unmoved x modify x keep.** Overwrite a record without moving it or
changing its label: the reference is unchanged. This is the clean baseline,
already measured in ``01a_03`` (accuracy 0.4637 over 49,456 queries, 78 donors,
intervals in ``01a_04``). Not an attack, and not missing.

### 5.4b The dose ladder -- one grid for every celltype-scope attack type

**Agreed 2026-09-24.** All sixteen celltype-scope attack types marked to run use
the SAME fourteen doses. This is not a property of individual rows to be
extended one at a time; it is a property of the design, and it is what makes
insert against modify a comparison rather than two adjacent curves.

    0.002  0.005  0.01  0.025  0.05  0.1  0.25  0.5  0.7  0.8  0.85  0.9  1.0  1.25

as a fraction of the SOURCE CELLTYPE (``dose_basis = celltype``, 5.5). What that
means in absolute terms across the 26 attackable celltypes, which run 10,438 to
368,537 records:

  dose   % of celltype     records            % of the 7,913,892-record atlas
  0.002       0.2%        21 -       737       0.0003% - 0.009%
  0.05        5.0%       522 -    18,427       0.0066% - 0.233%
  0.25       25.0%     2,610 -    92,134       0.0330% - 1.164%
  0.9        90.0%     9,394 -   331,683       0.1187% - 4.191%
  1.25      125.0%    13,048 -   460,671       0.1649% - 5.821%

Four orders of magnitude, 21 records to 460,671.

**The ladder is a UNION of the two arms' existing ladders, plus three points
below them.** That is the property that matters, and it was chosen over an
evenly log-spaced ladder for two measured reasons.

*Placement, not count, is what a dose-response ladder has to get right.* The
two arms turn on in different places, and both places are measured:

  ``02b_01``, insert/relabel, median within-celltype net attack success:
    0.25 0.000 | 0.50 0.000 | 0.70 0.000 | 0.80 0.006 | 0.85 0.020
    0.90 0.053 | 1.00 0.245 | 1.25 0.390

  ``02b_05``, modify/oracle-hub, median traffic captured by the edited records:
    0.01 0.694 | 0.05 1.000 | 0.10 1.000 | 0.25 1.000

Insert does nothing at all until 0.7 and then rises by a factor of forty across
0.8-1.25; every point that resolves that rise lies in a span half a decade wide.
Modify is already at 69% traffic capture at the LOWEST dose anyone has run and
saturates by 0.05; its entire unmeasured region is below 0.01. A ladder that is
evenly log-spaced is dense where insert is flat and sparse exactly where it
moves. The six points at 0.7-1.25 and the three at 0.002-0.025 are there because
that is where the two curves respectively live.

*Nothing already measured is discarded.* The ladder contains all eight doses of
``02b_01`` and all four of ``02b_05``, so 1,144 existing insert conditions carry
forward instead of being re-run. An evenly spaced ladder shares only four points
with ``02b_01`` and two with ``02b_02``, throwing away 572 conditions -- about
7 h of compute -- to buy a worse curve.

**Atlas-scope attack types keep their own ladder** and that is forced, not
chosen: their budget divides by 7,913,892 rather than by a celltype, so the two
are different quantities (5.5, HANDOFF 3.22). ``atlas_share`` and ``n_poison``
are on every row of both, which is what lets a reader put them on one axis.

**Cost, at 26 celltypes**, using the median 45.2 s per insert condition measured
in ``02b_01``'s own ``seconds`` column rather than an estimate:

  insert   12 sweeps (8 types, flood x5 anchors)   4,368 conditions
           of which 1,144 already measured         3,224 to run     40.5 h
  modify    8 sweeps                               2,912 conditions  ~3 h
                                                                    ~44 h serial

Three concurrent insert workers hold ~45 GB of 64 and contend for cores, giving
roughly 16-18 h wall.

### 5.4d The alpha ladder -- how far a shifted record moves

**Agreed 2026-09-25, after ``02b_11`` measured the problem.**

``shifted`` geometry moves a record toward the target centroid by ``alpha``.
v01 only ever ran 0.5, and both v01 and ``02b_11`` measure it as INERT: net
attack success exactly 0.0000 at all thirteen doses, for all four modify
variants. That was read as "the attack does not work". It is not what the
number says.

``02b_11``'s retrieval column says it: 0.0006. The poison is never retrieved,
so it cannot win a vote it never enters. And the reason is geometric, now
measured rather than guessed. The genuine top-50 neighbourhood spans
``1 - cos = 0.0195`` (5.4c, 02b_13). At alpha 0.5 a shifted record moves
``1 - cos = 0.1514``, which is **776% of that radius** -- it has left the
retrievable region entirely.

So the question "is shifted a weak attack?" has never been asked. What was
measured is "a record thrown 8 radii away is not retrieved", which is
arithmetic, not a finding.

**The ladder**, chosen to bracket the neighbourhood edge the way 5.4b brackets
the dose range:

    0.005  0.01  0.02  0.05  0.1  0.2  0.5  0.75  1.0

with the displacement each produces, median over the 26 celltypes:

  alpha    1-cos moved   degrees   % of neighbourhood radius
  0.005        0.00001      0.26        0%     inside
  0.01         0.00004      0.52        0%     inside
  0.02         0.00016      1.04        1%     inside
  0.05         0.00106      2.64        5%     inside
  0.1          0.00449      5.43       23%     inside
  0.2          0.01984     11.43      102%     AT THE EDGE
  0.5          0.15139     31.94      776%     well outside  <- v01's only value
  0.75         0.34727     49.25     1781%     well outside
  1.0          0.55974     63.88     2870%     unretrievable

Six of the nine points sit inside or at the edge, which is the region v01 never
sampled. The three above 0.5 are kept deliberately: they DOCUMENT the collapse
instead of assuming it, and 0.5 anchors the ladder to every number v01 and
``02b_11`` already published.

**Note the shape of the expected answer.** Alpha trades reach against
displacement in the same way jitter trades distinctness against retrieval: a
record that has not moved is retrieved but carries the attacker's label onto a
vector indistinguishable from a genuine cell -- which is ``unmoved`` geometry,
already measured. If low alpha simply reproduces ``unmoved``, then ``shifted``
has no regime of its own and that is the finding. If there is a middle alpha
that beats both, that is a different and more interesting one. Either way the
curve is the result, not any single point on it.

### 5.4e What SCimilarity's own QC is, and why no attack here can trip it

**Established 2026-09-25 by reading ``scimilarity 0.4.0``.** Recorded because
the question "would their QC have caught this?" is the first one a reviewer
asks, and because this conversation reached the wrong answer twice before
reading the right file.

**The reference QC exists and is published.** It is the default filter of the
training data module, ``tiledb_data_models.py:253``, under a comment reading
``# get data with a limit to cells with counts and labels, and QC checks``:

    label_id != NaN  and  total_counts > 1000  and  n_genes_by_counts > 500
    and  pct_counts_mt < 20  and  predicted_doublets == 0

That is the ``cleaned_data`` named in the shipped ``metadata.json``. A looser
user-facing helper, ``utils.filter_cells`` (``min_genes=400``,
``mito_percent=30``), exists for cleaning one's own data before embedding.

**Every criterion is expression-space, and runs BEFORE the encoder.** Counts,
genes detected, mitochondrial fraction, doublet score. Not one is geometric:
nothing checks whether an embedding sits plausibly on the manifold, because at
the point QC runs there is no embedding yet.

**Consequence for this benchmark, by geometry:**

``unmoved``      relabel, duplicate, relabel_inplace. The vector IS a reference
                 cell's, so it already passed all five criteria to be in the
                 atlas. The QC cannot catch it -- not because it is evaded, but
                 because the record is QC-cleared data wearing a false label,
                 and cell QC does not inspect annotations.
``shifted``,     interpolate, flood, and the in-place variants. Not applicable:
``synthesised``  a constructed 128-d vector has no counts, no genes, no
                 mitochondrial reads and no doublet score to filter on, and it
                 never passes through the stage where QC happens.

So the reference QC is real, documented and citable -- and structurally unable
to see any attack in this project. That is a stronger and more defensible
sentence than "no QC exists", which is what an earlier draft of this section
said before the training module was read.

**Therefore ingestion QC is OUT OF SCOPE**, and with it the decoder round-trip
test that was proposed to answer it. There is no ingestion path in the shipped
package -- no ``add_items``, no ``build_index`` -- so there is nothing to model.
What remains is a DEFENCE question (would re-deriving embeddings from raw data
close rung 1?), and it belongs with the defence stage.

**What is in scope, and is not QC.** ``annotate_dataset`` writes six columns
per cell and acts on none of them: ``celltype_hint``, ``min_dist``,
``celltype_hits``, ``celltype_hits_weighted``, ``celltype_hint_stat``
(``vsAll``) and its weighted twin. The annotation path has no ``max_dist``
parameter and cannot be given one; all k=50 neighbours vote however far away
they are. On this project's own queries, 1,568 of 49,456 (3.2%) sit beyond the
0.03 bar SCimilarity suggests elsewhere, and they are annotated at 0.338
accuracy against 0.466 for the rest -- so the signal is real and unused.

That is the surface worth attacking: not a filter that poison evades, but
diagnostics that are computed, written down, and ignored. Whether poisoning
moves them in the attacker's favour -- ``min_dist`` down, ``vsAll`` up -- is
untested and needs only those two columns added to the per-query scoring.

### 5.4bb Simultaneous multi-celltype attacks -- OUT OF SCOPE

**Ruled out 2026-09-24.** v01 measured ``combo_interference``: one adversary
poisoning 2, 3 or 4 celltypes at once, with a row per MEMBER so that a member's
result inside the combo sits beside its result run alone. v02 will not reproduce
it.

It is not one of the 19 attack types and never was. The taxonomy is
``scope x access x geometry x label x selection``; a combo is *n* simultaneous
instances of an EXISTING type, which is a sixth axis -- how many victims at once
-- rather than a new construction. Excluding it therefore leaves the attack
design complete rather than partial.

The reason is cost against value. The celltype-scope grid is already 16 types x
14 doses x 26 celltypes; adding a victim-count axis multiplies a design that is
about to take ~55 h to run once. And v01's result does not suggest a hidden
effect worth chasing: across 351 rows, ``net_delta`` is EXACTLY zero in 45% of
them and its median is 0.000000 at every combo size and every attack. Attacking
several celltypes together changes no member's outcome.

**What it does change is retrieval, and that is recorded here rather than
re-measured.** ``ret_delta`` is non-zero in 99.4% of those rows, and retrieval
roughly doubles: ``relabel`` at two members goes 0.095 alone to 0.176 together
(1.87x), ``duplicate`` 0.067 to 0.122 (1.81x). So a member's poison enters far
more neighbourhoods when other celltypes are attacked at the same time, and
converts none of it -- the sharpest instance in the project of retrieval failure
and outcome failure moving independently. A defence watching retrieval would see
a multi-target campaign as twice the threat; one watching outcomes would see
nothing. That belongs in the defence discussion as a v01 result, cited, not as a
v02 sweep.

**Consequence for the code.** ``make_poison.build_combo_poison``,
``POISON_MEMBER`` and ``tests/test_combo_poison.py`` exist and pass, and nothing
calls them. They stay: they are tested, they cost nothing to keep, and the
exclusion is a scope decision rather than a judgement that the code is wrong.
``schemas.FOCUS_SOURCE``'s docstring describes the per-member row shape and is
left intact for the same reason. Anyone reversing this decision inherits a
working constructor.

### 5.4c The threshold dose -- what the ladder is FOR

A ladder of fourteen doses is not itself a result. The result it exists to
support is a single number per condition: **the dose at which the attack starts
working**, with an interval on it. Without that, the sweep produces fourteen
rates and leaves the reader to eyeball where the curve turns, which is not a
quantity anyone can compare across attacks, across celltypes, or against a
defence.

Step ``02b_10_both_threshold_dose`` computes it. ``both``, because it is the one
step that spans the insert and modify arms -- that is the whole point of putting
them on one ladder.

**The quantity.** ``D_tau`` = the smallest dose at which the response exceeds an
absolute level ``tau``. Reported for a FAMILY of levels, not one:

    tau in {0.01, 0.05, 0.10, 0.25, 0.50}

Reporting the family is deliberate. A single threshold makes one arbitrary
constant load-bearing; five make the shape of the dependence visible, and a
reader who disagrees with the headline ``tau = 0.05`` can read their own off the
same table.

**Two responses, never collapsed.** ``D_tau`` is computed separately on
``retrieval`` (a poisoned record entered the top-k) and on
``net_attack_success`` (the vote moved to the attacker's label). These are
different thresholds answering different defence questions -- whether a filter
should act on retrieval or on aggregation -- and a single "threshold dose" that
merges them would destroy the distinction the project exists to preserve.

**No parametric curve fit, and this is a measured decision.** The obvious move
is a 4PL / Hill fit and an EC50. It is wrong for this family. In ``02b_01``,
``duplicate`` plateaus near 0.10-0.11 from dose 0.5 onward -- a Hill fit would
work. ``relabel`` reads 0.245 at dose 1.0 and 0.390 at 1.25 and is still
climbing at the top of the ladder: its asymptote has never been observed. Fitting
one would invent the plateau, and every EC50 derived from it would be an
extrapolation reported as a measurement. Instead:

1. Pool the resampled donors' counts at each of the 14 doses -- numerator and
   denominator as integers, so the pooled rate is ``sum(num)/sum(den)`` and
   donors are weighted by how many queries they actually contributed.
2. Fit a monotone-increasing **isotonic regression (PAVA)** across the 14 doses.
   Non-parametric, assumes only that more poison does not help the defender,
   and assumes NOTHING about a ceiling.
3. Locate the bracketing pair of doses where the isotonic curve crosses ``tau``;
   interpolate log-linearly within that bracket for the point estimate. The
   bracket itself is reported alongside (``d_lo_bracket``, ``d_hi_bracket``) so a
   reader can ignore the interpolation and use the purely empirical interval.

**Censoring is a RESULT, not a missing value.** Two conditions have no threshold
inside the ladder, and both are real findings that must not surface as ``NaN``:

``right``   the isotonic curve never reaches ``tau`` by dose 1.25. Reported as
            ``> 1.25``. This is the honest form of "this attack does not work",
            and by the standing rule it is published as prominently as a
            threshold that exists.
``left``    the curve already exceeds ``tau`` at dose 0.002. Reported as
            ``< 0.002``. This is the expected outcome for hub-selected modify,
            which ``02b_05`` measures at 0.694 traffic captured at dose 0.01 --
            already most of the way up at the lowest dose ever run. A left-
            censored threshold is the strongest result in the project, not a
            failure to measure one.

Every row carries ``censor`` in ``{none, left, right}``. A row with
``censor != none`` has ``d_tau = NaN`` BY CONSTRUCTION, and no figure, caption or
sentence may fill it in.

**The interval: donor-paired bootstrap, B = N_BOOTSTRAP (2000).**

The donor is the unit of independence -- ``schemas.py`` already says so
(``DONOR_ID``: 78 subjects in GSE136831, "17,571 macrophage queries from 78
donors are closer to 78 observations than 17,571"). So:

- Resample the 78 donors WITH REPLACEMENT. Resample seeds with replacement.
- **Hold the draw FIXED across all 14 doses within a replicate.** This is the
  load-bearing detail, and the reason is CORRECTNESS rather than tightness: the
  pairing makes the interval WIDER, not narrower. The same donors are scored at
  every dose, so the per-dose errors are positively correlated through a common
  donor effect. Pairing reproduces that correlation. Independent per-dose draws
  destroy it, letting the common donor effect average out across the ladder --
  and a crossing point is most sensitive to exactly that, a coherent shift of
  the whole curve. An unpaired bootstrap therefore UNDERSTATES the interval:
  measured at 0.21 against 0.16 at 78 donors, and narrower in all nine
  configurations tried (``test_unpaired_draw_understates_the_interval``). An
  earlier draft of this section asserted the opposite; the test is what caught
  it.
- Re-run steps 1-3 on each replicate; take the 2.5th and 97.5th percentiles of
  the 2000 thresholds.

**The interval can itself be censored**, and is reported that way. Each row
carries ``frac_censor_right`` and ``frac_censor_left`` -- the share of the 2000
replicates that had no crossing. Rule: if either exceeds 0.5, ``d_tau``,
``ci_lo`` and ``ci_hi`` are all ``NaN`` and only the censoring direction is
published. Between 0 and 0.5 the percentiles are taken over the UNCENSORED
replicates and the censored fraction is printed beside them, because an interval
computed from 1,200 of 2,000 replicates means something different from one
computed from all 2,000 and the reader is entitled to know which they are
holding.

**Dependency -- this step cannot run today.** ``D_tau`` needs a per-donor rate at
every dose, and no committed artefact has one: ``02b_01``'s graph table is
already aggregated past the donor, and ``v02/2_results/<DS>/00_raw/`` is empty.
The sweep must first emit

    02b_01_..._by_donor.parquet
    keys:   attack, source, target, dose, seed, donor_id, scope
    values: n_retrieval, n_success, n_queries   (integer counts, NOT rates)

Long format on every axis except ``scope``, which stays on the row per the
standing SCOPE exception. Counts rather than rates because the bootstrap must
re-pool them under a new donor draw, which a pre-divided rate cannot support.
Size is trivial -- 26 celltypes x 14 doses x 78 donors x 12 sweeps is under
350k rows -- and it is written by the sweep that already holds the per-query
data in memory, so it costs no extra retrieval.

**Ordering.** ``02b_10`` runs after the ladder sweep and before the figure
(its figure, numbered after the analysis steps). It reads ``by_donor`` only, computes the
thresholds, and writes ``02b_10_both_threshold_dose.parquet``:

    keys     attack, source, target, selection, geometry, label, tau, response
    values   within_d_tau, within_ci_lo, within_ci_hi, within_censor,
             within_d_lo_bracket, within_d_hi_bracket,
             within_frac_censor_left, within_frac_censor_right,
             and the identical global_* family

**Schema escalation.** None of ``d_tau``, ``censor``, ``tau``, ``response``,
``d_lo_bracket``, ``d_hi_bracket``, ``frac_censor_left``, ``frac_censor_right``
or ``n_retrieval``/``n_success``/``n_queries`` exists in
``1_scripts/00_common/schemas.py`` yet. They are orchestrator-owned additions and
land there before either script is written.

Separately, ``schemas.py`` currently documents ``CI_LO`` as "over donors and
seeds" and ``CI_HI`` as "over seeds" for the same interval. One of the two is
wrong; the code is the authority and the docstrings are corrected to match it in
the same change.

**Ordered work to reach symmetry.**

1. **A shared dose ladder.** Nothing else matters first, and it is now SETTLED
   -- the fourteen points of 5.4b, for every celltype-scope attack type. On the
   modify side a new dose is a re-vote over cached neighbours, seconds per
   condition; on the insert side it is 45.2 s measured. Converts one comparable
   point into fourteen, and discards no condition already run.
2. **``hub`` selection in the insert arm.** Both arms then share ``uniform`` and
   ``hub``. Hub targeting is worth 19x at fixed budget under modify (94.7% of
   retrieval traffic against 5.1%); whether it transfers to insertion is
   unmeasured, and it is the obvious way insertion could become cheap. If it does
   NOT transfer, "insertion is expensive" survives its strongest objection.
3. **``synthesised`` in the modify arm.** The only geometry present in one arm and
   absent from the other. Unlike ``relabel_inplace`` it moves vectors, so it needs
   the merge treatment v01 used for ``embed_shift`` rather than a plain re-vote.
4. **``keep`` in the insert arm.** Genuine records submitted under their own
   label: volume without the lie. It is the control for every insert result --
   the claim that harm comes from the false annotation rather than the added
   volume is currently asserted and not measured -- and it completes the
   ``label`` axis in both arms.

5. **Alpha for ``shifted`` + ``keep``, at 0.1 increments.** The claim "moving a
   vector while keeping its label does nothing" rests on ONE alpha (0.5, 936
   rows), and accuracy there is flat at 0.4648-0.4653 across nine budgets against
   a 0.4637 baseline. That measures alpha, not the threat model. Run alpha in
   {0.0, 0.1, ... 1.0} x dose {0.05, 0.25, 1.0} x {uniform, hub} x 26 celltypes
   = 1,716 conditions, merge cost. ``alpha = 0`` must reproduce the clean
   baseline EXACTLY, which makes the ladder self-checking.

   Prediction recorded before running, so the result can falsify it: flat to
   alpha ~0.5, and any fall concentrated between 0.8 and 1.0 at dose 1.0 --
   the only configuration where no same-celltype record remains to fill the
   vacated neighbour slot.

   Three attack types are DOWNSTREAM of this and marked ``NO -- pending`` until
   it runs: both atlas ``shifted / keep`` rows and ``synthesised / keep / hub``.
   If displacement stays flat at alpha 1.0, none of them is worth running.

**One asymmetry is principled and stays:** ``concentrated`` is insert-only, for
the reason given above. Every other blank in the grid is work not yet done.

### 5.5 One quantity, one name -- so the two arms can be read side by side

**The rule.** If two attack types do the same thing, the column that describes it
carries the SAME NAME in both. A reviewer should not have to learn that ``budget``
and ``rate`` are the same quantity, or that ``threat_variant`` and ``selection``
are the same axis. Symmetry in the design is worthless if the tables disguise it.

**Offences currently in the schema.**

``budget`` vs ``rate``. Both are "how much poison, as a fraction". The atlas arm
calls it ``budget`` and divides by 7,913,892; the celltype arm calls it ``rate``
and divides by the attacked celltype. Two names for one quantity, differing only
in denominator -- and the denominator is exactly what CLAUDE.md 10 and
HANDOFF 3.22 warn must never be left implicit.

  Fix: one column ``dose``, plus ``dose_basis`` in {``celltype``, ``atlas``}.
  ``n_poison`` and ``atlas_share`` stay as they are -- both already appear on
  every v02 row -- so a reader can always recover the other denominator without
  recomputing it.

``threat_variant`` vs ``selection``. ``02b_05`` carries BOTH, with identical
values (``oracle``/``split``/``random``). One is redundant.

  Fix: keep ``selection``, drop ``threat_variant``. ``selection`` generalises to
  the insert arm, where the choice is currently hidden inside the attack name
  (``relabel`` = uniform, ``duplicate`` = concentrated) rather than being a
  column at all.

``attack`` as a single opaque name. ``relabel``, ``duplicate`` and ``uniform``
are the same attack at three parameter settings, and nothing in the column says
so.

  Fix: ``geometry``, ``access``, ``label_change``, ``selection`` as separate
  columns, per 5.2's axes. "Show me everything unmoved" becomes one predicate
  instead of a list of names.

**Migration.** These are committed columns. Follow the precedent set when
``insert``/``modify`` became ``no_access``/``with_access``: change the schema,
leave v01 alone, and put the correspondence in ``schemas.py`` so a v01 table stays
readable from v02. ``check_schema.py`` already enforces one such translation
(``V01_VALUES``) and should enforce these too.

**Why this is a spec item and not a style preference.** The point of 5.4 is that
insert and modify become comparable. Two arms measured at the same doses, with
the same selection rules, reported in columns with different names, are still not
comparable to a reader -- only to whoever wrote the pipeline.

---

**The section below is superseded and kept for the record.** It was the plan
before the ontology graph was wired in and before the celltype set was derived
rather than listed. It specified 3 attacks, 7 rates, 6 hand-picked pairs on a
near/mid/far ladder and 10 seeds, for 1,260 conditions. None of those four axes
survives above. Nothing in it has been edited.

| Axis | Values | n |
|---|---|---|
| attack | `relabel`, `duplicate`, `interpolate` | 3 |
| rate | 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2 | 7 |
| pair | 6 (below) | 6 |
| seed | 0–9 | 10 |

**1,260 conditions.** Seeds are 10 because `ci_lo`/`ci_hi` bootstrap over seeds;
5 replicates produce an interval that mostly reports having 5 replicates.

Candidate pairs, chosen from CL terms present in the query set with enough cells
to attack. **Integer distances are placeholders pending the CL graph, and every
pair must also exist in the reference vocabulary — to be confirmed in stage 0.**

| Class | Source | Target |
|---|---|---|
| near | `CL:0000235` macrophage | `CL:0000583` alveolar macrophage |
| near | `CL:0000910` cytotoxic T cell | `CL:0000815` regulatory T cell |
| mid | `CL:0000235` macrophage | `CL:0000860` classical monocyte |
| mid | `CL:0000084` T cell | `CL:0000236` B cell |
| far | `CL:0000235` macrophage | `CL:1001567` lung endothelial cell |
| far | `CL:0000910` cytotoxic T cell | `CL:0002063` type II pneumocyte |

---

## 6. Schemas

Column names, dtypes and enums live in `contracts/schemas.py`, per `CLAUDE.md`
§5, and are imported by every lane. This spec adds two columns to the aggregated
table beyond CLAUDE.md §5:

- `clean_error_rate` — required by the baseline obligation in §1.
- `qc_evasion_rate` — required by `fig03`, which the figure agent's
  brief lists as a core figure but which no existing column supports.

> **Orchestrator action.** Both are aggregate-level only and derivable from
> existing per-cell columns; no change to `raw_result` is needed.

---

## 7. Build order

1. `contracts/schemas.py`, `config/config.yaml`, both profiles.
2. Stages 3–5 with golden-value tests on hand-built fixtures. *No data needed.*
3. Stages 0–2 once the tarball lands and verifies.
4. First `smoke` run to `results/summary.parquet`; `docs/RUNLOG.md` entry.
5. Figures 1–3. Stage 6 and `fig04` after.

---

## 8. Open questions

1. Does the tarball ship a Cell Ontology graph, or is `cl.obo` a stage-0 input?
2. Do all six pairs exist in the reference label vocabulary, with enough source
   cells at rate 1e-2 (79,000 rows at `full`)?
3. `interpolate` moves a source vector toward the target — toward the target
   **centroid**, as written above, or toward a sampled target cell? Centroid is
   the stronger attack and the more reproducible one; sampling is more realistic.
   Recommend centroid, revisit if it saturates.
4. Ties in the k-NN vote are broken by lowest CL id. Deterministic and seed-free,
   but arbitrary. Alternative: nearest-neighbour distance. Recommend as written.
5. **Naming hazard, flag before `schemas.py` is written.** `CLAUDE.md` §5 defines
   `min_poison_score` as "SCimilarity score of nearest poison". Because
   `score = 1/(1-c)` increases with similarity, the score of the *nearest* poison
   is the **maximum** score over the poison set, not the minimum. The name reads
   as the opposite of what it holds. Semantics follow CLAUDE.md — nearest poison
   — and the docstring in `schemas.py` must say so explicitly, or someone will
   eventually implement `min()` and produce a QC evasion rate that is wrong in a
   way that looks plausible.

---

## 9. Deviations from CLAUDE.md — orchestrator edits required

Consolidated, so the contract owner can act on them in one pass.

| # | CLAUDE.md says | This spec does | Why |
|---|---|---|---|
| 1 | §8 `smoke` = "synthetic 500k reference, no downloads" | 500k fixed-seed subsample of the real reference; stage 0 runs once | Real CL labels, same size and runtime, no generator to maintain and later discard |
| 2 | §5 `summary` adds 4 columns | adds 6: `clean_error_rate`, `qc_evasion_rate` | Baseline obligation (§1) and `fig03`, which the figure brief requires but no column supported |
| 3 | §3 tree shows `config/profiles/smoke.yaml` as a file | `config/profiles/smoke/config.yaml` — a directory | `snakemake --profile` takes a directory, as §8's own command line does. The tree and the command contradict each other; the command is correct |
| 4 | §5 names `min_poison_score` | kept verbatim, documented as holding the **maximum** score | Renaming breaks the contract; leaving it undocumented invites a wrong number that looks right |


---

## 10. Resolutions of contract ambiguities

Raised by the doc lane reading the code and the contract side by side.

**`alpha` direction.** Larger `alpha` means closer to the *target*. §4 stage 3
previously said the opposite. The implementation and its monotonicity test are
authoritative; the spec text was wrong and is corrected above.

**`provenance` under `interpolate`.** `CLAUDE.md` §5 permits "source reference
cell ID, or `synthetic`". An interpolated vector *is* derived from one real
source cell, moved toward a target centroid — so the source id is both
permitted and strictly more informative than the constant `synthetic`, which
would discard the only traceability the record has. All three attacks therefore
record the source cell id. `synthetic` remains reserved for a future attack that
fabricates vectors from nothing.

**`SCORE_QC` is measured, never applied.** Nothing in this pipeline drops a hit.
The sweep records `poison_survives_qc` and votes over every neighbour. The
threshold exists so we can report what SCimilarity's own filter *would* have
caught — applying it ourselves would answer a different question and hide the
evasion rate we are trying to measure.

---

## 11. Capability ladder

Adopted from the protein arm of BioRAG-Guard, where mixing tiers on one
unlabelled curve is explicitly prohibited. The rung is a labelled axis, not a
hidden assumption.

**Replaces the binary realistic/ceiling tier — revised 2026-09-22.** The binary
said whether an attack was realistic; it did not say *which* capability made it
unrealistic, and that is the only part a reader can act on. `flood` needs the
attacker to write embedding coordinates. `embed_to_query_ceiling` needs that
*and* sight of the held-out queries. Those are different assumptions, and a
deployment may grant one and not the other. The ladder names the capability so
the reader decides which rungs their own ingestion path allows.

| Rung | The attacker can write | no_access attacks | with_access attacks |
|---|---|---|---|
| **0 — annotation** | the label only; vectors are genuine reference embeddings, unmodified | `relabel`, `duplicate` | `relabel_inplace` |
| **1 — coordinates** | embedding coordinates of their choosing, computed from public data | `interpolate`, `flood` | `embed_shift_relabel`, `embed_shift_keeplabel` |
| **2 — coordinates + query knowledge** | coordinates, *and* they know where the queries land | *(none built)* | `embed_to_query_ceiling` |

### What each rung asks of a deployment

**Rung 0** is available to anyone who can get a record accepted. No geometric
check can act: the vectors are genuine, so re-deriving embeddings from raw counts
reproduces them exactly and a checksum over the index still passes. Only
label-level reasoning is left — label consistency, donor voting, volume caps.

**Rung 1** requires writing coordinates no count matrix produces. The question a
reader must answer about their own system is whether the ingestion path re-derives
embeddings from submitted raw data or accepts submitted vectors. If it re-derives,
rung 1 is closed and every number at that rung is inapplicable to them.

**Rung 2** additionally requires knowing the query workload. Nothing stored in the
reference can defend against it, because the attack is defined relative to data
the reference does not contain. Its role is to bound the axis; see §12.

### What each rung is worth — measured

Mean net attack success, v02, 26 celltypes, matched rate (`02b_01`, `02b_02`):

| rate | rung 0 best | rung 1 best (`flood`) | ratio |
|---|---|---|---|
| 0.5 | 0.118 | 0.271 | 2.3x |
| 0.8 | 0.143 | 0.327 | 2.3x |
| 0.9 | 0.125 | 0.355 | 2.8x |
| 1.0 | 0.340 | 0.342 | 1.0x |

**Coordinate control is worth 2-3x below parity and nothing at parity.** That is
the finding the ladder exists to produce, and neither rung states it alone.
Reading rung 0 by itself gives "an insertion attacker needs the whole celltype",
which understates the risk by a factor of three at half the budget.

For the with_access arm, rung 1 to rung 2 (v01, exact search, pre-unified vote --
levels indicative, not quotable as v02 numbers): `relabel_inplace` reaches 0.469
at full budget against `embed_to_query_ceiling`'s 0.774. The ceiling is
near-flat in budget, reaching 0.564 at rate 0.002 and gaining 21 pp over a 500x
budget increase, which is itself the result: for an unconstrained attacker the
binding constraint is placement, not budget.

### A rung is permission, not achievement

Carried forward from 2026-08-28 and still the most important caveat here. The
rung states what the attacker is *permitted* to do, never how much damage they
achieve. `interpolate` sits at rung 1 and achieves 0.004 -- it drags vectors
toward the target centroid while the queries under attack are *source* cells, so
it lands away from the neighbourhood it must occupy. A rung-1 attack can lose to
a rung-0 one, and `interpolate` does. No figure may imply that a higher rung is a
higher number.

**Open question of 2026-08-28, now closed.** That entry asked whether to add a
query-targeted construction, noting a genuine upper bound "would place synthetic
vectors at the query positions, labelled as the target. That is untested here."
It was subsequently built and run in v01 as `embed_to_query_ceiling`, and it is
rung 2 above. The mentor's `direct_vector` tier corresponds to rung 1.

**Reporting rule.** Reporting a higher-rung number beside rung-0 ones without
saying so converts an unfalsifiable "attacks are possible" into something that
reads as measured risk. Every table carries the rung; no lane restates it.

**Pending code change.** `1_scripts/00_common/schemas.py` still encodes the
binary `AttackTier` (`realistic` / `ceiling`). The ladder above is the contract;
the enum and the `attack_tier` column follow in an orchestrator edit, and until
they do, a table's `attack_tier` value is the binary and this section is the
authority. `flood` and `interpolate` both map to rung 1; `relabel` and
`duplicate` to rung 0.

## 12. Prohibited claims

Pre-committed, following the protein arm's practice of stating what the work
does not claim.

- **Do not claim any named atlas is vulnerable.** The tested system is
  SCimilarity's encoder and reference with k-NN label transfer, on one tissue.
  CELLxGENE, HLCA and every other named resource are out of scope.
- **Do not claim SCimilarity misannotates cells.** The vote here is ours, not
  its shipped `get_predictions_knn`. The claim is about a SCimilarity-style
  retrieval system, not the packaged tool end to end.
- **Do not present `interpolate` results as realistic risk.** See §11.
- **Do not claim a wrong label implies a wrong biological conclusion** without
  showing the downstream analysis actually changes. Cell-type calls feed
  differential expression, composition tests and trajectory inference; none of
  that has been measured here.
- **Do not quote outcome failure without the clean baseline.** It sits near 53%
  before any poisoning, because query and reference annotate at different
  granularities.
- **Do not quote intervals as if cells were independent.** The unit of
  independence is the donor: 78 subjects, not 49,456 cells.
- **RETRACTED 2026-09-10 — do not claim donor voting reduces attack success.**
  The "~24x at zero false positives" figure held only because
  `run_defence_table.py` assigned every poisoned record the same donor string,
  collapsing the whole poison set to one vote by construction. An attacker
  supplies their own sample metadata and can hold as many fabricated donors as
  they like. `results/defence_identity_sweep.parquet` sweeps that: against
  B cell / `duplicate` the defence is **worse than no defence at n >= 10
  identities** (16.5% vs 11.98%), reaching 20.95% at n = 500. Against
  `relabel` it turns a 0.00% attack into **37.82%**.
  The mechanism is arithmetic: donor voting deduplicates the *genuine*
  neighbours too, and a real 50-neighbour set holds only **12-17 distinct
  donors** (measured). An attacker with more identities than that keeps more
  votes than the honest cells do.
  **What may still be claimed:** donor voting raises clean accuracy 0.816 ->
  0.931 at zero false positives on unpoisoned data, because it corrects the
  6.5% duplicate records already in the shipped reference. That is a curation
  fix and is independent of any attacker. Say curation, never defence.

- **Do not quote a graph OUTCOME difference smaller than the reproducibility
  band** — added 2026-09-28. `hnswlib.add_items` defaults to `num_threads=-1`
  and concurrent insertion is scheduler-dependent, so a `graph` run is not
  reproducible in its outcome metrics. `02b_34` measures the band: retrieval
  rates move by exactly 0.000000, `within_accuracy` and
  `within_net_attack_success` by up to 0.0168. Retrieval numbers may be quoted
  as measured; outcome numbers carry **±0.02 within-celltype and ±0.001
  global**. A claimed effect below that is not distinguishable from the thread
  schedule. See HANDOFF §3.41.
- **Do not quote a difference between two of your own runs unless both are
  committed** — added 2026-09-28. `02b_30` ran twice seven minutes apart and the
  two disagree by up to 0.019; only the second is committed, so any A-vs-B
  quantity is not re-derivable from `2_results/` and is therefore a fabricated
  number under CLAUDE.md §1, however honestly it was observed.
- **Do not describe spillover by grouping on the author's label** — added
  2026-09-28. Both ends of a before/after comparison must be *predictions*.
  Grouping changed queries by `y_true` inflates apparent spillover by 2.4x and
  counts the model's own pre-existing granularity disagreement as attacker
  damage. Retracted once in HANDOFF §3.40 and reproduced in a new figure eleven
  days later; §3.42 carries the corrected table.

### Novelty claims — added 2026-08-28 after a literature check

Six bodies of prior work bear on this project. None of them cover what it does,
and all six must be cited rather than ignored. The novelty claim is
correspondingly narrow. Full review and sources in
[`docs/RELATED_WORK.md`](docs/RELATED_WORK.md).

- **Do not claim poisoning of single-cell models is unstudied.** *Unveiling
  potential threats: backdoor attacks in single-cell pre-trained models*
  (Cell Discovery, 2024; `BioX-NKU/scBackdoor`) poisons training data and
  fine-tunes scGPT, Geneformer and scBERT, reaching 97.6–100% attack success.
  Their attack requires retraining and a trigger present in the query at
  inference; ours requires neither, and no retrieval appears anywhere in their
  paper.
- **Do not claim corpus poisoning of a retrieval system is novel.** It is an
  established text-domain field — PoisonedRAG, BadRAG and a substantial defence
  literature. What is untested is the *modality*: a single-cell embedding atlas
  rather than a text corpus.
- **Do not claim the retrieval/outcome dissociation as a discovery.** The
  robustness of k-NN majority voting to data poisoning is a known result with
  certified bounds (Jia et al., *Certified Robustness of Nearest Neighbors
  against Data Poisoning and Backdoor Attacks*, AAAI 2022). Our measurement is
  consistent with it and gives it a domain-specific magnitude; it does not
  establish it. Cite the theory, then report what it does not predict.
- **Do not claim label-flipping poisoning is new.** It is a classical attack
  with its own literature.
- **Do not claim adversarial attacks on single-cell annotation are unstudied.**
  *adverSCarial* (Bioinformatics, 2025) attacks scRNA-seq classifiers — scType,
  CHETAH, scAnnotatR and others — and is the closest prior work to this project.
  It perturbs the **query cell**; we leave the query untouched and perturb the
  **reference**. Evasion versus poisoning. This is the right comparison for an
  introduction, ahead of scBackdoor.
- **Do not claim reference-atlas quality problems are unreported.** Contamination
  and low-quality cells in the Human Cell Atlas, Tabula Sapiens and Tabula Muris
  are documented (*BMC Genomics*, 2024). The duplicate-record finding is a
  specific instance, and its value here is as the **mechanism explaining why
  donor voting improves clean accuracy** — not as a discovery about atlas
  quality.

**What may be claimed as new**, pending peer review, is domain-specific and
none of it follows from the generic theory: vulnerability varying ~10× by
attacked cell type and unexplained after three rejected hypotheses; ontology
distance as a measured axis; evasion measured against the vendor's own shipped
QC threshold rather than a hypothetical filter; the 6.5% redundant records
found in the released reference; and donor-level voting as a defence that
exploits biological metadata structure and *improves* clean accuracy.
