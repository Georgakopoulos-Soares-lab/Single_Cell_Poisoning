# Stage 6 — `validate_shortcut`, in v02

**Status: never run in v02, and this is defensible rather than outstanding.**
There is no `shortcut_validation.parquet` under `v02/2_results/`, and none of
the 39 logs names `validate_shortcut`. The only stage 6 run in this repository
is v01's — Runs 3–4 of 2026-08-26/27, 96 conditions.

`CLAUDE.md` §6 used to say "Stage 6 is not optional. Publishing sweep results
without it is not permitted." Read literally, that made every v02 figure
unpublishable. The clause was reconciled on 2026-09-28, and the reconciliation
is the substance of this file.

## The obligation, stated precisely

The merge-and-rerank shortcut is exact for exhaustive search but only
approximate under HNSW, because a poisoned node can alter graph traversal. So:
**a result produced by the shortcut may not be published until something has
measured the shortcut against a real rebuild.**

That is the rule. What changed is the observation that in v02, **nothing
published rests on the shortcut** — so the rule is satisfied vacuously rather
than waived.

## Why it does not bite here

Verified 2026-09-28 by listing every parquet each figure script opens by name
and reading its `retrieval_path`. Not one carries `exact`.

**Most v02 results never took the approximation.** `02a_01`, `02b_01`,
`02b_02`, `02b_13`, `02b_14`, `02b_30`, `02b_34` (`hnsw`) and `02b_11_graph`,
`02b_15_graph` (`graph`) insert poison into the shipped `labelled_kNN.bin` and
re-query it. There is no approximation to validate because none was taken.

**`02a_03` and `02b_05` are exact by construction.** They are label-only edits.
No vector moves, so the index is byte-identical and the cached clean neighbours
*are* the poisoned ones. `cached_clean` is not a shortcut; it is the same
computation with a proof that it need not be repeated.

## `02b_16` corroborates this, and is supplementary rather than load-bearing

The point above alone discharges the obligation. `02b_16` is the methods result
that measured the shortcut anyway — and at a far larger scale than stage 6 ever
did: **2,964 conditions × 2 metrics = 5,928 rows**, against stage 6's 96.

From `02b_16_with_access_exact_vs_graph.parquet`:

| step | metric | n | identical | \|diff\| > 0.01 | mean diff |
|---|---|---|---|---|---|
| `02b_11` | `within_accuracy` | 2,028 | 44.0% | 11.4% | −0.00146 |
| `02b_11` | `within_retrieval` | 2,028 | 60.5% | 15.6% | −0.00112 |
| `02b_15` | `within_accuracy` | 936 | 41.2% | 9.1% | −0.00110 |
| `02b_15` | `within_retrieval` | 936 | 46.4% | 13.4% | −0.00393 |

**This retracted a standing claim.** The earlier position — HANDOFF §5 before
2026-09-27 — was that the `exact` tables "understate the attack by an unknown
per-celltype margin", based on four conditions. Over the full grid the
disagreement runs in **both** directions and nearly cancels: the graph attack is
stronger in 33.4% of `02b_11`'s accuracy conditions and **weaker in 22.6%**.
`remerge` does not systematically understate.

The case for running faithfully survives on different grounds: the disagreement
is concentrated and unpredictable, and nothing tells you in advance which
celltype will disagree. But the old wording was wrong and `docs/RUNLOG.md`
Run 26 records the retraction.

## What is still required

Any **new** v02 result resting on the shortcut needs the same treatment before
publication — measured against the faithful path over the conditions it covers,
not a spot check.

Better: run faithfully in the first place. It costs about **12.8× per
condition** and removes the question entirely. Every v02 step since 2026-09-26
does this.

## The honest caveat

Faithful retrieval removed the shortcut question and introduced a different one.
The `hnsw` and `graph` paths insert with `add_items`, which is non-deterministic
by default — so v02 traded an approximation it could bound for a variance it
has now measured (stage 4's v02 file, HANDOFF §3.41). Retrieval rates are
unaffected; outcome rates carry ±0.02 within-celltype. Neither problem is
hidden, and they are not the same problem.
