# Stage 3 — `make_poison`, in v02

**Status: the concept survived; the implementation grew from three attacks to a
family, and moved out of the DAG into `00_common/attacks.py`.**

## What changed

v01's stage 3 built `relabel`, `duplicate` and `interpolate`. v02 has one entry
point — `attacks.py`, "one entry point that builds any attack's poison set" —
behind which sit the modules the attack family needs:

| module | what it owns |
|---|---|
| `attacks.py` | the entry point; dispatches on attack name |
| `selection.py` | **how an attacker chooses which records to draw from** |
| `modify.py` | overwriting existing records, embeddings and all (SPEC.md §5.4) |
| `atlas_attack.py` | choosing records and forging labels at atlas scope |
| `flood_common.py` | the cluster-flood attack's geometry |
| `ontology.py` | Cell Ontology parsing and tree distance |

**`selection.py` is the module v01 did not have and the project's central
result rests on.** v01 varied dose. v02 varies dose *and selection*, and
selection dominates: hub selection captures 36.5% of a celltype's top-50 traffic
for 0.2% of its records — 182× its budget — while random selection captures
exactly its dose (`02b_24`). A benchmark that only sweeps rate is measuring the
weaker axis.

## Poison lives at the SHARED cache tier, and that is a decision

`cache/poison/` sits at the repository root, not under `cache/<DATASET>/`,
because **poison is drawn from REFERENCE cells**. The same
`(attack, pair, rate, seed)` therefore produces byte-identical poison whatever
is being queried. Sharing it makes a cross-dataset comparison *exact* rather
than merely matched — the two datasets are attacked with the same records, not
with equivalent ones.

13,142 entries are on disk. It is gitignored: poison is recomputable from the
reference and the seed, and that is the test of whether something belongs in
`cache/`.

## Seeds, and the digest

Seeds are passed in, never generated — `np.random.default_rng(seed)` locally, no
module-level RNG (`CLAUDE.md` §7). v02 adds a **`poison_digest`** column so a
repeat can prove it used the same poison rather than merely the same seed.
`02b_34` relies on it: its claim is that six repeats differ *only* in insertion,
and the digest is what makes that checkable instead of assumed.

## The knobs have interior optima, which is a finding about the attacker

v01 fixed the geometry knobs at whatever values it happened to use. v02 swept
them, and three settings turned out to be wrong:

- **Flood spread**: the committed default of 0.99 delivers **61%** of what 0.999
  achieves. Every flood number in the project was produced at three-fifths
  strength (`02b_17`).
- **Jitter**: never helps, at any level. 0.2775 → 0.1326 → 0.0000 as it grows
  (`02b_17`).
- **Displacement alpha**: peaks at 0.05 and collapses by 0.5. `02b_11` fixed
  alpha at 0.5 and reported the attack inert; it was measuring one setting off
  the end of its own ladder (`02b_18`).

An attack reported as a null at one parameter setting is not a null. Stage 4's
v02 file carries the same caution for the metric side.
