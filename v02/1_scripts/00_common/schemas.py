"""Column names, dtypes and enums shared by every lane of biorag-guard-sc.

Orchestrator-owned (``CLAUDE.md`` §4). Pipeline, figure and doc lanes import
from here so that nobody hardcodes a column name and all three agree without
coordinating. This bottleneck is deliberate.

Nothing in this module reads a file, computes a statistic, or holds state. It
declares the contract and the constants the contract is defined in terms of.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Fixed constants (SPEC.md §3)
# ─────────────────────────────────────────────────────────────────────────────

EMB_DIM: Final[int] = 128
"""Embedding dimension. Reference vectors are unit-normalised onto S^127."""

K_NEIGHBORS: Final[int] = 50
"""Neighbours in the k-NN vote. Also the number retrieved in stage 2, which
SPEC.md §4.2 proves is sufficient rather than approximate under exhaustive
search."""

SCORE_QC: Final[float] = 33.0
"""SCimilarity's shipped similarity bar, borrowed from cell *search*.

Provenance, verified against scimilarity 0.4.0 source rather than assumed:
``max_dist = 0.03`` is the default of ``CellQuery.search_exhaustive`` and its
two centroid variants (``cell_query.py``), where it genuinely drops hits
(``nn_idxs[row].append(nn_idx[hits])``). ``33.0 ~ 1/0.03`` converts it to the
score scale. Note 1/0.03 = 33.33, so 33.0 is marginally *more permissive* than
the vendor bar (cosine 0.969697 vs 0.97); the gap is ~1% and moves no result,
but it is a deliberate rounding, not an accident.

**The annotation path does not apply it.** ``CellAnnotation.get_predictions_knn``
and ``annotate_dataset`` take no ``max_dist`` argument at all — there is no
distance filter to enable, and every one of the k neighbours votes. The
approximate search methods (``search_nearest`` and friends) default it to
``None``, i.e. off. So the bar fires only in the exhaustive search methods.

We therefore apply it as a **counterfactual**: "would the vendor's own bar have
caught this poison, had the annotation path had one?" That is what
``poison_survives_qc`` records.

**Nothing in this pipeline drops anything.** The sweep votes over every
neighbour. Applying the threshold ourselves would both misrepresent SCimilarity
and hide the evasion rate we exist to measure. Do not implement a drop to make
the code match a misread of this constant.

**DO NOT CALL THIS "QC" IN PROSE. Established 2026-09-25 by reading the source.**

0.03 really is a QC constant in scimilarity -- ``utils.pseudobulk_anndata``
takes ``qc_filters={"mito_percent": 20.0, "min_counts": 1000, "min_genes": 500,
"max_nn_dist": 0.03}`` and drops rows on each. So the constant's provenance is
not in doubt.

What is in doubt is the DIRECTION. Every QC scimilarity ships is QUERY-side:

* ``pseudobulk_anndata`` filters YOUR annotated cells, dropping any whose
  nearest reference neighbour is further than 0.03 -- its docstring says "max
  nearest neighbor distance to a reference label FOR PREDICTED LABELS", i.e. it
  discards annotations it cannot trust.
* ``search_exhaustive(max_dist=0.03)`` drops returned hits, at query time.
* ``qc_stats["query_coherence"]`` reports whether YOUR query set clusters
  coherently. This is the only thing scimilarity itself names ``qc``.

**Nothing scimilarity ships ever inspects the reference.** The atlas is treated
as ground truth. So a poisoned record is not "caught" or "missed" by QC -- it is
never looked at, and that absence is the finding rather than an evasion rate.

What :data:`qc_pass` measures is the distance from a QUERY to its nearest
POISON. That is a retrieval property, not a verdict on the poison's quality. A
caption saying the poison "survived QC" claims a test that was never run, and
collides with the vendor's own use of the word. v01's ``fig03_qc_evasion`` said
exactly that; its NUMBER was sound and honestly caveated, its LABEL was not.

Say "within scimilarity's search distance bar", or name the quantity. The
columns keep the ``qc_`` prefix because the tables are committed and renaming a
column breaks its readers -- the fix is to the language, not the schema."""

COSINE_QC: Final[float] = 1.0 - 1.0 / SCORE_QC
"""The QC threshold expressed as cosine similarity: 0.969697.

Ranking is done on cosine (monotone with score, and cheaper); the threshold is
*defined* on the score, per ``CLAUDE.md`` §1. Both forms are given so no lane
has to re-derive one from the other and risk disagreeing at the boundary."""

N_BOOTSTRAP: Final[int] = 2000
"""Bootstrap resamples for ci_lo / ci_hi. Resampling is over SEEDS, not cells:
seeds are the independent replicates."""

EMB_DTYPE: Final = np.float32
"""float32 throughout. float64 doubles memory on large arrays for no benefit."""


def scimilarity_score(cosine: np.ndarray) -> np.ndarray:
    """SCimilarity score from cosine similarity: ``1 / (1 - c)``.

    Defined here, in the contract, because the pipeline computes it and the
    docs quote it; two definitions would eventually disagree.

    Args:
        cosine: (...,) float32, cosine similarity in [-1, 1].

    Returns:
        (...,) float32 score. Monotone increasing in ``cosine``.

    Note:
        ``c == 1`` yields ``inf``. This is reachable, not hypothetical: the
        ``duplicate`` attack copies reference vectors exactly. ``inf`` is a
        legal float32 and compares correctly against :data:`SCORE_QC`. Do not
        clip it and do not guard it — see ``CLAUDE.md`` §7 on failing loudly.
    """
    return (1.0 / (1.0 - cosine.astype(EMB_DTYPE))).astype(EMB_DTYPE)


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class Attack(StrEnum):
    """Poisoning strategies. Ontology distance is a separate axis, not a
    strength parameter of these (``CLAUDE.md`` §1)."""

    RELABEL = "relabel"
    """Reference vectors kept verbatim, attacker-chosen label attached."""

    DUPLICATE = "duplicate"
    """Exact copies of reference vectors. Produces cosine == 1 against the
    source cell, hence infinite score."""

    INTERPOLATE = "interpolate"
    """Source vector moved toward the target, then re-normalised. ``alpha``
    records how far. Note it aims AWAY from the source, which is why it is
    rarely retrieved by the source's own queries."""

    FLOOD = "flood"
    """Anchors drawn from the source pool, rotated ``spread`` toward the pool's
    OWN centroid, then scattered with ``jitter``. Unlike the other three it stays
    near the source manifold -- where the queries are -- while the jitter defeats
    near-duplicate screening.

    It carries an extra axis the others do not: ``n_anchors``, i.e. how many
    clusters the budget is split across. v01 measured net attack success rising
    monotonically with it -- 0.002 at 1 anchor, 0.295 at 1,024, 0.400 at 16,384,
    which was the single best attack condition in the project. Any flood result
    is therefore meaningless without its anchor count, so ``n_anchors`` is a
    required column wherever ``attack == FLOOD`` and null elsewhere.

    It was NOT an Attack member in v01 -- it lived in ``flood_common`` with its
    own scripts and its own tables, purely because ``build_flood`` takes three
    parameters the others do not. That is a weak reason for a taxonomy to bend,
    and it made the strongest attack in the project the hardest one to compare
    against the rest."""


class ThreatModel(StrEnum):
    """Whether the attacker can WRITE to the reference, or only submit to it.

    An axis, not a migration: both arms are required, and the CONTRAST is the
    result. Reporting either alone overstates or understates the risk, because
    the two face different ceilings (``HANDOFF.md`` §3.24).

    Every analysis table carries this column. It is defined here rather than
    written as a literal in each script so that the two arms cannot drift into
    disagreeing about what they mean.

    .. admonition:: Reading a v01 table

       **v01 spells these values differently.** It used ``insert`` and
       ``modify``, naming the MECHANISM; v02 uses ``no_access`` and
       ``with_access``, naming the CAPABILITY the attacker needs, which is the
       thing a reader actually wants to know. The concepts are identical and
       the correspondence is exact:

       ==================  ==================  ====================================
       v01 value           v02 value           same thing?
       ==================  ==================  ====================================
       ``insert``          ``no_access``       yes -- records appended, never edited
       ``modify``          ``with_access``     yes -- existing records overwritten
       ==================  ==================  ====================================

       v01 is frozen and its tables keep the old values; they are read through
       v01's own copy of this file, which still defines them. Translate with the
       table above when comparing a v01 number against a v02 one -- nothing else
       about the arms changed."""

    NO_ACCESS = "no_access"
    """The attacker cannot write to the reference. They can only get records
    ACCEPTED into it -- through the ingestion path, or by a maintainer taking a
    submission. The genuine records remain and keep voting, so the attacker only
    ADDS a wrong vote. Two ceilings follow: the genuine twin survives, and only
    own-class neighbour slots are contestable. Measured cost: ~100% of a class
    (poison records equal to class size) for 42.0% attack success
    (``HANDOFF.md`` §3.32, recorded there as ``insert``).

    Against a graph-backed index this arm MUST alter the graph -- hnswlib
    returns only nodes it contains, so unindexed poison is never retrieved and
    the attack is inert. That is why it costs ~57 s per condition to measure
    honestly, against ~0.3 s for the merge shortcut."""

    WITH_ACCESS = "with_access"
    """The attacker can OVERWRITE records already in the index -- label,
    embedding, or both. Editing REMOVES a correct vote as well as adding a wrong
    one, so it escapes both ceilings above. Applied in memory only: the
    reference on disk is read-only, and §3.27's guarantee that no attack has
    ever mutated it must survive.

    Relabelling in place leaves every embedding where it was, so
    ``labelled_kNN.bin`` is BYTE-IDENTICAL and a checksum on it still passes.
    Only ``reference_labels.tsv`` differs, and nothing in SCimilarity checks it
    against anything. The graph is therefore queried once and reused across
    every condition -- this arm runs on the real index at full grid density for
    the cost of a single query."""


class ThreatModelVariant(StrEnum):
    """How the attacker spends a MODIFY budget.

    Separated from :class:`ThreatModel` because the choice of WHICH records to
    edit matters more than how many, and conflating them would hide that."""

    RANDOM = "random"
    """Records chosen uniformly at random within the class. The naive baseline;
    the number to beat, not a realistic attacker."""

    HUB = "hub"
    """Records ranked by retrieval frequency -- how many query-neighbour slots
    each occupies -- and overwritten from the top. Retrieval is extremely
    concentrated, so this is the efficient attack and the one a real adversary
    would find."""


#: Column name for :class:`ThreatModel`. Present on every analysis table.
THREAT_MODEL: Final = "threat_model"

#: Column name for :class:`ThreatModelVariant`. Empty string where it does not
#: apply -- an INSERT row has no edit-selection strategy.
THREAT_VARIANT: Final = "threat_variant"


class AttackTier(StrEnum):
    """What kind of claim a construction supports.

    Mixing tiers on one unlabelled curve overstates realistic risk, so the tier
    is a labelled axis rather than a hidden assumption."""

    REALISTIC = "realistic"
    """An attacker submitting genuine records could do this. `relabel` and
    `duplicate` both consist of real reference embeddings; nothing about the
    vectors is fabricated, only the labels attached to them."""

    CEILING = "ceiling"
    """An upper bound, not a threat. `interpolate` synthesises vectors that no
    count matrix produces, so any defender who re-derives embeddings from raw
    data rejects them outright. Its value is the GAP: the distance between the
    ceiling and the realistic tiers measures exactly what re-derivation buys."""


class DefenceTier(StrEnum):
    """What kind of claim a DEFENCE result supports.

    The mirror of :class:`AttackTier`, and needed for the same reason. An
    unlabelled attack ceiling flatters the attacker; an unlabelled defence
    ceiling flatters the defender. Both are unfalsifiable, and a reader cannot
    tell either from a measurement by looking at the number.

    As with attacks, the value is the GAP: the distance between a ceiling layer
    and a realistic one measures what the missing capability would be worth if
    someone built it."""

    REALISTIC = "realistic"
    """A defender holding only what SCimilarity ships can run this. The
    annotation bundle is an HNSW graph and a label column; ``cache/reference.npz``
    adds embeddings and cell ids. Anything computable from vectors and labels
    alone qualifies."""

    NONE = "none"
    """Not a defence. The undefended control, which carries no tier claim."""

    CEILING = "ceiling"
    """An upper bound on what the layer would buy, not a measurement of what it
    does buy. Either the admission decision is asserted rather than computed, or
    the layer needs information the shipped artefact does not carry. Reporting
    one beside a realistic layer without saying so converts "we have a defence"
    into a claim the evidence does not support."""


DEFENCE_TIER: Final[dict[str, str]] = {
    # Layer names drifted across scripts written at different times. Both
    # spellings map to the same tier rather than being renamed, because the
    # tables are committed and renaming a value breaks their readers.
    "none": DefenceTier.NONE,
    "label consistency": DefenceTier.REALISTIC,
    "labelcons": DefenceTier.REALISTIC,
    "de-duplication": DefenceTier.REALISTIC,
    "dedup": DefenceTier.REALISTIC,
    "re-encode": DefenceTier.CEILING,
    "recompute": DefenceTier.CEILING,
    "donor vote": DefenceTier.CEILING,
    "donor": DefenceTier.CEILING,
    "quarantine": DefenceTier.CEILING,
    "volume cap": DefenceTier.CEILING,
    # Composites inherit the weakest tier they contain: a stack is only as
    # deployable as its least deployable layer.
    "full stack": DefenceTier.CEILING,
    "recompute+labelcons+donor": DefenceTier.CEILING,
    "quarantine + donor vote": DefenceTier.CEILING,
    "donor vote (n=50)": DefenceTier.CEILING,
    # Vote weighting, not a filter: it needs only the cosines already in hand,
    # so it is deployable. Deployable is not the same as useful -- HANDOFF 3.12
    # found it inert -- and the tier says nothing about whether a layer works.
    "uniform k=50": DefenceTier.REALISTIC,
    "uniform k=10": DefenceTier.REALISTIC,
    "rank k=50": DefenceTier.REALISTIC,
}
"""Which tier each defence layer belongs to, and why each ceiling is one.

``re-encode``
    ``run_defence_matrix.py`` sets ``rec = ~P`` -- every poisoned record fails
    and every genuine one passes, asserted rather than recomputed. It also
    assumes the attacker submits counts at all, that re-derivation is exact, and
    that counts cannot be engineered to encode where the attacker wants. That
    last is an untested attack, not a settled question. The raw count matrices
    do not ship, so no downstream user can run this regardless.

``donor vote``
    Needs donor or sample ids. The shipped annotation bundle has none;
    ``cache/reference_provenance.npz`` recovers them by an unscripted join at
    95.52% coverage. It is a ceiling twice over, because under MODIFY the
    attacker inherits each record's genuine ids (HANDOFF 3.39) -- so even a
    perfect provenance join defeats nothing.

``quarantine``
    ``run_quarantine_sweep.py`` CONSTRUCTS its ``seasoned`` array from the swept
    ``aged`` parameter. There is no timestamp anywhere in the shipped artefact,
    so it measures "if a curator could perfectly separate new records from old,
    how much would that help".

``volume cap``
    Needs submission events and rates, which are an ingestion-time policy the
    shipped artefact does not record.
"""

#: Column name for :class:`DefenceTier`. Every defence table carries it.
DEFENCE_TIER_COL: Final = "defence_tier"


ATTACK_TIER: Final[dict[str, str]] = {
    Attack.RELABEL: AttackTier.REALISTIC,
    Attack.DUPLICATE: AttackTier.REALISTIC,
    Attack.INTERPOLATE: AttackTier.CEILING,
    # Flood rotates real anchors toward a centroid and jitters them, so the
    # vectors it submits are NOT reference embeddings -- same grounds as
    # interpolate. Consequence worth stating: this makes the strongest attack
    # measured in the project (net 0.400) a CEILING, not a threat. That is the
    # tier system working rather than a demotion -- the GAP between flood and
    # the realistic tier is exactly what embedding-space control is worth to an
    # attacker, and reporting it unlabelled beside relabel would overstate
    # realistic risk, which is the failure AttackTier exists to prevent.
    Attack.FLOOD: AttackTier.CEILING,
}
"""Which tier each attack belongs to. Reporting a ceiling result beside
realistic ones without saying so converts an unfalsifiable "attacks are
possible" into a number that looks like measured risk and is not."""


class DistanceClass(StrEnum):
    """Coarse grouping of ontology pairs, for faceting. The integer
    ``ontology_dist`` remains the quantitative axis."""

    NEAR = "near"
    MID = "mid"
    FAR = "far"


# ─────────────────────────────────────────────────────────────────────────────
# poison_set — cache/poison/{attack}_{pair}_{rate}_{seed}.npz
# ─────────────────────────────────────────────────────────────────────────────

POISON_EMB: Final = "emb"
POISON_LABEL: Final = "label"
POISON_PROVENANCE: Final = "provenance"
POISON_ATTACK: Final = "attack"
POISON_ALPHA: Final = "alpha"

POISON_FIELDS: Final[dict[str, str]] = {
    POISON_EMB: "float32 (n, 128), unit norm",
    POISON_LABEL: "str (n,), attacker-chosen Cell Ontology term",
    POISON_PROVENANCE: "str (n,), source reference cell id, or 'synthetic'",
    POISON_ATTACK: "str, one of Attack",
    POISON_ALPHA: "float32 (n,), interpolation strength; nan for other attacks",
}

# ─────────────────────────────────────────────────────────────────────────────
# raw_result — 2_results/GSE136831/00_raw/raw/{attack}_{pair}_{rate}_{seed}.parquet
# One row per query cell. Long format.
# ─────────────────────────────────────────────────────────────────────────────

QUERY_ID: Final = "query_id"
DONOR_ID: Final = "donor_id"
Y_TRUE: Final = "y_true"
Y_PRED_CLEAN: Final = "y_pred_clean"
Y_PRED_POISONED: Final = "y_pred_poisoned"
N_POISON_IN_TOPK: Final = "n_poison_in_topk"
MIN_POISON_SCORE: Final = "min_poison_score"
POISON_SURVIVES_QC: Final = "poison_survives_qc"
ONTOLOGY_DIST: Final = "ontology_dist"

# condition keys, present in raw_result and summary alike
ATTACK: Final = "attack"
PAIR: Final = "pair"
RATE: Final = "rate"
SEED: Final = "seed"

CONDITION_KEYS: Final[tuple[str, ...]] = (ATTACK, PAIR, RATE, SEED)
"""Full condition identity. One raw_result file per combination."""

GROUP_KEYS: Final[tuple[str, ...]] = (ATTACK, PAIR, RATE)
"""What :data:`SUMMARY_DTYPES` aggregates over — every key except SEED, because
seeds are the replicates being collapsed."""

RAW_RESULT_DTYPES: Final[dict[str, str]] = {
    QUERY_ID: "string",
    DONOR_ID: "category",
    Y_TRUE: "category",
    Y_PRED_CLEAN: "category",
    Y_PRED_POISONED: "category",
    N_POISON_IN_TOPK: "int16",
    MIN_POISON_SCORE: "float32",
    POISON_SURVIVES_QC: "bool",
    ATTACK: "category",
    PAIR: "category",
    RATE: "float32",
    SEED: "int16",
    ONTOLOGY_DIST: "int8",
}

RAW_RESULT_DOC: Final[dict[str, str]] = {
    QUERY_ID: "cell barcode in the held-out dataset",
    DONOR_ID: (
        "the donor this cell came from (the column named by "
        "config.query.donor_column; in GSE136831 that is obs.sample, 78 "
        "subjects, one disease state each). THE UNIT OF INDEPENDENCE. "
        "The principle is dataset-independent; every count in this entry is "
        "GSE136831's and will differ for another dataset. Cells from one donor "
        "share genotype, batch, dissociation protocol and sequencing run, so "
        "17,571 macrophage queries from 78 donors are closer to 78 observations "
        "than 17,571. The bootstrap resamples donors for this reason."
    ),
    Y_TRUE: "author-provided Cell Ontology term (obs.celltype_id)",
    Y_PRED_CLEAN: "k-NN vote with no poisoning — the baseline",
    Y_PRED_POISONED: "k-NN vote after merging the poison set",
    N_POISON_IN_TOPK: "count of poisoned records in the merged top-k",
    MIN_POISON_SCORE: (
        "SCimilarity score of the NEAREST poison, whether or not it entered "
        "the top-k. Because score = 1/(1-c) increases with similarity, the "
        "score of the nearest poison is the MAXIMUM over the poison set, not "
        "the minimum. The name inverts its meaning; it is kept for "
        "compatibility with CLAUDE.md §5. Implementing this as min() yields a "
        "QC evasion rate that is wrong and looks entirely plausible."
    ),
    POISON_SURVIVES_QC: f"nearest poison has score >= {SCORE_QC}",
    ONTOLOGY_DIST: "tree distance between source and target Cell Ontology term",
}

# ─────────────────────────────────────────────────────────────────────────────
# summary — 2_results/GSE136831/02_a_attack_global/summary.parquet
# Aggregated over seeds. One row per (attack, pair, rate).
# ─────────────────────────────────────────────────────────────────────────────

RETRIEVAL_FAIL_RATE: Final = "retrieval_fail_rate"
OUTCOME_FAIL_RATE: Final = "outcome_fail_rate"
CLEAN_ERROR_RATE: Final = "clean_error_rate"
QC_EVASION_RATE: Final = "qc_evasion_rate"
FOCUS_SOURCE: Final = "focus_source"
QC_EVASION_FOCUS: Final = "qc_evasion_rate_focus"
RETRIEVAL_FAIL_FOCUS: Final = "retrieval_fail_rate_focus"
ATTACK_SUCCESS_RATE: Final = "attack_success_rate"
CLEAN_TARGET_RATE: Final = "clean_target_rate"
NET_ATTACK_SUCCESS: Final = "net_attack_success"
CI_LO: Final = "ci_lo"
CI_HI: Final = "ci_hi"
N_SEEDS: Final = "n_seeds"

SUMMARY_DTYPES: Final[dict[str, str]] = {
    ATTACK: "category",
    PAIR: "category",
    RATE: "float32",
    ONTOLOGY_DIST: "int8",
    RETRIEVAL_FAIL_RATE: "float32",
    OUTCOME_FAIL_RATE: "float32",
    CLEAN_ERROR_RATE: "float32",
    QC_EVASION_RATE: "float32",
    FOCUS_SOURCE: "category",
    QC_EVASION_FOCUS: "float32",
    RETRIEVAL_FAIL_FOCUS: "float32",
    ATTACK_SUCCESS_RATE: "float32",
    CLEAN_TARGET_RATE: "float32",
    NET_ATTACK_SUCCESS: "float32",
    CI_LO: "float32",
    CI_HI: "float32",
    N_SEEDS: "int16",
}

SUMMARY_DOC: Final[dict[str, str]] = {
    RETRIEVAL_FAIL_RATE: "fraction of queries with >= 1 poison in the top-k",
    OUTCOME_FAIL_RATE: "fraction of queries where y_pred_poisoned != y_true",
    CLEAN_ERROR_RATE: (
        "fraction of queries where y_pred_clean != y_true. The classifier's own "
        "error, independent of poisoning. Outcome failure must never be quoted "
        "without it — the difference is the attack effect; the level is not."
    ),
    QC_EVASION_RATE: (
        "fraction of ALL queries whose nearest poison scores >= 33. Dominated "
        "by how much of the query set the attacked type represents: macrophages "
        "are 36% of these queries and NK cells 3.8%, so this measures reach "
        "across the dataset, not how exposed the attacked type is."
    ),
    QC_EVASION_FOCUS: (
        "the same fraction among queries whose TRUE label is focus_source. This "
        "is the one that says how exposed the attacked cell type is, and it "
        "orders the types completely differently from the all-query version."
    ),
    RETRIEVAL_FAIL_FOCUS: (
        "fraction of queries whose TRUE label is focus_source with >= 1 poison "
        "in the top-k. retrieval_fail_rate uses ALL queries as its denominator, "
        "which dilutes it by however much of the query set the attacked type "
        "happens to be -- macrophages are 36% of these queries and NK cells "
        "3.8%, so the all-query version understates capture on a small type by "
        "more than an order of magnitude. Any statement of the form 'poison "
        "entered the top-k for X% of queries' must use THIS column, and it is "
        "the one that shares a denominator with qc_evasion_rate_focus and "
        "net_attack_success -- the three are only plottable on one axis "
        "because of that."
    ),
    FOCUS_SOURCE: (
        "which source class this row's attack_success_rate is about. A single "
        "attack yields one row; a simultaneous multi-type attack yields one row "
        "PER MEMBER, so a member's success under a combo is directly comparable "
        "to the same member attacked alone at the same rate — which is the only "
        "way to tell whether concurrent attacks compound or interfere."
    ),
    ATTACK_SUCCESS_RATE: (
        "fraction of queries whose TRUE label is focus_source and whose "
        "poisoned vote landed on the attacker's target. This is what the "
        "attacker wants, and unlike outcome_fail_rate it is not saturated by "
        "annotation-granularity mismatch between query and reference."
    ),
    CLEAN_TARGET_RATE: (
        "the same fraction with no poisoning. attack_success_rate must never be "
        "quoted without it: some source cells are called the target anyway, and "
        "the difference is the attack effect."
    ),
    NET_ATTACK_SUCCESS: (
        "attack_success_rate minus clean_target_rate: how much the poisoning "
        "moved the vote, rather than where the vote ended up. This is what "
        "ci_lo/ci_hi bracket, and it is estimated PAIRED WITHIN DONOR -- each "
        "donor's own cells supply both the poisoned and the clean rate, so "
        "donor-to-donor variation in baseline accuracy cancels instead of "
        "swamping the effect."
    ),
    CI_LO: (
        f"bootstrap {N_BOOTSTRAP}-resample 2.5th percentile, over DONORS. Not "
        f"over seeds: flood_common.donor_ratio_ci resamples the donor axis and "
        f"nothing else, and 02b_01 runs a single seed, so there is no seed axis "
        f"to resample. These two entries previously read 'over donors and seeds' "
        f"and 'over seeds' for the same interval; both were wrong."
    ),
    CI_HI: f"bootstrap {N_BOOTSTRAP}-resample 97.5th percentile, over DONORS",
    N_SEEDS: "replicates actually aggregated — never assume the grid was complete",
}

CI_TARGET: Final = NET_ATTACK_SUCCESS
"""Which quantity ci_lo/ci_hi bracket.

Was OUTCOME_FAIL_RATE, following CLAUDE.md §5. Changed once the intervals were
measured: bracketing outcome failure produced a width of ~7.4 percentage points
that barely moved across four orders of magnitude of contamination or across
cell types, because it was dominated by donor-to-donor variation in the ~53%
BASELINE rather than by anything the attack did. An interval an order of
magnitude wider than the effect it is meant to bound cannot distinguish a
working defence from an inert one, which is what the next stage needs it for."""


# ─────────────────────────────────────────────────────────────────────────────
# by_donor — 2_results/<DS>/02_b_attack_celltype/*_by_donor.parquet
# One row per (condition, donor). The substrate D_tau is estimated from.
# ─────────────────────────────────────────────────────────────────────────────

NUM_SUFFIX: Final = "_num"
DEN_SUFFIX: Final = "_den"
N_QUERIES: Final = "n_queries"

BY_DONOR_KEYS: Final[tuple[str, ...]] = (
    ATTACK, "source", "target", RATE, SEED, DONOR_ID,
)
"""What identifies a by_donor row. ``scope`` is NOT among them.

Within- and global-scope live on ONE row as ``within_*``/``global_*`` column
pairs, per the SCOPE exception in CLAUDE.md §7: they are two views of a single
condition -- same poison, same retrieval, same donor -- not two observations.
Splitting them would make a paired scope difference a self-join.
"""

BY_DONOR_DOC: Final[str] = f"""Per-donor NUMERATOR and DENOMINATOR sums, never rates.

Every indicator ``x`` that :func:`celltype_sweep.scope` reduces to a point and an
interval is also emitted here, unreduced, as ``{{scope}}_x{NUM_SUFFIX}`` and
``{{scope}}_x{DEN_SUFFIX}``. The pooled rate is ``sum(num) / sum(den)`` over
whichever donors a caller has in hand.

**Counts, not rates, and that is the whole point.** A bootstrap that resamples
donors must RE-POOL them under a new draw. A pre-divided per-donor rate cannot
be re-pooled correctly: averaging per-donor rates weights a donor contributing
9 queries the same as one contributing 4,000, which is a different estimator
from the ratio estimator every published interval in this project already uses.
Storing the two sums keeps the downstream bootstrap identical to the one in
``flood_common.donor_ratio_ci``.

A denominator of 0 means the donor contributed no query to that scope -- common
in ``within_``, where only donors possessing the attacked celltype appear. It is
a structural zero, not a missing value, and it must stay 0 rather than becoming
NaN: ``sum(num)/sum(den)`` then remains correct without special-casing.

Indicators whose name ends ``_delta`` or ``_net`` are per-query DIFFERENCES, so
their numerator can be negative. Do not clip it.

``{N_QUERIES}`` is carried per scope as the plain query count, so a reader can
weight or filter donors without reconstructing it from a denominator that may
mean something else (``mean_error_dist`` denominates by reachability, not by
query count).
"""

# ─────────────────────────────────────────────────────────────────────────────
# threshold — 2_results/<DS>/02_b_attack_celltype/02b_10_both_threshold_dose.parquet
# The dose at which an attack starts working, with a donor-paired interval.
# SPEC.md §5.4c.
# ─────────────────────────────────────────────────────────────────────────────

TAU: Final = "tau"
RESPONSE: Final = "response"
D_TAU: Final = "d_tau"
D_LO_BRACKET: Final = "d_lo_bracket"
D_HI_BRACKET: Final = "d_hi_bracket"
CENSOR: Final = "censor"
FRAC_CENSOR_LEFT: Final = "frac_censor_left"
FRAC_CENSOR_RIGHT: Final = "frac_censor_right"

TAU_LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.10, 0.25, 0.50)
"""Absolute response levels D_tau is reported at.

A FAMILY, not one level, so that no single arbitrary constant is load-bearing:
a reader who disagrees with the headline 0.05 reads their own off the same
table. SPEC.md §5.4c.
"""

THRESHOLD_RESPONSES: Final[tuple[str, ...]] = ("retrieval", NET_ATTACK_SUCCESS)
"""The responses D_tau is computed on, SEPARATELY and never merged.

Retrieval failure and outcome failure answer different defence questions --
whether a filter should act on retrieval or on aggregation -- and a single
"threshold dose" collapsing them would destroy the distinction this project
exists to preserve (CLAUDE.md §1).
"""


class Censor(StrEnum):
    """Why a condition has no threshold inside the ladder.

    Censoring is a RESULT, not a missing value. A censored row carries
    ``d_tau = NaN`` BY CONSTRUCTION and no figure, caption or sentence may fill
    it in.
    """

    NONE = "none"
    LEFT = "left"
    """Already above tau at the lowest dose. Reported as ``< 0.002``.

    The expected reading for hub-selected modify, which 02b_05 measures at 0.694
    traffic captured at dose 0.01 -- already most of the way up at the lowest
    dose ever run. This is the strongest result in the project, not a failure to
    measure one.
    """

    RIGHT = "right"
    """Never reaches tau by the top dose. Reported as ``> 1.25``.

    The honest form of "this attack does not work", published as prominently as
    a threshold that exists (CLAUDE.md §1).
    """


CENSOR_MAX_FRAC: Final[float] = 0.5
"""Above this share of censored bootstrap replicates, no threshold is published.

Between 0 and this, percentiles are taken over the UNCENSORED replicates and the
censored fraction is reported beside them -- an interval computed from 1,200 of
2,000 replicates means something different from one computed from all 2,000, and
the reader is entitled to know which they are holding.
"""

THRESHOLD_DTYPES: Final[dict[str, str]] = {
    ATTACK: "category",
    "source": "category",
    "target": "category",
    "selection": "category",
    "geometry": "category",
    "label": "category",
    TAU: "float64",
    RESPONSE: "category",
    **{f"{p}_{c}": t
       for p in ("within", "global")
       for c, t in ((D_TAU, "float64"), (CI_LO, "float64"), (CI_HI, "float64"),
                    (D_LO_BRACKET, "float64"), (D_HI_BRACKET, "float64"),
                    (CENSOR, "category"),
                    (FRAC_CENSOR_LEFT, "float32"),
                    (FRAC_CENSOR_RIGHT, "float32"))},
}
"""``tau`` and every dose column are float64, not float32.

They are DOSES -- the same quantity as ``rate``, which
:data:`celltype_sweep.KEY_COLUMNS` already exempts from the float32 cast because
it reaches ``seed_of`` through ``repr()``. A threshold that round-trips to
0.699999988079071 cannot be compared against a ladder point, which is the only
thing anyone will do with it.
"""

THRESHOLD_DOC: Final[dict[str, str]] = {
    D_TAU: (
        "smallest dose at which the isotonic response exceeds tau, "
        "log-linearly interpolated inside the bracketing pair of ladder doses. "
        "NaN wherever censor != none."
    ),
    D_LO_BRACKET: (
        "the ladder dose below the crossing. With d_hi_bracket this is the "
        "PURELY EMPIRICAL interval -- no interpolation, no fit -- and a reader "
        "who distrusts the interpolation should quote this pair instead."
    ),
    D_HI_BRACKET: "the ladder dose at or above the crossing",
    CENSOR: "none | left | right; see Censor. A result, never a missing value.",
    CI_LO: (
        f"{N_BOOTSTRAP}-resample percentile interval on d_tau, resampling "
        f"donors with replacement and HOLDING THE DRAW FIXED ACROSS ALL DOSES "
        f"within a replicate -- for correctness, not tightness: the pairing "
        f"makes this interval WIDER. The SAME donors are scored at every dose, "
        f"so the per-dose errors are positively correlated through a common "
        f"donor effect. Pairing reproduces that correlation; independent "
        f"per-dose draws destroy it, letting the common donor effect average "
        f"out across the ladder -- and a crossing point is most sensitive to "
        f"exactly that, a coherent shift of the whole curve. An unpaired "
        f"bootstrap therefore UNDERSTATES this interval."
    ),
    CI_HI: "the 97.5th percentile of the same replicates as ci_lo",
    FRAC_CENSOR_LEFT: "share of replicates already above tau at the lowest dose",
    FRAC_CENSOR_RIGHT: "share of replicates never reaching tau by the top dose",
}


#: Which taxonomy axes (SPEC.md 5.4) each attack name occupies.
#:
#: The sweeps write an attack NAME; the threshold table and every cross-arm
#: figure need the AXES, because "insert vs modify" and "unmoved vs shifted" are
#: what the design compares and a name is not decomposable. Held here rather
#: than rebuilt per consumer so the two arms cannot disagree about what
#: ``duplicate`` is.
#:
#: TEMPORARY. SPEC.md 5.5 has the sweeps write ``access``/``geometry``/``label``
#: directly, at which point this mapping is deleted rather than maintained. It
#: exists because the tables that must be read TODAY predate that change.
ATTACK_AXES: Final[dict[str, dict[str, str]]] = {
    "relabel": {"access": "insert", "geometry": "unmoved", "label": "relabel"},
    "duplicate": {"access": "insert", "geometry": "unmoved", "label": "relabel"},
    "interpolate": {"access": "insert", "geometry": "shifted", "label": "relabel"},
    "flood": {"access": "insert", "geometry": "synthesised", "label": "relabel"},
    # 02b_05 overwrites labels in place and moves no embedding
    "relabel_inplace": {"access": "modify", "geometry": "unmoved",
                        "label": "relabel"},
    # 02b_11 and 02b_15 overwrite the embedding too. Their ``label`` here is the
    # DEFAULT only: both sweeps run relabel AND keep, and write ``label_change``
    # per row, so a caller with that column must prefer it. Added 2026-09-27,
    # when their graph by_donor tables first reached 02b_10 and it raised
    # KeyError on a name it had never seen.
    "shifted_inplace": {"access": "modify", "geometry": "shifted",
                        "label": "relabel"},
    "synthesised_inplace": {"access": "modify", "geometry": "synthesised",
                            "label": "relabel"},
}

#: How each sweep's own selection term maps onto the taxonomy's selection axis.
#:
#: ``duplicate`` draws from a 1% pool and copies, which is ``concentrated``;
#: ``flood`` piles onto a fixed anchor count, likewise. 02b_05's three terms are
#: NOT three selections: ``oracle`` and ``split`` are both HUB selection and
#: differ in the attacker's KNOWLEDGE (capability ladder, SPEC.md 11), while
#: ``random`` is uniform. Collapsing oracle and split to one row would erase the
#: split-donor result; they stay distinct via ``knowledge``.
SELECTION_AXES: Final[dict[str, tuple[str, str]]] = {
    "relabel": ("uniform", "n/a"),
    "duplicate": ("concentrated", "n/a"),
    "interpolate": ("uniform", "n/a"),
    "flood": ("concentrated", "n/a"),
    "oracle": ("hub", "sees the scored donors' traffic"),
    "split": ("hub", "sees other donors' traffic only"),
    "random": ("uniform", "none"),
    # 02b_11 and 02b_15 name the selection axis directly rather than through a
    # sweep-specific term, so these map to themselves. Their ``hub`` is not
    # 02b_05's: it ranks on the reference's own clean top-k traffic with no
    # donor split, which is why the knowledge string differs. Added 2026-09-27.
    "uniform": ("uniform", "none"),
    "hub": ("hub", "sees the reference's own top-k traffic"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Attack axes — SPEC.md §5.2, §5.4, §5.5.
# The four things that describe an attack. `attack` as a single opaque name is
# the offence §5.5 names: relabel and duplicate are ONE construction at two
# selection settings, and nothing in the name says so.
# ─────────────────────────────────────────────────────────────────────────────

ACCESS: Final = "access"
GEOMETRY: Final = "geometry"
LABEL_CHANGE: Final = "label_change"
SELECTION: Final = "selection"
DOSE: Final = "dose"
DOSE_BASIS: Final = "dose_basis"


class Access(StrEnum):
    """Whether the attacker may overwrite reference records or only add them."""

    INSERT = "insert"
    MODIFY = "modify"


class Geometry(StrEnum):
    """Where a poisoned vector comes from."""

    UNMOVED = "unmoved"
    """A genuine reference vector, copied bit for bit. No geometric check can
    reject it -- rung 0 of the capability ladder."""

    SHIFTED = "shifted"
    """A genuine vector moved toward another celltype's centroid."""

    SYNTHESISED = "synthesised"
    """Constructed: anchors rotated toward the source centroid and jittered."""


class LabelChange(StrEnum):
    """Whether the record's label lies."""

    RELABEL = "relabel"
    KEEP = "keep"
    """The record keeps its TRUE label. Volume without the lie, and the control
    every relabel result needs: the claim that harm comes from the false
    annotation rather than from the added records is currently asserted and
    unmeasured (SPEC.md §5.4)."""


class Selection(StrEnum):
    """How the attacker chooses which reference records to draw from."""

    UNIFORM = "uniform"
    CONCENTRATED = "concentrated"
    """A small pool, copied. ``duplicate``'s rule: 1% of the celltype, drawn
    with replacement, so raising the dose deepens a few points instead of
    covering more of the celltype."""

    HUB = "hub"
    """The busiest records first, by top-k occupancy. Worth 19x at fixed budget
    under modify (94.7% of retrieval traffic against 5.1%); whether it transfers
    to insertion is the open question ``build 2`` exists to answer."""


DOSE_BASES: Final[tuple[str, ...]] = ("celltype", "atlas")
"""What ``dose`` divides by. Never implicit -- CLAUDE.md §10, HANDOFF §3.22.

An atlas share drawn from one celltype's pool runs out above ~0.1%: sampling
falls back to with-replacement and silently turns ``relabel`` into
``duplicate``, so the attack changes identity along the dose axis instead of
getting stronger.
"""


class Sites(StrEnum):
    """Where a flood's clusters are placed -- the axis v01 called ``mode``.

    v01's ``cluster_flood`` reports three "constructions", but two of them --
    ``anchor exact`` and ``anchor directed`` -- differ only in how far the
    vector is displaced, which is the ``spread`` parameter and is swept by
    ``02b_13``. The genuine third axis is WHERE the clusters sit.
    """

    ANCHOR = "anchor"
    """Real reference cells. A source cell sits at cosine ~0.979 to a typical
    query against a 50th genuine neighbour at ~0.980 -- it clears the retrieval
    bar because that is where queries actually land."""

    KMEANS = "kmeans_centres"
    """Cluster centres of the attacked celltype. An average, and in 128
    dimensions the average of a cloud is not in the cloud: v01 measured a
    centroid at cosine 0.9496 against a 0.9796 fiftieth neighbour, i.e. BELOW
    the bar. It captures more neighbourhoods than an anchor and converts far
    fewer -- median 0.066 capture against 0.075 win-given-capture, where
    anchors reach 0.68 -- so a retrieval-based filter would flag it hardest
    while it is the least dangerous construction."""
