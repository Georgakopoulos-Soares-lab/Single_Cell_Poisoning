"""Step 01a_03 — how accurate is the deployed system before any attacker exists?

Every effect in this project is reported net of this control. v01 computed it --
``net_attack_success`` is ``attack_success_rate`` minus ``clean_target_rate``, so
the baseline was always in the arithmetic -- but it reported it from
``y_pred_clean``, which came from EXACT brute-force retrieval. That answers "how
accurate is exhaustive search over these vectors", not "how accurate is the
system a user installs".

This answers the second question. The vote here is taken over
``01a_01_no_access_hnsw_neighbors.npz``: the shipped ``labelled_kNN.bin``,
queried the way ``CellAnnotation.get_predictions_knn`` queries it. 01a_02
measured how far apart the two retrievals land; this one reports the deployed
side of that gap as the control every later arm is scored against.

Three things are measured, all on every distinct true celltype rather than only
the four the attack grid names -- an attacked type's number means nothing without
the distribution it sits in:

**accuracy** — does the k = 50 vote return the term the authors assigned?

**accuracy at ontology tolerance** — the number that actually answers "how
accurate is SCimilarity". Exact match punishes the model for returning
*alveolar macrophage* where the author wrote *macrophage*, which is a
granularity disagreement between two annotators, not a retrieval failure. So
accuracy is reported again counting a prediction correct when it is an ancestor
or descendant of the true term (``lineage``), and at tree distance <= 1, 2 and 3.
The gap between exact and lineage-tolerant is the share of "errors" that are
really label-granularity mismatch, and it is large. 01a_04 puts a donor-paired
interval on that gap.

**reachability** — is the query's true term even present in the labelled
reference? A term the reference cannot say is a ceiling on accuracy that no
defence can lift, and it separates "retrieval was wrong" from "the answer was
never available".

**confusion** — when it is wrong, what does it say instead, and how far away is
that in the tree? A neighbouring subtype is a different failure from an
unrelated lineage, and that distinction decides whether a wrong label would
survive a human reading the output.

Ties are broken by ``config.vote.tie_break``, which is ``lowest_cl_id``. **This
differs from 01a_02**, which deliberately left ties on argmax order and reported
them rather than resolving them, on the grounds that neither rule is defensible
as an answer. Both are reported here too: ``tie_rate`` bounds how much the choice
can move ``clean_accuracy``, and the bound is the whole of the possible
disagreement between this table and 01a_02's ``accuracy_hnsw``.

No intervals are written here. They are 01a_04's job, because the bootstrap is
2,000 resamples x ~200 celltypes x 10 quantities and this step is the part worth
rerunning when the retrieval changes.

**THREAT MODEL: none.** This is the unpoisoned control; no attacker exists. The
``threat_model`` column is still written, because a table without it cannot be
attributed later and v01 lost that for 35 of 39 tables.

Writes, under ``2_results/<dataset>/01_a_baseline/``:
  ``01a_03_no_access_clean_baseline.parquet``   one row per celltype, plus ``__all__``
  ``01a_03_no_access_per_donor.parquet``        donor x celltype counts -- 01a_04's input
  ``01a_03_no_access_clean_confusion.parquet``  top wrong answers per celltype
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_common"))
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dataset_config import cache_root, load, query_cache  # noqa: E402
from ontology import Ontology  # noqa: E402
from provenance import emit  # noqa: E402
from schemas import (  # noqa: E402
    DONOR_ID,
    K_NEIGHBORS,
    ONTOLOGY_DIST,
    THREAT_MODEL,
    Y_TRUE,
    ThreatModel,
)
from vote import vote  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[2]
CFG = load(BASE)
DS = CFG["dataset"]
CACHE = cache_root(BASE, CFG)      # shared: reference.npz, poison/
QCACHE = query_cache(BASE, CFG)    # this dataset only -- see dataset_config
OUT = BASE / "2_results" / DS / "01_a_baseline"

#: Tolerance ladder, in ``is_a`` hops. Reported alongside exact match, never
#: instead of it: a tolerant number quoted alone is an accuracy claim about a
#: looser question than the one the reader is asking.
TOLERANCES = (1, 2, 3)

#: Wrong answers listed per celltype in the confusion table. Three is enough to
#: show whether the errors concentrate on one neighbour or scatter.
TOP_CONFUSIONS = 3

#: Distance written when two terms are unreachable in the ``is_a`` graph. A
#: sentinel, not a distance -- large enough that no tolerance ever admits it,
#: and distinguishable from a real hop count on inspection.
UNREACHABLE = 99

#: The columns the per-donor table carries, and that 01a_04 bootstraps.
METRICS = ("correct", "lineage") + tuple(f"within{t}" for t in TOLERANCES) + ("reachable",)


def score_pairs(truth: np.ndarray, pred: np.ndarray,
                onto: Ontology) -> tuple[np.ndarray, np.ndarray]:
    """Lineage relation and tree distance for every query, via distinct pairs.

    Computed once per (true, predicted) pair rather than once per query: there
    are ~49,000 queries but only a few hundred distinct pairs, and
    :meth:`Ontology.distance` is a breadth-first search.

    Args:
        truth: (n_q,) str, author-assigned CL ids. pred: (n_q,) str, the vote.
        onto: parsed Cell Ontology.

    Returns:
        lineage: (n_q,) float32 in {0, 1}, 1 where one term generalises the
            other, which is a granularity disagreement rather than a wrong
            answer. dist: (n_q,) int16 hop count, :data:`UNREACHABLE` where the
            two terms are not connected.
    """
    pairs = pd.DataFrame({"y": truth, "p": pred}).drop_duplicates()
    rel: dict[tuple[str, str], bool] = {}
    dist: dict[tuple[str, str], int] = {}
    for y, p in pairs.itertuples(index=False):
        rel[(y, p)] = bool(y == p or onto.is_ancestor_or_descendant(y, p))
        d = 0 if y == p else onto.distance(y, p)
        dist[(y, p)] = UNREACHABLE if d is None else int(d)
    key = list(zip(truth, pred))
    return (np.array([rel[k] for k in key], dtype=np.float32),
            np.array([dist[k] for k in key], dtype=np.int16))


def main() -> None:
    t0 = time.time()
    emit(__file__)          # which code ran, HEAD or dirty tree
    if CFG["vote"]["tie_break"] != "lowest_cl_id":
        raise ValueError(
            f"config.vote.tie_break is {CFG['vote']['tie_break']!r}; "
            "plurality_vote implements lowest_cl_id only"
        )

    onto = Ontology.from_obo(BASE.parent / CFG["ontology"]["obo"])
    names = dict(onto.id_to_name)
    # config's term_names covers only the terms the experiment happens to name;
    # the confusion targets are whatever the vote returned, so names come from
    # the ontology itself. A bare CL id in a results table is unreadable, and
    # unreadable is how a granularity artefact gets mistaken for an error.
    names.update(CFG.get("term_names", {}))

    def nm(term: str) -> str:
        """Human-readable name for a CL id, falling back to the id itself."""
        return names.get(term, term)

    hn = np.load(OUT / "01a_01_no_access_hnsw_neighbors.npz", allow_pickle=True)
    meta = pd.read_parquet(QCACHE / "query_meta.parquet")
    lab = hn["label"].astype(str)
    if lab.shape != (len(meta), K_NEIGHBORS):
        raise ValueError(f"neighbours are {lab.shape}, expected "
                         f"{(len(meta), K_NEIGHBORS)}")

    ref_terms = set(np.unique(np.load(CACHE / "reference.npz",
                                      allow_pickle=True)["label"]).astype(str))
    print(f"[{time.time() - t0:.0f}s] {len(ref_terms)} distinct terms the "
          f"reference can say", file=sys.stderr)

    truth = meta[Y_TRUE].to_numpy().astype(str)
    donor = meta[DONOR_ID].to_numpy().astype(str)
    _r = vote(lab, pivotal=truth)
    pred, tied = _r.pred, _r.pivotal_in_tie
    lineage, dist = score_pairs(truth, pred, onto)

    d = pd.DataFrame({
        "celltype": truth,
        DONOR_ID: donor,
        "predicted": pred,
        "tied": tied.astype(np.float32),
        "correct": (pred == truth).astype(np.float32),
        "lineage": lineage,
        ONTOLOGY_DIST: dist,
        "reachable": np.isin(truth, list(ref_terms)).astype(np.float32),
    })
    for t in TOLERANCES:
        d[f"within{t}"] = (dist <= t).astype(np.float32)
    print(f"[{time.time() - t0:.0f}s] {len(d):,} queries, "
          f"{d[DONOR_ID].nunique()} donors, {d.celltype.nunique()} true celltypes",
          file=sys.stderr)

    # ── per-donor counts: 01a_04's input ────────────────────────────────────
    # Donors are the unit of independence, so the interval is taken over these
    # rows and not over queries. __all__ is deliberately absent: it is the sum
    # over celltypes within a donor, and storing it would let the two disagree.
    g = d.groupby(["celltype", DONOR_ID], observed=True)
    per_donor = g[list(METRICS)].sum().reset_index()
    per_donor["n_queries"] = g.size().to_numpy()
    per_donor.insert(0, THREAT_MODEL, ThreatModel.NO_ACCESS)

    OUT.mkdir(parents=True, exist_ok=True)
    per_donor.to_parquet(OUT / "01a_03_no_access_per_donor.parquet", index=False)
    print(f"[{time.time() - t0:.0f}s] per-donor table {per_donor.shape}",
          file=sys.stderr)

    def rate(col: pd.Series) -> float:
        """Mean of a 0/1 indicator column, accumulated in float64.

        Args:
            col: (n_q,) float32 in {0, 1}.

        Returns:
            The exact ratio as a Python float.
        """
        return float(col.to_numpy(dtype=np.float64).mean())

    # ── point estimates, per celltype and pooled ────────────────────────────
    rows = []
    for ct in sorted(pd.unique(truth)) + ["__all__"]:
        sub = d if ct == "__all__" else d[d.celltype == ct]
        row = {
            THREAT_MODEL: ThreatModel.NO_ACCESS,
            "celltype": ct,
            "label": "all queries" if ct == "__all__" else nm(ct),
            "n_queries": int(len(sub)),
            "n_donors": int(sub[DONOR_ID].nunique()),
            # float64, deliberately, over float32 indicator columns. The 0/1
            # values are exact in float32 and stored that way, but their MEAN is
            # not: accumulating 49,456 of them in float32 lands within float32
            # eps (~6e-8) of the true ratio rather than on it, which corrupts the
            # eighth significant digit of every rate published here. float32 is
            # this project's rule for the 7.9M x 128 embedding arrays, where it
            # halves real memory; an indicator column is not that case. 01a_04
            # recomputes these same rates from the per-donor counts in float64
            # and compares at 1e-9 -- it is that check which caught this.
            "clean_accuracy": rate(sub["correct"]),
            "tie_rate": rate(sub["tied"]),
        }
        row.update({m: rate(sub[m]) for m in METRICS if m != "correct"})
        rows.append(row)
    baseline = pd.DataFrame(rows)
    baseline.to_parquet(OUT / "01a_03_no_access_clean_baseline.parquet", index=False)

    # ── confusion: when it is wrong, what does it say instead ───────────────
    conf = []
    for ct in sorted(pd.unique(truth)):
        sub = d[(d.celltype == ct) & (d["correct"] == 0)]
        if sub.empty:
            continue
        n_ct = int((d.celltype == ct).sum())
        for term, n in sub["predicted"].value_counts().head(TOP_CONFUSIONS).items():
            conf.append({
                THREAT_MODEL: ThreatModel.NO_ACCESS,
                "celltype": ct,
                "label": nm(ct),
                "predicted": term,
                "predicted_label": nm(term),
                # The distance is what separates "a neighbouring subtype" from
                # "an unrelated lineage", which is the whole point of the table.
                ONTOLOGY_DIST: int(sub.loc[sub["predicted"] == term,
                                           ONTOLOGY_DIST].iloc[0]),
                "n": int(n),
                "share_of_errors": float(n / len(sub)),
                "share_of_queries": float(n / n_ct),
            })
    confusion = pd.DataFrame(conf)
    confusion.to_parquet(OUT / "01a_03_no_access_clean_confusion.parquet", index=False)

    a = baseline[baseline.celltype == "__all__"].iloc[0]
    print(f"\n[{time.time() - t0:.0f}s] pooled over {int(a.n_queries):,} queries, "
          f"{int(a.n_donors)} donors:", file=sys.stderr)
    print(f"  exact          {a.clean_accuracy:.4f}", file=sys.stderr)
    print(f"  same lineage   {a.lineage:.4f}", file=sys.stderr)
    for t in TOLERANCES:
        print(f"  dist <= {t}      {a[f'within{t}']:.4f}", file=sys.stderr)
    print(f"  reachable      {a.reachable:.4f}", file=sys.stderr)
    print(f"  tied votes     {a.tie_rate:.4f}   (bounds disagreement with 01a_02)",
          file=sys.stderr)
    print(f"[{time.time() - t0:.0f}s] wrote {len(baseline)} baseline rows and "
          f"{len(confusion)} confusion rows", file=sys.stderr)


if __name__ == "__main__":
    main()
