"""Step 01a_02 — does exact retrieval match what SCimilarity actually returns?

The whole benchmark rests on an answer nobody has measured. Every sweep
condition is the exact clean top-50 with poison merged in, so if exact retrieval
and SCimilarity's approximate retrieval disagree on CLEAN data, every downstream
number describes a system slightly different from the deployed one.

Stage 6 compares the shortcut to a rebuilt graph, but only under attack -- its
rates start at 0.01. This is the rate-0 case it never covers.

Three quantities, because they answer different questions:

``overlap``   how many of the 50 retrieved cells are the same set. Measures the
              approximation directly.
``agreement`` whether the k-NN VOTE lands on the same celltype. A query can lose
              several neighbours and keep its answer, so this is the one that
              matters for a label.
``accuracy``  each path scored against ``y_true`` independently. If exact is more
              accurate the benchmark is optimistic about the deployed system; if
              less, pessimistic. Reported either way.

Intervals are donor-paired (``flood_common.donor_ratio_ci``): cells from one
donor share genotype, batch and dissociation, so the donor is the unit of
independence. Resampling cells would claim ~49,000 independent observations
where there are 78.

Ties are reported, never silently broken. A tie among labels that are all wrong
is already determined, so only a tie CONTAINING the pivotal label is ambiguous;
those are counted and bounded rather than resolved by argmax order, which is
what SCimilarity itself does (``max`` returns the first maximal key).

**THREAT MODEL: none.** Clean retrieval, no attacker. ``threat_model`` is still
written so the table can be attributed.

Reads ``01a_01_no_access_hnsw_neighbors.npz`` and ``cache/<dataset>/clean_neighbors.npz``.
Writes ``01a_02_no_access_exact_vs_hnsw.parquet`` -- one row per celltype, plus a
pooled row.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_common"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dataset_config import cache_root, load, query_cache  # noqa: E402
from flood_common import donor_ratio_ci  # noqa: E402
from provenance import emit  # noqa: E402
from schemas import K_NEIGHBORS, THREAT_MODEL, ThreatModel  # noqa: E402
from vote import vote  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[2]
CFG = load(BASE)
DS = CFG["dataset"]
CACHE = cache_root(BASE, CFG)      # shared: reference.npz, poison/
QCACHE = query_cache(BASE, CFG)    # this dataset only -- see dataset_config
OUT = BASE / "2_results" / DS / "01_a_baseline"
#: Deterministic across reruns; the bootstrap must be re-derivable.
SEED = 0


def main() -> None:
    emit(__file__)          # which code ran, HEAD or dirty tree
    hn = np.load(OUT / "01a_01_no_access_hnsw_neighbors.npz", allow_pickle=True)
    ex = np.load(QCACHE / "clean_neighbors.npz", allow_pickle=True)
    meta = pd.read_parquet(QCACHE / "query_meta.parquet")

    for name, arr in (("hnsw", hn["idx"]), ("exact", ex["idx"])):
        if arr.shape != (len(meta), K_NEIGHBORS):
            raise ValueError(f"{name} is {arr.shape}, expected "
                             f"{(len(meta), K_NEIGHBORS)}")

    # Set overlap per query: how much of the retrieved neighbourhood survives.
    overlap = np.array([len(np.intersect1d(a, b, assume_unique=False))
                        for a, b in zip(hn["idx"], ex["idx"])], dtype=np.int16)

    truth0 = meta["y_true"].to_numpy().astype(str)
    r_hn = vote(hn["label"], pivotal=truth0)
    r_ex = vote(ex["label"], pivotal=truth0)
    p_hn, p_ex = r_hn.pred, r_ex.pred
    tie_hn, tie_ex = r_hn.pivotal_in_tie, r_ex.pivotal_in_tie
    truth = meta["y_true"].to_numpy().astype(str)
    donor = meta["donor_id"].to_numpy()

    agree = (p_hn == p_ex).astype(float)
    acc_hn = (p_hn == truth).astype(float)
    acc_ex = (p_ex == truth).astype(float)
    ambiguous = (tie_hn | tie_ex).astype(float)
    ones = np.ones(len(meta))

    rows = []
    for ct in list(pd.unique(truth)) + ["__pooled__"]:
        m = np.ones(len(truth), bool) if ct == "__pooled__" else truth == ct
        if m.sum() == 0:
            continue
        ov, ov_lo, ov_hi = donor_ratio_ci(overlap[m], ones[m] * K_NEIGHBORS,
                                          donor[m], SEED)
        ag, ag_lo, ag_hi = donor_ratio_ci(agree[m], ones[m], donor[m], SEED)
        ah, ah_lo, ah_hi = donor_ratio_ci(acc_hn[m], ones[m], donor[m], SEED)
        ae, ae_lo, ae_hi = donor_ratio_ci(acc_ex[m], ones[m], donor[m], SEED)
        rows.append({
            THREAT_MODEL: ThreatModel.NO_ACCESS,
            "celltype": ct,
            "n_queries": int(m.sum()),
            "n_donors": int(pd.unique(donor[m]).size),
            "overlap": ov, "overlap_lo": ov_lo, "overlap_hi": ov_hi,
            "min_overlap": int(overlap[m].min()),
            "vote_agreement": ag, "agree_lo": ag_lo, "agree_hi": ag_hi,
            "accuracy_hnsw": ah, "acc_hnsw_lo": ah_lo, "acc_hnsw_hi": ah_hi,
            "accuracy_exact": ae, "acc_exact_lo": ae_lo, "acc_exact_hi": ae_hi,
            # Signed: positive means EXACT is the more accurate of the two, so
            # the benchmark is optimistic relative to what users deploy.
            "accuracy_delta": ae - ah,
            "tie_rate": float(ambiguous[m].mean()),
        })

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "01a_02_no_access_exact_vs_hnsw.parquet", index=False)

    p = df[df.celltype == "__pooled__"].iloc[0]
    print(f"pooled over {int(p.n_queries):,} queries, {int(p.n_donors)} donors:",
          file=sys.stderr)
    print(f"  top-{K_NEIGHBORS} overlap  {p.overlap:.4f} "
          f"[{p.overlap_lo:.4f}, {p.overlap_hi:.4f}]  min {int(p.min_overlap)}",
          file=sys.stderr)
    print(f"  vote agreement   {p.vote_agreement:.4f} "
          f"[{p.agree_lo:.4f}, {p.agree_hi:.4f}]", file=sys.stderr)
    print(f"  accuracy exact   {p.accuracy_exact:.4f}   hnsw {p.accuracy_hnsw:.4f}"
          f"   delta {p.accuracy_delta:+.4f}", file=sys.stderr)
    print(f"  ambiguous ties   {p.tie_rate:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
