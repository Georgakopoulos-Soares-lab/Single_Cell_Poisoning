"""Step 03b_04 — which integrity check catches which attack, and what it costs.

# No `from __future__ import annotations`: snakemake's script directive
# prepends a preamble, which would push it off line 1 and make it a SyntaxError.

Every other defence in this project is CONTENT-BASED: it inspects a record and
decides whether to trust it. `03b_01` shows that family has a hard floor --
against an in-place relabel, `recompute` is exactly zero on all 208 conditions,
because the attacker forged nothing inspectable. The embedding is genuine, the
donor is genuine, the record has been in the atlas for years. There is no
property of the record that is wrong.

The defence that DOES catch it is not content-based at all: compare the atlas to
a trusted copy. That is obvious, and this step is not here to discover it. It is
here because the OPERATIONAL question -- what do you hash, how often, and what
does it cost -- has a non-obvious answer that a maintainer can act on.

WHAT RE-EMBEDDING DOES NOT DO, stated because the project's own notes got this
loose. Re-deriving embeddings from counts catches a moved VECTOR. It cannot
catch an in-place relabel, because the attack never touches an embedding:
`reference_labels.tsv` is a SEPARATE artefact from `labelled_kNN.bin`, and
re-embedding reproduces the vectors exactly while saying nothing about the
labels attached to them. `recompute`'s 0.00 is that fact measured.

THE FINDING THIS STEP EXISTS TO MEASURE. Because the labels ship separately
from the index, **checksumming the index alone misses the one attack no
content-based defence can catch.** A maintainer who verifies `labelled_kNN.bin`
against the release hash and stops there has verified the artefact the attacker
did not need to touch. The cheap check catches the expensive problem, but only
if it is pointed at the right file.

COST IS THE OTHER HALF, and it inverts the intuition. Hashing is seconds over
7.9M records; re-embedding is hours to days and needs count matrices a
downstream user does not have. So the undefendable attack is caught by the
cheapest available check, and the attacks that re-embedding does catch are
already caught by `recompute` for free.

WHAT IS NOT MEASURED HERE, and must not be implied. Re-embedding wall time is
NOT measured: the reference count matrices are not in this repository, so there
is nothing to re-embed. It is reported as unavailable rather than estimated.
And the EXPOSURE WINDOW -- poison submitted the day after a quarterly check is
live for up to 90 days -- is an operational parameter, not a measurement; the
recommendation turns on it and it is stated as an assumption.
"""
import hashlib
import pathlib
import sys
import time

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "00_common"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from attacks import build_attack  # noqa: E402
from celltype_sweep import Bench, seed_of  # noqa: E402
from dataset_config import load  # noqa: E402
from modify import build_modify  # noqa: E402
from provenance import emit  # noqa: E402
from schemas import (GEOMETRY, LABEL_CHANGE, POISON_EMB, POISON_LABEL,  # noqa: E402
                     Geometry)
from selection import choose  # noqa: E402

BASE = _HERE.parents[2]
CFG = load(BASE)
DS = CFG["dataset"]
TAB = BASE / "2_results" / DS / "03_b_defence_vs_modify"
OUT = TAB / "03b_04_both_integrity.parquet"

SEED = 0
N_CELLTYPES = 6
RATE = 0.25
FLOOD_ANCHORS, FLOOD_SPREAD, FLOOD_JITTER = 16_384, 0.999, 1.0
ALPHA = 0.05
#: (label, arm, kind). Four types spanning the 2x2 that decides the answer:
#: whether the attack APPENDS or EDITS, and whether it touches vectors.
CASES = (("relabel", "insert", "attack"), ("flood", "insert", "attack"),
         ("unmoved_inplace", "modify", "modify"),
         ("shifted_inplace", "modify", "modify"))


def digest(a: np.ndarray) -> str:
    """sha256 of an array's bytes, the check a maintainer would actually run."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(a).view(np.uint8).reshape(-1).tobytes())
    return h.hexdigest()


def edit_in_place(geom: str, *, ref_emb: np.ndarray, ref_lab: np.ndarray,
                  ref_ids: np.ndarray, source: str, target: str, rate: float,
                  seed: int, hub_nbr: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, int]:
    """The atlas after a with-access attack: same length, records overwritten.

    ``unmoved`` is built here and NOT through ``build_modify``, which refuses it
    by design: it moves no vector, so the modify machinery has nothing to do.
    Its rows are drawn exactly as 03b_01's cached_clean arm draws them --
    ``choose`` over the source pool at ``max(1, round(rate * n_src))`` -- so the
    record set whose edit this step checksums is the one 03b_01 scored. That
    path was missing, and the run of 2026-10-01 (RUNLOG Run 44) died on it.

    Args:
        geom: a :class:`schemas.Geometry` value -- unmoved, shifted or
            synthesised.
        ref_emb: (n_ref, 128) float32, unit norm. Not modified.
        ref_lab: (n_ref,) str. Not modified.
        hub_nbr: (n_q, k) observed neighbour rows, for hub selection.

    Returns:
        ``(new_emb, new_lab, n_touched)`` -- copies of the two artefacts with the
        attack applied, and the number of records overwritten.
    """
    new_emb, new_lab = ref_emb.copy(), ref_lab.copy()
    if geom == Geometry.UNMOVED:
        pool = np.flatnonzero(ref_lab == source)
        n_p = max(1, int(round(rate * len(pool))))
        rows = choose("hub", pool, n_p, np.random.default_rng(seed),
                      hub_nbr=hub_nbr, n_ref=len(ref_lab))
        new_lab[rows] = target
        return new_emb, new_lab, len(rows)
    p = build_modify(
        ref_emb=ref_emb, ref_labels=ref_lab, ref_ids=ref_ids,
        source=source, target=target, dose=rate, seed=seed,
        selection="hub", geometry=geom, label_change="relabel",
        alpha_range=(ALPHA, ALPHA), hub_nbr=hub_nbr,
        **({"n_anchors": FLOOD_ANCHORS, "spread": FLOOD_SPREAD,
            "jitter": FLOOD_JITTER} if geom == Geometry.SYNTHESISED else {}))
    rows = p["rows"]
    new_lab[rows] = p[POISON_LABEL]
    new_emb[rows] = p[POISON_EMB]
    return new_emb, new_lab, len(rows)


def main() -> None:
    t0 = time.time()
    emit(__file__)
    b = Bench.load(CFG, BASE, t0)
    ref_emb, ref_lab = b.ref_emb, b.ref_label

    # Baseline digests of the two artefacts a release ships. Timed, because the
    # cost claim is the point and an unmeasured "seconds" is an assertion.
    t = time.time()
    h_emb0 = digest(ref_emb)
    dt_emb = time.time() - t
    t = time.time()
    h_lab0 = digest(ref_lab)
    dt_lab = time.time() - t
    print(f"[{time.time()-t0:.0f}s] baseline digests over {len(ref_lab):,} records: "
          f"embeddings {dt_emb:.1f}s, labels {dt_lab:.1f}s", file=sys.stderr)

    rows: list[dict] = []
    for source in list(b.sources)[:N_CELLTYPES]:
        target, odist = b.target_of[source]
        b.clean_target(target)
        cond_seed = seed_of(source, RATE, SEED)
        for name, arm, kind in CASES:
            t1 = time.time()
            if kind == "attack":
                p = build_attack(
                    name, ref_emb=ref_emb, ref_labels=ref_lab, ref_ids=b.ref_id,
                    source=source, target=target, rate=RATE, seed=cond_seed,
                    selection="hub", hub_nbr=b.clean_idx,
                    **({"n_anchors": FLOOD_ANCHORS, "spread": FLOOD_SPREAD,
                        "jitter": FLOOD_JITTER} if name == "flood" else {}))
                # An insert APPENDS: both artefacts grow.
                new_emb = np.concatenate([ref_emb, p[POISON_EMB]])
                new_lab = np.concatenate([ref_lab, p[POISON_LABEL]])
                n_touched = len(p[POISON_EMB])
            else:
                new_emb, new_lab, n_touched = edit_in_place(
                    name.split("_")[0], ref_emb=ref_emb, ref_lab=ref_lab,
                    ref_ids=b.ref_id, source=source, target=target, rate=RATE,
                    seed=cond_seed, hub_nbr=b.clean_idx)

            t_e = time.time()
            h_emb = digest(new_emb)
            dt_e = time.time() - t_e
            t_l = time.time()
            h_lab = digest(new_lab)
            dt_l = time.time() - t_l
            # Element-wise comparison: says WHICH records moved, not merely
            # that something did. The expensive-but-actionable version.
            t_d = time.time()
            lab_changed = (int((new_lab[:len(ref_lab)] != ref_lab).sum())
                           + max(len(new_lab) - len(ref_lab), 0))
            emb_changed = (int((new_emb[:len(ref_emb)] != ref_emb).any(axis=1).sum())
                           + max(len(new_emb) - len(ref_emb), 0))
            dt_diff = time.time() - t_d

            rows.append({
                "threat_model": "no_access" if arm == "insert" else "with_access",
                "access": "insert" if arm == "insert" else "overwrite",
                "attack": name, GEOMETRY: "" if arm == "insert" else name.split("_")[0],
                LABEL_CHANGE: "relabel", "selection": "hub",
                "source": source, "target": target, "ontology_dist": odist,
                "rate": RATE, "seed": SEED, "condition_seed": cond_seed,
                "n_records_touched": n_touched,
                # The four checks a maintainer could run.
                "index_hash_detects": bool(h_emb != h_emb0),
                "label_hash_detects": bool(h_lab != h_lab0),
                "label_diff_detects": bool(lab_changed > 0),
                "emb_diff_detects": bool(emb_changed > 0),
                "n_labels_changed": lab_changed,
                "n_embeddings_changed": emb_changed,
                "seconds_index_hash": dt_e, "seconds_label_hash": dt_l,
                "seconds_elementwise_diff": dt_diff,
                # Re-embedding is NOT measured: the reference count matrices are
                # not in this repository. Recorded as missing, never estimated.
                "seconds_reembed": np.nan,
                "n_reference_records": len(ref_lab),
            })
            print(f"[{time.time()-t0:7.0f}s] {source} {name:16s} touched "
                  f"{n_touched:>8,}  index_hash={h_emb != h_emb0!s:5s} "
                  f"label_hash={h_lab != h_lab0!s:5s} ({time.time()-t1:.0f}s)",
                  file=sys.stderr)
            del new_emb, new_lab

    d = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT, index=False)
    print(f"\n[{time.time()-t0:.0f}s] wrote {len(d)} rows to {OUT.name}", file=sys.stderr)
    summ = (d.groupby("attack", observed=True)
             .agg(n=("attack", "size"),
                  index_hash=("index_hash_detects", "mean"),
                  label_hash=("label_hash_detects", "mean"),
                  labels_changed=("n_labels_changed", "median"),
                  embs_changed=("n_embeddings_changed", "median")))
    print(summ.to_string(), file=sys.stderr)
    miss = d[~d.index_hash_detects]
    if len(miss):
        print(f"\n  {len(miss)} of {len(d)} conditions are INVISIBLE to an index "
              f"checksum: {sorted(miss.attack.unique())}", file=sys.stderr)
        print("  -- the label file must be verified separately", file=sys.stderr)


if __name__ == "__main__":
    main()
