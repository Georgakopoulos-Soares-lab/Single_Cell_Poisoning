"""Stage 7 — run a defence over a neighbourhood and measure what it costs.

# No `from __future__ import annotations` here: snakemake's script directive
# prepends a preamble, which would push it off line 1 and make it a SyntaxError.

Stage 6 asks whether the MEASUREMENT is sound. This asks whether the attack can
be STOPPED, which is a different question and a stage CLAUDE.md's original
six-stage table does not contain, because the spec stopped at characterisation.

A defence is only a result once its costs are measured beside its benefit. A
filter that rejects everything drives attack success to zero and is useless, so
every configuration reports four numbers:

    attack_success   the vote landed on the attacker's chosen label
    clean_accuracy   what the same filter costs on unpoisoned data
    false_positive   genuine records the filter rejected
    abstention       queries declined rather than answered

Abstention is tracked separately from error throughout. Declining to answer is a
refusal, not a wrong answer, and conflating them lets a defence that answers
almost nothing look almost perfect.
"""

import numpy as np

from defense import (
    DefenseConfig,
    admit,
    defended_vote,
    survivors,
)


def evaluate_defense(
    labels: np.ndarray,
    is_poison: np.ndarray,
    y_true: np.ndarray,
    target: str,
    cfg: DefenseConfig,
    recompute_ok: np.ndarray = None,
    label_consistency: np.ndarray = None,
    donor: np.ndarray = None,
    seasoned: np.ndarray = None,
    cos: np.ndarray = None,
    emb: np.ndarray = None,
    per_query: bool = False,
) -> dict:
    """Apply a defence to every query's neighbourhood and score the result.

    Args:
        labels: (n_q, k) labels of each query's merged top-k, ranked.
        is_poison: (n_q, k) bool, which of those records are poisoned.
        y_true: (n_q,) the queries' true labels.
        target: the label the attacker is trying to induce.
        cfg: which defence layers are active.
        recompute_ok: (n_q, k) bool, required when ``cfg.recompute``.
        label_consistency: (n_q, k) float, required when ``cfg.label_consistency``.
        donor: (n_q, k) str, required when ``cfg.donor_vote``.
        seasoned: (n_q, k) bool, required when ``cfg.quarantine`` — False for a
            record added within the quarantine window.
        cos: (n_q, k) float cosines, ranked descending, required when
            ``cfg.weight`` is not ``uniform``.
        emb: (n_q, k, d) float32 neighbour vectors, required when
            ``cfg.dedup_tau`` is set.
        per_query: also return the per-query boolean outcomes, so a caller can
            bootstrap over donors. The unit of independence is the donor, not
            the cell, and that resampling cannot be done from the means alone.

    Returns:
        ``attack_success`` — share of queries voted as ``target``.
        ``clean_accuracy`` — share voted correctly with poison removed, the same
        filter still applied, so the filter's own cost is visible.
        ``false_positive`` — share of GENUINE records the filter rejected.
        ``abstention`` — share of queries declined.
        With ``per_query``, additionally ``hit`` (n_q,) bool — voted as
        ``target`` — ``correct`` (n_q,) bool — voted correctly with poison
        removed and the same filter still applied — and ``pred`` (n_q,) object,
        the label actually returned, so a caller can compare against the clean
        run cell by cell.

    Note:
        clean_accuracy is computed by dropping the poisoned records and
        re-running the SAME filter, not by skipping the filter. Skipping it
        would report the undefended baseline and hide what the defence costs
        ordinary annotation.
    """
    n_q, k = labels.shape

    def col(arr, i):
        return None if arr is None else np.asarray(arr)[i]

    preds, clean_preds = [], []
    keep_mask = np.zeros(labels.shape, dtype=bool)
    fp_num = fp_den = 0

    for i in range(n_q):
        preds.append(
            defended_vote(labels[i], cfg, recompute_ok=col(recompute_ok, i),
                          label_consistency=col(label_consistency, i),
                          donor=col(donor, i), seasoned=col(seasoned, i),
                          cos=col(cos, i), emb=col(emb, i))
        )
        kept_i = survivors(labels[i], cfg, recompute_ok=col(recompute_ok, i),
                           label_consistency=col(label_consistency, i),
                           donor=col(donor, i), seasoned=col(seasoned, i),
                           emb=col(emb, i))
        keep_mask[i, kept_i] = True
        genuine = ~np.asarray(is_poison[i])
        if genuine.any():
            clean_preds.append(
                defended_vote(
                    labels[i][genuine], cfg,
                    recompute_ok=col(recompute_ok, i)[genuine] if recompute_ok is not None else None,
                    label_consistency=col(label_consistency, i)[genuine] if label_consistency is not None else None,
                    donor=col(donor, i)[genuine] if donor is not None else None,
                    seasoned=col(seasoned, i)[genuine] if seasoned is not None else None,
                    cos=col(cos, i)[genuine] if cos is not None else None,
                    emb=col(emb, i)[genuine] if emb is not None else None,
                )
            )
        else:
            clean_preds.append(None)

        # false positives: genuine records the record-level filters rejected
        for j in np.flatnonzero(genuine):
            fp_den += 1
            if not admit(
                cfg,
                recompute_ok=bool(recompute_ok[i][j]) if recompute_ok is not None else True,
                label_consistency=float(label_consistency[i][j]) if label_consistency is not None else 1.0,
                seasoned=bool(seasoned[i][j]) if seasoned is not None else True,
            ):
                fp_num += 1

    preds = np.array(preds, dtype=object)
    clean_preds = np.array(clean_preds, dtype=object)
    answered = preds != None  # noqa: E711 — object array, `is not None` won't vectorise

    out_extra = {}
    if per_query:
        out_extra = {
            "kept": keep_mask,
            "hit": (preds == target),
            # The vote that was actually RETURNED, poison included, is right.
            # This is accuracy under attack. An abstention (pred None) counts
            # as not correct: a declined query is not a correct annotation.
            "accurate": (preds == np.asarray(y_true, dtype=object)),
            # NOT accuracy under attack: the vote with the poison REMOVED and
            # the same filter re-applied, i.e. the per-query form of
            # ``clean_accuracy`` -- what the filter costs on unpoisoned data.
            # Until 2026-10-04 every caller stored this as ``accuracy``; see
            # diagnosis/2026-10-04_attack_vs_defence_none.
            "correct": (clean_preds == np.asarray(y_true, dtype=object)),
            "pred": preds,
        }

    return {
        **out_extra,
        "defense": cfg.name,
        "attack_success": float((preds == target).mean()),
        "accuracy": float((preds == np.asarray(y_true, dtype=object)).mean()),
        "clean_accuracy": float((clean_preds == np.asarray(y_true, dtype=object)).mean()),
        "false_positive": float(fp_num / fp_den) if fp_den else 0.0,
        "abstention": float((~answered).mean()),
        "n_queries": int(n_q),
    }
