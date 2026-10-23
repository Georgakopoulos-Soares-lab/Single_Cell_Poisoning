"""One entry point that builds any attack's poison set.

Flood was not an ``Attack`` member in v01. It lived in ``flood_common`` with its
own drivers and its own tables, purely because ``build_flood`` takes three
parameters the other three constructors do not. The result was that the
strongest attack measured in the project -- net attack success 0.400, higher
than anything ``relabel``, ``duplicate`` or ``interpolate`` reached -- was the
one hardest to place beside the others, and it sat outside every sweep that
called itself a comparison of attacks.

A differing signature is a reason for a dispatcher, not for a taxonomy to bend.
This is that dispatcher: one call, one return shape, attack-specific parameters
carried explicitly and reported as columns so a reader can see which knobs were
turned. ``build_poison`` and ``build_flood`` are untouched and still own their
own construction -- nothing is reimplemented here.

**Attack-specific parameters are not optional extras.** ``n_anchors`` moves
flood's net attack success from 0.002 to 0.400; a flood number quoted without it
says nothing. :func:`params_of` returns them for every attack so the caller can
write them into the table, null where they do not apply.
"""
from __future__ import annotations

import sys
from typing import Any

import numpy as np

from schemas import EMB_DTYPE, Attack, LabelChange, Sites

sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/00_pipeline")

#: Flood has NO defaults, deliberately. An earlier draft of this module set
#: ``FLOOD_JITTER = 0.25``, read out of ``run_cluster_flood.py``'s plan tuples --
#: which unpack as ``(mode, m, spread, RATE)``. The 0.25 is a rate. That script
#: passes no jitter at all. Since ``build_flood``'s ``jitter`` is the cosine
#: RETAINED, 0.25 would have meant ~97% random direction: poison at cosine 0.244
#: to the source centroid against a 50th genuine neighbour at ~0.980, never
#: retrieved, and the project's strongest attack would have measured as inert.
#: Caught by review before it ran.
#:
#: So all three flood parameters are REQUIRED. ``n_anchors`` moves net attack
#: success from 0.024 (64 anchors) to 0.400 (16,384) in v01's anchor-directed
#: series; spread and jitter decide whether the poison lands on the manifold at
#: all. None of the three has a defensible default, and a default here is a
#: silent choice of result.


def build_attack(attack: str, *, ref_emb: np.ndarray, ref_labels: np.ndarray,
                 ref_ids: np.ndarray, source: str, target: str, rate: float,
                 seed: int, alpha_range: tuple[float, float] = (0.3, 0.9),
                 n_anchors: int | None = None,
                 spread: float | None = None,
                 jitter: float | None = None,
                 label_change: str = LabelChange.RELABEL,
                 selection: str | None = None,
                 hub_nbr: np.ndarray | None = None,
                 sites: str = Sites.ANCHOR) -> dict[str, Any]:
    """One poison set, for any attack, in the ``poison_set`` contract shape.

    Args:
        attack: an :class:`schemas.Attack` value.
        ref_emb: (n_ref, 128) float32 unit-norm reference vectors.
        ref_labels: (n_ref,) str reference labels. ref_ids: (n_ref,) str cell ids.
        source: celltype being poisoned. target: label the attacker forges.
        rate: poison count as a fraction of the SOURCE CELLTYPE, so the attack
            cannot silently change identity as the budget grows.
        seed: passed in, never generated.
        alpha_range: ``interpolate`` only -- how far toward the target.
        n_anchors: ``flood`` only, REQUIRED there -- how many clusters the budget
            splits across. spread/jitter: ``flood`` only.
        label_change: ``relabel`` forges ``target``; ``keep`` leaves the record's
            true label. Applies to EVERY geometry including flood, which is what
            makes ``synthesised x keep`` -- the geometry control for flood --
            expressible at all.
        selection: how records are drawn; None derives it from ``attack``.
            Ignored by ``flood``, whose anchors are its own selection rule.
        hub_nbr: (n_q, k) observed neighbour rows, ``hub`` selection only.
        sites: ``flood`` only -- where its clusters sit, on real reference cells
            or on cluster centres of the celltype (:class:`schemas.Sites`).

    Returns:
        ``{emb, label, provenance, attack, alpha}`` as CLAUDE.md §5 specifies.
        Flood records are synthetic -- rotated and jittered, matching no single
        reference cell -- so their provenance is ``"synthetic"``, which is what
        that field's contract already reserves for constructed vectors.

    Raises:
        ValueError: unknown attack, or ``flood`` without ``n_anchors``.
    """
    from schemas import (POISON_ALPHA, POISON_ATTACK, POISON_EMB,  # noqa: PLC0415
                         POISON_LABEL, POISON_PROVENANCE)

    if attack == Attack.FLOOD:
        missing = [n for n, v in (("n_anchors", n_anchors), ("spread", spread),
                                  ("jitter", jitter)) if v is None]
        if missing:
            raise ValueError(
                f"flood requires {', '.join(missing)} explicitly. None has a "
                "defensible default: n_anchors ran 0.024 to 0.400 across v01's "
                "anchor-directed series, and spread/jitter decide whether the "
                "poison lands on the source manifold at all.")
        from flood_common import build_flood  # noqa: PLC0415
        pool = ref_emb[np.flatnonzero(ref_labels == source)]
        if pool.size == 0:
            raise ValueError(f"no reference cells carry the source term {source!r}")
        emb = build_flood(pool, n_anchors, rate, spread, jitter, seed,
                          sites=sites)
        n_p = len(emb)
        return {POISON_EMB: emb.astype(EMB_DTYPE),
                POISON_LABEL: np.full(
                    n_p, target if label_change == LabelChange.RELABEL
                    else source),
                POISON_PROVENANCE: np.full(n_p, "synthetic"),
                POISON_ATTACK: str(attack),
                POISON_ALPHA: np.full(n_p, np.nan, dtype=EMB_DTYPE)}

    from make_poison import build_poison  # noqa: PLC0415
    return build_poison(ref_emb=ref_emb, ref_labels=ref_labels, ref_ids=ref_ids,
                        attack=attack, source=source, target=target, rate=rate,
                        seed=seed, alpha_range=alpha_range,
                        label_change=label_change, selection=selection,
                        hub_nbr=hub_nbr)


def params_of(attack: str, *, n_anchors: int | None = None,
              alpha_range: tuple[float, float] = (0.3, 0.9),
              spread: float | None = None,
              jitter: float | None = None,
              sites: str | None = None) -> dict[str, Any]:
    """The attack-specific knobs, for writing into a results table.

    Null where they do not apply, so one table holds every attack without a
    reader having to know which columns belong to which.
    """
    f = attack == Attack.FLOOD
    i = attack == Attack.INTERPOLATE
    return {"n_anchors": int(n_anchors) if f and n_anchors else None,
            "flood_spread": float(spread) if f else None,
            "flood_jitter": float(jitter) if f else None,
            "flood_sites": str(sites) if f and sites else None,
            "alpha_lo": float(alpha_range[0]) if i else None,
            "alpha_hi": float(alpha_range[1]) if i else None}
