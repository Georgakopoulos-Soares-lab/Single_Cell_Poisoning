"""Stage 3 — build a poison set for one experimental condition.

Embedding space only. Poisoned records are drawn from, or interpolated between,
reference vectors that already exist; nothing is re-embedded here and the
reference index is never touched (``CLAUDE.md`` §1).
"""
# 1_scripts cannot be a package (leading digit), so the shared library in
# 00_common is put on the path and imported flat.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "00_common"))

# No `from __future__ import annotations` here: snakemake's script directive
# prepends a preamble, which would push it off line 1 and make it a SyntaxError.
# Python 3.12 evaluates the annotations used below natively.

import numpy as np

from schemas import (
    EMB_DTYPE,
    POISON_ALPHA,
    POISON_ATTACK,
    POISON_EMB,
    POISON_LABEL,
    POISON_PROVENANCE,
    Attack,
    LabelChange,
    Selection,
)
from selection import choose

DUPLICATE_POOL_FRACTION = 0.01
"""Fraction of the source pool that `duplicate` copies from.

Superseded by ``selection.CONCENTRATED_POOL_FRACTION``, which holds the same
0.01 and is what :func:`build_poison` now reads. Kept because v01 tables and
docs quote this name; the two are asserted equal in ``test_selection.py``.


`duplicate` amplifies a few records many times; `relabel` mislabels many
records once. Without a restricted pool the two coincide whenever the source
class is large, and the attack axis stops distinguishing anything."""


def build_poison(
    ref_emb: np.ndarray,
    ref_labels: np.ndarray,
    ref_ids: np.ndarray,
    attack: str,
    source: str,
    target: str,
    rate: float,
    seed: int,
    alpha_range: tuple[float, float] = (0.3, 0.9),
    label_change: str = LabelChange.RELABEL,
    selection: str | None = None,
    hub_nbr: np.ndarray | None = None,
) -> dict[str, np.ndarray | str]:
    """Construct the poisoned records for one condition.

    Args:
        ref_emb: (n_ref, d) float32, unit norm.
        ref_labels: (n_ref,) str, Cell Ontology terms.
        ref_ids: (n_ref,) str, reference cell identifiers.
        attack: one of :class:`~contracts.schemas.Attack`.
        source: CL term the attacker draws from.
        target: CL term the attacker claims. Every poisoned record gets it.
        rate: poisoned records as a fraction of the SOURCE CLASS, in (0, 1].
        seed: passed in, never generated.
        alpha_range: (lo, hi) interpolation strength, `interpolate` only.
        label_change: `relabel` forges `target`; `keep` leaves the record's TRUE
            label on it. `keep` is the control for every relabel result -- the
            claim that harm comes from the false annotation rather than from the
            added volume is otherwise asserted and unmeasured (SPEC.md 5.4).
        selection: how records are drawn. None derives it from `attack`, which
            is what the attack name encoded before selection became an axis:
            `duplicate` is concentrated, the others uniform. Passing it
            explicitly is what makes `relabel x hub` expressible.
        hub_nbr: (n_q, k) observed neighbour rows, `hub` selection only.

    Returns:
        Mapping matching the ``poison_set`` contract: ``emb`` (n_p, d) float32
        unit norm, ``label``, ``provenance``, ``attack``, ``alpha``.

    Raises:
        ValueError: the source term is absent from the reference, or the rate
            yields no records. Both would otherwise produce an empty poison set
            that reads as a robust system rather than a broken condition.
    """
    rng = np.random.default_rng(seed)

    src_idx = np.flatnonzero(ref_labels == source)
    if src_idx.size == 0:
        raise ValueError(f"no reference cells carry the source term {source!r}")

    # rate is a fraction of the SOURCE CLASS, not of the whole reference.
    # "a quarter of the macrophages in this atlas are mislabelled" is a threat
    # model an attacker could realise with one bad submission. "a quarter of
    # the whole atlas is poisoned macrophages" is not: the reference holds
    # 86,983 macrophages, and a quarter of 7.9M would need 1,978,473 of them.
    # Under the old basis every rate above ~1.1% forced `relabel` to reuse
    # cells, which silently turns it into `duplicate`.
    n_p = int(round(rate * src_idx.size))
    if n_p < 1:
        raise ValueError(
            f"rate {rate} over {src_idx.size:,} {source} cells yields no poison"
        )

    # Selection is ORTHOGONAL to geometry (SPEC.md 5.5). Derived from the
    # attack name when not given, so every existing caller draws exactly the
    # records it drew before -- asserted byte for byte in test_selection.py.
    if selection is None:
        selection = (Selection.CONCENTRATED if attack == Attack.DUPLICATE
                     else Selection.UNIFORM)
    chosen = choose(selection, src_idx, n_p, rng,
                    hub_nbr=hub_nbr, n_ref=len(ref_labels))

    if attack in (Attack.RELABEL, Attack.DUPLICATE):
        emb = ref_emb[chosen].astype(EMB_DTYPE)
        alpha = np.full(n_p, np.nan, dtype=EMB_DTYPE)
    elif attack == Attack.INTERPOLATE:
        tgt_idx = np.flatnonzero(ref_labels == target)
        if tgt_idx.size == 0:
            raise ValueError(f"no reference cells carry the target term {target!r}")
        centroid = ref_emb[tgt_idx].mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        alpha = rng.uniform(*alpha_range, size=n_p).astype(EMB_DTYPE)
        emb = (1.0 - alpha)[:, None] * ref_emb[chosen] + alpha[:, None] * centroid
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb.astype(EMB_DTYPE)
    else:
        raise ValueError(f"unknown attack {attack!r}")

    # The ONE place the label lies, and now the one place it may not.
    return {
        POISON_EMB: emb,
        POISON_LABEL: np.full(n_p, target if label_change == LabelChange.RELABEL
                              else source),
        POISON_PROVENANCE: ref_ids[chosen],
        POISON_ATTACK: str(attack),
        POISON_ALPHA: alpha,
    }


POISON_MEMBER = "member_source"
"""Which member of a combo each poisoned record came from.

Not part of the single-attack contract. Without it a combo's poison set is
unattributable, and the per-member comparison that the experiment exists to
make would be impossible."""


def build_combo_poison(
    ref_emb: np.ndarray,
    ref_labels: np.ndarray,
    ref_ids: np.ndarray,
    attack: str,
    members: list[dict[str, str]],
    rate: float,
    seed: int,
    alpha_range: tuple[float, float] = (0.3, 0.9),
) -> dict[str, np.ndarray | str]:
    """Poison several source classes at once.

    Args:
        ref_emb: (n_ref, d) float32, unit norm.
        ref_labels: (n_ref,) str, Cell Ontology terms.
        ref_ids: (n_ref,) str.
        attack: one of :class:`~contracts.schemas.Attack`, applied to every member.
        members: ``[{"source": CL id, "target": CL id}, ...]``, at least two.
        rate: fraction of EACH member's own source class. A member with more
            cells therefore contributes more records, which is what a single
            adversary poisoning several types at one intensity would produce.
        seed: passed in. Each member is offset from it so that two members of
            equal size do not draw the same row indices — that would be an
            artefact of the seeding, not a property of the attack.
        alpha_range: interpolation strength, `interpolate` only.

    Returns:
        The ``poison_set`` fields, concatenated across members, plus
        :data:`POISON_MEMBER` naming each record's originating source.

    Raises:
        ValueError: fewer than two members — that is a single attack, and
            running it through this path would make the two incomparable.
    """
    if len(members) < 2:
        raise ValueError(
            f"a combo needs at least two members, got {len(members)}; "
            f"use build_poison for a single source"
        )

    parts = []
    for offset, m in enumerate(members):
        part = build_poison(
            ref_emb=ref_emb, ref_labels=ref_labels, ref_ids=ref_ids,
            attack=attack, source=m["source"], target=m["target"],
            rate=rate,
            # Distinct per member: identical seeds would select identical row
            # indices whenever two source classes are the same size.
            seed=seed * 1000 + offset,
            alpha_range=alpha_range,
        )
        part[POISON_MEMBER] = np.full(len(part[POISON_EMB]), m["source"])
        parts.append(part)

    return {
        POISON_EMB: np.vstack([p[POISON_EMB] for p in parts]).astype(EMB_DTYPE),
        POISON_LABEL: np.concatenate([p[POISON_LABEL] for p in parts]),
        POISON_PROVENANCE: np.concatenate([p[POISON_PROVENANCE] for p in parts]),
        POISON_ATTACK: str(attack),
        POISON_ALPHA: np.concatenate([p[POISON_ALPHA] for p in parts]),
        POISON_MEMBER: np.concatenate([p[POISON_MEMBER] for p in parts]),
    }


# ── snakemake entry point ────────────────────────────────────────────────────
if "snakemake" in globals():  # pragma: no cover - exercised by the workflow
    import numpy as _np

    _ref = _np.load(snakemake.input.reference)  # noqa: F821
    _w = snakemake.wildcards  # noqa: F821
    _all = {p["name"]: p for p in snakemake.config["pairs"]}  # noqa: F821
    _pair = _all[_w.pair]
    _kw = dict(
        ref_emb=_ref["emb"],
        ref_labels=_ref["label"],
        ref_ids=_ref["cell_id"],
        attack=_w.attack,
        rate=float(_w.rate),
        seed=int(_w.seed),
        alpha_range=tuple(snakemake.config["poison"]["alpha_range"]),  # noqa: F821
    )
    if "members" in _pair:
        _out = build_combo_poison(
            members=[
                {"source": _all[m]["source"], "target": _all[m]["target"]}
                for m in _pair["members"]
            ],
            **_kw,
        )
    else:
        _out = build_poison(source=_pair["source"], target=_pair["target"], **_kw)
    _np.savez_compressed(snakemake.output[0], **_out)  # noqa: F821
