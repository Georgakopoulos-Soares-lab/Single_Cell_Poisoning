"""Save and reload one condition's graph retrieval.

Added 2026-10-04. The graph path is not reproducible (CLAUDE.md §1:
multi-threaded ``add_items``), so a retrieval thrown away cannot be recovered
by re-running the condition. Until now every defence step discarded its
neighbour lists after scoring, so when the accuracy metric turned out to be the
wrong quantity (diagnosis/2026-10-04_attack_vs_defence_none) the only fix was a
multi-day re-run. With the retrieval kept, any metric or defence can be
re-scored from disk without touching the index.

What is stored is exactly what cannot be regenerated: the neighbour ids and
cosines the attacked graph returned. The poison itself is NOT stored -- it is
deterministic in ``condition_seed`` and is rebuilt by the producing step's own
``build_attack`` / ``build_modify`` call -- but its SHA-1 digest is, so a
reload can prove the rebuilt poison is the one that was retrieved against.

One compressed ``.npz`` per condition under
``<cache>/retrievals/<step>_<tag>/``, written atomically (temp file, then
rename) so a shard killed mid-write never leaves a truncated file that looks
complete. Opt-in: a step writes only when ``BIORAG_SAVE_RETRIEVAL=1``.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
from typing import Any

import numpy as np

ENV = "BIORAG_SAVE_RETRIEVAL"


def enabled() -> bool:
    """True when the run asked for retrievals to be kept."""
    return os.environ.get(ENV, "") == "1"


def digest(a: np.ndarray) -> str:
    """SHA-1 of an array's bytes, the poison's identity."""
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


def path_for(root: pathlib.Path, step: str, tag: str, key: dict[str, Any]) -> pathlib.Path:
    """Deterministic file name for one condition.

    ``key`` holds the condition's identifying values (source, attack,
    selection, rate, ...). They are joined in sorted-key order so the same
    condition always maps to the same file.
    """
    def clean(v: Any) -> str:
        return str(v).replace(":", "").replace("/", "-").replace(" ", "")
    stem = "__".join(f"{k}-{clean(key[k])}" for k in sorted(key))
    return root / "retrievals" / f"{step}_{tag or 'default'}" / f"{stem}.npz"


def save(root: pathlib.Path, step: str, tag: str, key: dict[str, Any], *,
         ids: np.ndarray, cos: np.ndarray, meta: dict[str, Any],
         extra: dict[str, np.ndarray] | None = None) -> pathlib.Path:
    """Write one retrieval. Returns the path written.

    Args:
        ids: (n_q, k) neighbour ids as the graph returned them (poison at
            ids >= n_ref for an insert step). Stored as int32.
        cos: (n_q, k) cosines of those neighbours. Stored as float16, which is
            ample for re-ranking and halves the file.
        meta: JSON-serialisable facts needed to re-score: condition_seed,
            n_poison, poison_digest, target, n_ref, k, ef ... Stored verbatim.
        extra: further small arrays a step needs to rebuild its labels, e.g.
            the edited rows of a modify condition.
    """
    p = path_for(root, step, tag, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"ids": np.asarray(ids, dtype=np.int32),
              "cos": np.asarray(cos, dtype=np.float16),
              "meta": np.array(json.dumps({**meta, "key": key}, sort_keys=True,
                                          default=str))}
    for k, v in (extra or {}).items():
        arrays[f"x_{k}"] = np.asarray(v)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, p)
    return p


def load(path: pathlib.Path) -> dict[str, Any]:
    """Read one retrieval back: ``ids``, ``cos``, ``meta`` and any extras."""
    with np.load(path, allow_pickle=False) as z:
        out = {"ids": z["ids"].astype(np.int64), "cos": z["cos"].astype(np.float32),
               "meta": json.loads(str(z["meta"]))}
        for k in z.files:
            if k.startswith("x_"):
                out[k[2:]] = z[k]
    return out


def verify_poison(rec: dict[str, Any], poison_emb: np.ndarray) -> None:
    """Raise unless ``poison_emb`` is the poison this retrieval was taken with."""
    want = rec["meta"].get("poison_digest")
    got = digest(poison_emb)
    if want != got:
        raise ValueError(f"rebuilt poison digest {got[:12]} != stored {str(want)[:12]}; "
                         "this is not the poison the retrieval was taken against")
