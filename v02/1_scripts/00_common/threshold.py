"""The threshold dose and its donor-paired interval. SPEC.md §5.4c.

A ladder of doses is not a result; the result it supports is one number per
condition -- the dose at which the attack starts working -- with an interval on
it. This module is the estimator, kept apart from the step that drives it so the
statistics can be tested on hand-built curves with no sweep output in sight.

**No parametric fit, and that is a measured decision.** The obvious move is a
4PL/Hill fit and an EC50. It is wrong for this family: ``02b_01``'s ``duplicate``
plateaus near 0.10 from dose 0.5 and would fit fine, but ``relabel`` reads 0.245
at dose 1.0 and 0.390 at 1.25 and is still climbing at the top of the ladder.
Its asymptote has never been observed, so fitting one would invent the plateau
and every EC50 from it would be extrapolation reported as measurement. Isotonic
regression assumes only that more poison does not help the defender.

**Censoring is a result, not a missing value.** See :class:`schemas.Censor`.
"""
from __future__ import annotations

import numpy as np

from schemas import CENSOR_MAX_FRAC, N_BOOTSTRAP, Censor


def isotonic(y: np.ndarray) -> np.ndarray:
    """Least-squares monotone-increasing fit, column-wise, equal weights.

    Uses the max-min formula, ``yhat[i] = max_{j<=i} min_{k>=i} mean(y[j..k])``,
    which is exact for L2 isotonic regression and -- unlike a pool-adjacent-
    violators loop -- vectorises over every bootstrap replicate at once. The
    ladder has 14 points, so the O(n^2) window table is 196 entries per replicate
    and the quadratic cost is irrelevant beside avoiding 2000 Python loops.

    Equal weights are correct HERE and would not be in general: the denominator
    of ``retrieval`` and ``net_attack_success`` is the number of queries in
    scope, and the same queries are scored at every dose, so every point on the
    curve carries the same weight. :func:`threshold_curve` asserts this rather
    than assuming it.

    Args:
        y: (n_doses, n_cols) float64, ordered by increasing dose. NaN-free.

    Returns:
        (n_doses, n_cols) the monotone-increasing fit of each column.
    """
    n = len(y)
    cs = np.concatenate([np.zeros((1, y.shape[1])), np.cumsum(y, axis=0)])
    j = np.arange(n)[:, None, None]
    k = np.arange(n)[None, :, None]
    # (n, n, n_cols); entries with k < j are never read, see the slice below
    with np.errstate(divide="ignore", invalid="ignore"):
        m = (cs[np.arange(n) + 1][None, :, :] - cs[np.arange(n)][:, None, :]) \
            / (k - j + 1)
    out = np.empty_like(y)
    for i in range(n):
        out[i] = m[:i + 1, i:, :].min(axis=1).max(axis=0)
    return out


def crossing(dose: np.ndarray, g: np.ndarray, tau: float
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Where each monotone curve first exceeds ``tau``.

    Interpolates log-linearly inside the bracketing pair, because the ladder is
    roughly geometric at the bottom (0.002, 0.005, 0.01, ...) and a linear
    interpolation between 0.002 and 0.005 would sit in the wrong place by more
    than the gap itself.

    Args:
        dose: (n_doses,) strictly increasing, all > 0.
        g: (n_doses, n_cols) monotone-increasing curves.
        tau: the level to cross.

    Returns:
        ``(d_tau, lo_bracket, hi_bracket, censor_code)``, each (n_cols,).
        ``censor_code`` is 0 none, 1 left, 2 right; ``d_tau`` and the brackets
        are NaN wherever it is nonzero.
    """
    n_cols = g.shape[1]
    d = np.full(n_cols, np.nan)
    lo = np.full(n_cols, np.nan)
    hi = np.full(n_cols, np.nan)
    code = np.zeros(n_cols, np.int8)

    code[g[0] > tau] = 1                       # already above at the lowest dose
    code[g[-1] <= tau] = 2                     # never gets there
    live = code == 0
    if not live.any():
        return d, lo, hi, code

    # first index that exceeds tau; guaranteed to exist and to be >= 1 for live
    # columns, because g[0] <= tau < g[-1] and g is non-decreasing
    i = (g[:, live] > tau).argmax(axis=0)
    col = np.flatnonzero(live)
    g_lo, g_hi = g[i - 1, col], g[i, col]
    d_lo, d_hi = dose[i - 1], dose[i]
    # g_hi > tau >= g_lo, so the denominator is strictly positive
    t = (tau - g_lo) / (g_hi - g_lo)
    d[col] = np.exp(np.log(d_lo) + t * (np.log(d_hi) - np.log(d_lo)))
    lo[col], hi[col] = d_lo, d_hi
    return d, lo, hi, code


def threshold_curve(dose: np.ndarray, num: np.ndarray, den: np.ndarray,
                    tau: float, boot_seed: int,
                    n_boot: int = N_BOOTSTRAP) -> dict[str, float]:
    """D_tau and its donor-paired bootstrap interval for one condition.

    The donor draw is held FIXED across every dose within a replicate, and the
    reason is correctness rather than tightness -- the pairing makes this
    interval WIDER. The SAME donors are scored at every dose, so the per-dose
    errors are positively correlated through a common donor effect. Pairing
    reproduces that correlation. Independent per-dose draws destroy it, letting
    the common donor effect average out across the ladder -- and a crossing
    point is most sensitive to exactly that, a coherent shift of the whole
    curve. So an unpaired bootstrap UNDERSTATES this interval; measured at 0.21
    against 0.16 at 78 donors, and narrower in all nine configurations tried.
    Pairing is required for the interval to be honest, not to make it small.

    Implemented by drawing one (n_boot, n_donors) multiplicity matrix and
    applying it to every dose at once.

    Donors contributing no query to this scope are dropped BEFORE resampling.
    They reach here as structural 0/0 rows (``schemas.BY_DONOR_DOC``); leaving
    them in would draw a random number of them per replicate and quietly vary
    the effective sample size.

    Args:
        dose: (n_doses,) strictly increasing.
        num: (n_doses, n_donors) per-donor numerator sums.
        den: (n_doses, n_donors) per-donor denominator sums.
        tau: the level D_tau is defined against.
        boot_seed: passed in, never generated.
        n_boot: bootstrap replicates.

    Returns:
        ``{d_tau, ci_lo, ci_hi, d_lo_bracket, d_hi_bracket, censor,
        frac_censor_left, frac_censor_right, n_donors}``. Every dose value is
        NaN wherever ``censor`` is not ``none``; ``censor`` is None where the
        scope held no scored query at all, which is NOT censoring.
    """
    nan = float("nan")
    keep = den.sum(axis=0) > 0
    out: dict[str, float] = {
        "d_tau": nan, "ci_lo": nan, "ci_hi": nan, "d_lo_bracket": nan,
        "d_hi_bracket": nan, "censor": None, "frac_censor_left": nan,
        "frac_censor_right": nan, "n_donors": int(keep.sum()),
    }
    if not keep.any():
        return out
    num, den = num[:, keep], den[:, keep]

    # Equal-weight isotonic is only right when every dose carries the same
    # denominator. It does here -- the same queries are scored at every dose --
    # and a silent violation would bias the fit toward the heavier doses.
    tot = den.sum(axis=1)
    if not np.allclose(tot, tot[0]):
        raise ValueError(
            f"denominator varies across doses ({tot.min()} to {tot.max()}); "
            "equal-weight isotonic regression is not valid for this indicator")

    point = isotonic((num.sum(axis=1) / tot)[:, None])
    d, lo, hi, code = crossing(dose, point, tau)

    rng = np.random.default_rng(boot_seed)
    n_d = num.shape[1]
    pick = rng.integers(0, n_d, size=(n_boot, n_d))
    # multiplicity matrix: C[b, j] = times donor j was drawn in replicate b.
    # Applying it as a matmul re-pools every dose under ONE draw.
    flat = (np.arange(n_boot)[:, None] * n_d + pick).ravel()
    c = np.bincount(flat, minlength=n_boot * n_d).reshape(n_boot, n_d).astype(float)
    bn, bd = num @ c.T, den @ c.T                       # (n_doses, n_boot)
    good = bd[0] > 0
    gb = isotonic(bn[:, good] / bd[:, good])
    db, _, _, cb = crossing(dose, gb, tau)

    out["frac_censor_left"] = float((cb == 1).mean())
    out["frac_censor_right"] = float((cb == 2).mean())

    if code[0] == 1 or out["frac_censor_left"] > CENSOR_MAX_FRAC:
        out["censor"] = str(Censor.LEFT)
        return out
    if code[0] == 2 or out["frac_censor_right"] > CENSOR_MAX_FRAC:
        out["censor"] = str(Censor.RIGHT)
        return out

    ok = np.isfinite(db)
    out.update(censor=str(Censor.NONE), d_tau=float(d[0]),
               d_lo_bracket=float(lo[0]), d_hi_bracket=float(hi[0]),
               ci_lo=float(np.percentile(db[ok], 2.5)),
               ci_hi=float(np.percentile(db[ok], 97.5)))
    return out
