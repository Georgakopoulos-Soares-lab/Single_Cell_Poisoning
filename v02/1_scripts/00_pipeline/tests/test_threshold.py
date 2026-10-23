"""The threshold estimator, on curves whose answer is known by hand.

SPEC.md §5.4c. These run on constructed curves rather than sweep output, so the
statistics are checked before any 40-hour run exists to check them against.
"""
import numpy as np
import pytest

from schemas import Censor
from threshold import crossing, isotonic, threshold_curve

LADDER = np.array([0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
                   0.5, 0.7, 0.8, 0.85, 0.9, 1.0, 1.25])


# ── isotonic ────────────────────────────────────────────────────────────────

def test_isotonic_leaves_a_monotone_curve_alone() -> None:
    y = np.array([0.0, 0.1, 0.2, 0.5, 0.9])[:, None]
    assert np.allclose(isotonic(y), y)


def test_isotonic_pools_one_violation_to_its_mean() -> None:
    """[0, 0.6, 0.2, 0.9] -> the middle pair pools to 0.4 each, by hand."""
    y = np.array([0.0, 0.6, 0.2, 0.9])[:, None]
    assert np.allclose(isotonic(y).ravel(), [0.0, 0.4, 0.4, 0.9])


def test_isotonic_flattens_a_decreasing_curve_to_its_mean() -> None:
    y = np.array([1.0, 0.8, 0.6, 0.2])[:, None]
    assert np.allclose(isotonic(y).ravel(), 0.65)


def test_isotonic_is_columnwise() -> None:
    """Columns are independent; a violation in one must not touch the other."""
    y = np.column_stack([np.array([0.0, 0.6, 0.2, 0.9]),
                         np.array([0.0, 0.1, 0.2, 0.3])])
    got = isotonic(y)
    assert np.allclose(got[:, 0], [0.0, 0.4, 0.4, 0.9])
    assert np.allclose(got[:, 1], [0.0, 0.1, 0.2, 0.3])


def test_isotonic_matches_a_reference_pava() -> None:
    """Agrees with a plain pool-adjacent-violators loop on random input."""
    def pava(v: np.ndarray) -> np.ndarray:
        v = v.astype(float).copy()
        w = np.ones(len(v))
        i = 0
        while i < len(v) - 1:
            if v[i] <= v[i + 1]:
                i += 1
                continue
            tot = w[i] + w[i + 1]
            v[i] = (v[i] * w[i] + v[i + 1] * w[i + 1]) / tot
            w[i] = tot
            v = np.delete(v, i + 1)
            w = np.delete(w, i + 1)
            i = max(i - 1, 0)
        return np.repeat(v, w.astype(int))

    rng = np.random.default_rng(0)
    for _ in range(25):
        y = rng.random(14)
        assert np.allclose(isotonic(y[:, None]).ravel(), pava(y))


# ── crossing ────────────────────────────────────────────────────────────────

def test_crossing_brackets_and_interpolates() -> None:
    """tau exactly midway in LOG dose between two points with rates 0 and 1.

    dose 0.01 -> 0.0 and dose 0.1 -> 1.0, tau 0.5, so t = 0.5 and the answer is
    exp(mean(log(0.01), log(0.1))) = sqrt(0.001) ~ 0.0316, NOT the arithmetic
    midpoint 0.055.
    """
    dose = np.array([0.01, 0.1])
    g = np.array([0.0, 1.0])[:, None]
    d, lo, hi, code = crossing(dose, g, 0.5)
    assert code[0] == 0
    assert (lo[0], hi[0]) == (0.01, 0.1)
    assert d[0] == pytest.approx(np.sqrt(0.001))
    assert d[0] < 0.055


def test_crossing_reports_left_and_right_censoring() -> None:
    dose = np.array([0.01, 0.1, 1.0])
    above = np.array([0.9, 0.95, 0.99])[:, None]
    below = np.array([0.0, 0.01, 0.02])[:, None]
    assert crossing(dose, above, 0.5)[3][0] == 1
    assert crossing(dose, below, 0.5)[3][0] == 2
    for g in (above, below):
        assert np.isnan(crossing(dose, g, 0.5)[0][0])


def test_crossing_at_exactly_tau_is_not_a_crossing() -> None:
    """D_tau is where the curve EXCEEDS tau, so touching it is right-censored."""
    dose = np.array([0.01, 0.1])
    g = np.array([0.0, 0.5])[:, None]
    assert crossing(dose, g, 0.5)[3][0] == 2


# ── threshold_curve ─────────────────────────────────────────────────────────

def _world(rates: np.ndarray, n_donors: int = 40, n_per: int = 50):
    """Per-donor counts realising a target rate at each dose, exactly."""
    den = np.full((len(rates), n_donors), float(n_per))
    num = np.round(den * rates[:, None])
    return num, den


def _world_het(rates: np.ndarray, n_donors: int = 40, n_per: int = 50,
               spread: float = 0.25):
    """The same, with donor-to-donor variation so the bootstrap has something to find.

    Each donor carries a fixed offset applied at EVERY dose, which keeps the
    denominator constant across doses -- :func:`threshold_curve` rejects a
    dose-varying denominator, and rightly.
    """
    rng = np.random.default_rng(1)
    off = rng.uniform(-spread, spread, n_donors)
    den = np.full((len(rates), n_donors), float(n_per))
    num = np.round(den * np.clip(rates[:, None] + off[None, :], 0.0, 1.0))
    return num, den


def test_threshold_recovers_a_known_step() -> None:
    """A curve that jumps 0 -> 1 between two ladder points brackets them."""
    rates = np.where(LADDER < 0.5, 0.0, 1.0)
    num, den = _world(rates)
    got = threshold_curve(LADDER, num, den, 0.5, boot_seed=0, n_boot=200)
    assert got["censor"] == str(Censor.NONE)
    assert (got["d_lo_bracket"], got["d_hi_bracket"]) == (0.25, 0.5)
    assert 0.25 < got["d_tau"] <= 0.5
    assert got["ci_lo"] <= got["d_tau"] <= got["ci_hi"]


def test_threshold_left_censors_a_saturated_attack() -> None:
    """The modify arm's shape: already high at the lowest dose."""
    num, den = _world(np.full(len(LADDER), 0.9))
    got = threshold_curve(LADDER, num, den, 0.5, boot_seed=0, n_boot=200)
    assert got["censor"] == str(Censor.LEFT)
    assert np.isnan(got["d_tau"]) and np.isnan(got["ci_lo"])
    assert got["frac_censor_left"] == 1.0


def test_threshold_right_censors_an_inert_attack() -> None:
    """interpolate's shape: flat and near zero everywhere. A finding, not a gap."""
    num, den = _world(np.full(len(LADDER), 0.004))
    got = threshold_curve(LADDER, num, den, 0.05, boot_seed=0, n_boot=200)
    assert got["censor"] == str(Censor.RIGHT)
    assert np.isnan(got["d_tau"])
    assert got["frac_censor_right"] == 1.0


def test_threshold_is_none_not_censored_when_the_scope_is_empty() -> None:
    """No scored query is NOT censoring, and must not read as a finding."""
    den = np.zeros((len(LADDER), 12))
    got = threshold_curve(LADDER, den.copy(), den, 0.05, boot_seed=0, n_boot=50)
    assert got["censor"] is None
    assert got["n_donors"] == 0


def test_threshold_drops_structural_zero_donors_before_resampling() -> None:
    """Padding with 0/0 donors must not change the answer.

    02b_05 emits the full 78-donor roster with the attacker's own donors at 0/0.
    If those were resampled, the effective sample size would vary per replicate
    and the interval would widen for a reason unrelated to the data.
    """
    rates = np.where(LADDER < 0.5, 0.0, 1.0)
    num, den = _world(rates, n_donors=20)
    pad = np.zeros((len(LADDER), 30))
    a = threshold_curve(LADDER, num, den, 0.5, boot_seed=0, n_boot=300)
    b = threshold_curve(LADDER, np.hstack([num, pad]), np.hstack([den, pad]),
                        0.5, boot_seed=0, n_boot=300)
    assert a["n_donors"] == b["n_donors"] == 20
    assert a == b


def test_threshold_rejects_a_dose_varying_denominator() -> None:
    """Equal-weight isotonic is invalid there, and must fail loudly."""
    num, den = _world(np.linspace(0, 1, len(LADDER)))
    den[3] *= 2
    with pytest.raises(ValueError, match="denominator varies across doses"):
        threshold_curve(LADDER, num, den, 0.5, boot_seed=0, n_boot=50)


def test_identical_donors_give_a_degenerate_interval() -> None:
    """With no donor-to-donor variation the interval is a point, and should be.

    ``_world`` gives every donor the same counts, so every bootstrap replicate
    rebuilds the same curve. An interval of width zero is the honest answer
    there; a nonzero one would mean the resampling had found variation that is
    not in the data.
    """
    num, den = _world(np.linspace(0.0, 0.9, len(LADDER)))
    got = threshold_curve(LADDER, num, den, 0.5, boot_seed=7, n_boot=200)
    assert got["ci_lo"] == got["ci_hi"] == got["d_tau"]


def test_threshold_interval_is_reproducible_from_the_seed() -> None:
    """Same seed, same interval; different seed, a different one.

    Needs DONOR HETEROGENEITY to be a real test -- see
    :func:`test_identical_donors_give_a_degenerate_interval`.
    """
    num, den = _world_het(np.linspace(0.0, 0.9, len(LADDER)))
    a = threshold_curve(LADDER, num, den, 0.5, boot_seed=7, n_boot=400)
    b = threshold_curve(LADDER, num, den, 0.5, boot_seed=7, n_boot=400)
    c = threshold_curve(LADDER, num, den, 0.5, boot_seed=8, n_boot=400)
    assert a == b
    assert a["ci_lo"] != c["ci_lo"]
    assert a["ci_lo"] < a["d_tau"] < a["ci_hi"]


def test_unpaired_draw_understates_the_interval() -> None:
    """The fixed-draw rule earns its place, measured rather than asserted.

    SPEC.md §5.4c holds one donor draw across every dose, and the reason is
    CORRECTNESS rather than tightness -- the pairing makes the interval WIDER.

    The same donors are scored at every dose, so the per-dose errors are
    positively correlated through a common donor effect. Pairing reproduces that
    correlation; independent per-dose draws destroy it, letting the common donor
    effect average out across the ladder. A crossing point is most sensitive to
    precisely that -- a coherent shift of the whole curve up or down -- so the
    unpaired bootstrap comes back too narrow.

    This test asserts the direction. An earlier version of this file asserted
    the opposite and failed, which is how the claim got checked.
    """
    num, den = _world_het(np.linspace(0.0, 0.9, len(LADDER)))
    paired = threshold_curve(LADDER, num, den, 0.5, boot_seed=0, n_boot=400)

    rng = np.random.default_rng(0)
    scrambled = np.stack([num[d, rng.permutation(num.shape[1])]
                          for d in range(len(LADDER))])
    unpaired = threshold_curve(LADDER, scrambled, den, 0.5, boot_seed=0,
                               n_boot=400)
    assert (paired["ci_hi"] - paired["ci_lo"]) > (unpaired["ci_hi"]
                                                  - unpaired["ci_lo"])


def test_threshold_rises_with_tau() -> None:
    """A harder bar is reached later. Monotone in tau, by construction."""
    num, den = _world(np.linspace(0.0, 1.0, len(LADDER)))
    d = [threshold_curve(LADDER, num, den, t, boot_seed=0, n_boot=100)["d_tau"]
         for t in (0.05, 0.25, 0.5)]
    assert d[0] < d[1] < d[2]
