"""The oracle against two independent implementations of the same curve.

hdr-toys is a GPU shader, libplacebo a C library. Neither is a source of
truth, but three implementations agreeing on the same numbers rules out a
transcription mistake in any one of them.
"""

import numpy as np
import pytest

import hdrtoys_port
from bt2390_ref import Eetf, pq_eotf, pq_inverse_eotf

CASES = [
    # LB, LW, Lmin, Lmax
    (0.0, 1000.0, 0.0, 203.0),
    (0.0, 10000.0, 0.0, 203.0),
    (0.0001, 1000.0, 0.0, 203.0),
    (0.0001, 4000.0, 0.0, 100.0),
    (0.005, 737.0, 0.0, 203.0),
    (0.0, 1000.0, 0.5, 203.0),
    (0.0001, 1000.0, 1.0, 203.0),
]


@pytest.mark.parametrize("lb,lw,lmin,lmax", CASES)
def test_curve_matches_the_hdrtoys_port(lb, lw, lmin, lmax):
    """Same four luminances, and inside the domain where the port's clamp is idle."""
    curve = Eetf(lb, lw, lmin, lmax)
    e = np.linspace(curve.pq_lb, curve.pq_lw, 200001)
    theirs = hdrtoys_port.f(
        e,
        iw=curve.pq_lw,
        ib=curve.pq_lb,
        ow=float(pq_inverse_eotf(lmax)),
        ob=float(pq_inverse_eotf(lmin)),
    )
    ours = curve(e)
    # The port clamps its result to [ob, ow]. A positive black lift pushes the
    # peak just above ow, so the two differ only where that clamp bites.
    overshoot = (
        max(curve.min_lum, 0.0)
        * (1.0 - curve.max_lum) ** 4
        * (curve.pq_lw - curve.pq_lb)
    )
    assert np.max(np.abs(ours - theirs)) <= overshoot + 1e-15
    idle = ours <= float(pq_inverse_eotf(lmax))
    assert np.max(np.abs(ours[idle] - theirs[idle])) < 1e-15


def libplacebo_curve(e, pq_lb, pq_lw, pq_min, pq_max, knee_offset=0.5):
    """The BT.2390 curve libplacebo implements, transcribed from its bt2390().

    Three things differ from the report. The knee position is a parameter,
    KS = (1 + k) maxLum - k, and BT.2390 prints the k = 0.5 instance. The black
    lift takes the exponent min(1 / minLum, 4) rather than a fixed 4, which
    only bites for a target black above a quarter of the source span. And the
    lifted curve is then rescaled by a gain that puts the peak back on maxLum,
    which the report does not do, so the two curves differ wherever minLum is
    not zero. The lift itself applies whatever the sign of minLum, so black
    lands at PQ(Lmin) here exactly as it does in the report.

    This is the curve before pl_tone_map_generate clamps the finished lookup
    table to the output range, which is what a comparison of curves wants.
    Nothing is copied from libplacebo; only the parameterisation is.
    """
    span = pq_lw - pq_lb
    min_lum = (pq_min - pq_lb) / span
    max_lum = (pq_max - pq_lb) / span
    ks = (1.0 + knee_offset) * max_lum - knee_offset
    bp = min(1.0 / min_lum, 4.0) if min_lum > 0.0 else 4.0
    gain_inv = 1.0 + min_lum / max_lum * (1.0 - max_lum) ** bp
    gain = 1.0 / gain_inv if max_lum < 1.0 else 1.0

    x = np.clip((np.asarray(e) - pq_lb) / span, 0.0, 1.0)
    if ks < 1.0:
        t = np.clip((x - ks) / (1.0 - ks), 0.0, None)
        t2, t3 = t * t, t * t * t
        spline = (
            (2.0 * t3 - 3.0 * t2 + 1.0) * ks
            + (t3 - 2.0 * t2 + t) * (1.0 - ks)
            + (-2.0 * t3 + 3.0 * t2) * max_lum
        )
        x = np.where(x >= ks, spline, x)
    lifted = x + min_lum * (1.0 - x) ** bp
    x = np.where(x < 1.0, gain * (lifted - min_lum) + min_lum, x)
    return x * span + pq_lb


def bt1886_codes(pq_values, lmax, levels=256):
    """PQ code values as the integer levels a BT.1886 display would show.

    The curves are compared in PQ, which is perceptual but not a display
    quantity; this says what a difference there is worth once the result is
    encoded for an SDR display.
    """
    v = pq_eotf(np.asarray(pq_values)) / lmax
    return np.rint(np.clip(v, 0.0, 1.0) ** (1.0 / 2.4) * (levels - 1))


@pytest.mark.parametrize("lb,lw,lmin,lmax", CASES)
def test_curve_matches_the_libplacebo_form(lb, lw, lmin, lmax):
    """Identical where the lift is idle, within an 8-bit code where it is not."""
    curve = Eetf(lb, lw, lmin, lmax)
    e = np.linspace(curve.pq_lb, curve.pq_lw, 200001)
    ours = curve(e)
    theirs = libplacebo_curve(
        e,
        curve.pq_lb,
        curve.pq_lw,
        float(pq_inverse_eotf(lmin)),
        float(pq_inverse_eotf(lmax)),
        knee_offset=0.5,
    )
    if curve.min_lum == 0.0:
        assert np.max(np.abs(ours - theirs)) < 1e-15
        return

    # With a lift of either sign the two part company, because libplacebo
    # rescales the lifted curve and the report does not. Black is the one
    # point that still has to agree: both put E1 = 0 at PQ(Lmin).
    assert abs(ours[0] - theirs[0]) < 1e-15
    assert np.max(np.abs(ours - theirs)) < 3.5e-4
    assert np.max(np.abs(bt1886_codes(ours, lmax) - bt1886_codes(theirs, lmax))) <= 1


def test_a_different_knee_offset_is_a_different_curve():
    """Guards the comparison above from passing for the wrong reason."""
    curve = Eetf(0.0, 1000.0, 0.0, 203.0)
    e = np.linspace(curve.pq_lb, curve.pq_lw, 2001)
    other = libplacebo_curve(
        e,
        curve.pq_lb,
        curve.pq_lw,
        float(pq_inverse_eotf(0.0)),
        float(pq_inverse_eotf(203.0)),
        knee_offset=1.0,
    )
    assert np.max(np.abs(curve(e) - other)) > 1e-3


def test_port_keeps_its_own_defaults():
    """The port falls back to 0.001 and 1000 cd/m2. The plugin raises instead."""
    p = hdrtoys_port.Params()
    assert float(hdrtoys_port.get_min_i(p)) == float(pq_inverse_eotf(0.001))
    assert float(hdrtoys_port.get_max_i(p)) == float(pq_inverse_eotf(1000.0))


@pytest.mark.parametrize(
    "representation", ["ictcp", "ycbcr", "yrgb", "prergb", "maxrgb"]
)
def test_port_runs_every_representation(representation):
    rng = np.random.default_rng(61)
    rgb = rng.random((512, 3)) * 5.0
    out = hdrtoys_port.tone_map(rgb, representation)
    assert out.shape == rgb.shape
    assert np.all(np.isfinite(out))
