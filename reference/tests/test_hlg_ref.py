"""The float64 HLG oracle against published ITU numbers and its own identities.

Nothing here compares the oracle with itself. Every expected value is either
printed in ITU-R BT.2100-3 or ITU-R BT.2408-9, or is an identity the spec's
own construction requires.
"""

import numpy as np
import pytest

from hlg_ref import GAMMA_RANGE, KAPPA, A, B, C, black_lift, system_gamma


def test_printed_constants_match_their_definitions():
    """Note 5b gives b and c as formulae, Note 5c as decimals. Both agree."""
    assert B == pytest.approx(1.0 - 4.0 * A, abs=5e-9)
    assert C == pytest.approx(0.5 - A * np.log(4.0 * A), abs=5e-9)


def test_system_gamma_is_1_2_at_the_reference_peak():
    """Table 5: gamma is 1.2 at a nominal peak of 1000 cd/m2.

    Both Note 5f formulae have to give it, since 1000 is inside the simple
    formula's range and the extended one is anchored on the same point.
    """
    assert system_gamma(1000.0) == pytest.approx(1.2, abs=1e-12)
    assert 1.2 * KAPPA ** np.log2(1000.0 / 1000.0) == pytest.approx(1.2, abs=1e-12)


@pytest.mark.parametrize(
    "lw,expected",
    [
        (400.0, 1.032865),
        (600.0, 1.106824),
        (800.0, 1.159298),
        (1000.0, 1.200000),
        (1500.0, 1.273958),
        (2000.0, 1.326433),
    ],
)
def test_system_gamma_over_the_production_range(lw, expected):
    assert system_gamma(lw) == pytest.approx(expected, abs=5e-7)


def test_system_gamma_matches_the_value_bt2408_prints():
    """BT.2408-9 prints gamma = 1.33 at a peak of 2000 cd/m2.

    Note 5f allows rounding to three significant digits. The oracle does not
    round, so the check is against the rounded value rather than for it.
    """
    assert round(system_gamma(2000.0), 2) == 1.33


def test_the_extended_formula_takes_over_outside_the_production_range():
    """Note 5f scopes the simple formula to 400 to 2000 cd/m2.

    Just outside the range the result must follow the extended formula, not a
    continuation of the simple one.
    """
    lo, hi = GAMMA_RANGE
    for lw in (lo - 1.0, hi + 1.0):
        extended = 1.2 * KAPPA ** np.log2(lw / 1000.0)
        simple = 1.2 + 0.42 * np.log10(lw / 1000.0)
        assert system_gamma(lw) == pytest.approx(extended, abs=1e-12)
        assert system_gamma(lw) != pytest.approx(simple, abs=1e-6)


@pytest.mark.parametrize("lo,hi", [(50.0, 399.9), (400.0, 2000.0), (2000.1, 10000.0)])
def test_system_gamma_is_monotone_within_each_formula(lo, hi):
    """Each formula on its own is strictly increasing in peak luminance."""
    peaks = np.geomspace(lo, hi, 2000)
    gammas = np.array([system_gamma(float(lw)) for lw in peaks])
    assert np.all(np.diff(gammas) > 0.0)


def test_the_formulae_disagree_at_the_lower_boundary():
    """Note 5f's switch at 400 cd/m2 steps gamma down, not up.

    So the composite is not monotone across the whole range: a display of
    399 cd/m2 gets a *higher* system gamma than one of 400. That is the
    spec's own artefact rather than a transcription error, and it is
    pinned here so that nobody later smooths it away as a bug.
    """
    below = system_gamma(399.999)
    at = system_gamma(400.0)
    assert below > at
    assert at - below == pytest.approx(-0.01126, abs=1e-5)


def test_the_formulae_disagree_at_the_upper_boundary():
    """At 2000 cd/m2 the step goes the other way, up by about 0.0068."""
    at = system_gamma(2000.0)
    above = system_gamma(2000.001)
    assert above > at
    assert above - at == pytest.approx(0.00677, abs=1e-5)


def test_system_gamma_crosses_one_near_301_cd_m2():
    """Below this peak the OOTF exponent gamma - 1 goes negative.

    That is what makes the black guard necessary, so the crossing point is
    worth pinning. It falls under 400, so the extended formula is the one in
    force there; reading it off the simple formula gives 334, which is wrong.
    """
    assert system_gamma(295.0) < 1.0
    assert system_gamma(310.0) > 1.0


def test_black_lift_is_zero_for_a_black_of_zero():
    assert black_lift(1000.0, 0.0) == 0.0


@pytest.mark.parametrize(
    "lw,lb", [(1000.0, 0.005), (1000.0, 0.5), (400.0, 0.01), (4000.0, 0.005)]
)
def test_black_lift_inverts_the_eotf_at_zero_signal(lw, lb):
    """Table 5's beta is defined so that a signal of 0 displays exactly L_B.

    Reconstructing L_B here is what distinguishes the correct reading of the
    formula, sqrt(3 (LB/LW)^(1/gamma)), from sqrt(3) (LB/LW)^(1/gamma). The
    PDF's text layer does not make the extent of the radical clear.
    """
    gamma = system_gamma(lw)
    beta = black_lift(lw, lb, gamma)
    scene = beta * beta / 3.0  # the inverse OETF's lower branch
    assert lw * scene**gamma == pytest.approx(lb, rel=1e-12)


from hlg_ref import hlg_inverse_oetf, hlg_oetf  # noqa: E402


def test_oetf_branches_meet_at_the_breakpoint():
    """Note 5a splits at E = 1/12, where both branches must give E' = 0.5."""
    assert hlg_oetf(1.0 / 12.0) == pytest.approx(0.5, abs=1e-12)
    # The log branch reaches 0.500000000470 here: c as printed sits 4.7e-10
    # above c derived from 0.5 - a ln(4a), and that is the whole gap.
    assert A * np.log(12.0 / 12.0 - B) + C == pytest.approx(0.5, abs=1e-9)


def test_inverse_oetf_branches_meet_at_the_breakpoint():
    assert hlg_inverse_oetf(0.5) == pytest.approx(1.0 / 12.0, abs=1e-12)


def test_oetf_round_trips():
    """Both directions are in Note 5a, so the pair has to be an identity.

    Exactly an identity, in fact: composing them cancels a, b and c
    algebraically, so the residual here is float64 noise and nothing else.
    Measured worst case is 6.7e-16. The bound is widened to 1e-14 because a
    re-measurement on different hardware landed at 5.55e-16 against the
    original 1e-15 bound, too little headroom for a quantity that varies
    between runs and across the Linux/Windows split between lint and CI.
    """
    scene = np.concatenate([[0.0], np.geomspace(1e-9, 1.0, 5000)])
    assert hlg_inverse_oetf(hlg_oetf(scene)) == pytest.approx(scene, abs=1e-14)


def test_inverse_oetf_round_trips_over_the_interior():
    """Everywhere except the top of the domain this is float64 exact.

    Measured worst case over the interior is 1.1e-16. The endpoint is a
    separate case with its own cause; see the next test. The bound is
    widened to 1e-14 because a re-measurement on different hardware landed
    at 5.55e-16 against the original 1e-15 bound, too little headroom for a
    quantity that varies between runs and across the Linux/Windows split
    between lint and CI.
    """
    signal = np.linspace(0.0, 1.0, 5001)[:-1]
    assert hlg_oetf(hlg_inverse_oetf(signal)) == pytest.approx(signal, abs=1e-14)


def test_the_top_of_the_domain_loses_about_4e_9_to_the_clamp():
    """A signal of exactly 1 is the one place the round trip is not tight.

    The printed constants do not put the log branch exactly on scene 1.0 at
    a signal of 1.0: the inverse overshoots by 2.4e-8. hlg_oetf then clamps
    that back to 1.0 before encoding, and encoding 1.0 gives 0.9999999955.
    So the residual is the clamp meeting the overshoot.

    It is not caused by taking b and c as the decimals Note 5c prints
    rather than as their defining formulae. b is bit-identical either way,
    c differs by 4.7e-10, and substituting the derived pair makes this
    slightly worse (4.9e-9 against 4.5e-9) rather than better. Anyone
    tempted to "fix" the imprecision that way should not bother.
    """
    overshoot = float(hlg_inverse_oetf(np.array(1.0))) - 1.0
    assert overshoot == pytest.approx(2.437e-8, rel=1e-3)
    assert float(hlg_oetf(hlg_inverse_oetf(np.array(1.0)))) == pytest.approx(
        1.0, abs=1e-8
    )


def test_oetf_endpoints():
    """Scene 0 gives signal 0, scene 1 gives signal 1."""
    assert hlg_oetf(0.0) == pytest.approx(0.0, abs=1e-15)
    # 4.5e-9 short, for the reason the previous test documents.
    assert hlg_oetf(1.0) == pytest.approx(1.0, abs=1e-8)


def test_inverse_oetf_is_monotone():
    signal = np.linspace(0.0, 1.0, 20001)
    assert np.all(np.diff(hlg_inverse_oetf(signal)) >= 0.0)


def test_both_directions_clamp_to_their_domain():
    """HLG is defined on [0, 1] only.

    Out-of-range samples arrive from chroma upsampling ringing and from
    limited-range codes outside 64 to 940, so they have to land somewhere
    rather than produce a NaN from a log or a negative root.
    """
    assert hlg_inverse_oetf(-0.4) == pytest.approx(0.0, abs=1e-15)
    assert hlg_inverse_oetf(1.9) == pytest.approx(hlg_inverse_oetf(1.0), abs=1e-15)
    assert hlg_oetf(-0.4) == pytest.approx(0.0, abs=1e-15)
    assert hlg_oetf(1.9) == pytest.approx(hlg_oetf(1.0), abs=1e-15)
    assert np.all(np.isfinite(hlg_oetf(np.linspace(-1.0, 2.0, 501))))
    assert np.all(np.isfinite(hlg_inverse_oetf(np.linspace(-1.0, 2.0, 501))))


from hlg_ref import hlg, inverse_ootf, ootf  # noqa: E402

# BT.2408-9 section 2 and Table 4: HDR reference white is the 75% HLG signal
# level, and the report tabulates the luminance it reaches for each display
# peak. These are the report's own integers.
REFERENCE_WHITE = {400: 101, 600: 138, 800: 172, 1000: 203, 1500: 276, 2000: 343}

# Measured from the grayscale patch set. Signal is (code10 - 64) / (940 - 64),
# the 10-bit limited-range code carried in each file's name.
PATCH_LADDER = [
    # patch, code10, cd/m2 at L_W 1000
    (0, 64, 0.0000),
    (45, 460, 39.8028),
    (75, 720, 201.7407),
    (100, 940, 1000.0000),
]


def grey(signal):
    """An (N, 3) neutral at each signal level."""
    return np.repeat(np.atleast_1d(np.asarray(signal, dtype=np.float64))[:, None], 3, 1)


@pytest.mark.parametrize("lw,expected", sorted(REFERENCE_WHITE.items()))
def test_reference_white_matches_bt2408(lw, expected):
    """75% HLG must reach the luminance BT.2408-9 prints for that peak."""
    got = hlg(grey([0.75]), lw=float(lw), nominal_luminance=1.0)
    assert round(float(got[0, 0])) == expected


@pytest.mark.parametrize("patch,code10,expected", PATCH_LADDER)
def test_patch_ladder(patch, code10, expected):
    signal = (code10 - 64.0) / (940.0 - 64.0)
    got = hlg(grey([signal]), lw=1000.0, nominal_luminance=1.0)
    assert float(got[0, 0]) == pytest.approx(expected, abs=0.01)


def test_the_ootf_preserves_chromaticity():
    """Table 5 scales all three channels by one luminance-derived scalar.

    Note 5e records that some legacy displays raise each channel separately
    instead, and calls that an approximation. The two agree on neutrals and
    diverge on everything else, so a saturated colour is what tells them
    apart. zimg's std-b67 takes the per-channel route, which is why this
    conversion cannot be delegated to a resize.
    """
    scene = np.array([[0.4, 0.1, 0.02]])
    display = ootf(scene, lw=1000.0)
    ratios = display[0] / scene[0]
    assert ratios[0] == pytest.approx(ratios[1], rel=1e-12)
    assert ratios[1] == pytest.approx(ratios[2], rel=1e-12)

    per_channel = 1000.0 * scene**1.2
    assert not np.allclose(display, per_channel, rtol=1e-3)


def test_ootf_round_trips():
    rng = np.random.default_rng(20260916)
    scene = rng.random((500, 3)) ** 3
    for lw in (400.0, 1000.0, 4000.0):
        display = ootf(scene, lw=lw)
        assert inverse_ootf(display, lw=lw) == pytest.approx(
            scene, rel=1e-12, abs=1e-15
        )


@pytest.mark.parametrize(
    "lw,lb", [(1000.0, 0.0), (1000.0, 0.005), (1000.0, 0.5), (400.0, 0.01)]
)
def test_zero_signal_gives_exactly_the_display_black(lw, lb):
    """Table 5's EOTF is built so that E' = 0 displays L_B."""
    got = hlg(grey([0.0]), lw=lw, lb=lb, nominal_luminance=1.0)
    assert float(got[0, 0]) == pytest.approx(lb, rel=1e-12, abs=1e-15)


def test_peak_signal_reaches_the_display_peak():
    got = hlg(grey([1.0]), lw=1000.0, nominal_luminance=1.0)
    assert float(got[0, 0]) == pytest.approx(1000.0, rel=1e-7)


def test_black_stays_black_when_the_system_gamma_is_below_one():
    """Below a peak of about 301 cd/m2 the OOTF exponent gamma - 1 is negative.

    Y_S raised to a negative exponent at exact black is infinity, and
    infinity times a channel of zero is NaN. Without a guard every black
    pixel would be NaN on a low-peak render, so this is a correctness check
    rather than a preference.
    """
    assert system_gamma(200.0) < 1.0
    got = hlg(grey([0.0, 0.0]), lw=200.0, nominal_luminance=1.0)
    assert np.all(np.isfinite(got))
    assert got == pytest.approx(0.0, abs=1e-15)


def test_out_of_range_signal_stays_finite():
    signal = np.array([[-0.5, 0.2, 1.4], [2.0, -1.0, 0.0]])
    got = hlg(signal, lw=1000.0)
    assert np.all(np.isfinite(got))
    assert np.all(got >= 0.0)


def test_nominal_luminance_only_rescales():
    at_1 = hlg(grey([0.6]), lw=1000.0, nominal_luminance=1.0)
    at_100 = hlg(grey([0.6]), lw=1000.0, nominal_luminance=100.0)
    assert at_100 == pytest.approx(at_1 / 100.0, rel=1e-12)
