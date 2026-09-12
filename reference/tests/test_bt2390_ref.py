"""Checks on the PQ functions, the matrices and the EETF.

Every assertion here is a property the ITU text states or that follows from
it by derivation. The plugin is later held to the same numbers.
"""

import hdrtoys_port
import numpy as np
import pytest
from bt2390_ref import (
    C1,
    C2,
    C3,
    CB_DIVISOR,
    CR_DIVISOR,
    ICTCP_TO_LMSP,
    LMSP_TO_ICTCP,
    LUMA_BT2020_EXACT,
    LUMA_BT2020_PRINTED,
    M2,
    PRIMARIES_BT709,
    PRIMARIES_BT2020,
    REPRESENTATIONS,
    RGB709_TO_XYZ,
    RGB2020_TO_LMS,
    RGB2020_TO_XYZ,
    Eetf,
    apply_matrix,
    bt2390,
    chroma_ratio,
    pq_eotf,
    pq_inverse_eotf,
    rgb_to_xyz_matrix,
)

# Printed to four decimals in BT.2087-0 Annex 1 (block M2) and in BT.2407-0
# equation (1). They exist to catch a wrong primary or a transposed matrix,
# not to pin down digits.
BT2087_RGB2020_TO_XYZ = np.array(
    [[0.6370, 0.1446, 0.1689], [0.2627, 0.6780, 0.0593], [0.0000, 0.0281, 1.0610]]
)
BT2087_RGB709_TO_XYZ = np.array(
    [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
)
BT2087_RGB709_TO_RGB2020 = np.array(
    [[0.6274, 0.3293, 0.0433], [0.0691, 0.9195, 0.0114], [0.0164, 0.0880, 0.8956]]
)
BT2407_RGB2020_TO_RGB709 = np.array(
    [[1.6605, -0.5876, -0.0728], [-0.1246, 1.1329, -0.0083], [-0.0182, -0.1006, 1.1187]]
)


# --- PQ, BT.2100-3 Table 4 -------------------------------------------------


def test_pq_round_trip():
    luminance = np.geomspace(1e-4, 1e4, 4001)
    back = pq_eotf(pq_inverse_eotf(luminance))
    assert np.max(np.abs(back / luminance - 1.0)) < 1e-12


def test_pq_constants_are_consistent():
    assert C1 == C3 - C2 + 1.0
    assert pq_inverse_eotf(10000.0) == 1.0
    assert float(pq_inverse_eotf(203.0)) == pytest.approx(0.5807, abs=5e-5)


def test_pq_of_zero_is_not_zero():
    """PQ(0) is c1^m2, which the LB = 0 normalisation has to use as it stands."""
    assert float(pq_inverse_eotf(0.0)) == C1**M2
    assert float(pq_inverse_eotf(0.0)) == pytest.approx(7.31e-7, rel=1e-3)
    # The EOTF's own max(., 0) maps everything below that back to black.
    assert float(pq_eotf(0.5 * pq_inverse_eotf(0.0))) == 0.0


def test_negative_linear_input_is_black():
    assert pq_inverse_eotf(-5.0) == pq_inverse_eotf(0.0)
    assert pq_inverse_eotf(1e9) == pq_inverse_eotf(10000.0)


# --- Matrices, BT.2100-3 Tables 6 and 7 ------------------------------------


def test_lms_matrix_row_sums():
    """Rows summing to 4096 mean white gives L = M = S."""
    assert np.all(RGB2020_TO_LMS.sum(axis=1) == 1.0)


def test_ictcp_matrix_row_sums():
    sums = LMSP_TO_ICTCP.sum(axis=1)
    assert sums[0] == 1.0
    assert sums[1] == 0.0
    assert sums[2] == 0.0


def test_lms_matrix_equals_the_xyz_path():
    derived = hdrtoys_port.XYZ_TO_LMS @ hdrtoys_port.RGB_TO_XYZ
    assert np.max(np.abs(derived - RGB2020_TO_LMS)) < 1e-15


def test_ictcp_inverse_matches_the_published_decimals():
    assert np.max(np.abs(ICTCP_TO_LMSP - hdrtoys_port.ICTCP_TO_LMS)) < 1e-15


def test_derived_matrices_match_the_printed_ones():
    assert np.allclose(RGB2020_TO_XYZ, BT2087_RGB2020_TO_XYZ, atol=5e-5)
    assert np.allclose(RGB709_TO_XYZ, BT2087_RGB709_TO_XYZ, atol=5e-5)
    to_709 = np.linalg.inv(RGB709_TO_XYZ) @ RGB2020_TO_XYZ
    assert np.allclose(to_709, BT2407_RGB2020_TO_RGB709, atol=5e-5)
    assert np.allclose(np.linalg.inv(to_709), BT2087_RGB709_TO_RGB2020, atol=5e-5)
    assert np.max(np.abs(RGB2020_TO_XYZ - hdrtoys_port.RGB_TO_XYZ)) < 1e-15


def test_bt709_cube_nests_inside_bt2020():
    """No negative entry and rows summing to 1, so the effective gamuts nest."""
    to_2020 = np.linalg.inv(RGB2020_TO_XYZ) @ RGB709_TO_XYZ
    assert np.all(to_2020 >= 0.0)
    assert np.allclose(to_2020.sum(axis=1), 1.0, atol=1e-12)


def test_luminance_coefficients():
    """The printed row and the colorimetric row differ above float32 tolerance."""
    assert CB_DIVISOR == pytest.approx(1.8814, abs=1e-12)
    assert CR_DIVISOR == pytest.approx(1.4746, abs=1e-12)
    assert LUMA_BT2020_PRINTED.sum() == 1.0
    relative = np.abs(LUMA_BT2020_EXACT / LUMA_BT2020_PRINTED - 1.0)
    assert relative.max() == pytest.approx(2.9e-5, rel=0.1)
    assert relative.max() > 1.2e-7  # a float32 ULP


def test_primary_derivation_puts_white_at_unity():
    for primaries in (PRIMARIES_BT2020, PRIMARIES_BT709):
        m = rgb_to_xyz_matrix(primaries)
        white = apply_matrix(m, np.ones(3))
        assert white[1] == pytest.approx(1.0, abs=1e-15)


# --- The EETF, BT.2408-9 Annex 5 -------------------------------------------


def normalised_curve(curve, e1):
    """E3 for a given E1, undoing the step 1 and step 5 normalisation."""
    span = curve.pq_lw - curve.pq_lb
    return (curve(np.asarray(e1) * span + curve.pq_lb) - curve.pq_lb) / span


def test_spline_endpoints_and_slopes():
    curve = Eetf(0.0, 1000.0, 0.0, 203.0)  # b = 0, so E3 is E2
    ks, max_lum = curve.ks, curve.max_lum

    assert normalised_curve(curve, ks) == pytest.approx(ks, abs=1e-12)
    assert normalised_curve(curve, 1.0) == pytest.approx(max_lum, abs=1e-12)

    # dP/dE1 = (1 - T)^2: slope 1 at the knee, slope 0 at E1 = 1.
    e1 = np.linspace(ks, 1.0, 20001)
    h = 1e-7
    slope = (normalised_curve(curve, e1 + h) - normalised_curve(curve, e1 - h)) / (2 * h)
    t = (e1 - ks) / (1.0 - ks)
    assert np.max(np.abs(slope - (1.0 - t) ** 2)) < 1e-6
    assert slope[0] == pytest.approx(1.0, abs=1e-6)
    assert slope[-1] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("src_max", [203.0001, 400.0, 1000.0, 4000.0, 10000.0])
@pytest.mark.parametrize("dst_min", [0.0, 0.05, 0.5])
def test_curve_is_monotone_and_bounded(src_max, dst_min):
    curve = Eetf(0.0, src_max, dst_min, 203.0)
    e1 = np.linspace(0.0, 1.0, 200001)
    e3 = normalised_curve(curve, e1)
    assert np.all(np.diff(e3) >= 0.0)

    # E2 lies in [0, maxLum]. The black lift then adds b (1 - E2)^4, which is
    # why the peak sits slightly above maxLum whenever b is positive.
    lift = curve.min_lum * (1.0 - curve.max_lum) ** 4
    assert e3.min() == pytest.approx(curve.min_lum, abs=1e-12)
    assert e3.max() == pytest.approx(curve.max_lum + lift, abs=1e-12)


@pytest.mark.parametrize(
    "src_min,dst_min", [(0.0, 0.0), (0.005, 0.5), (0.0001, 1.0), (0.01, 0.0)]
)
def test_e4_range_including_the_black_lift_overshoot(src_min, dst_min):
    """E4 starts exactly at PQ(Lmin), and with a positive b it ends above PQ(Lmax).

    Annex 5 adds b (1 - E2)^4 after the spline, so the peak sits above maxLum
    by b (1 - maxLum)^4 whenever b is positive, which is exactly when dst_min
    exceeds src_min. The spec has no output clamp and neither does this.
    """
    src_max, dst_max = 1000.0, 203.0
    curve = Eetf(src_min, src_max, dst_min, dst_max)
    e4 = curve(np.linspace(curve.pq_lb, curve.pq_lw, 200001))
    overshoot = curve.min_lum * (1.0 - curve.max_lum) ** 4 * (curve.pq_lw - curve.pq_lb)

    assert (overshoot > 0.0) == (dst_min > src_min)
    assert e4.min() == pytest.approx(
        float(pq_inverse_eotf(dst_min)), rel=1e-12, abs=1e-15
    )
    assert e4.max() == pytest.approx(
        float(pq_inverse_eotf(dst_max)) + overshoot, rel=1e-12, abs=1e-15
    )


def test_the_overshoot_is_small_but_real():
    """0.39% in luminance for LW 1000, Lmin 1, Lmax 203."""
    curve = Eetf(0.0, 1000.0, 1.0, 203.0)
    peak = float(pq_eotf(curve(curve.pq_lw)))
    assert peak > 203.0
    assert peak / 203.0 - 1.0 == pytest.approx(0.0039, abs=0.0002)


def test_black_lift_stays_monotone_exactly_to_a_quarter():
    """E3 = E2 + b (1 - E2)^4 has slope 1 - 4b at E2 = 0."""
    e2 = np.linspace(0.0, 1.0, 200001)
    assert np.all(np.diff(e2 + 0.25 * (1.0 - e2) ** 4) >= 0.0)
    assert np.any(np.diff(e2 + 0.2501 * (1.0 - e2) ** 4) < 0.0)


def test_eetf_rejects_a_non_monotone_black_lift():
    pq_lb = float(pq_inverse_eotf(0.0))
    pq_lw = float(pq_inverse_eotf(1000.0))
    boundary = float(pq_eotf(0.25 * (pq_lw - pq_lb) + pq_lb))
    assert boundary == pytest.approx(1.99, abs=0.01)
    assert Eetf(0.0, 1000.0, boundary, 203.0).min_lum == pytest.approx(0.25, abs=1e-9)
    with pytest.raises(ValueError, match="monotone"):
        Eetf(0.0, 1000.0, boundary * 1.05, 203.0)


def test_knee_table():
    """LW 1000, Lmax 203, LB and Lmin 0."""
    curve = Eetf(0.0, 1000.0, 0.0, 203.0)
    assert curve.knee_luminance() == pytest.approx(87.8, abs=0.05)
    assert float(pq_eotf(curve(pq_inverse_eotf(203.0)))) == pytest.approx(159.0, abs=0.05)
    assert float(pq_eotf(curve(pq_inverse_eotf(1000.0)))) == pytest.approx(
        203.0, abs=1e-9
    )


def test_degenerate_range_never_evaluates_the_spline():
    curve = Eetf(0.0, 1000.0, 0.0, 1000.0)
    assert curve.degenerate
    assert curve.ks >= 1.0
    e = np.linspace(curve.pq_lb, curve.pq_lw, 10001)
    assert np.allclose(curve(e), e, rtol=0.0, atol=1e-15)


def test_e1_is_clamped_to_the_domain():
    curve = Eetf(0.1, 1000.0, 0.0, 203.0)
    assert float(curve(pq_inverse_eotf(0.0))) == float(curve(pq_inverse_eotf(0.1)))
    assert float(curve(pq_inverse_eotf(9000.0))) == float(curve(pq_inverse_eotf(1000.0)))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(src_min=0.0, src_max=0.0, dst_min=0.0, dst_max=203.0),
        dict(src_min=0.0, src_max=-1.0, dst_min=0.0, dst_max=203.0),
        dict(src_min=0.0, src_max=1000.0, dst_min=0.0, dst_max=0.0),
        dict(src_min=1000.0, src_max=1000.0, dst_min=0.0, dst_max=203.0),
        dict(src_min=2000.0, src_max=1000.0, dst_min=0.0, dst_max=203.0),
        dict(src_min=0.0, src_max=1000.0, dst_min=203.0, dst_max=203.0),
        dict(src_min=0.0, src_max=1000.0, dst_min=300.0, dst_max=203.0),
        dict(src_min=0.0, src_max=1000.0, dst_min=50.0, dst_max=203.0),
        dict(src_min=0.0, src_max=np.inf, dst_min=0.0, dst_max=203.0),
        dict(src_min=np.nan, src_max=1000.0, dst_min=0.0, dst_max=203.0),
        # Outside the PQ domain. Two values above 10000 both encode to 1.0,
        # which makes the span zero.
        dict(src_min=0.0, src_max=20000.0, dst_min=0.0, dst_max=203.0),
        dict(src_min=20000.0, src_max=30000.0, dst_min=0.0, dst_max=203.0),
        dict(src_min=-5.0, src_max=1000.0, dst_min=0.0, dst_max=203.0),
        dict(src_min=0.0, src_max=1000.0, dst_min=-1.0, dst_max=203.0),
        dict(src_min=0.0, src_max=1000.0, dst_min=0.0, dst_max=20000.0),
        # KS below 0, where the whole curve is spline and P(0) is negative.
        dict(src_min=0.0, src_max=10000.0, dst_min=0.0, dst_max=5.0),
        dict(src_min=0.0, src_max=1000.0, dst_min=0.0, dst_max=3.0),
    ],
)
def test_parameter_errors(kwargs):
    with pytest.raises(ValueError):
        Eetf(**kwargs)


def test_ks_below_zero_is_rejected_because_e2_goes_negative():
    """maxLum below 1/3 puts the whole domain on the spline, and P(0) < 0.

    E2 then leaves [0, maxLum], E4 falls below PQ(Lmin) and the chroma ratio
    changes sign. Realistic targets are nowhere near it: with LB 0, LW 1000
    puts the bound at Lmax 5.20 cd/m2 and LW 10000 at 15.13.
    """
    for src_max, expected in ((1000.0, 5.1985), (10000.0, 15.1343)):
        pq_lb = float(pq_inverse_eotf(0.0))
        pq_lw = float(pq_inverse_eotf(src_max))
        bound = float(pq_eotf(pq_lb + (pq_lw - pq_lb) / 3.0))
        assert bound == pytest.approx(expected, abs=0.001)
        assert Eetf(0.0, src_max, 0.0, bound * 1.02).ks >= 0.0
        with pytest.raises(ValueError, match="KS"):
            Eetf(0.0, src_max, 0.0, bound * 0.98)


def test_every_luminance_must_lie_in_the_pq_domain():
    Eetf(0.0, 10000.0, 0.0, 203.0)  # both ends exactly on the domain
    for name in ("src_min", "src_max", "dst_min", "dst_max"):
        ok = dict(src_min=0.0, src_max=1000.0, dst_min=0.0, dst_max=203.0)
        with pytest.raises(ValueError, match="0 to 10000"):
            Eetf(**{**ok, name: 10001.0})


def test_target_black_below_mastering_black_is_allowed():
    """b < 0 expands blacks and the curve stays monotone."""
    curve = Eetf(0.01, 1000.0, 0.0, 203.0)
    assert curve.min_lum < 0.0
    e1 = np.linspace(0.0, 1.0, 100001)
    assert np.all(np.diff(normalised_curve(curve, e1)) >= 0.0)


# --- The five representations ----------------------------------------------


def random_rgb(seed, n=20000):
    """Linear BT.2020 with 1.0 meaning 100 cd/m2, up to 1000 cd/m2 per channel.

    Biased towards the dark end, where most of a real frame sits, and ending
    with the three primaries at the mastering peak and with black.
    """
    rng = np.random.default_rng(seed)
    rgb = rng.random((n, 3)) ** 3 * 10.0
    return np.vstack([rgb, np.eye(3) * 10.0, np.zeros((1, 3))])


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_identity_when_the_target_matches_the_source(representation):
    rgb = random_rgb(3)
    out = bt2390(
        rgb, 0.0, 1000.0, dst_min=0.0, dst_max=1000.0, representation=representation
    )
    assert np.allclose(out, rgb * 100.0 / 1000.0, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_black_stays_black(representation):
    out = bt2390(np.zeros((1, 3)), 0.0, 1000.0, representation=representation)
    assert np.all(out == 0.0)


@pytest.mark.parametrize("dst_min", [0.0, 0.5, 1.0])
def test_black_maps_to_the_curves_own_black_in_every_representation(dst_min):
    """A driving value of zero has no ratio, so the pixel takes the curve's black.

    The other three representations put an exact-black input at Lmin, so
    yrgb and maxrgb have to agree, otherwise a positive black lift leaves
    exact black as the only unlifted pixel in the frame.
    """
    outputs = {
        representation: bt2390(
            np.zeros((1, 3)),
            0.0,
            1000.0,
            dst_min=dst_min,
            representation=representation,
        )
        for representation in REPRESENTATIONS
    }
    expected = dst_min / 203.0
    for representation, out in outputs.items():
        assert out == pytest.approx(expected, abs=1e-12), representation


@pytest.mark.parametrize(
    "representation,driving",
    [
        ("yrgb", lambda rgb: rgb @ LUMA_BT2020_PRINTED),
        ("maxrgb", lambda rgb: rgb.max(axis=-1)),
    ],
)
def test_the_ratio_representations_are_continuous_in_their_driving_value(
    representation, driving
):
    """The quantity the ratio is built from lands on Lmin at and near black.

    Individual channels do not: a saturated near-black is expanded by the
    ratio, so yrgb takes (1e-7, 0, 0) cd/m2 to a red channel of Lmin/0.2627.
    That is Annex 5's formula. What must not happen is exact black dropping
    to zero while everything around it sits at Lmin.
    """
    tiny = 1e-9
    steps = np.array([[0.0, 0.0, 0.0], [tiny, 0.0, 0.0], [tiny, tiny, tiny]])
    out = bt2390(steps, 0.0, 1000.0, dst_min=0.5, representation=representation)
    # The tolerance is the curve's own slope over that interval, not slack:
    # before the fix exact black read 0 against 0.5 for its neighbours.
    assert np.allclose(driving(out) * 203.0, 0.5, rtol=1e-3)


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_negative_input_is_treated_as_black(representation):
    out = bt2390(np.full((1, 3), -0.5), 0.0, 1000.0, representation=representation)
    assert np.allclose(out, 0.0, atol=1e-12)


def test_output_volume_per_representation():
    """Table A5-1: rgb and maxrgb stay inside the target volume, the others do not."""
    rgb = random_rgb(5)
    for representation in ("rgb", "maxrgb"):
        out = bt2390(rgb, 0.0, 1000.0, representation=representation)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-12
    for representation in ("ictcp", "ycbcr", "yrgb"):
        out = bt2390(rgb, 0.0, 1000.0, representation=representation)
        assert out.max() > 1.0


@pytest.mark.parametrize("dst_min", [0.0, 0.5])
def test_ictcp_chroma_scaling_is_a_scaling_of_lmsp(dst_min):
    """Scaling CT and CP by I2/I1 scales the whole L'M'S' vector by I2/I1.

    That identity holds on the compressing branch. Where the curve lifts,
    which only happens near black with a positive b, the ratio is I1/I2 and
    the result is no longer a pure scaling. What has to hold on both is that
    the reconstructed L'M'S' never goes negative, so the EOTF needs no clamp
    beyond its own.
    """
    rgb = np.clip(random_rgb(9) * 100.0, 0.0, 10000.0)
    lmsp = pq_inverse_eotf(apply_matrix(RGB2020_TO_LMS, rgb))
    ictcp = apply_matrix(LMSP_TO_ICTCP, lmsp)
    curve = Eetf(0.0, 1000.0, dst_min, 203.0)

    i1 = ictcp[..., 0]
    i2 = curve(i1)
    k = chroma_ratio(i1, i2)
    assert np.all(k <= 1.0 + 1e-15)

    mapped = apply_matrix(
        ICTCP_TO_LMSP, np.stack([i2, k * ictcp[..., 1], k * ictcp[..., 2]], axis=-1)
    )
    assert np.all(mapped >= -1e-15)

    compressing = i2 < i1
    # A relative margin, because the E1 round trip can land a ULP high
    # at the very bottom of the range where the curve is the identity.
    lifting = i2 > i1 * (1.0 + 1e-9)
    assert compressing.any()
    assert lifting.any() == (dst_min > 0.0)

    scale = np.where(compressing, i2 / np.where(compressing, i1, 1.0), 1.0)
    assert np.allclose(
        mapped[compressing], (scale[..., None] * lmsp)[compressing], atol=1e-14
    )


@pytest.mark.parametrize("dst_min", [0.0, 0.5])
def test_ycbcr_chroma_scaling_is_a_scaling_of_rgbp(dst_min):
    """The Y'CbCr counterpart: the same identity on PQ-encoded R'G'B'."""
    rgb = np.clip(random_rgb(13) * 100.0, 0.0, 10000.0)
    rgbp = pq_inverse_eotf(rgb)
    curve = Eetf(0.0, 1000.0, dst_min, 203.0)

    y1 = rgbp @ LUMA_BT2020_PRINTED
    y2 = curve(y1)
    k = chroma_ratio(y1, y2)
    assert np.all(k <= 1.0 + 1e-15)

    cb = (rgbp[..., 2] - y1) / CB_DIVISOR
    cr = (rgbp[..., 0] - y1) / CR_DIVISOR
    rp = y2 + CR_DIVISOR * k * cr
    bp = y2 + CB_DIVISOR * k * cb
    kr, kg, kb = LUMA_BT2020_PRINTED
    gp = (y2 - kr * rp - kb * bp) / kg
    mapped = np.stack([rp, gp, bp], axis=-1)
    assert np.all(mapped >= -1e-15)

    compressing = y2 < y1
    # A relative margin, because the E1 round trip can land a ULP high
    # at the very bottom of the range where the curve is the identity.
    lifting = y2 > y1 * (1.0 + 1e-9)
    assert compressing.any()
    assert lifting.any() == (dst_min > 0.0)

    scale = np.where(compressing, y2 / np.where(compressing, y1, 1.0), 1.0)
    assert np.allclose(
        mapped[compressing], (scale[..., None] * rgbp)[compressing], atol=1e-14
    )
