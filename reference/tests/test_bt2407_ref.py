"""Checks on the BT.2407 gamut conversion.

The boundary crossing is checked against bisection, the soft clip against the
parametric Bezier it is derived from, and the whole projection against the
invariants Annex 5 claims for it.
"""

import numpy as np
import pytest
from bt2390_ref import (
    D65,
    PRIMARIES_BT709,
    PRIMARIES_BT2020,
    PRIMARIES_P3D65,
    RGB2020_TO_XYZ,
    apply_matrix,
    bt2390,
    rgb_to_xyz_matrix,
)
from bt2407_ref import (
    RGB2020_TO_RGB709,
    WHITE_UV,
    XYZ_TO_RGB709,
    boundary_t,
    bt2407,
    clip,
    mastering_gamut,
    resolve_source_gamut,
    soft_clip,
    softclip,
    valid_gamut,
    xy_to_uv,
)

XYZ_TO_RGB2020 = np.linalg.inv(RGB2020_TO_XYZ)
XYZ_TO_P3D65 = np.linalg.inv(rgb_to_xyz_matrix(PRIMARIES_P3D65))


def random_rays(seed, n=6000):
    """Chromaticities of random in-cube BT.2020 colours, away from white."""
    rng = np.random.default_rng(seed)
    rgb = rng.random((4 * n, 3))
    xyz = apply_matrix(RGB2020_TO_XYZ, rgb)
    y = xyz[..., 1]
    denom = xyz[..., 0] + 15.0 * xyz[..., 1] + 3.0 * xyz[..., 2]
    u = 4.0 * xyz[..., 0] / denom
    v = 9.0 * xyz[..., 1] / denom
    du = u - WHITE_UV[0]
    dv = v - WHITE_UV[1]
    keep = (y > 0.02) & (y < 0.9) & (np.hypot(du, dv) > 1e-3)
    return y[keep][:n], du[keep][:n], dv[keep][:n]


def inside(xyz_to_rgb, y, du, dv, t):
    uw, vw = WHITE_UV
    u = uw + t * du
    v = vw + t * dv
    x = 9.0 * y * u / (4.0 * v)
    z = y * (12.0 - 3.0 * u - 20.0 * v) / (4.0 * v)
    rgb = apply_matrix(xyz_to_rgb, np.stack([x, y, z], axis=-1))
    return np.all((rgb >= -1e-12) & (rgb <= 1.0 + 1e-12), axis=-1)


@pytest.mark.parametrize("name", ["bt2020", "bt709", "p3d65"])
def test_boundary_matches_bisection(name):
    matrix = {
        "bt2020": XYZ_TO_RGB2020,
        "bt709": XYZ_TO_RGB709,
        "p3d65": XYZ_TO_P3D65,
    }[name]
    y, du, dv = random_rays(17)
    assert np.all(inside(matrix, y, du, dv, np.zeros_like(y)))

    hi = np.ones_like(y)
    for _ in range(40):
        hi = np.where(inside(matrix, y, du, dv, hi), hi * 2.0, hi)
    assert not np.any(inside(matrix, y, du, dv, hi))

    lo = np.zeros_like(y)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        ok = inside(matrix, y, du, dv, mid)
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)

    t = boundary_t(matrix, y, du, dv)
    assert np.all(np.isfinite(t))
    assert np.max(np.abs(t / (0.5 * (lo + hi)) - 1.0)) < 1e-9


def test_achromatic_ray_never_leaves_the_gamut():
    zero = np.zeros(4)
    t = boundary_t(XYZ_TO_RGB709, np.full(4, 0.3), zero, zero)
    assert np.all(np.isinf(t))


def test_alpha_is_non_negative_with_bt2020_as_source():
    y, du, dv = random_rays(23)
    alpha = (
        boundary_t(XYZ_TO_RGB2020, y, du, dv) / boundary_t(XYZ_TO_RGB709, y, du, dv) - 1.0
    )
    assert alpha.min() >= 0.0
    assert alpha.min() == pytest.approx(0.048, abs=0.01)


# --- The soft clip ---------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.0, 0.05, 0.2, 0.2000001, 1.0, 5.0])
@pytest.mark.parametrize("beta", [0.0, 0.05, 0.2, 0.5])
def test_soft_clip_matches_the_parametric_bezier(alpha, beta):
    """Bezier through (1-beta, 1-beta), control (1, 1), end (1+alpha, 1)."""
    s = np.linspace(1e-9, 1.0, 20001)
    x = (1.0 - beta) + 2.0 * beta * s + (alpha - beta) * s * s
    y = (1.0 - beta) + 2.0 * beta * s - beta * s * s
    assert np.allclose(soft_clip(x, alpha, beta), y, atol=1e-12)


@pytest.mark.parametrize("alpha", [0.0, 0.05, 0.2, 1.0, 5.0])
def test_soft_clip_properties(alpha):
    beta = 0.2
    r = np.linspace(0.0, 4.0 + alpha, 400001)
    f = soft_clip(r, alpha, beta)

    assert np.all(np.diff(f) >= -1e-15)  # monotone
    identity = r <= 1.0 - beta
    assert np.array_equal(f[identity], r[identity])
    assert f.max() <= 1.0 + 1e-15
    assert float(soft_clip(1.0 + alpha, alpha, beta)) == pytest.approx(1.0, abs=1e-12)
    assert float(soft_clip(1.0 - beta, alpha, beta)) == pytest.approx(1.0 - beta)

    h = 1e-7
    lower = (
        float(soft_clip(1.0 - beta + h, alpha, beta))
        - float(soft_clip(1.0 - beta, alpha, beta))
    ) / h
    assert lower == pytest.approx(1.0, abs=1e-5)
    if alpha > 0.0:
        upper = (
            float(soft_clip(1.0 + alpha, alpha, beta))
            - float(soft_clip(1.0 + alpha - h, alpha, beta))
        ) / h
        assert upper == pytest.approx(0.0, abs=1e-5)


def test_soft_clip_with_no_headroom_is_a_hard_clip():
    r = np.linspace(0.0, 3.0, 10001)
    assert np.allclose(soft_clip(r, 0.0, 0.2), np.minimum(r, 1.0), atol=1e-15)


def test_soft_clip_rejects_an_out_of_range_margin():
    for beta in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            soft_clip(np.array([0.5]), np.array([1.0]), beta)


# --- The whole projection --------------------------------------------------


def tone_mapped(seed, representation="ictcp", n=30000):
    """A spread of BT.2390 output, which is what the gamut mapper receives."""
    rng = np.random.default_rng(seed)
    rgb = np.vstack([rng.random((n, 3)) ** 3 * 10.0, np.eye(3) * 10.0])
    return bt2390(rgb, 0.0, 1000.0, representation=representation)


def test_tone_mapper_output_reaches_outside_the_target_volume():
    mapped = tone_mapped(31)
    xyz = apply_matrix(RGB2020_TO_XYZ, mapped)
    assert mapped.max() > 4.0
    assert xyz[..., 1].max() > 1.0


def test_softclip_output_is_inside_the_unit_cube():
    out = softclip(tone_mapped(31))
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_softclip_preserves_luminance_and_hue_direction():
    mapped = tone_mapped(31)
    xyz = apply_matrix(RGB2020_TO_XYZ, mapped)
    y = xyz[..., 1]
    denom = xyz[..., 0] + 15.0 * xyz[..., 1] + 3.0 * xyz[..., 2]
    projected = (y > 1e-6) & (y < 1.0 - 1e-6) & (denom > 0.0)
    assert projected.mean() > 0.5

    out = softclip(mapped)
    out_xyz = apply_matrix(np.linalg.inv(XYZ_TO_RGB709), out)
    assert np.allclose(out_xyz[projected, 1], y[projected], rtol=1e-9, atol=1e-12)

    uw, vw = WHITE_UV
    du = 4.0 * xyz[..., 0] / denom - uw
    dv = 9.0 * xyz[..., 1] / denom - vw
    out_denom = out_xyz[..., 0] + 15.0 * out_xyz[..., 1] + 3.0 * out_xyz[..., 2]
    du2 = 4.0 * out_xyz[..., 0] / out_denom - uw
    dv2 = 9.0 * out_xyz[..., 1] / out_denom - vw
    cross = du[projected] * dv2[projected] - dv[projected] * du2[projected]
    dot = du[projected] * du2[projected] + dv[projected] * dv2[projected]
    assert np.max(np.abs(cross)) < 1e-9
    assert np.all(dot >= -1e-15)


def test_softclip_leaves_the_inner_region_alone_and_equals_the_hard_clip_there():
    rng = np.random.default_rng(41)
    rgb = rng.random((40000, 3))
    beta = 0.2

    xyz = apply_matrix(RGB2020_TO_XYZ, rgb)
    y = xyz[..., 1]
    denom = xyz[..., 0] + 15.0 * xyz[..., 1] + 3.0 * xyz[..., 2]
    du = 4.0 * xyz[..., 0] / denom - WHITE_UV[0]
    dv = 9.0 * xyz[..., 1] / denom - WHITE_UV[1]
    r = 1.0 / boundary_t(XYZ_TO_RGB709, y, du, dv)
    untouched = (r <= 1.0 - beta) & (y > 0.0) & (y < 1.0)

    # The report's hatched area: roughly a third of random in-cube colours.
    assert 0.2 < untouched.mean() < 0.5

    out = softclip(rgb, beta=beta)
    assert np.allclose(out[untouched], clip(rgb)[untouched], rtol=1e-9, atol=1e-12)
    assert np.allclose(
        out[untouched],
        apply_matrix(RGB2020_TO_RGB709, rgb)[untouched],
        rtol=1e-9,
        atol=1e-12,
    )


def test_luminance_policies_at_the_ends():
    black = softclip(np.array([[0.0, 0.0, 0.0], [-0.3, -0.1, -0.2]]))
    assert np.all(black == 0.0)
    white = softclip(np.array([[1.2, 1.2, 1.2], [3.0, 0.9, 0.4]]))
    assert np.all(white == 1.0)


def test_softclip_converges_on_white_as_luminance_approaches_one():
    """The Y >= 1 policy is the projection's own limit, not a discontinuity."""
    saturated = np.array([[0.0, 1.0, 0.0]])
    out = [
        softclip(saturated * y / apply_matrix(RGB2020_TO_XYZ, saturated)[0, 1])[0]
        for y in (0.9, 0.99, 0.999, 0.99999)
    ]
    distance = [float(np.max(np.abs(o - 1.0))) for o in out]
    assert distance == sorted(distance, reverse=True)
    assert distance[-1] < 1e-3


def test_hard_clip_takes_over_where_there_is_no_chromaticity():
    """A negative channel can make X + 15Y + 3Z non-positive with Y still above 0."""
    rgb = np.array([[0.0, 1.0, -5.0]])
    xyz = apply_matrix(RGB2020_TO_XYZ, rgb)
    assert xyz[0, 1] > 0.0
    assert xyz[0, 0] + 15.0 * xyz[0, 1] + 3.0 * xyz[0, 2] <= 0.0
    assert np.all(softclip(rgb) == clip(rgb))


# --- Source gamut selection ------------------------------------------------


def test_a_source_boundary_colour_reaches_the_target_boundary():
    """alpha is set so the source boundary maps exactly onto the target boundary.

    A colour sitting on the P3 boundary has r = 1 + alpha when P3 is the
    declared source, so f(r) is 1 and it lands on the BT.709 boundary with a
    channel at zero. Declaring the wider BT.2020 source instead leaves alpha
    too large, and the same colour stops short of the boundary.
    """
    on_p3_boundary = apply_matrix(
        np.linalg.inv(RGB2020_TO_XYZ),
        apply_matrix(rgb_to_xyz_matrix(PRIMARIES_P3D65), np.array([[0.0, 0.35, 0.0]])),
    )
    narrow = softclip(on_p3_boundary, src_primaries=PRIMARIES_P3D65)
    wide = softclip(on_p3_boundary, src_primaries=PRIMARIES_BT2020)
    assert narrow.min() == pytest.approx(0.0, abs=1e-12)
    assert wide.min() > 1e-5


MASTERING_P3 = {
    "MasteringDisplayPrimariesX": [0.680, 0.265, 0.150],
    "MasteringDisplayPrimariesY": [0.320, 0.690, 0.060],
    "MasteringDisplayWhitePointX": 0.3127,
    "MasteringDisplayWhitePointY": 0.3290,
}


def test_mastering_gamut_is_read_and_validated():
    primaries, white = mastering_gamut(MASTERING_P3)
    assert np.allclose(primaries, PRIMARIES_P3D65)
    assert white == D65


@pytest.mark.parametrize(
    "broken",
    [
        {},
        {"MasteringDisplayPrimariesX": [0.68, 0.265, 0.15]},  # y missing
        dict(MASTERING_P3, MasteringDisplayPrimariesX=[0.68, 0.265]),  # wrong count
        dict(MASTERING_P3, MasteringDisplayPrimariesY=[0.0, 0.69, 0.06]),  # y = 0
        dict(MASTERING_P3, MasteringDisplayPrimariesX=[0.9, 0.265, 0.15]),  # x + y > 1
        dict(MASTERING_P3, MasteringDisplayPrimariesY=[0.32, 0.32, 0.32]),  # no area
        dict(MASTERING_P3, MasteringDisplayWhitePointX=0.9),  # white outside
        dict(MASTERING_P3, MasteringDisplayWhitePointY=float("nan")),
    ],
)
def test_broken_mastering_metadata_counts_as_absent(broken):
    assert mastering_gamut(broken) is None


def test_valid_gamut_accepts_the_standard_sets():
    for primaries in (PRIMARIES_BT2020, PRIMARIES_BT709, PRIMARIES_P3D65):
        assert valid_gamut(primaries, D65)


def test_primaries_on_the_z_zero_line_are_accepted():
    """BT.2020 red (0.708, 0.292) and P3 red (0.680, 0.320) sit on x + y = 1.

    A strict x + y < 1 rejects both, which means falling back to BT.2020 on
    exactly the P3-mastered content the source-gamut rule exists to serve.
    """
    for red in (PRIMARIES_BT2020[0], PRIMARIES_P3D65[0]):
        assert red[0] + red[1] == 1.0
    assert valid_gamut(PRIMARIES_BT2020, D65)
    assert valid_gamut(PRIMARIES_P3D65, D65)

    # The same values as a container carries them, in counts of 0.00002.
    container = {
        "MasteringDisplayPrimariesX": [34000 * 2e-5, 13250 * 2e-5, 7500 * 2e-5],
        "MasteringDisplayPrimariesY": [16000 * 2e-5, 34500 * 2e-5, 3000 * 2e-5],
        "MasteringDisplayWhitePointX": 15635 * 2e-5,
        "MasteringDisplayWhitePointY": 16450 * 2e-5,
    }
    found = mastering_gamut(container)
    assert found is not None
    assert np.allclose(found[0], PRIMARIES_P3D65, atol=1e-12)


def test_a_primary_on_the_y_axis_is_accepted_but_y_zero_is_not():
    """x may be 0; y may not, because the matrix derivation divides by it."""
    assert valid_gamut(((0.0, 0.9), (0.6, 0.35), (0.15, 0.06)), D65)
    assert valid_gamut(((0.708, 0.0), (0.170, 0.797), (0.131, 0.046)), D65) is False


@pytest.mark.parametrize(
    "src_gamut,props,label",
    [
        ("auto", MASTERING_P3, "mastering"),
        ("auto", None, "bt2020"),
        ("auto", {}, "bt2020"),
        ("bt2020", MASTERING_P3, "bt2020"),
        ("p3d65", None, "p3d65"),
    ],
)
def test_resolve_source_gamut(src_gamut, props, label):
    assert resolve_source_gamut(src_gamut, props)[2] == label


def test_bt2407_dispatch():
    rgb = tone_mapped(53)
    hard, label = bt2407(rgb, method="clip")
    assert label == "bt2020"
    assert np.array_equal(hard, clip(rgb))

    soft, label = bt2407(rgb, method="softclip", src_gamut="auto", props=MASTERING_P3)
    assert label == "mastering"
    assert np.array_equal(soft, softclip(rgb, src_primaries=PRIMARIES_P3D65))

    with pytest.raises(ValueError):
        bt2407(rgb, method="nonsense")
    with pytest.raises(ValueError):
        bt2407(rgb, src_gamut="nonsense")


def test_white_point_uv():
    assert xy_to_uv(*D65) == pytest.approx((0.1978, 0.4683), abs=5e-5)
