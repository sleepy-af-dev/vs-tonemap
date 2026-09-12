"""float64 reference for the BT.2407 gamut conversion from BT.2020 to BT.709.

Two methods. `clip` is section 2 of the report: the matrix, then a hard clamp
per channel. `softclip` is Annex 5: keep the luminance and the hue, move the
chromaticity along the ray from the white point towards the BT.709 boundary,
and roll the last stretch off with a quadratic Bezier so the mapping stays
reversible near the boundary.

Input is linear BT.2020 with SDR white at 1.0, that is, the output of the
BT.2390 stage. Channels above 1.0 are expected there and are handled.

Source: ITU-R BT.2407-0, section 2, section 3 and Annex 5.
"""

import numpy as np
from bt2390_ref import (
    D65,
    PRIMARIES_BT2020,
    PRIMARIES_P3D65,
    RGB709_TO_XYZ,
    RGB2020_TO_XYZ,
    apply_matrix,
    rgb_to_xyz_matrix,
)

XYZ_TO_RGB709 = np.linalg.inv(RGB709_TO_XYZ)
RGB2020_TO_RGB709 = XYZ_TO_RGB709 @ RGB2020_TO_XYZ

METHODS = ("clip", "softclip")
SRC_GAMUTS = ("auto", "bt2020", "p3d65")


def xy_to_uv(x, y):
    """CIE 1931 xy to CIE 1976 u'v'."""
    d = -2.0 * np.asarray(x, dtype=np.float64) + 12.0 * np.asarray(y, np.float64) + 3.0
    return 4.0 * x / d, 9.0 * y / d


WHITE_UV = xy_to_uv(*D65)


def clip(rgb):
    """BT.2407 section 2: matrix to BT.709, then clamp each channel."""
    return np.clip(apply_matrix(RGB2020_TO_RGB709, rgb), 0.0, 1.0)


# --- Source gamut selection, BT.2407 section 3 -----------------------------


def _signed_area(tri):
    (x0, y0), (x1, y1), (x2, y2) = tri
    return 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))


def valid_gamut(primaries, white):
    """True when three primaries and a white point can define a gamut.

    Every chromaticity has to be a physically meaningful one (x >= 0, y > 0
    and x + y <= 1, so that z is not negative), the primaries have to form a
    triangle with a non-zero area, and the white point has to lie strictly
    inside that triangle. Anything else counts as absent metadata.

    The bound on x + y is inclusive because BT.2020's red primary
    (0.708, 0.292) and P3's (0.680, 0.320) both sit exactly on z = 0. An
    exclusive bound rejects the two most common mastering gamuts there are.
    y stays strictly positive because rgb_to_xyz_matrix divides by it.
    """
    pts = np.asarray(primaries, dtype=np.float64)
    w = np.asarray(white, dtype=np.float64)
    if pts.shape != (3, 2) or w.shape != (2,):
        return False
    if not (np.all(np.isfinite(pts)) and np.all(np.isfinite(w))):
        return False
    for x, y in np.vstack([pts, w[None, :]]):
        if not (x >= 0.0 and y > 0.0 and x + y <= 1.0):
            return False
    orientation = np.sign(_signed_area(pts))
    if orientation == 0.0:
        return False
    for i in range(3):
        a, b = pts[i], pts[(i + 1) % 3]
        edge = (b[0] - a[0]) * (w[1] - a[1]) - (b[1] - a[1]) * (w[0] - a[0])
        if np.sign(edge) != orientation:
            return False
    return True


def mastering_gamut(props):
    """The mastering display gamut from frame properties, or None.

    props holds the values VapourSynth reports: MasteringDisplayPrimariesX
    and MasteringDisplayPrimariesY as three each in R, G, B order, and
    MasteringDisplayWhitePointX and MasteringDisplayWhitePointY as one each.
    Anything missing or failing validation returns None.
    """
    if not props:
        return None
    try:
        xs = np.asarray(props["MasteringDisplayPrimariesX"], dtype=np.float64)
        ys = np.asarray(props["MasteringDisplayPrimariesY"], dtype=np.float64)
        white = (
            float(props["MasteringDisplayWhitePointX"]),
            float(props["MasteringDisplayWhitePointY"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if xs.shape != (3,) or ys.shape != (3,):
        return None
    primaries = tuple(zip(xs.tolist(), ys.tolist()))
    if not valid_gamut(primaries, white):
        return None
    return primaries, white


def resolve_source_gamut(src_gamut="auto", props=None):
    """Pick the source gamut for the soft clip.

    Returns (primaries, white, label), where the label is what the plugin
    records in the TonemapSourceGamut frame property.
    """
    if src_gamut not in SRC_GAMUTS:
        raise ValueError(
            f"src_gamut must be one of {', '.join(SRC_GAMUTS)}, got {src_gamut!r}"
        )
    if src_gamut == "bt2020":
        return PRIMARIES_BT2020, D65, "bt2020"
    if src_gamut == "p3d65":
        return PRIMARIES_P3D65, D65, "p3d65"
    found = mastering_gamut(props)
    if found is None:
        return PRIMARIES_BT2020, D65, "bt2020"
    return found[0], found[1], "mastering"


# --- Annex 5 projection ----------------------------------------------------


def boundary_t(xyz_to_rgb, y, du, dv):
    """Where the ray from the white point leaves a gamut, as a multiple of (du, dv).

    Along c(t) = w + t (du, dv) at fixed luminance y, multiplying through by
    4 v'(t) makes both the XYZ triple and the denominator affine in t, so
    each of the six constraints 0 <= channel <= 1 becomes one linear
    inequality. The answer is the smallest upper bound among them, and inf
    when the colour is achromatic and the ray does not move.
    """
    y = np.asarray(y, dtype=np.float64)
    du = np.asarray(du, dtype=np.float64)
    dv = np.asarray(dv, dtype=np.float64)
    uw, vw = WHITE_UV

    base = np.stack(
        [9.0 * y * uw, 4.0 * y * vw, y * (12.0 - 3.0 * uw - 20.0 * vw)], axis=-1
    )
    step = np.stack([9.0 * y * du, 4.0 * y * dv, y * (-3.0 * du - 20.0 * dv)], axis=-1)
    a = apply_matrix(xyz_to_rgb, base)
    b = apply_matrix(xyz_to_rgb, step)
    d0 = 4.0 * vw
    d1 = (4.0 * dv)[..., None]

    with np.errstate(divide="ignore", invalid="ignore"):
        lower = np.where(b < 0.0, -a / b, np.inf)  # channel >= 0
        den = b - d1
        upper = np.where(den > 0.0, -(a - d0) / den, np.inf)  # channel <= 1
    return np.minimum(lower, upper).min(axis=-1)


def soft_clip(r, alpha, beta):
    """The Annex 5 roll-off: identity below 1 - beta, flat at 1 above 1 + alpha.

    The quadratic Bezier through (1 - beta, 1 - beta), control point (1, 1)
    and end (1 + alpha, 1). With s the Bezier parameter, the x coordinate is
    (1 - beta) + 2 beta s + (alpha - beta) s^2, so s solves a quadratic in r
    and the y coordinate reduces to r - alpha s^2.

    Equation (5-4) of the report prints the bracket unsquared, which gives
    3.17 rather than 1 at r = 1 + alpha. The squared form here is what the
    report's own Bezier construction produces and what every stated property
    of the function requires.

    s is taken in the rationalised form q / (sqrt(beta^2 + k q) + beta)
    rather than (sqrt(beta^2 + k q) - beta) / k, which is the same number but
    stays finite as k = alpha - beta approaches zero.
    """
    r = np.asarray(r, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    if not np.all(np.isfinite(beta)) or not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")

    q = r + beta - 1.0
    k = alpha - beta
    roll = (q > 0.0) & (r <= 1.0 + alpha)
    disc = np.where(roll, np.maximum(beta * beta + k * q, 0.0), 1.0)
    den = np.sqrt(disc) + beta
    s = np.where(den > 0.0, q / np.where(den > 0.0, den, 1.0), 0.0)
    return np.where(roll, r - alpha * s * s, np.where(r > 1.0 + alpha, 1.0, r))


def softclip(rgb, beta=0.2, src_primaries=PRIMARIES_BT2020, src_white=D65):
    """BT.2407 Annex 5: luminance-preserving projection with a reversible roll-off."""
    rgb = np.asarray(rgb, dtype=np.float64)
    xyz_to_src = np.linalg.inv(rgb_to_xyz_matrix(src_primaries, src_white))
    uw, vw = WHITE_UV

    xyz = apply_matrix(RGB2020_TO_XYZ, rgb)
    y = xyz[..., 1]
    denom = xyz[..., 0] + 15.0 * xyz[..., 1] + 3.0 * xyz[..., 2]

    # A colour with a non-positive denominator has no chromaticity to project
    # along, which only a negative input channel can produce. Those pixels
    # take the hard clip, the one mapping that is defined everywhere.
    projectable = (y > 0.0) & (y < 1.0) & (denom > 0.0)
    safe_denom = np.where(projectable, denom, 1.0)
    safe_y = np.where(projectable, y, 0.5)

    u = np.where(projectable, 4.0 * xyz[..., 0] / safe_denom, uw)
    v = np.where(projectable, 9.0 * xyz[..., 1] / safe_denom, vw)
    du = u - uw
    dv = v - vw

    t_src = boundary_t(xyz_to_src, safe_y, du, dv)
    t_709 = boundary_t(XYZ_TO_RGB709, safe_y, du, dv)

    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = t_src / t_709 - 1.0
        r = 1.0 / t_709
    # An achromatic ray leaves both gamuts at infinity, so alpha is 0/0. Those
    # pixels have r = 0 and take the identity branch, where alpha is unused.
    alpha = np.maximum(np.where(np.isfinite(alpha), alpha, 0.0), 0.0)
    r = np.where(np.isfinite(r), r, 0.0)

    with np.errstate(invalid="ignore"):
        scale = np.where(r <= 1.0 - beta, 1.0, t_709 * soft_clip(r, alpha, beta))

    u2 = uw + du * scale
    v2 = vw + dv * scale
    x2 = 9.0 * safe_y * u2 / (4.0 * v2)
    z2 = safe_y * (12.0 - 3.0 * u2 - 20.0 * v2) / (4.0 * v2)
    out = np.clip(
        apply_matrix(XYZ_TO_RGB709, np.stack([x2, safe_y, z2], axis=-1)), 0.0, 1.0
    )

    # Y at or above the target peak is white: the effective gamut shrinks to
    # the white point as Y approaches 1 and the projection already converges
    # there, so this is the method's own limit. Y at or below 0 is black.
    out = np.where(projectable[..., None], out, clip(rgb))
    out = np.where((y >= 1.0)[..., None], 1.0, out)
    return np.where((y <= 0.0)[..., None], 0.0, out)


def bt2407(rgb, method="softclip", beta=0.2, src_gamut="auto", props=None):
    """Convert linear BT.2020 with SDR white at 1.0 to linear BT.709 in [0, 1].

    Returns (rgb709, label) where the label is the source gamut actually
    used, as the plugin records it in TonemapSourceGamut. The hard clip has
    no source gamut and reports bt2020.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}, got {method!r}")
    if method == "clip":
        return clip(rgb), "bt2020"
    primaries, white, label = resolve_source_gamut(src_gamut, props)
    return softclip(rgb, beta=beta, src_primaries=primaries, src_white=white), label
