"""float64 reference for the BT.2390 EETF as ITU-R BT.2408-9 Annex 5 states it.

This module is the oracle. The plugin is measured against it and never
against itself, so nothing here is written for speed. Every value is
float64, every luminance is cd/m2 unless the name says otherwise, and every
constant comes from the ITU text rather than from a re-derivation.

Sources:
  ITU-R BT.2100-3, Table 2 (primaries), Table 4 (PQ), Table 6 (Y'CbCr),
  Table 7 (ICtCp matrices).
  ITU-R BT.2408-9, Annex 5 (the EETF and the five colour representations).
  ITU-R BT.2087-0 (deriving an RGB to XYZ matrix from primaries).
"""

import numpy as np

# --- PQ, BT.2100-3 Table 4 -------------------------------------------------

M1 = 2610.0 / 16384.0
M2 = 2523.0 / 4096.0 * 128.0
C1 = 3424.0 / 4096.0
C2 = 2413.0 / 4096.0 * 32.0
C3 = 2392.0 / 4096.0 * 32.0

PQ_PEAK = 10000.0  # cd/m2 represented by a PQ code of 1.0


def pq_inverse_eotf(luminance):
    """Absolute luminance in cd/m2 to a PQ code.

    The input is clamped to [0, 10000] first: PQ is defined on that range
    only, so negative linear light (resampling ringing, sub-black codes) is
    treated as black.
    """
    y = np.clip(np.asarray(luminance, dtype=np.float64) / PQ_PEAK, 0.0, 1.0)
    ym = y**M1
    return ((C1 + C2 * ym) / (1.0 + C3 * ym)) ** M2


def pq_eotf(code):
    """PQ code to absolute luminance in cd/m2.

    Codes below PQ(0) = 7.31e-7 map to 0 through the spec's own max(., 0).
    Negative codes are outside the domain and are treated as 0.
    """
    t = np.maximum(np.asarray(code, dtype=np.float64), 0.0) ** (1.0 / M2)
    return PQ_PEAK * (np.maximum(t - C1, 0.0) / (C2 - C3 * t)) ** (1.0 / M1)


# --- Matrices, BT.2100-3 Tables 6 and 7 ------------------------------------


def apply_matrix(m, v):
    """Apply a 3x3 matrix to the last axis of v."""
    return np.asarray(v, dtype=np.float64) @ np.asarray(m, dtype=np.float64).T


RGB2020_TO_LMS = (
    np.array([[1688.0, 2146.0, 262.0], [683.0, 2951.0, 462.0], [99.0, 309.0, 3688.0]])
    / 4096.0
)
LMS_TO_RGB2020 = np.linalg.inv(RGB2020_TO_LMS)

LMSP_TO_ICTCP = (
    np.array(
        [
            [2048.0, 2048.0, 0.0],
            [6610.0, -13613.0, 7003.0],
            [17933.0, -17390.0, -543.0],
        ]
    )
    / 4096.0
)
ICTCP_TO_LMSP = np.linalg.inv(LMSP_TO_ICTCP)

# Annex 5 and BT.2100 Table 6 print these three luminance coefficients. They
# are signal coefficients, not CIE luminance, so `yrgb` and `ycbcr` use them
# exactly as printed. LUMA_BT2020_EXACT below is the colorimetric row and is
# what the gamut mapper uses.
LUMA_BT2020_PRINTED = np.array([0.2627, 0.6780, 0.0593])
_KR, _KG, _KB = LUMA_BT2020_PRINTED
CB_DIVISOR = 2.0 * (1.0 - _KB)  # 1.8814
CR_DIVISOR = 2.0 * (1.0 - _KR)  # 1.4746

# --- Primaries, BT.2100-3 Table 2 and BT.709 -------------------------------

D65 = (0.3127, 0.3290)
PRIMARIES_BT2020 = ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046))
PRIMARIES_BT709 = ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060))
PRIMARIES_P3D65 = ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060))


def rgb_to_xyz_matrix(primaries, white=D65):
    """RGB to XYZ from primary and white chromaticities, by the BT.2087 method.

    P holds the primaries as (x, y, 1 - x - y) columns. Solving P s = W for
    the white point as XYZ with Y = 1 gives the column scalings that make
    RGB (1, 1, 1) land on the white point.
    """
    p = np.array([[x, y, 1.0 - x - y] for x, y in primaries], dtype=np.float64).T
    wx, wy = white
    w = np.array([wx / wy, 1.0, (1.0 - wx - wy) / wy])
    return p * np.linalg.solve(p, w)


RGB2020_TO_XYZ = rgb_to_xyz_matrix(PRIMARIES_BT2020)
RGB709_TO_XYZ = rgb_to_xyz_matrix(PRIMARIES_BT709)
LUMA_BT2020_EXACT = RGB2020_TO_XYZ[1].copy()


# --- The EETF, BT.2408-9 Annex 5 steps 1 to 5 ------------------------------


class Eetf:
    """The Annex 5 EETF for one set of mastering and target luminances.

    src_min and src_max are the mastering display black and white (LB, LW).
    dst_min and dst_max are the target black and white (Lmin, Lmax). All four
    are cd/m2. The four PQ conversions and the derived constants happen once,
    here, which is what the plugin does per frame.
    """

    def __init__(self, src_min, src_max, dst_min, dst_max):
        # The domain check comes first. PQ clamps anything outside [0, 10000]
        # to its ends, so two values above the peak would both encode to 1.0
        # and leave the span at zero.
        for name, value in (
            ("src_min", src_min),
            ("src_max", src_max),
            ("dst_min", dst_min),
            ("dst_max", dst_max),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= PQ_PEAK:
                raise ValueError(
                    f"{name} must be a luminance from 0 to 10000 cd/m2, got {value}"
                )
        for name, value in (("src_max", src_max), ("dst_max", dst_max)):
            if value <= 0.0:
                raise ValueError(f"{name} must be a positive luminance in cd/m2")
        if src_min >= src_max:
            raise ValueError("src_min must be below src_max")
        if dst_min >= dst_max:
            raise ValueError("dst_min must be below dst_max")

        self.pq_lb = float(pq_inverse_eotf(src_min))
        self.pq_lw = float(pq_inverse_eotf(src_max))
        span = self.pq_lw - self.pq_lb
        self.min_lum = (float(pq_inverse_eotf(dst_min)) - self.pq_lb) / span
        self.max_lum = (float(pq_inverse_eotf(dst_max)) - self.pq_lb) / span
        self.ks = 1.5 * self.max_lum - 0.5

        # The black lift E3 = E2 + b (1 - E2)^4 has slope 1 - 4b at E2 = 0, so
        # it is only monotone for b <= 0.25. The spec does not state the bound.
        # A non-monotone tone curve is never what a caller wants, so this is a
        # parameter error rather than something to clamp.
        if self.min_lum > 0.25:
            raise ValueError(
                "dst_min is too high for the black lift to stay monotone: "
                f"minLum is {self.min_lum:.4f}, the bound is 0.25"
            )

        # Below KS = 0 the whole domain is spline and P(0) is negative, so E2
        # leaves [0, maxLum], E4 drops below PQ(Lmin) and the chroma ratio
        # changes sign. Same class of error as the black-lift bound, and just
        # as far from any realistic target.
        if self.ks < 0.0:
            raise ValueError(
                "dst_max is too low for the tone curve to stay in range: "
                f"maxLum is {self.max_lum:.4f} and KS is {self.ks:.4f}, "
                "which must not be below 0 (maxLum at least 1/3)"
            )

    @property
    def degenerate(self):
        """True when the target peak reaches the mastering peak, so KS >= 1.

        Every E1 then takes the identity branch and the spline is never
        evaluated. The branch has to be taken before T is computed, because
        T divides by 1 - KS.
        """
        return self.max_lum >= 1.0

    def knee_luminance(self):
        """The luminance in cd/m2 at which the spline takes over from identity."""
        return float(pq_eotf(self.ks * (self.pq_lw - self.pq_lb) + self.pq_lb))

    def __call__(self, code):
        """Map a PQ code through the EETF to a PQ code."""
        span = self.pq_lw - self.pq_lb
        # E1, clamped to the domain the spec defines the curve on. Above the
        # mastering peak is treated as the peak, below mastering black as black.
        e1 = np.clip((np.asarray(code, dtype=np.float64) - self.pq_lb) / span, 0.0, 1.0)

        if self.degenerate:
            e2 = e1
        else:
            t = (e1 - self.ks) / (1.0 - self.ks)
            t2 = t * t
            t3 = t2 * t
            p = (
                (2.0 * t3 - 3.0 * t2 + 1.0) * self.ks
                + (t3 - 2.0 * t2 + t) * (1.0 - self.ks)
                + (-2.0 * t3 + 3.0 * t2) * self.max_lum
            )
            e2 = np.where(e1 < self.ks, e1, p)

        e3 = e2 + self.min_lum * (1.0 - e2) ** 4
        return e3 * span + self.pq_lb


# --- The five colour representations, Annex 5 options 1 to 5 ---------------

REPRESENTATIONS = ("ictcp", "ycbcr", "yrgb", "rgb", "maxrgb")


def chroma_ratio(v1, v2):
    """min(v1/v2, v2/v1), taken as 1 where either value is zero."""
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    ok = (v1 != 0.0) & (v2 != 0.0)
    d1 = np.where(ok, v1, 1.0)
    d2 = np.where(ok, v2, 1.0)
    return np.where(ok, np.minimum(v1 / d2, v2 / d1), 1.0)


def _scale_rgb(rgb, v1, v2, black):
    """rgb * (v2 / v1), taking the curve's own black where v1 is not positive.

    A driving value of zero leaves the ratio undefined. Annex 5 says nothing
    about it, but the other three representations put an exact-black input at
    Lmin, so these two have to as well. Returning literal black instead would
    make exact black the one unlifted pixel in a frame whenever dst_min is
    above src_min, a visible step one LSB wide.
    """
    ok = v1 > 0.0
    k = np.where(ok, v2 / np.where(ok, v1, 1.0), 0.0)
    return np.where(ok[..., None], rgb * k[..., None], black)


def bt2390(
    rgb,
    src_min,
    src_max,
    dst_min=0.0,
    dst_max=203.0,
    nominal_luminance=100.0,
    representation="ictcp",
    curve=None,
):
    """Tone map linear BT.2020 RGB.

    Input has 1.0 meaning nominal_luminance cd/m2. Output has 1.0 meaning
    dst_max cd/m2. Channels above 1.0 are possible in the ictcp, ycbcr and
    yrgb representations. rgb and maxrgb stay inside [0, 1] when dst_min is
    at or below src_min; a positive black lift takes all five above 1.0 by
    the Annex 5 overshoot, b (1 - maxLum)^4 in PQ.

    curve replaces the EETF with another callable on PQ codes, which is how a
    different implementation's tone curve is run through the same colour
    handling so that the curve is the only thing that differs.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(
            f"representation must be one of {', '.join(REPRESENTATIONS)}, "
            f"got {representation!r}"
        )
    if not np.isfinite(nominal_luminance) or nominal_luminance <= 0.0:
        raise ValueError("nominal_luminance must be positive")

    if curve is None:
        curve = Eetf(src_min, src_max, dst_min, dst_max)
    rgb = np.clip(np.asarray(rgb, dtype=np.float64) * nominal_luminance, 0.0, PQ_PEAK)
    # What an exact-black input becomes: E1 is 0, so E4 is PQ(Lmin) and the
    # luminance is dst_min. The two ratio representations fall back on it.
    black = float(pq_eotf(curve(pq_inverse_eotf(0.0))))

    if representation == "ictcp":
        ictcp = apply_matrix(
            LMSP_TO_ICTCP, pq_inverse_eotf(apply_matrix(RGB2020_TO_LMS, rgb))
        )
        i1 = ictcp[..., 0]
        i2 = curve(i1)
        k = chroma_ratio(i1, i2)
        mapped = np.stack([i2, k * ictcp[..., 1], k * ictcp[..., 2]], axis=-1)
        out = apply_matrix(LMS_TO_RGB2020, pq_eotf(apply_matrix(ICTCP_TO_LMSP, mapped)))

    elif representation == "ycbcr":
        rgbp = pq_inverse_eotf(rgb)
        y1 = rgbp @ LUMA_BT2020_PRINTED
        cb = (rgbp[..., 2] - y1) / CB_DIVISOR
        cr = (rgbp[..., 0] - y1) / CR_DIVISOR
        y2 = curve(y1)
        k = chroma_ratio(y1, y2)
        rp = y2 + CR_DIVISOR * k * cr
        bp = y2 + CB_DIVISOR * k * cb
        gp = (y2 - _KR * rp - _KB * bp) / _KG
        out = pq_eotf(np.stack([rp, gp, bp], axis=-1))

    elif representation == "yrgb":
        y1 = rgb @ LUMA_BT2020_PRINTED
        y2 = pq_eotf(curve(pq_inverse_eotf(y1)))
        out = _scale_rgb(rgb, y1, y2, black)

    elif representation == "rgb":
        out = pq_eotf(curve(pq_inverse_eotf(rgb)))

    else:  # maxrgb
        m1 = rgb.max(axis=-1)
        m2 = pq_eotf(curve(pq_inverse_eotf(m1)))
        out = _scale_rgb(rgb, m1, m2, black)

    return out / dst_max
