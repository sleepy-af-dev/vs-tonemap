"""float64 reference for HLG as ITU-R BT.2100-3 Table 5 states it.

This module is an oracle. The plugin is measured against it and never against
itself, so nothing here is written for speed. Every value is float64, every
luminance is cd/m2 unless the name says otherwise, and every constant comes
from the ITU text rather than from a re-derivation.

HLG is scene-referred: the signal encodes a fraction of the light that fell on
the sensor, not a display luminance. Turning it into display light needs the
OOTF, which is driven by scene luminance and so touches all three channels at
once. That is why this cannot be done with a per-channel transfer function,
and why Note 5e calls the per-channel form an approximation used by legacy
displays.

Sources:
  ITU-R BT.2100-3, Table 5 and Notes 5a to 5i (the OETF, the OOTF, the EOTF,
  the system gamma formulae and the black level lift).
  ITU-R BT.2408-9, section 2 and Table 4 (HDR reference white against peak).
"""

import numpy as np

from bt2390_ref import LUMA_BT2020_PRINTED

# --- Constants, BT.2100-3 Table 5 Notes 5b and 5c --------------------------

A = 0.17883277
# Note 5b defines b = 1 - 4a and c = 0.5 - a ln(4a); Note 5c prints both as
# the decimals below. The printed decimals are used, so that this module and
# the compiled filter agree bit for bit rather than to whatever precision a
# re-derivation happens to land on.
B = 0.28466892
C = 0.55991073

# Note 5f scopes its first gamma formula to the usual production monitoring
# range and directs the extended formula outside it.
GAMMA_RANGE = (400.0, 2000.0)
KAPPA = 1.111


def system_gamma(lw):
    """The system gamma for a display of nominal peak lw cd/m2, Note 5f.

    Two formulae, and which one applies is set by lw rather than by taste.
    They agree at lw = 1000, where both give 1.2, and differ by about 1% at
    the ends of the production range, so the switch is a small discontinuity.
    That discontinuity is the spec's own and is not smoothed over.
    """
    lw = float(lw)
    if not np.isfinite(lw) or lw <= 0.0:
        raise ValueError(f"lw must be a positive luminance in cd/m2, got {lw}")
    if GAMMA_RANGE[0] <= lw <= GAMMA_RANGE[1]:
        return 1.2 + 0.42 * np.log10(lw / 1000.0)
    return 1.2 * KAPPA ** np.log2(lw / 1000.0)


def black_lift(lw, lb, gamma=None):
    """Table 5's beta, the user black level lift.

    beta = sqrt(3 (LB/LW)^(1/gamma)). The radical covers the whole expression
    including the 3; the PDF's text layer is ambiguous on that point, and this
    is the reading under which the EOTF returns exactly L_B for a signal of 0.
    """
    lw = float(lw)
    lb = float(lb)
    if not np.isfinite(lb) or lb < 0.0:
        raise ValueError(f"lb must be a non-negative luminance in cd/m2, got {lb}")
    if lb >= lw:
        raise ValueError(f"lb ({lb}) must be below lw ({lw})")
    if gamma is None:
        gamma = system_gamma(lw)
    return float(np.sqrt(3.0 * (lb / lw) ** (1.0 / gamma)))


# --- The OETF and its inverse, BT.2100-3 Table 5 Note 5a -------------------


def hlg_oetf(e):
    """Scene linear [0, 1] to HLG signal [0, 1].

    Not used by the filter, which only ever decodes. It is here so tests can
    build HLG signal from known scene light and so the round trip can be
    checked in both directions.
    """
    e = np.clip(np.asarray(e, dtype=np.float64), 0.0, 1.0)
    # np.where evaluates both branches, and 12e - b is negative for e below
    # b/12 = 0.0237. The floor keeps the log defined; the value is discarded.
    log_arg = np.maximum(12.0 * e - B, np.finfo(np.float64).tiny)
    return np.where(e <= 1.0 / 12.0, np.sqrt(3.0 * e), A * np.log(log_arg) + C)


def hlg_inverse_oetf(ep):
    """HLG signal [0, 1] to scene linear [0, 1].

    The input is clamped to [0, 1] because that is the whole domain HLG
    defines. Chroma upsampling ringing and limited-range codes outside 64 to
    940 both produce samples beyond it.
    """
    ep = np.clip(np.asarray(ep, dtype=np.float64), 0.0, 1.0)
    return np.where(ep <= 0.5, ep * ep / 3.0, (np.exp((ep - C) / A) + B) / 12.0)


# --- The OOTF and the reference EOTF, BT.2100-3 Table 5 --------------------


def _luminance_scalar(y, exponent):
    """y**exponent, taken as 0 where y is not positive.

    Below a peak of about 301 cd/m2 the system gamma falls under 1, so the
    exponent is negative and y**exponent at y = 0 is infinity. Multiplied by
    a channel of zero that is NaN, which would make every black pixel NaN on
    a low-peak render. Exact black is the only input that reaches this,
    because the scene values are clamped non-negative upstream.
    """
    ok = y > 0.0
    return np.where(ok, np.where(ok, y, 1.0) ** exponent, 0.0)


def ootf(scene, lw=1000.0, gamma=None):
    """Scene linear RGB to display linear RGB in cd/m2.

    Table 5: F_D = alpha Y_S^(gamma - 1) E, with alpha equal to L_W. One
    scalar derived from scene luminance scales all three channels, which is
    what leaves chromaticity untouched. alpha is L_W and not L_W - L_B; the
    black level is carried by beta in the EOTF, and the two together put a
    signal of 0 on exactly L_B.
    """
    if gamma is None:
        gamma = system_gamma(lw)
    scene = np.asarray(scene, dtype=np.float64)
    ys = scene @ LUMA_BT2020_PRINTED
    return lw * _luminance_scalar(ys, gamma - 1.0)[..., None] * scene


def inverse_ootf(display, lw=1000.0, gamma=None):
    """Display linear RGB in cd/m2 to scene linear RGB.

    Not used by the filter. It is here so tests can build HLG signal from
    known display light, which is how the round trip is checked.
    """
    if gamma is None:
        gamma = system_gamma(lw)
    display = np.asarray(display, dtype=np.float64)
    yd = display @ LUMA_BT2020_PRINTED
    scalar = _luminance_scalar(yd / lw, (1.0 - gamma) / gamma)
    return (display / lw) * scalar[..., None]


def hlg(signal, lw=1000.0, lb=0.0, nominal_luminance=100.0):
    """The HLG Reference EOTF of Table 5, rescaled for the tone mapper.

    Input is the HLG signal, an (..., 3) array of R'G'B' in [0, 1]. Output is
    linear BT.2020 scaled so 1.0 means nominal_luminance cd/m2, which is what
    BT2390 expects on its input and what resize's nominal_luminance means.

    Table 5 states the EOTF as
    F_D = OOTF[ OETF^-1[ max(0, (1 - beta) E' + beta) ] ]. The max is applied
    to the lifted signal rather than to E', which matters only when beta is
    non-zero and the input is negative.
    """
    if not np.isfinite(nominal_luminance) or nominal_luminance <= 0.0:
        raise ValueError(f"nominal_luminance must be positive, got {nominal_luminance}")
    gamma = system_gamma(lw)
    beta = black_lift(lw, lb, gamma)
    signal = np.asarray(signal, dtype=np.float64)
    lifted = np.maximum(0.0, (1.0 - beta) * signal + beta)
    return ootf(hlg_inverse_oetf(lifted), lw, gamma) / nominal_luminance
