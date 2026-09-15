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
