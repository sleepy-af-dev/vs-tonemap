"""Shared plumbing for the compiled-filter tests.

Both filters are measured the same way, against the same oracle, with the
same two gates, so the machinery lives here rather than twice over.
"""

from pathlib import Path

import numpy as np
import vapoursynth as vs

PLUGIN = Path(__file__).resolve().parents[2] / "build" / "tonemapper.dll"

# Two gates, because the fixture inputs are float64 and a clip can only carry
# float32, so a filter never sees the fixture value exactly.
#
# The pipeline gate is section 5 as written: it measures the filter plus the
# quantisation of its input, which is what a script actually gets.
ABSOLUTE_GATE = 1e-6
RELATIVE_GATE = 1e-5
# The filter gate is the same comparison with the oracle re-run on the float32
# the clip carries, so it measures only the arithmetic. Both numbers are one
# float32 ULP. The absolute bound applies only at or below SDR white, because
# above that a half-ULP of the float32 store is itself larger than 1e-6.
FILTER_ABSOLUTE_GATE = 1.2e-7
FILTER_RELATIVE_GATE = 1.2e-7
ABSOLUTE_CEILING = 1.0
RELATIVE_FLOOR = 1e-3

GATES = {
    "pipeline": (ABSOLUTE_GATE, RELATIVE_GATE),
    "filter": (FILTER_ABSOLUTE_GATE, FILTER_RELATIVE_GATE),
}

# The tags every clip these filters accept has to carry, or leave absent.
LINEAR_BT2020 = {"_Transfer": 8, "_Primaries": 9, "_Range": 1}

core = vs.core


def make_clip(rows, props=None):
    """A one-row RGBS clip carrying rows, an (N, 3) array, and props."""
    rows = np.asarray(rows, dtype=np.float32)
    blank = core.std.BlankClip(
        format=vs.RGBS, width=rows.shape[0], height=1, length=1, keep=True
    )

    def fill(n, f):
        out = f.copy()
        for plane in range(3):
            np.asarray(out[plane])[0, :] = rows[:, plane]
        return out

    clip = core.std.ModifyFrame(blank, blank, fill)
    return core.std.SetFrameProps(clip, **props) if props else clip


def run(clip):
    """The filter's output as an (N, 3) float64 array, plus the frame props."""
    frame = clip.get_frame(0)
    rows = np.stack([np.asarray(frame[p])[0, :] for p in range(3)], axis=-1)
    return rows.astype(np.float64), dict(frame.props)


def evaluate(clip):
    """Force evaluation so a per-frame error surfaces as an exception."""
    clip.get_frame(0)


def error_stats(got, expected):
    """Absolute error at or below SDR white, relative error above the floor.

    The two cover different parts of the range. Scoping the absolute one is
    what keeps it meaningful: the output is stored as float32, whose half-ULP
    at value v is v times 2^-24, so an absolute bound of 1e-6 is unreachable
    above v = 16.8 whatever the arithmetic does.
    """
    error = np.abs(got - expected)
    absolute = np.where(np.abs(expected) <= ABSOLUTE_CEILING, error, 0.0)
    big = np.abs(expected) > RELATIVE_FLOOR
    relative = np.zeros_like(error)
    relative[big] = error[big] / np.abs(expected[big])
    return absolute, relative


def float32_ulp_distance(got, expected):
    """How many representable float32 steps apart the two values are.

    Ordering the bit patterns this way is monotone across zero and through the
    denormals, which a division by np.spacing is not.
    """

    def ordered(x):
        bits = np.asarray(x, dtype=np.float32).view(np.int32).astype(np.int64)
        return np.where(bits < 0, np.int64(-(2**31)) - bits, bits)

    return np.abs(ordered(got) - ordered(expected))


def report_errors(label, report, worst, ulps):
    """The per-case table, the proposed thresholds and the ULP histogram."""
    print(f"\n{label}: error against the float64 oracle")
    print(f"  {'':16}{'pipeline (float32 in)':>26}{'filter alone':>26}")
    print(f"  {'case':<24}{'max abs':>13}{'max rel':>13}{'max abs':>13}{'max rel':>13}")
    for name, pa, pr, fa, fr in report:
        print(f"  {name:<24}{pa:>13.3e}{pr:>13.3e}{fa:>13.3e}{fr:>13.3e}")
    print(
        f"  {'worst':<24}{worst['pipeline'][0]:>13.3e}{worst['pipeline'][1]:>13.3e}"
        f"{worst['filter'][0]:>13.3e}{worst['filter'][1]:>13.3e}"
    )

    # Section 5 asks for this for information only; nothing is gated on it. A
    # ULP distance is meaningless where both values are cancellation noise
    # around zero, so the histogram covers what an output format can resolve
    # and the rest is reported as an absolute number.
    steps, magnitude, absolute = np.concatenate(ulps, axis=1)
    floor = 1e-6  # a fifteenth of a 16-bit step at SDR white
    big = magnitude > floor
    print(
        f"  filter vs oracle in float32 ULP, {int(big.sum())} values above {floor:g}:"
    )
    for edge in (0, 1, 2, 4):
        print(
            f"    <= {edge:>2} ULP  {float((steps[big] <= edge).mean()) * 100.0:6.2f}%"
        )
    print(f"    max      {int(steps[big].max())} ULP")
    if (~big).any():
        print(
            f"  the other {int((~big).sum())} values are cancellation noise around "
            f"zero: largest magnitude {magnitude[~big].max():.2e}, "
            f"largest difference {absolute[~big].max():.2e}"
        )


def ulp_columns(got, expected):
    """The three rows report_errors needs from one case."""
    return np.stack(
        [
            float32_ulp_distance(got, expected).ravel(),
            np.abs(expected).ravel(),
            np.abs(got - expected).ravel(),
        ]
    )
