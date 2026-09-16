"""The compiled HLG filter against the float64 oracle.

The oracle is the expected value; the plugin is never its own reference. The
two gates and the shared clip plumbing are in vsharness.
"""

import numpy as np
import pytest

from hlg_ref import hlg, system_gamma
from vsharness import evaluate, make_clip, run

# What an HLG clip carries once resize has done the matrix and range
# conversion but left the transfer alone. 18 is ARIB STD-B67.
HLG_BT2020 = {"_Transfer": 18, "_Primaries": 9, "_Range": 1}

REFERENCE_WHITE = {400: 101, 600: 138, 800: 172, 1000: 203, 1500: 276, 2000: 343}

PATCH_LADDER = [
    (0, 64, 0.0000),
    (45, 460, 39.8028),
    (75, 720, 201.7407),
    (100, 940, 1000.0000),
]


def grey(signal):
    return np.repeat(np.atleast_1d(np.asarray(signal, dtype=np.float64))[:, None], 3, 1)


# simd=0 throughout this module: the scalar kernel is the reference, tested
# here against the published numbers and the float64 oracle. The vector
# kernel is covered by being bit-identical to it (test_plugin_simd.py), except
# for the accuracy gate below, which both paths must clear.


@pytest.mark.parametrize("lw,expected", sorted(REFERENCE_WHITE.items()))
def test_reference_white_matches_bt2408(plugin, lw, expected):
    got, _ = run(
        plugin.HLG(
            make_clip(grey([0.75]), HLG_BT2020),
            lw=float(lw),
            nominal_luminance=1.0,
            simd=0,
        )
    )
    assert round(float(got[0, 0])) == expected


@pytest.mark.parametrize("patch,code10,expected", PATCH_LADDER)
def test_patch_ladder(plugin, patch, code10, expected):
    signal = (code10 - 64.0) / (940.0 - 64.0)
    got, _ = run(
        plugin.HLG(
            make_clip(grey([signal]), HLG_BT2020),
            lw=1000.0,
            nominal_luminance=1.0,
            simd=0,
        )
    )
    assert float(got[0, 0]) == pytest.approx(expected, abs=0.01)


def test_black_stays_finite_when_the_system_gamma_is_below_one(plugin):
    assert system_gamma(200.0) < 1.0
    got, _ = run(plugin.HLG(make_clip(grey([0.0, 0.0]), HLG_BT2020), lw=200.0, simd=0))
    assert np.all(np.isfinite(got))
    assert got == pytest.approx(0.0, abs=1e-15)


def test_output_is_tagged_linear_and_carries_the_peak(plugin):
    _, props = run(plugin.HLG(make_clip(grey([0.5]), HLG_BT2020), lw=1000.0, lb=0.005))
    assert props["_Transfer"] == 8
    assert props["_Primaries"] == 9
    assert props["_Range"] == 1
    assert props["MasteringDisplayMaxLuminance"] == pytest.approx(1000.0)
    assert props["MasteringDisplayMinLuminance"] == pytest.approx(0.005)


def test_the_peak_is_read_from_the_frame_properties(plugin):
    """lw falls back to MasteringDisplayMaxLuminance, as src_max does."""
    props = dict(HLG_BT2020, MasteringDisplayMaxLuminance=2000.0)
    got, _ = run(plugin.HLG(make_clip(grey([0.75]), props), nominal_luminance=1.0))
    assert round(float(got[0, 0])) == REFERENCE_WHITE[2000]


def test_the_argument_beats_the_frame_property(plugin):
    props = dict(HLG_BT2020, MasteringDisplayMaxLuminance=2000.0)
    got, _ = run(
        plugin.HLG(make_clip(grey([0.75]), props), lw=1000.0, nominal_luminance=1.0)
    )
    assert round(float(got[0, 0])) == REFERENCE_WHITE[1000]


def test_the_peak_defaults_to_1000_when_nothing_says_otherwise(plugin):
    got, _ = run(plugin.HLG(make_clip(grey([0.75]), HLG_BT2020), nominal_luminance=1.0))
    assert round(float(got[0, 0])) == REFERENCE_WHITE[1000]


def test_a_linear_clip_is_rejected(plugin):
    """A PQ-style linear clip is the mistake this filter most expects."""
    clip = plugin.HLG(make_clip(grey([0.5]), {"_Transfer": 8, "_Primaries": 9}))
    with pytest.raises(Exception) as excinfo:
        evaluate(clip)
    assert "_Transfer" in str(excinfo.value)


def test_an_untagged_clip_is_accepted(plugin):
    """Absent is not a contradiction, matching the other two filters."""
    got, _ = run(plugin.HLG(make_clip(grey([0.75])), nominal_luminance=1.0))
    assert round(float(got[0, 0])) == REFERENCE_WHITE[1000]


@pytest.mark.parametrize(
    "kwargs,fragment",
    [
        (dict(lw=0.0), "lw"),
        (dict(lw=-5.0), "lw"),
        (dict(lb=-1.0), "lb"),
        (dict(lw=100.0, lb=200.0), "lb"),
        (dict(nominal_luminance=0.0), "nominal_luminance"),
    ],
)
def test_bad_arguments_fail_at_create_time(plugin, kwargs, fragment):
    """Every check whose inputs are all arguments runs when the filter is
    created, so a bad parameter fails at script evaluation rather than on the
    first frame."""
    with pytest.raises(Exception) as excinfo:
        plugin.HLG(make_clip(grey([0.5]), HLG_BT2020), **kwargs)
    assert fragment in str(excinfo.value)


@pytest.mark.parametrize(
    "lw,lb",
    [
        (1000.0, 0.0),  # the reference display BT.2100 is written around
        (1000.0, 0.005),  # a realistic display black, so beta is non-zero
        # A black high enough to separate alpha = L_W from alpha = L_W - L_B.
        # That misreading scales every output by (lw - lb)/lw, a uniform
        # relative error of lb/lw: 5e-6 at lb 0.005, a 42x margin against the
        # 1.2e-7 gate and comfortably caught, and 5e-4 here at lb 0.5.
        (1000.0, 0.5),
        (4000.0, 0.0),  # outside [400, 2000], so the extended gamma applies
        # Gamma below 1, so the OOTF exponent is negative. lb=0 means the
        # black guard only fires at signal 0, which another test already
        # covers; what this case adds is oracle-exact agreement along a
        # curve with a negative exponent.
        (200.0, 0.0),
    ],
)
def test_matches_the_oracle_on_a_dense_ramp(plugin, lw, lb):
    signal = np.linspace(0.0, 1.0, 4096)
    rows = grey(signal)
    got, _ = run(
        plugin.HLG(
            make_clip(rows, HLG_BT2020),
            lw=lw,
            lb=lb,
            nominal_luminance=100.0,
            simd=0,
        )
    )
    expected = hlg(
        rows.astype(np.float32).astype(np.float64),
        lw=lw,
        lb=lb,
        nominal_luminance=100.0,
    )
    error = np.abs(got - expected)
    relative = np.where(
        np.abs(expected) > 1e-3, error / np.maximum(np.abs(expected), 1e-12), 0.0
    )
    assert error[np.abs(expected) <= 1.0].max() <= 1.2e-7
    assert relative.max() <= 1.2e-7


@pytest.mark.parametrize("lw,lb", [(1000.0, 0.005), (1000.0, 0.5), (400.0, 0.01)])
def test_zero_signal_gives_exactly_the_display_black(plugin, lw, lb):
    """Table 5 builds beta so that a signal of 0 displays exactly L_B.

    This is the assertion that separates alpha = L_W, which is what Table 5
    says, from alpha = L_W - L_B, which is the plausible misreading. The
    wrong one gives lb - lb**2/lw instead.
    """
    got, _ = run(
        plugin.HLG(
            make_clip(grey([0.0, 0.0]), HLG_BT2020),
            lw=lw,
            lb=lb,
            nominal_luminance=1.0,
            simd=0,
        )
    )
    assert float(got[0, 0]) == pytest.approx(lb, rel=1e-6)


def test_the_extended_gamma_formula_is_used_outside_the_production_range(plugin):
    """A peak of 4000 is outside Note 5f's 400 to 2000 range.

    So the extended kappa formula applies, giving gamma 1.481185 and putting
    75% signal on 559.36 cd/m2. Had the simple formula been used instead,
    gamma would be 1.452865 and the answer 580.80, a 3.8% difference. This
    pins which branch runs rather than merely touching it.
    """
    got, _ = run(
        plugin.HLG(
            make_clip(grey([0.75]), HLG_BT2020),
            lw=4000.0,
            nominal_luminance=1.0,
            simd=0,
        )
    )
    assert float(got[0, 0]) == pytest.approx(559.3574, rel=1e-5)


from vsharness import GATES, error_stats, report_errors, ulp_columns  # noqa: E402


@pytest.mark.parametrize("simd", [0, 1])
def test_every_case_meets_the_accuracy_gate(plugin, fixtures, simd, record_property):
    meta, arrays = fixtures
    report = []
    ulps = []
    worst = {"pipeline": [0.0, 0.0], "filter": [0.0, 0.0]}

    for name, params in meta["hlg"].items():
        rows = arrays[f"hlg/{name}/in"]
        got, _ = run(plugin.HLG(make_clip(rows, HLG_BT2020), simd=simd, **params))

        # Two references. The fixture input is float64; a clip can only carry
        # float32, so the filter never sees the fixture value exactly.
        # Against the fixture the measurement includes that quantisation,
        # which is the pipeline's error. Against the oracle re-run on the
        # float32 the clip actually carries, it is the filter's arithmetic.
        against = {
            "pipeline": arrays[f"hlg/{name}/out"],
            "filter": hlg(rows.astype(np.float32).astype(np.float64), **params),
        }

        row = [name]
        for gate in ("pipeline", "filter"):
            absolute, relative = error_stats(got, against[gate])
            row += [absolute.max(), relative.max()]
            worst[gate][0] = max(worst[gate][0], absolute.max())
            worst[gate][1] = max(worst[gate][1], relative.max())
            assert absolute.max() <= GATES[gate][0], f"{name} {gate} absolute"
            assert relative.max() <= GATES[gate][1], f"{name} {gate} relative"
        report.append(tuple(row))
        ulps.append(ulp_columns(got, against["filter"]))

    report_errors(f"HLG simd={simd}", report, worst, ulps)
    record_property("worst_filter_relative", worst["filter"][1])
