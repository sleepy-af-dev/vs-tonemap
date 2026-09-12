"""The compiled BT2390 filter against the float64 oracle.

The oracle is the expected value; the plugin is never its own reference. The
gate is section 5 of the design: absolute error at most 1e-6 on output
normalised to SDR white, and relative error at most 1e-5 where the oracle
value exceeds 1e-3.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import vapoursynth as vs
from bt2390_ref import bt2390
from fixtures import build

# Two gates, because the fixture inputs are float64 and a clip can only carry
# float32, so the filter never sees the fixture value exactly.
#
# The pipeline gate is section 5 as written: it measures the filter plus the
# quantisation of its input, which is what a script actually gets.
ABSOLUTE_GATE = 1e-6
RELATIVE_GATE = 1e-5
# The filter gate is the same comparison with the oracle re-run on the float32
# the clip carries, so it measures only the arithmetic. Frozen after Phase 2
# at twice the measured maximum, which was 4.75e-07 and 5.94e-08.
FILTER_ABSOLUTE_GATE = 9.5e-7
FILTER_RELATIVE_GATE = 1.2e-7
RELATIVE_FLOOR = 1e-3

PLUGIN = Path(__file__).resolve().parents[2] / "build" / "tonemapper.dll"

core = vs.core


@pytest.fixture(scope="session", autouse=True)
def plugin():
    if not PLUGIN.exists():
        pytest.skip(f"{PLUGIN.name} is not built")
    if not hasattr(core, "tonemapper"):
        core.std.LoadPlugin(path=str(PLUGIN))
    return core.tonemapper


@pytest.fixture(scope="module")
def fixtures():
    return build()


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


def error_stats(got, expected):
    absolute = np.abs(got - expected)
    big = np.abs(expected) > RELATIVE_FLOOR
    relative = np.zeros_like(absolute)
    relative[big] = absolute[big] / np.abs(expected[big])
    return absolute, relative


def float32_ulp_distance(got, expected):
    """How many representable float32 steps apart the two values are.

    Ordering the bit patterns this way is monotone across zero and through
    the denormals, which a division by np.spacing is not.
    """

    def ordered(x):
        bits = np.asarray(x, dtype=np.float32).view(np.int32).astype(np.int64)
        return np.where(bits < 0, np.int64(-(2**31)) - bits, bits)

    return np.abs(ordered(got) - ordered(expected))


# The mastering metadata of the test clip, as a source filter attaches it.
MASTERED = {
    "MasteringDisplayMinLuminance": 0.0001,
    "MasteringDisplayMaxLuminance": 1000.0,
    "_Transfer": 8,
    "_Primaries": 9,
    "_Range": 1,
}


def case_props(params):
    """Only the tags. The luminances go in as arguments unless a test says so."""
    return {"_Transfer": 8, "_Primaries": 9, "_Range": 1}


def filter_args(params, representation):
    return dict(
        src_min=params["src_min"],
        src_max=params["src_max"],
        dst_min=params["dst_min"],
        dst_max=params["dst_max"],
        nominal_luminance=params["nominal_luminance"],
        representation=representation,
    )


IMPLEMENTED = ("ictcp",)


@pytest.mark.parametrize("representation", IMPLEMENTED)
def test_every_case_meets_the_accuracy_gate(
    plugin, fixtures, representation, record_property
):
    meta, arrays = fixtures
    report = []
    ulps = []
    worst = {"pipeline": [0.0, 0.0], "filter": [0.0, 0.0]}

    for name, params in meta["tone"].items():
        rows = arrays[f"tm/{name}/in"]
        got, _ = run(
            plugin.BT2390(
                make_clip(rows, case_props(params)),
                **filter_args(params, representation),
            )
        )

        # Two references. The fixture input is float64; a clip can only carry
        # float32, so the filter never sees the fixture value exactly. Against
        # the fixture the measurement includes that quantisation, which is the
        # pipeline's error. Against the oracle re-run on the float32 the clip
        # actually carries, it is the filter's own arithmetic.
        against = {
            "pipeline": arrays[f"tm/{name}/{representation}"],
            "filter": bt2390(
                rows.astype(np.float32).astype(np.float64),
                representation=representation,
                **{k: v for k, v in params.items()},
            ),
        }
        gates = {
            "pipeline": (ABSOLUTE_GATE, RELATIVE_GATE),
            "filter": (FILTER_ABSOLUTE_GATE, FILTER_RELATIVE_GATE),
        }
        row = [name]
        for which, expected in against.items():
            absolute, relative = error_stats(got, expected)
            worst[which][0] = max(worst[which][0], float(absolute.max()))
            worst[which][1] = max(worst[which][1], float(relative.max()))
            row += [float(absolute.max()), float(relative.max())]
            assert absolute.max() <= gates[which][0], (name, which)
            assert relative.max() <= gates[which][1], (name, which)
        report.append(row)
        ulps.append(
            np.stack(
                [
                    float32_ulp_distance(got, against["filter"]).ravel(),
                    np.abs(against["filter"]).ravel(),
                    np.abs(got - against["filter"]).ravel(),
                ]
            )
        )

    print(f"\n{representation}: error against the float64 oracle")
    print(f"  {'':16}{'pipeline (float32 in)':>26}{'filter alone':>26}")
    print(f"  {'case':<16}{'max abs':>13}{'max rel':>13}{'max abs':>13}{'max rel':>13}")
    for name, pa, pr, fa, fr in report:
        print(f"  {name:<16}{pa:>13.3e}{pr:>13.3e}{fa:>13.3e}{fr:>13.3e}")
    print(
        f"  {'worst':<16}{worst['pipeline'][0]:>13.3e}{worst['pipeline'][1]:>13.3e}"
        f"{worst['filter'][0]:>13.3e}{worst['filter'][1]:>13.3e}"
    )
    print(
        f"  twice the measured maximum: pipeline "
        f"{2 * worst['pipeline'][0]:.3e} / {2 * worst['pipeline'][1]:.3e}, "
        f"filter {2 * worst['filter'][0]:.3e} / {2 * worst['filter'][1]:.3e}"
    )

    # Section 5 asks for this for information only; nothing is gated on it.
    # A ULP distance is meaningless where both values are cancellation noise
    # around zero, so the histogram covers what an output format can resolve
    # and the rest is reported as an absolute number.
    steps, magnitude, absolute = np.concatenate(ulps, axis=1)
    floor = 1e-6  # a fifteenth of a 16-bit step at SDR white
    big = magnitude > floor
    print(f"  filter vs oracle in float32 ULP, {int(big.sum())} values above {floor:g}:")
    for edge in (0, 1, 2, 4):
        print(f"    <= {edge:>2} ULP  {float((steps[big] <= edge).mean()) * 100.0:6.2f}%")
    print(f"    max      {int(steps[big].max())} ULP")
    print(
        f"  the other {int((~big).sum())} values are cancellation noise around zero: "
        f"largest magnitude {magnitude[~big].max():.2e}, "
        f"largest difference {absolute[~big].max():.2e}"
    )

    record_property("max_absolute", worst["filter"][0])
    record_property("max_relative", worst["filter"][1])


# --- Frame properties ------------------------------------------------------


def test_the_source_range_can_come_from_the_frame_properties(plugin, fixtures):
    """Omitting src_min and src_max reads the mastering display metadata."""
    meta, arrays = fixtures
    params = meta["tone"]["mastered_1000"]
    rows = arrays["tm/mastered_1000/in"]
    expected = arrays["tm/mastered_1000/ictcp"]

    got, _ = run(
        plugin.BT2390(
            make_clip(rows, MASTERED),
            dst_min=params["dst_min"],
            dst_max=params["dst_max"],
            nominal_luminance=params["nominal_luminance"],
        )
    )
    absolute, relative = error_stats(got, expected)
    assert absolute.max() <= ABSOLUTE_GATE
    assert relative.max() <= RELATIVE_GATE


def test_an_argument_overrides_the_property(plugin, fixtures):
    _, arrays = fixtures
    rows = arrays["tm/maxcll_737/in"]
    got, _ = run(plugin.BT2390(make_clip(rows, MASTERED), src_max=737.0, src_min=0.0001))
    absolute, _ = error_stats(got, arrays["tm/maxcll_737/ictcp"])
    assert absolute.max() <= ABSOLUTE_GATE


def test_output_properties(plugin):
    stale = dict(
        MASTERED,
        ContentLightLevelMax=737.0,
        ContentLightLevelAverage=130.0,
        MasteringDisplayPrimariesX=[0.680, 0.265, 0.150],
        MasteringDisplayPrimariesY=[0.320, 0.690, 0.060],
        MasteringDisplayWhitePointX=0.3127,
        MasteringDisplayWhitePointY=0.3290,
    )
    _, props = run(plugin.BT2390(make_clip(np.full((4, 3), 0.5), stale)))

    for gone in (
        "MasteringDisplayMinLuminance",
        "MasteringDisplayMaxLuminance",
        "ContentLightLevelMax",
        "ContentLightLevelAverage",
        "DolbyVisionRPU",
        "HDR10Plus",
        "_ColorRange",
    ):
        assert gone not in props, gone
    # BT2407 reads these, so they survive.
    assert len(props["MasteringDisplayPrimariesX"]) == 3
    assert props["MasteringDisplayWhitePointX"] == pytest.approx(0.3127)
    assert props["_Transfer"] == 8
    assert props["_Primaries"] == 9
    assert int(props["_Range"]) == 1


# --- Error cases, section 4.1 ----------------------------------------------


def evaluate(clip):
    """Force evaluation so a per-frame error surfaces as an exception."""
    clip.get_frame(0)


def test_a_non_rgbs_clip_is_rejected(plugin):
    yuv = core.std.BlankClip(format=vs.YUV420P10, width=8, height=8, length=1)
    with pytest.raises(vs.Error, match="RGBS"):
        plugin.BT2390(yuv, src_min=0.0, src_max=1000.0)


def test_missing_mastering_metadata_is_an_error(plugin):
    clip = make_clip(np.full((4, 3), 0.5), case_props(None))
    with pytest.raises(vs.Error, match="src_max"):
        evaluate(plugin.BT2390(clip))
    with pytest.raises(vs.Error, match="src_min"):
        evaluate(plugin.BT2390(clip, src_max=1000.0))
    evaluate(plugin.BT2390(clip, src_max=1000.0, src_min=0.0))


@pytest.mark.parametrize(
    "value",
    [0.0, -1.0, float("nan"), float("inf")],
    ids=["zero", "negative", "nan", "inf"],
)
def test_an_unusable_mastering_peak_counts_as_absent(plugin, value):
    """A curve needs a positive peak, so these read as no metadata at all."""
    clip = make_clip(
        np.full((4, 3), 0.5),
        dict(
            case_props(None),
            MasteringDisplayMinLuminance=0.0,
            MasteringDisplayMaxLuminance=value,
        ),
    )
    with pytest.raises(vs.Error, match="src_max"):
        evaluate(plugin.BT2390(clip))


def test_a_zero_mastering_black_is_valid(plugin):
    clip = make_clip(
        np.full((4, 3), 0.5),
        dict(
            case_props(None),
            MasteringDisplayMinLuminance=0.0,
            MasteringDisplayMaxLuminance=1000.0,
        ),
    )
    evaluate(plugin.BT2390(clip))


@pytest.mark.parametrize(
    "props,expected",
    [
        ({"_Transfer": 16}, "_Transfer"),  # PQ rather than linear
        ({"_Primaries": 1}, "_Primaries"),  # BT.709 rather than BT.2020
        ({"_Range": 0}, "_Range"),  # limited rather than full
        # The deprecated spelling, which the core rewrites into _Range with the
        # value inverted, so the filter sees and reports the current one.
        ({"_ColorRange": 1}, "_Range"),
    ],
)
def test_a_contradicting_tag_is_an_error(plugin, props, expected):
    clip = make_clip(np.full((4, 3), 0.5), props)
    with pytest.raises(vs.Error, match=expected):
        evaluate(plugin.BT2390(clip, src_min=0.0, src_max=1000.0))


def test_absent_tags_are_accepted(plugin):
    """Consistency is demanded, tagging is not."""
    evaluate(plugin.BT2390(make_clip(np.full((4, 3), 0.5)), src_min=0.0, src_max=1000.0))


def test_the_pre_r74_range_key_cannot_be_reached_from_here():
    """The _ColorRange fallback in the filter is untestable on this core.

    `_Range` arrived in R74. The filter falls back to `_ColorRange` with the
    old convention, 0 for full, when `_Range` is absent, which keeps range
    validation working on R55 to R73. From R74 on the core rewrites
    `_ColorRange` into `_Range` inside the map itself and inverts the value,
    so the old key cannot be put on a frame here by any route: setting it
    through SetFrameProps or through a ModifyFrame copy both come back as
    `_Range`. The fallback is exercised the first time this is built against
    an older core.
    """
    clip = core.std.BlankClip(format=vs.RGBS, width=4, height=1, length=1)

    def rewrite(n, f):
        out = f.copy()
        del out.props["_Range"]
        out.props["_ColorRange"] = 0
        return out

    props = dict(core.std.ModifyFrame(clip, clip, rewrite).get_frame(0).props)
    assert "_ColorRange" not in props
    assert int(props["_Range"]) == 1


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (dict(src_min=0.0, src_max=0.0), "src_max must be a positive luminance"),
        (dict(src_min=1000.0, src_max=1000.0), "src_min must be below src_max"),
        (dict(src_min=2000.0, src_max=1000.0), "src_min must be below src_max"),
        (dict(src_min=0.0, src_max=1000.0, dst_min=300.0), "dst_min must be below"),
        (dict(src_min=0.0, src_max=1000.0, dst_min=50.0), "monotone"),
        (dict(src_min=0.0, src_max=1000.0, dst_max=3.0), "KS"),
        (dict(src_min=0.0, src_max=20000.0), "0 to 10000"),
        (dict(src_min=-1.0, src_max=1000.0), "0 to 10000"),
        (dict(src_min=0.0, src_max=1000.0, nominal_luminance=0.0), "nominal_luminance"),
        (dict(src_min=0.0, src_max=1000.0, representation="oklab"), "representation"),
    ],
)
def test_parameter_errors_are_caught_at_script_time(plugin, kwargs, expected):
    """Anything independent of the frame properties fails before frame 0."""
    clip = make_clip(np.full((4, 3), 0.5))
    with pytest.raises(vs.Error, match=expected):
        plugin.BT2390(clip, **kwargs)


def test_a_bad_property_derived_range_fails_per_frame(plugin):
    """The same checks run again once the properties supply the numbers."""
    clip = make_clip(
        np.full((4, 3), 0.5),
        dict(
            case_props(None),
            MasteringDisplayMinLuminance=500.0,
            MasteringDisplayMaxLuminance=100.0,
        ),
    )
    with pytest.raises(vs.Error, match="src_min must be below src_max"):
        evaluate(plugin.BT2390(clip))


def test_the_documented_script_runs(plugin):
    """Section 4.3 end to end, so zimg's own tags have to satisfy the contract.

    This is the shape every user will write. It catches a disagreement with
    the resize filter over _Transfer, _Primaries or _Range, which the
    hand-tagged clips elsewhere in this file cannot.
    """
    src = core.std.BlankClip(
        format=vs.YUV420P10, width=64, height=64, length=1, color=[700, 512, 512]
    )
    src = core.std.SetFrameProps(src, _Matrix=9, _Transfer=16, _Primaries=9, _Range=0)
    lin = core.resize.Bicubic(
        src,
        format=vs.RGBS,
        transfer_in_s="st2084",
        transfer_s="linear",
        primaries_in_s="2020",
        primaries_s="2020",
        nominal_luminance=100,
    )
    sdr = plugin.BT2390(lin, src_min=0.0, src_max=1000.0, nominal_luminance=100)
    out = core.resize.Bicubic(
        sdr, format=vs.YUV420P10, matrix_s="709", transfer_s="709", primaries_s="709"
    )
    frame = out.get_frame(0)
    luma = np.asarray(frame[0])
    assert np.all(np.isfinite(luma))
    assert luma.min() > 0
    assert int(frame.props["_Transfer"]) == 1


def test_the_filter_survives_a_multi_row_frame(plugin, fixtures):
    """The row loop and the stride arithmetic, which a 1-row clip never exercises."""
    meta, arrays = fixtures
    params = meta["tone"]["default"]
    rows = arrays["tm/default/in"]
    height = 8
    wide = np.tile(rows, (height, 1))

    blank = core.std.BlankClip(
        format=vs.RGBS, width=rows.shape[0], height=height, length=1, keep=True
    )

    def fill(n, f):
        out = f.copy()
        for plane in range(3):
            np.asarray(out[plane])[:, :] = wide[:, plane].reshape(height, -1)
        return out

    clip = core.std.SetFrameProps(
        core.std.ModifyFrame(blank, blank, fill), **case_props(params)
    )
    frame = plugin.BT2390(clip, **filter_args(params, "ictcp")).get_frame(0)
    got = np.stack([np.asarray(frame[p]) for p in range(3)], axis=-1).astype(np.float64)

    expected = np.tile(arrays["tm/default/ictcp"], (height, 1)).reshape(height, -1, 3)
    assert np.abs(got - expected).max() <= ABSOLUTE_GATE
    assert math.isfinite(float(np.abs(got).max()))
