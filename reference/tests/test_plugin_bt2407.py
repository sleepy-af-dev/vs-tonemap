"""The compiled BT2407 filter against the float64 oracle.

Same two gates as BT2390. The output is BT.709 in [0, 1] throughout, so the
absolute bound covers every value here rather than only the lower part of
the range.
"""

import numpy as np
import pytest
import vapoursynth as vs
from bt2390_ref import PRIMARIES_BT2020, PRIMARIES_P3D65
from bt2407_ref import bt2407
from vsharness import (
    ABSOLUTE_GATE,
    GATES,
    LINEAR_BT2020,
    core,
    error_stats,
    evaluate,
    make_clip,
    report_errors,
    run,
    ulp_columns,
)

MASTERING_P3 = {
    "MasteringDisplayPrimariesX": [0.680, 0.265, 0.150],
    "MasteringDisplayPrimariesY": [0.320, 0.690, 0.060],
    "MasteringDisplayWhitePointX": 0.3127,
    "MasteringDisplayWhitePointY": 0.3290,
}


def filter_args(params):
    """The oracle's case parameters as the filter's arguments."""
    args = {"method": params["method"]}
    if "beta" in params:
        args["beta"] = params["beta"]
    if "src_gamut" in params:
        args["src_gamut"] = params["src_gamut"]
    return args


def clip_props(params):
    """The tags, plus the mastering primaries when the case supplies them."""
    props = dict(LINEAR_BT2020)
    if params.get("props"):
        props.update(params["props"])
    return props


def test_every_gamut_case_meets_the_accuracy_gate(plugin, fixtures):
    meta, arrays = fixtures
    report = []
    ulps = []
    worst = {"pipeline": [0.0, 0.0], "filter": [0.0, 0.0]}

    for name, params in meta["gamut"].items():
        rows = arrays[f"gm/{name}/in"]
        clip = make_clip(rows, clip_props(params))
        got, props = run(plugin.BT2407(clip, **filter_args(params)))

        assert props["TonemapSourceGamut"] == params["label"], name

        oracle_args = {k: v for k, v in params.items() if k != "label"}
        against = {
            "pipeline": arrays[f"gm/{name}/out"],
            "filter": bt2407(rows.astype(np.float32).astype(np.float64), **oracle_args)[
                0
            ],
        }
        row = [name]
        for which, expected in against.items():
            absolute, relative = error_stats(got, expected)
            worst[which][0] = max(worst[which][0], float(absolute.max()))
            worst[which][1] = max(worst[which][1], float(relative.max()))
            row += [float(absolute.max()), float(relative.max())]
            assert absolute.max() <= GATES[which][0], (name, which)
            # The pipeline column is gated on the absolute bound alone here.
            # This output is confined to [0, 1], so that bound is the complete
            # statement, and a relative bound is not: the projection sends a
            # channel near the gamut boundary to a small difference of larger
            # numbers, where quantising the input to float32 moves the result
            # by 1.6e-05 relative but only 7.8e-08 absolute. The filter column
            # keeps both bounds, and passes them, because it is the claim
            # about the arithmetic.
            if which == "filter":
                assert relative.max() <= GATES[which][1], (name, which)
        report.append(row)
        ulps.append(ulp_columns(got, against["filter"]))

        # Annex 5 output is BT.709 inside the unit cube, always.
        assert got.min() >= 0.0 and got.max() <= 1.0, name

    report_errors("bt2407", report, worst, ulps)


def test_the_inner_region_is_untouched_and_equals_the_hard_clip(plugin, fixtures):
    """The report's hatched area: below 1 - beta nothing moves."""
    _, arrays = fixtures
    rows = arrays["gm/softclip_bt2020/in"]
    clip = make_clip(rows, LINEAR_BT2020)
    soft, _ = run(plugin.BT2407(clip, method="softclip", src_gamut="bt2020"))
    hard, _ = run(plugin.BT2407(clip, method="clip"))

    # Where the hard clip did not clamp anything, the colour was inside BT.709
    # already; a good part of that is inside the roll-off's identity region too.
    untouched = np.all((hard > 0.0) & (hard < 1.0), axis=-1)
    agree = np.all(np.abs(soft - hard) < 1e-7, axis=-1)
    assert (agree & untouched).sum() > 0.2 * untouched.sum()


def test_a_source_boundary_colour_reaches_the_target_boundary(plugin):
    """alpha is set so the source boundary lands exactly on the target boundary.

    A colour on the P3 boundary has r = 1 + alpha when P3 is the declared
    source, so it lands on the BT.709 boundary with a channel at zero.
    Declaring the wider BT.2020 source leaves alpha too large and the same
    colour stops short.
    """
    from bt2390_ref import RGB2020_TO_XYZ, apply_matrix, rgb_to_xyz_matrix

    on_p3_boundary = apply_matrix(
        np.linalg.inv(RGB2020_TO_XYZ),
        apply_matrix(rgb_to_xyz_matrix(PRIMARIES_P3D65), np.array([[0.0, 0.35, 0.0]])),
    )
    clip = make_clip(on_p3_boundary, LINEAR_BT2020)
    narrow, _ = run(plugin.BT2407(clip, src_gamut="p3d65"))
    wide, _ = run(plugin.BT2407(clip, src_gamut="bt2020"))
    assert narrow.min() == pytest.approx(0.0, abs=1e-7)
    assert wide.min() > 1e-5


def test_luminance_is_preserved_through_the_projection(plugin, fixtures):
    """Annex 5's whole point: the chromaticity moves, Y does not."""
    from bt2390_ref import RGB709_TO_XYZ, RGB2020_TO_XYZ, apply_matrix

    _, arrays = fixtures
    rows = arrays["gm/softclip_bt2020/in"]
    got, _ = run(plugin.BT2407(make_clip(rows, LINEAR_BT2020), src_gamut="bt2020"))

    xyz = apply_matrix(RGB2020_TO_XYZ, rows)
    y_in = xyz[..., 1]
    y_out = apply_matrix(RGB709_TO_XYZ, got)[..., 1]
    denom = xyz[..., 0] + 15.0 * xyz[..., 1] + 3.0 * xyz[..., 2]
    # Only pixels the projection actually handled. The other three policies,
    # black, white and the hard clip where there is no chromaticity, do not
    # preserve luminance and are not meant to.
    projected = (y_in > 1e-6) & (y_in < 1.0 - 1e-6) & (denom > 0.0)
    assert projected.mean() > 0.5
    assert np.allclose(y_out[projected], y_in[projected], rtol=1e-6, atol=1e-7)


# --- Source gamut selection ------------------------------------------------


@pytest.mark.parametrize(
    "src_gamut,props,label",
    [
        ("auto", MASTERING_P3, "mastering"),
        ("auto", None, "bt2020"),
        ("bt2020", MASTERING_P3, "bt2020"),
        ("p3d65", None, "p3d65"),
    ],
)
def test_the_source_gamut_label_records_what_was_used(plugin, src_gamut, props, label):
    clip = make_clip(np.full((4, 3), 0.5), dict(LINEAR_BT2020, **(props or {})))
    _, out = run(plugin.BT2407(clip, src_gamut=src_gamut))
    assert out["TonemapSourceGamut"] == label


def test_auto_with_mastering_primaries_matches_forcing_p3(plugin):
    """The test clip's primaries are P3, so auto has to land on the P3 answer."""
    rng = np.random.default_rng(71)
    rows = rng.random((4096, 3)) * 1.2
    auto, props = run(
        plugin.BT2407(
            make_clip(rows, dict(LINEAR_BT2020, **MASTERING_P3)), src_gamut="auto"
        )
    )
    forced, _ = run(plugin.BT2407(make_clip(rows, LINEAR_BT2020), src_gamut="p3d65"))
    assert props["TonemapSourceGamut"] == "mastering"
    assert np.array_equal(auto, forced)


@pytest.mark.parametrize(
    "broken",
    [
        {"MasteringDisplayPrimariesX": [0.68, 0.265, 0.15]},  # y missing
        dict(MASTERING_P3, MasteringDisplayPrimariesX=[0.68, 0.265]),  # wrong count
        dict(MASTERING_P3, MasteringDisplayPrimariesY=[0.0, 0.69, 0.06]),  # y = 0
        dict(MASTERING_P3, MasteringDisplayPrimariesX=[0.9, 0.265, 0.15]),  # x + y > 1
        dict(MASTERING_P3, MasteringDisplayPrimariesY=[0.32, 0.32, 0.32]),  # no area
        dict(MASTERING_P3, MasteringDisplayWhitePointX=0.9),  # white outside
        dict(MASTERING_P3, MasteringDisplayWhitePointY=float("nan")),
        dict(MASTERING_P3, MasteringDisplayPrimariesX="not a number"),
    ],
)
def test_unusable_primaries_fall_back_to_bt2020(plugin, broken):
    """Absent or invalid metadata is not an error; the label is the only signal."""
    clip = make_clip(np.full((4, 3), 0.5), dict(LINEAR_BT2020, **broken))
    _, props = run(plugin.BT2407(clip, src_gamut="auto"))
    assert props["TonemapSourceGamut"] == "bt2020"


def test_the_standard_primaries_are_accepted(plugin):
    """BT.2020 red and P3 red both sit exactly on x + y = 1."""
    for primaries in (PRIMARIES_BT2020, PRIMARIES_P3D65):
        assert primaries[0][0] + primaries[0][1] == 1.0
        props = dict(
            LINEAR_BT2020,
            MasteringDisplayPrimariesX=[p[0] for p in primaries],
            MasteringDisplayPrimariesY=[p[1] for p in primaries],
            MasteringDisplayWhitePointX=0.3127,
            MasteringDisplayWhitePointY=0.3290,
        )
        _, out = run(
            plugin.BT2407(make_clip(np.full((4, 3), 0.5), props), src_gamut="auto")
        )
        assert out["TonemapSourceGamut"] == "mastering"


# --- Properties and errors -------------------------------------------------


def test_output_properties(plugin):
    stale = dict(LINEAR_BT2020, **MASTERING_P3)
    _, props = run(plugin.BT2407(make_clip(np.full((4, 3), 0.5), stale)))

    for gone in (
        "MasteringDisplayPrimariesX",
        "MasteringDisplayPrimariesY",
        "MasteringDisplayWhitePointX",
        "MasteringDisplayWhitePointY",
    ):
        assert gone not in props, gone
    assert props["_Primaries"] == 1  # BT.709
    assert props["_Transfer"] == 8  # still linear
    assert int(props["_Range"]) == 1


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (dict(method="oklab"), "method must be one of"),
        (dict(src_gamut="p3dci"), "src_gamut must be one of"),
        (dict(beta=1.0), r"beta must be in \[0, 1\)"),
        (dict(beta=-0.1), r"beta must be in \[0, 1\)"),
        (dict(beta=float("nan")), r"beta must be in \[0, 1\)"),
        # Checked whatever the method, so a typo is never swallowed.
        (dict(method="clip", beta=5.0), r"beta must be in \[0, 1\)"),
        (dict(method="clip", src_gamut="nonsense"), "src_gamut must be one of"),
    ],
)
def test_parameter_errors_are_caught_at_script_time(plugin, kwargs, expected):
    clip = make_clip(np.full((4, 3), 0.5), LINEAR_BT2020)
    with pytest.raises(vs.Error, match=expected):
        plugin.BT2407(clip, **kwargs)


@pytest.mark.parametrize("clip_format", [vs.YUV420P10, vs.RGB24, vs.RGBH], ids=str)
def test_a_clip_of_the_wrong_format_is_rejected(plugin, clip_format):
    wrong = core.std.BlankClip(format=clip_format, width=8, height=8, length=1)
    with pytest.raises(vs.Error, match="RGBS"):
        plugin.BT2407(wrong)


@pytest.mark.parametrize(
    "props,expected",
    [
        ({"_Transfer": 16}, "_Transfer"),
        ({"_Primaries": 1}, "_Primaries"),
        ({"_Range": 0}, "_Range"),
    ],
)
def test_a_contradicting_tag_is_an_error(plugin, props, expected):
    clip = make_clip(np.full((4, 3), 0.5), props)
    with pytest.raises(vs.Error, match=expected):
        evaluate(plugin.BT2407(clip))


def test_the_two_filters_chain(plugin, fixtures):
    """BT2390 then BT2407, which is what a script writes."""
    _, arrays = fixtures
    rows = arrays["tm/default/in"]
    chained = plugin.BT2407(
        plugin.BT2390(make_clip(rows, LINEAR_BT2020), src_min=0.0, src_max=1000.0)
    )
    got, props = run(chained)
    assert got.min() >= 0.0 and got.max() <= 1.0
    assert props["_Primaries"] == 1
    assert np.all(np.isfinite(got))
    # BT2390 removed the luminances; BT2407 removed the primaries.
    for gone in ("MasteringDisplayMaxLuminance", "MasteringDisplayPrimariesX"):
        assert gone not in props


def test_a_nan_free_output_on_pathological_input(plugin):
    """Negative channels, huge values and exact black, none of which may NaN."""
    rows = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.2, 1.2, 1.2],
            [0.0, 1.6, 0.0],
            [-0.05, 0.4, 0.02],
            [0.0, 1.0, -5.0],
            [0.0, 4.0, -10.0],
            [0.0, 0.0, -1.0],
            [1e30, -1e30, 1e-30],
        ]
    )
    for method in ("clip", "softclip"):
        got, _ = run(plugin.BT2407(make_clip(rows, LINEAR_BT2020), method=method))
        assert np.all(np.isfinite(got)), method
        assert got.min() >= 0.0 and got.max() <= 1.0, method


def test_the_beta_ends_behave(plugin, fixtures):
    """beta 0 is a hard clip in chromaticity; a larger beta moves more colours."""
    _, arrays = fixtures
    rows = arrays["gm/softclip_bt2020/in"]
    clip = make_clip(rows, LINEAR_BT2020)
    hard, _ = run(plugin.BT2407(clip, method="clip"))
    zero, _ = run(plugin.BT2407(clip, method="softclip", src_gamut="bt2020", beta=0.0))
    half, _ = run(plugin.BT2407(clip, method="softclip", src_gamut="bt2020", beta=0.5))

    moved_zero = np.any(np.abs(zero - hard) > 1e-6, axis=-1).sum()
    moved_half = np.any(np.abs(half - hard) > 1e-6, axis=-1).sum()
    assert moved_half > moved_zero


def test_the_documented_script_runs(plugin):
    """Section 4.3 end to end, with zimg's own tags at both ends."""
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
    sdr = plugin.BT2407(
        plugin.BT2390(lin, src_min=0.0, src_max=1000.0, nominal_luminance=100)
    )
    out = core.resize.Bicubic(
        sdr, format=vs.YUV420P10, matrix_s="709", transfer_s="709", primaries_s="709"
    )
    frame = out.get_frame(0)
    assert np.all(np.isfinite(np.asarray(frame[0])))
    assert int(frame.props["_Primaries"]) == 1


def test_the_gamut_output_stays_in_range_at_the_absolute_gate(plugin, fixtures):
    """Every BT2407 value is at or below 1, so the absolute bound covers all of it."""
    _, arrays = fixtures
    for name in ("clip", "softclip_bt2020", "softclip_p3d65"):
        expected = arrays[f"gm/{name}/out"]
        assert expected.max() <= 1.0
        got, _ = run(
            plugin.BT2407(
                make_clip(arrays[f"gm/{name}/in"], LINEAR_BT2020),
                **filter_args(
                    {
                        "method": "clip" if name == "clip" else "softclip",
                        "src_gamut": "bt2020" if name.endswith("bt2020") else "p3d65",
                    }
                    if name != "clip"
                    else {"method": "clip"}
                ),
            )
        )
        assert np.abs(got - expected).max() <= ABSOLUTE_GATE, name
