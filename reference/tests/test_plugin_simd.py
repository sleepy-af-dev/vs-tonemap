"""The SIMD kernels against the scalar ones.

The scalar path is the reference here, because it is what was measured
against the float64 oracle. These check that vectorising changed nothing
beyond the reassociation a different order of operations allows.
"""

from contextlib import contextmanager

import numpy as np
import pytest
import vapoursynth as vs

from bt2390_ref import REPRESENTATIONS
from vsharness import (
    FILTER_ABSOLUTE_GATE,
    FILTER_RELATIVE_GATE,
    LINEAR_BT2020,
    PLUGIN,
    core,
    error_stats,
    float32_ulp_distance,
    make_clip,
    run,
)

# Every x86 target Highway builds into the one binary, in the order it would
# pick them. Any of these is a valid answer; the test only rejects a machine
# with AVX-512 quietly running SSE2.
X86_TARGETS = (
    "AVX3_SPR",
    "AVX3_ZEN4",
    "AVX3_DL",
    "AVX3",
    "AVX2",
    "SSE4",
    "SSSE3",
    "SSE2",
    "EMU128",
    "SCALAR",
)

# Double lanes follow from the target: 8 on AVX-512, 4 on AVX2, 2 on SSE.
DOUBLE_LANES = {
    "AVX3_SPR": 8,
    "AVX3_ZEN4": 8,
    "AVX3_DL": 8,
    "AVX3": 8,
    "AVX2": 4,
    "SSE4": 2,
    "SSSE3": 2,
    "SSE2": 2,
}


def available_targets():
    """Every compiled target this CPU can run, read at collection time.

    One machine only ever dispatches to one of them, so the comparison against
    the scalar path is run over all of them; that needs the list before the
    fixtures do.
    """
    if not PLUGIN.exists():
        return []
    if not hasattr(core, "tonemap"):
        core.std.LoadPlugin(path=str(PLUGIN))
    names = core.tonemap.Info()["available_targets"]
    # A one-element property comes back as the value itself.
    return [names] if isinstance(names, str) else list(names)


TARGETS = available_targets()


@contextmanager
def forced(plugin, name):
    """Dispatch pinned to one target, whatever this CPU would have chosen."""
    plugin.Info(target=name)
    try:
        yield name
    finally:
        plugin.Info(target="")


@pytest.fixture(params=TARGETS)
def target(request, plugin):
    with forced(plugin, request.param) as name:
        yield name


def test_dispatch_picked_a_real_target(plugin):
    info = plugin.Info()
    assert info["target"] in X86_TARGETS, info["target"]
    if info["target"] in DOUBLE_LANES:
        assert info["double_lanes"] == DOUBLE_LANES[info["target"]]
    # The list is best first, so the automatic choice is its first entry.
    assert TARGETS and info["target"] == TARGETS[0]
    print(
        f"\ndispatched to {info['target']}, {info['double_lanes']} double lanes; "
        f"built and supported here: {', '.join(TARGETS)}"
    )


def test_every_compiled_target_can_be_forced(plugin, target):
    info = plugin.Info()
    assert info["target"] == target
    if target in DOUBLE_LANES:
        assert info["double_lanes"] == DOUBLE_LANES[target]


def test_an_unknown_target_is_refused(plugin):
    with pytest.raises(vs.Error):
        plugin.Info(target="AVX9")
    # The refusal left the automatic choice alone.
    assert plugin.Info()["target"] == TARGETS[0]


def test_the_shipped_build_uses_double_lanes(plugin):
    """The float32 kernel is a measurement aid and must never ship enabled."""
    assert plugin.Info()["ictcp_float32_lanes"] == 0


def test_info_reports_the_version_in_full(plugin):
    """A plugin version packs no patch, so Info() is where a 0.1.1 names itself.

    Both come from the same three constants, so this fails only if that
    derivation is broken, which is exactly when a release would claim a
    version it is not.
    """
    major, minor, patch = (int(part) for part in plugin.Info()["version"].split("."))
    assert (major, minor) == (plugin.version.major, plugin.version.minor)
    assert patch >= 0


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_tone_mapping_simd_matches_scalar(plugin, fixtures, representation, target):
    meta, arrays = fixtures
    worst = [0.0, 0.0, 0]

    for name, params in meta["tone"].items():
        clip = make_clip(arrays[f"tm/{name}/in"], LINEAR_BT2020)
        args = dict(
            src_min=params["src_min"],
            src_max=params["src_max"],
            dst_min=params["dst_min"],
            dst_max=params["dst_max"],
            nominal_luminance=params["nominal_luminance"],
            representation=representation,
        )
        simd, _ = run(plugin.BT2390(clip, simd=1, **args))
        scalar, _ = run(plugin.BT2390(clip, simd=0, **args))

        absolute, relative = error_stats(simd, scalar)
        assert absolute.max() <= FILTER_ABSOLUTE_GATE, name
        assert relative.max() <= FILTER_RELATIVE_GATE, name
        worst[0] = max(worst[0], float(absolute.max()))
        worst[1] = max(worst[1], float(relative.max()))
        # A ULP distance means nothing where both values are cancellation
        # noise around zero, so it is taken only where an output format could
        # resolve the difference.
        steps = float32_ulp_distance(simd, scalar)
        resolvable = np.abs(scalar) > 1e-6
        if resolvable.any():
            worst[2] = max(worst[2], int(steps[resolvable].max()))

    print(
        f"\n{target:<10}{representation:<8} simd vs scalar: max abs {worst[0]:.3e}, "
        f"max rel {worst[1]:.3e}, max {worst[2]} float32 ULP"
    )


def test_gamut_simd_matches_scalar(plugin, fixtures, target):
    meta, arrays = fixtures
    worst = [0.0, 0.0, 0]

    for name, params in meta["gamut"].items():
        props = dict(LINEAR_BT2020)
        if params.get("props"):
            props.update(params["props"])
        clip = make_clip(arrays[f"gm/{name}/in"], props)
        args = {"method": params["method"]}
        if "beta" in params:
            args["beta"] = params["beta"]
        if "src_gamut" in params:
            args["src_gamut"] = params["src_gamut"]

        simd, _ = run(plugin.BT2407(clip, simd=1, **args))
        scalar, _ = run(plugin.BT2407(clip, simd=0, **args))

        absolute, relative = error_stats(simd, scalar)
        assert absolute.max() <= FILTER_ABSOLUTE_GATE, name
        assert relative.max() <= FILTER_RELATIVE_GATE, name
        worst[0] = max(worst[0], float(absolute.max()))
        worst[1] = max(worst[1], float(relative.max()))
        # A ULP distance means nothing where both values are cancellation
        # noise around zero, so it is taken only where an output format could
        # resolve the difference.
        steps = float32_ulp_distance(simd, scalar)
        resolvable = np.abs(scalar) > 1e-6
        if resolvable.any():
            worst[2] = max(worst[2], int(steps[resolvable].max()))

    print(
        f"\n{target:<10}{'gamut':<8} simd vs scalar: max abs {worst[0]:.3e}, "
        f"max rel {worst[1]:.3e}, max {worst[2]} float32 ULP"
    )


def test_the_tail_of_a_row_is_handled(plugin, fixtures):
    """Widths that are not a whole number of vectors, around the lane count.

    Run on the widest target and on the narrowest as well, because a width
    that is a whole number of vectors on one is a tail on the other.
    """
    _, arrays = fixtures
    rows = arrays["tm/default/in"]
    for name in dict.fromkeys([TARGETS[0], TARGETS[-1]]):
        with forced(plugin, name):
            for width in range(1, 20):
                clip = make_clip(rows[:width], LINEAR_BT2020)
                simd, _ = run(plugin.BT2390(clip, src_min=0.0, src_max=1000.0, simd=1))
                scalar, _ = run(
                    plugin.BT2390(clip, src_min=0.0, src_max=1000.0, simd=0)
                )
                assert np.abs(simd - scalar).max() <= FILTER_ABSOLUTE_GATE, (
                    name,
                    width,
                )


def test_a_multi_row_frame_matches_row_by_row(plugin, fixtures):
    """Stride handling: the second row must not read the first row's tail."""
    import vapoursynth as vs

    from vsharness import core

    _, arrays = fixtures
    rows = arrays["tm/default/in"][:1000]
    height = 4
    stacked = np.tile(rows, (height, 1))

    blank = core.std.BlankClip(
        format=vs.RGBS, width=rows.shape[0], height=height, length=1, keep=True
    )

    def fill(n, f):
        out = f.copy()
        for plane in range(3):
            np.asarray(out[plane])[:, :] = stacked[:, plane].reshape(height, -1)
        return out

    clip = core.std.SetFrameProps(
        core.std.ModifyFrame(blank, blank, fill), **LINEAR_BT2020
    )
    frame = plugin.BT2390(clip, src_min=0.0, src_max=1000.0, simd=1).get_frame(0)
    got = np.stack([np.asarray(frame[p]) for p in range(3)], axis=-1)
    for y in range(1, height):
        assert np.array_equal(got[0], got[y]), y
