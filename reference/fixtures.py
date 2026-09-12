"""Generate the fixtures the plugin is compared against.

Writes one npz into `_fixtures/`, which git ignores. Everything is derived
from a fixed seed and from the oracle, so it is reproducible and is never
committed. Run it with:

    uv run --project reference python reference/fixtures.py

Layout of the archive. `meta` is a JSON document listing the cases. Each
tone-mapping case `name` contributes `tm/<name>/in` and, for each of the five
representations, `tm/<name>/<representation>`. Each gamut case contributes
`gm/<name>/in` and `gm/<name>/out`. All arrays are (N, 3) float64, linear
BT.2020 RGB on input; the consumer decides how to shape them into a frame.
"""

import json
from pathlib import Path

import numpy as np
from bt2390_ref import (
    PRIMARIES_P3D65,
    REPRESENTATIONS,
    RGB709_TO_XYZ,
    RGB2020_TO_XYZ,
    apply_matrix,
    bt2390,
    rgb_to_xyz_matrix,
)
from bt2407_ref import (
    WHITE_UV,
    XYZ_TO_RGB709,
    boundary_t,
    bt2407,
)

OUT_DIR = Path(__file__).resolve().parent / "_fixtures"
ARCHIVE = OUT_DIR / "fixtures.npz"

NOMINAL = 100.0  # cd/m2 that 1.0 means on the tone mapper's input
XYZ_TO_RGB2020 = np.linalg.inv(RGB2020_TO_XYZ)

CORNERS = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],  # primaries
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],  # secondaries
    ],
    dtype=np.float64,
)

MASTERING_P3 = {
    "MasteringDisplayPrimariesX": [0.680, 0.265, 0.150],
    "MasteringDisplayPrimariesY": [0.320, 0.690, 0.060],
    "MasteringDisplayWhitePointX": 0.3127,
    "MasteringDisplayWhitePointY": 0.3290,
}


def scaled(colours, luminances):
    """Each colour scaled so its largest channel reaches each luminance."""
    colours = np.asarray(colours, dtype=np.float64)
    peak = colours.max(axis=-1, keepdims=True)
    unit = colours / np.where(peak > 0.0, peak, 1.0)
    return np.vstack([unit * level for level in luminances])


def along_709_rays(fractions, relative_luminances):
    """Colours at fractions of the distance from white to the BT.709 boundary.

    The hues are the BT.709 primaries and secondaries. A fraction of 1 lands
    exactly on the boundary, below 0.8 lands in the region the default soft
    clip leaves alone, and above 1 lands outside BT.709 but still inside
    BT.2020.
    """
    uw, vw = WHITE_UV
    rows = []
    for hue in CORNERS:
        xyz = apply_matrix(RGB709_TO_XYZ, hue)
        denom = xyz[0] + 15.0 * xyz[1] + 3.0 * xyz[2]
        du = 4.0 * xyz[0] / denom - uw
        dv = 9.0 * xyz[1] / denom - vw
        for y in relative_luminances:
            t709 = float(boundary_t(XYZ_TO_RGB709, np.float64(y), du, dv))
            for fraction in fractions:
                s = fraction * t709
                u2, v2 = uw + s * du, vw + s * dv
                rows.append(
                    [
                        9.0 * y * u2 / (4.0 * v2),
                        y,
                        y * (12.0 - 3.0 * u2 - 20.0 * v2) / (4.0 * v2),
                    ]
                )
    return apply_matrix(XYZ_TO_RGB2020, np.array(rows))


def tone_mapping_input():
    """Every interesting linear BT.2020 input, in cd/m2."""
    knee = 87.836267395426  # where the spline takes over for LW 1000, Lmax 203
    neutral = np.array(
        [0.0, 5e-5, 1e-4, 0.001, 1.0, knee, 203.0, 1000.0, 4000.0, 10000.0]
    )
    ramp = np.linspace(0.0, 1000.0, 256)
    rng = np.random.default_rng(20260912)

    return np.vstack(
        [
            neutral[:, None] * np.ones(3),  # greys at the levels that matter
            ramp[:, None] * CORNERS[0],  # a ramp per channel
            ramp[:, None] * CORNERS[1],
            ramp[:, None] * CORNERS[2],
            scaled(CORNERS, [1.0, 10.0, 203.0, 1000.0, 4000.0]),
            scaled(p3_corners(), [10.0, 203.0, 1000.0]),
            along_709_rays([0.5, 0.79, 0.81, 1.0, 1.05], [0.05, 0.3, 0.7]) * 1000.0,
            np.array([[-10.0, -10.0, -10.0], [-1.0, 50.0, 200.0], [1e-6, 0.0, 0.0]]),
            rng.random((2000, 3)) ** 3 * 1000.0,
        ]
    )


def p3_corners():
    """The P3 primaries and secondaries expressed in BT.2020."""
    return apply_matrix(
        XYZ_TO_RGB2020, apply_matrix(rgb_to_xyz_matrix(PRIMARIES_P3D65), CORNERS)
    )


TONE_CASES = {
    # The plugin's defaults with the spec's fallback mastering black.
    "default": dict(src_min=0.0, src_max=1000.0),
    # The test clip's mastering metadata, and its MaxCLL as an override.
    "mastered_1000": dict(src_min=0.0001, src_max=1000.0),
    "maxcll_737": dict(src_min=0.0001, src_max=737.0),
    # The fallback range BT.2408 names when nothing is known.
    "full_pq_range": dict(src_min=0.0, src_max=10000.0),
    # A black lift and a dimmer target.
    "lifted_black": dict(src_min=0.005, src_max=4000.0, dst_min=0.5, dst_max=100.0),
    # A target black below the mastering black, which expands blacks.
    "expanded_black": dict(src_min=0.01, src_max=1000.0, dst_min=0.0),
    # KS >= 1, where the spline is never evaluated.
    "degenerate": dict(src_min=0.0, src_max=1000.0, dst_max=1000.0),
    # The same curve reached through a different input scale.
    "nominal_203": dict(src_min=0.0, src_max=1000.0, nominal_luminance=203.0),
}

GAMUT_CASES = {
    "clip": dict(method="clip"),
    "softclip_bt2020": dict(method="softclip", src_gamut="bt2020"),
    "softclip_p3d65": dict(method="softclip", src_gamut="p3d65"),
    "softclip_auto_mastering": dict(
        method="softclip", src_gamut="auto", props=MASTERING_P3
    ),
    "softclip_auto_absent": dict(method="softclip", src_gamut="auto", props=None),
    "softclip_beta_0": dict(method="softclip", src_gamut="bt2020", beta=0.0),
    "softclip_beta_half": dict(method="softclip", src_gamut="bt2020", beta=0.5),
}


def gamut_input(tone_mapped):
    """Tone mapper output plus colours the report does not contemplate."""
    extra = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.2, 1.2, 1.2],  # Y above 1
            [0.0, 1.6, 0.0],  # a channel above 1, Y above 1
            [0.0, 0.9, 0.0],  # a channel below 1, Y below 1, far outside BT.709
            [-0.05, 0.4, 0.02],  # a negative channel
            [0.0, 1.0, -5.0],  # no usable chromaticity
        ]
    )
    return np.vstack(
        [
            tone_mapped,
            extra,
            along_709_rays([0.5, 0.79, 0.81, 1.0, 1.05], [0.05, 0.3, 0.7]),
        ]
    )


def build():
    arrays = {}
    meta = {"nominal_luminance": NOMINAL, "tone": {}, "gamut": {}}

    rgb_cd = tone_mapping_input()
    tone_in = rgb_cd / NOMINAL
    for name, params in TONE_CASES.items():
        full = dict(
            src_min=0.0,
            src_max=1000.0,
            dst_min=0.0,
            dst_max=203.0,
            nominal_luminance=NOMINAL,
        )
        full.update(params)
        scale = NOMINAL / full["nominal_luminance"]
        arrays[f"tm/{name}/in"] = tone_in * scale
        for representation in REPRESENTATIONS:
            arrays[f"tm/{name}/{representation}"] = bt2390(
                tone_in * scale, representation=representation, **full
            )
        meta["tone"][name] = full

    mapped = bt2390(tone_in, src_min=0.0, src_max=1000.0)
    gamut_in = gamut_input(mapped)
    for name, params in GAMUT_CASES.items():
        arrays[f"gm/{name}/in"] = gamut_in
        out, label = bt2407(gamut_in, **params)
        arrays[f"gm/{name}/out"] = out
        meta["gamut"][name] = dict(params, label=label)

    meta["points"] = {"tone": int(tone_in.shape[0]), "gamut": int(gamut_in.shape[0])}
    return meta, arrays


def main():
    meta, arrays = build()
    OUT_DIR.mkdir(exist_ok=True)
    np.savez_compressed(ARCHIVE, meta=np.array(json.dumps(meta, indent=1)), **arrays)
    print(
        f"{ARCHIVE.name}: {len(arrays)} arrays, "
        f"{meta['points']['tone']} tone points, {meta['points']['gamut']} gamut points"
    )


def load():
    """The archive, generating it first if it is not there."""
    if not ARCHIVE.exists():
        main()
    data = np.load(ARCHIVE)
    return json.loads(str(data["meta"])), data


if __name__ == "__main__":
    main()
