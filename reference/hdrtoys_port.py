"""Port of the hdr-toys BT.2390 mpv shader, used as a numeric second opinion.

Upstream: https://github.com/natural-harmonia-gropius/hdr-toys
File: shaders/hdr-toys/tone-mapping/bt2390.glsl

MIT License

Copyright (c) 2023 natural-harmonia-gropius

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This is a port, not a reimplementation. It keeps the shader's own behaviour
where that differs from the ITU text, so the differences stay visible: the
output clamp in f(), the 0.001 and 1000 cd/m2 fallbacks, the 1e-6 and 1e-3
guards, the chroma scaling knob, and the absence of any clamp on E1. Its
input and output are linear BT.2020 with 1.0 meaning reference_white, which
is the shader's convention and not the plugin's.

GLSL writes `v * mat3(...)` with the mat3 arguments in column-major order, so
the numbers as laid out in the shader source read as rows of an ordinary
matrix times a column vector. The matrices below keep that layout.
"""

from dataclasses import dataclass

import numpy as np

m1 = 2610.0 / 4096.0 / 4.0
m2 = 2523.0 / 4096.0 * 128.0
c1 = 3424.0 / 4096.0
c2 = 2413.0 / 4096.0 * 32.0
c3 = 2392.0 / 4096.0 * 32.0
pw = 10000.0


def pq_eotf_inv(x):
    t = np.power(np.asarray(x, dtype=np.float64) / pw, m1)
    return np.power((c1 + c2 * t) / (1.0 + c3 * t), m2)


def pq_eotf(x):
    t = np.power(np.asarray(x, dtype=np.float64), 1.0 / m2)
    return np.power(np.maximum(t - c1, 0.0) / (c2 - c3 * t), 1.0 / m1) * pw


RGB_TO_XYZ = np.array(
    [
        [0.6369580483012914, 0.14461690358620832, 0.1688809751641721],
        [0.2627002120112671, 0.6779980715188708, 0.05930171646986196],
        [0.0, 0.028072693049087428, 1.060985057710791],
    ]
)
XYZ_TO_RGB = np.array(
    [
        [1.716651187971268, -0.355670783776392, -0.25336628137366],
        [-0.666684351832489, 1.616481236634939, 0.0157685458139111],
        [0.017639857445311, -0.042770613257809, 0.942103121235474],
    ]
)
XYZ_TO_LMS = np.array(
    [
        [0.3592832590121217, 0.6976051147779502, -0.0358915932320290],
        [-0.1920808463704993, 1.1004767970374321, 0.0753748658519118],
        [0.0070797844607479, 0.0748396662186362, 0.8433265453898765],
    ]
)
LMS_TO_XYZ = np.array(
    [
        [2.0701522183894223, -1.3263473389671563, 0.2066510476294053],
        [0.3647385209748072, 0.6805660249472273, -0.0453045459220347],
        [-0.0497472075358123, -0.0492609666966131, 1.1880659249923042],
    ]
)
LMS_TO_ICTCP = (
    np.array(
        [
            [2048.0, 2048.0, 0.0],
            [6610.0, -13613.0, 7003.0],
            [17933.0, -17390.0, -543.0],
        ]
    )
    / 4096.0
)
ICTCP_TO_LMS = np.array(
    [
        [1.0, 0.0086090370379328, 0.1110296250030260],
        [1.0, -0.0086090370379328, -0.1110296250030260],
        [1.0, 0.5600313357106791, -0.3206271749873189],
    ]
)

y_coef = np.array([0.2627002120112671, 0.6779980715188708, 0.05930171646986196])

_a, _b, _c = y_coef
_d = 2.0 * (1.0 - _c)
_e = 2.0 * (1.0 - _a)

RGB_TO_YCBCR = np.array(
    [[_a, _b, _c], [-_a / _d, -_b / _d, 0.5], [0.5, -_b / _e, -_c / _e]]
)
YCBCR_TO_RGB = np.array(
    [[1.0, 0.0, _e], [1.0, -_c / _b * _d, -_a / _b * _e], [1.0, _d, 0.0]]
)


def _mul(m, v):
    return np.asarray(v, dtype=np.float64) @ m.T


@dataclass
class Params:
    """The shader's own parameters, with the shader's own defaults."""

    min_luma: float = 0.0
    max_luma: float = 0.0
    max_cll: float = 0.0
    scene_max_r: float = 0.0
    scene_max_g: float = 0.0
    scene_max_b: float = 0.0
    max_pq_y: float = 0.0
    reference_white: float = 203.0
    contrast_ratio: float = 1000.0
    chroma_correction_scaling: float = 1.0


def get_max_i(p):
    """The shader's peak selection.

    The ICtCp pass takes the scene maximum's luminance through RGB_to_XYZ
    and the other four through dot(rgb, y_coef). Those are the same three
    numbers, so one function covers all five passes.
    """
    if p.max_pq_y > 0.0:
        return p.max_pq_y
    scene_max = np.array([p.scene_max_r, p.scene_max_g, p.scene_max_b])
    if np.any(scene_max > 0.0):
        return pq_eotf_inv(scene_max @ y_coef)
    if p.max_cll > 0.0:
        return pq_eotf_inv(p.max_cll)
    if p.max_luma > 0.0:
        return pq_eotf_inv(p.max_luma)
    return pq_eotf_inv(1000.0)


def get_min_i(p):
    if p.min_luma > 0.0:
        return pq_eotf_inv(p.min_luma)
    return pq_eotf_inv(0.001)


def f(x, iw, ib, ow, ob):
    """The shader's EETF, including its clamp of the result to [ob, ow]."""
    x = np.asarray(x, dtype=np.float64)
    minLum = (ob - ib) / (iw - ib)
    maxLum = (ow - ib) / (iw - ib)

    KS = 1.5 * maxLum - 0.5
    b = minLum

    x = (x - ib) / (iw - ib)

    # GLSL takes the spline as a branch, so it never divides by 1 - KS when
    # KS is 1. np.where evaluates both sides, hence the scalar guard.
    TB = (x - KS) / (1.0 - KS if KS != 1.0 else 1.0)
    TB2 = TB * TB
    TB3 = TB * TB2
    PB = (
        (2.0 * TB3 - 3.0 * TB2 + 1.0) * KS
        + (TB3 - 2.0 * TB2 + TB) * (1.0 - KS)
        + (-2.0 * TB3 + 3.0 * TB2) * maxLum
    )
    x = np.where(KS <= x, PB, x)

    x = np.where(0.0 <= x, x + b * np.power(1.0 - x, 4.0), x)

    x = x * (iw - ib) + ib
    return np.clip(x, ob, ow)


def curve(x, p):
    ow = pq_eotf_inv(p.reference_white)
    ob = pq_eotf_inv(
        p.reference_white / p.contrast_ratio if p.contrast_ratio > 0.0 else 0.0
    )
    iw = max(float(get_max_i(p)), ow + 1e-3)
    ib = min(float(get_min_i(p)), ob - 1e-3)
    return f(x, iw, ib, ow, ob)


def chroma_correction(ab, i1, i2, scaling):
    r1 = i1 / np.maximum(i2, 1e-6)
    r2 = i2 / np.maximum(i1, 1e-6)
    mix = 1.0 + (np.minimum(r1, r2) - 1.0) * scaling
    return ab * mix[..., None]


def tone_map(rgb, representation="ictcp", p=None):
    """The shader's hook() for one representation.

    rgb is linear BT.2020 with 1.0 meaning reference_white, as the shader
    receives it. `prergb` is the shader's name for the R'G'B' pass.
    """
    p = p or Params()
    rgb = np.asarray(rgb, dtype=np.float64)

    if representation == "ictcp":
        lms = _mul(XYZ_TO_LMS, _mul(RGB_TO_XYZ, rgb * p.reference_white))
        iab = _mul(LMS_TO_ICTCP, pq_eotf_inv(lms))
        i1 = iab[..., 0]
        i2 = curve(i1, p)
        ab2 = chroma_correction(iab[..., 1:], i1, i2, p.chroma_correction_scaling)
        out = np.concatenate([i2[..., None], ab2], axis=-1)
        out = _mul(XYZ_TO_RGB, _mul(LMS_TO_XYZ, pq_eotf(_mul(ICTCP_TO_LMS, out))))
        return out / p.reference_white

    if representation == "ycbcr":
        ycc = _mul(RGB_TO_YCBCR, pq_eotf_inv(rgb * p.reference_white))
        y1 = ycc[..., 0]
        y2 = curve(y1, p)
        ab2 = chroma_correction(ycc[..., 1:], y1, y2, p.chroma_correction_scaling)
        out = np.concatenate([y2[..., None], ab2], axis=-1)
        return pq_eotf(_mul(YCBCR_TO_RGB, out)) / p.reference_white

    if representation == "yrgb":
        y1 = (rgb @ y_coef) * p.reference_white
        y2 = pq_eotf(curve(pq_eotf_inv(y1), p))
        return (y2 / np.maximum(y1, 1e-6))[..., None] * rgb

    if representation == "prergb":
        rgbp = pq_eotf_inv(rgb * p.reference_white)
        return pq_eotf(curve(rgbp, p)) / p.reference_white

    if representation == "maxrgb":
        v1 = rgb.max(axis=-1) * p.reference_white
        v2 = pq_eotf(curve(pq_eotf_inv(v1), p))
        return (v2 / np.maximum(v1, 1e-6))[..., None] * rgb

    raise ValueError(f"unknown representation {representation!r}")
