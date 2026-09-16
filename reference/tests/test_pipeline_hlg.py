"""The HLG filter over real HLG footage.

Everything this needs is local-only: a grayscale patch set and a source
filter that decodes it. The module skips itself when either is absent, so
the suite still runs on a clean checkout.

What this covers that the synthetic tests cannot is the handover from a
real decoder. The filter reads frame properties, never the source, so the
thing worth checking is that the tags a decoder actually writes survive the
resize stage and drive the decode. Ground truth is the 10-bit
limited-range code carried in each filename.
"""

import re
from pathlib import Path

import numpy as np
import pytest
import vapoursynth as vs

from hlg_ref import hlg
from vsharness import core

RESOURCES = Path(__file__).resolve().parents[2] / ".local" / "resources"
PATCHES = RESOURCES / "hlg-patches"

INDEXERS = {
    "bs": ("BestSource-R21-win64-msvc.dll", lambda p: core.bs.VideoSource(source=p)),
    "lsmas": ("LSMASHSource.dll", lambda p: core.lsmas.LWLibavSource(source=p)),
    "ffms2": ("ffms2.dll", lambda p: core.ffms2.Source(source=p)),
}
AVAILABLE = [name for name, (dll, _) in INDEXERS.items() if (RESOURCES / dll).exists()]


def patch_files():
    """(patch, code10, path) for each patch clip, or an empty list."""
    if not PATCHES.is_dir():
        return []
    found = []
    for path in sorted(PATCHES.glob("*.mp4")):
        m = re.search(r"Patch_(\d+)_C(\d+)_C(\d+)_", path.name)
        if m:
            found.append((int(m.group(1)), int(m.group(2)), path))
    return found


PATCH_FILES = patch_files()

pytestmark = pytest.mark.skipif(
    not PATCH_FILES or not AVAILABLE,
    reason="the HLG patch material or its source filters are not here",
)


@pytest.fixture(scope="module")
def loaded():
    for name in AVAILABLE:
        dll = INDEXERS[name][0]
        if not hasattr(core, name):
            core.std.LoadPlugin(path=str(RESOURCES / dll))


@pytest.mark.parametrize("indexer", AVAILABLE)
@pytest.mark.parametrize("patch,code10,path", PATCH_FILES)
def test_real_footage_decodes_to_the_luminance_its_code_implies(
    plugin, loaded, indexer, patch, code10, path
):
    src = INDEXERS[indexer][1](str(path))
    props = src.get_frame(0).props
    assert props["_Transfer"] == 18, "the decoder should report ARIB STD-B67"
    assert props["_Primaries"] == 9
    assert props["_Matrix"] == 9

    # Transfer in and out are the same, so zimg does the matrix and range
    # conversion only and hands the HLG signal over untouched. Asking it for
    # linear here would apply the per-channel approximation of Note 5e
    # instead of the OOTF, which is the whole reason this filter exists.
    centre = core.std.CropAbs(
        src, width=16, height=16, left=src.width // 2, top=src.height // 2
    )
    signal = core.resize.Bicubic(
        centre,
        format=vs.RGBS,
        matrix_in_s="2020ncl",
        transfer_in_s="std-b67",
        transfer_s="std-b67",
        primaries_in_s="2020",
        primaries_s="2020",
    )
    decoded = plugin.HLG(signal, lw=1000.0, nominal_luminance=1.0)

    frame = decoded.get_frame(0)
    got = np.stack([np.asarray(frame[p]) for p in range(3)], axis=-1).astype(np.float64)

    expected_signal = (code10 - 64.0) / (940.0 - 64.0)
    expected = hlg(np.full((1, 3), expected_signal), lw=1000.0, nominal_luminance=1.0)[
        0, 0
    ]
    assert got.mean() == pytest.approx(expected, rel=2e-6, abs=1e-6)
    assert frame.props["_Transfer"] == 8
