"""The documented script over real HDR content.

Everything this needs is local-only: the test clip and the source filter that
decodes it. The whole module skips when they are absent, so the suite still
runs on a clean checkout.

What this covers that the synthetic tests cannot is the handover from a real
decoder. The filters read frame properties, never the source, so the thing
worth checking is that the properties a decoder actually writes survive the
resize stage and drive the curve. Every source filter found is exercised,
because the property types differ between them: they do not agree on whether
ContentLightLevel is an integer or a float, and only some carry HDR10Plus.
"""

from pathlib import Path

import numpy as np
import pytest
import vapoursynth as vs
from vsharness import core

RESOURCES = Path(__file__).resolve().parents[2] / ".local" / "resources"

# A short cut of the clip if one has been made, otherwise the clip itself.
CLIP = next(
    (
        p
        for p in (RESOURCES / "segment.mkv", RESOURCES / "vs-tonemap-test-clip.mkv")
        if p.exists()
    ),
    None,
)

# Every indexer that might be here, with the call that opens a file.
INDEXERS = {
    "bs": (
        "BestSource-R21-win64-msvc.dll",
        lambda path: core.bs.VideoSource(source=path),
    ),
    "lsmas": ("LSMASHSource.dll", lambda path: core.lsmas.LWLibavSource(source=path)),
    "ffms2": ("ffms2.dll", lambda path: core.ffms2.Source(source=path)),
}
AVAILABLE = [name for name, (dll, _) in INDEXERS.items() if (RESOURCES / dll).exists()]

pytestmark = pytest.mark.skipif(
    CLIP is None or not AVAILABLE,
    reason="the test clip or its source filters are not here",
)


@pytest.fixture(scope="module")
def loaded():
    for name in AVAILABLE:
        dll, _ = INDEXERS[name]
        if not hasattr(core, name):
            core.std.LoadPlugin(path=str(RESOURCES / dll))
    return AVAILABLE


def to_linear(clip):
    """Section 4.3's first line: PQ BT.2020 to linear RGBS at 100 cd/m2."""
    return core.resize.Bicubic(
        clip,
        format=vs.RGBS,
        transfer_in_s="st2084",
        transfer_s="linear",
        primaries_in_s="2020",
        primaries_s="2020",
        nominal_luminance=100,
    )


def frame_rows(clip, n=0):
    frame = clip.get_frame(n)
    return np.stack([np.asarray(frame[p]) for p in range(3)], axis=-1), dict(frame.props)


@pytest.mark.parametrize("indexer", AVAILABLE)
def test_the_mastering_metadata_reaches_the_filter(plugin, loaded, indexer):
    """No luminance arguments: the curve comes from what the decoder attached."""
    source = INDEXERS[indexer][1](str(CLIP))
    lin = to_linear(source)

    props = dict(lin.get_frame(0).props)
    assert props["MasteringDisplayMaxLuminance"] == pytest.approx(1000.0)
    assert props["MasteringDisplayMinLuminance"] == pytest.approx(0.0001)
    assert len(props["MasteringDisplayPrimariesX"]) == 3

    sdr = plugin.BT2407(plugin.BT2390(lin, nominal_luminance=100))
    rows, out = frame_rows(sdr)
    assert np.all(np.isfinite(rows))
    assert rows.min() >= 0.0 and rows.max() <= 1.0
    assert int(out["_Primaries"]) == 1
    # P3 primaries on the clip, so `auto` has to pick them up.
    assert out["TonemapSourceGamut"] == "mastering"


def test_every_indexer_gives_the_same_result(plugin, loaded):
    """The filters read properties, not the source, so the decoder cannot matter."""
    if len(AVAILABLE) < 2:
        pytest.skip("only one source filter is here")
    results = {}
    for indexer in AVAILABLE:
        sdr = plugin.BT2407(
            plugin.BT2390(
                to_linear(INDEXERS[indexer][1](str(CLIP))), nominal_luminance=100
            )
        )
        results[indexer] = frame_rows(sdr)[0]
    first = AVAILABLE[0]
    for indexer in AVAILABLE[1:]:
        assert np.array_equal(results[first], results[indexer]), (first, indexer)


def test_the_maxcll_override_brightens_the_result(plugin, loaded):
    """MaxCLL is 737 on this clip, below the declared 1000 peak.

    The plugin never reads MaxCLL itself, by design; it is a heuristic rather
    than a spec value, and one line of script passes it. Declaring the lower
    peak means less of the range is compressed, so the result is brighter.
    """
    lin = to_linear(INDEXERS[AVAILABLE[0]][1](str(CLIP)))
    assert dict(lin.get_frame(0).props)["ContentLightLevelMax"] == pytest.approx(737)

    metadata, _ = frame_rows(plugin.BT2390(lin, nominal_luminance=100))
    override, _ = frame_rows(plugin.BT2390(lin, src_max=737.0, nominal_luminance=100))

    assert np.all(np.isfinite(override))
    assert override.mean() > metadata.mean()
    # Same black, so the difference is all at the top of the range.
    assert override.min() == pytest.approx(metadata.min(), abs=1e-6)
