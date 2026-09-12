"""The generated fixture archive.

Phase 2 onward measures the plugin against these arrays, so the archive needs
checking in its own right: that it builds, that its shapes line up, and that
the inputs really do contain the cases they are meant to cover.
"""

import numpy as np
import pytest
from bt2390_ref import (
    REPRESENTATIONS,
    RGB2020_TO_XYZ,
    Eetf,
    apply_matrix,
    pq_eotf,
    pq_inverse_eotf,
)
from bt2407_ref import clip
from fixtures import build


@pytest.fixture(scope="module")
def archive():
    return build()


def test_every_case_has_an_input_and_an_output(archive):
    meta, arrays = archive
    assert meta["tone"] and meta["gamut"]
    for name in meta["tone"]:
        rows = arrays[f"tm/{name}/in"]
        assert rows.ndim == 2 and rows.shape[1] == 3
        for representation in REPRESENTATIONS:
            assert arrays[f"tm/{name}/{representation}"].shape == rows.shape
    for name in meta["gamut"]:
        rows = arrays[f"gm/{name}/in"]
        assert arrays[f"gm/{name}/out"].shape == rows.shape


def test_nothing_in_the_archive_is_nan(archive):
    _, arrays = archive
    for key, value in arrays.items():
        assert np.all(np.isfinite(value)), key


def test_the_tone_input_spans_the_cases_it_claims(archive):
    _, arrays = archive
    cd = arrays["tm/default/in"] * 100.0
    assert cd.min() < 0.0  # negatives
    assert (cd == 0.0).all(axis=-1).any()  # black
    assert cd.max() >= 10000.0  # above LW
    assert np.any(np.abs(cd - 203.0) < 1e-9)  # SDR white
    assert np.any(np.abs(cd - 87.836267395426) < 1e-9)  # the knee


def test_a_gamut_input_has_no_usable_chromaticity(archive):
    """X + 15Y + 3Z <= 0 with Y > 0 takes the hard clip, per section 3.5."""
    _, arrays = archive
    rows = arrays["gm/softclip_bt2020/in"]
    xyz = apply_matrix(RGB2020_TO_XYZ, rows)
    denom = xyz[..., 0] + 15.0 * xyz[..., 1] + 3.0 * xyz[..., 2]
    undefined = (denom <= 0.0) & (xyz[..., 1] > 0.0)

    assert undefined.sum() >= 1
    assert np.array_equal(
        arrays["gm/softclip_bt2020/out"][undefined], clip(rows[undefined])
    )


def test_a_tone_case_lifts_the_black_above_the_mastering_black(archive):
    """dst_min above src_min is where the section 3.2 overshoot shows up."""
    meta, arrays = archive
    lifted = [n for n, p in meta["tone"].items() if p["dst_min"] > p["src_min"]]
    assert lifted

    for name in lifted:
        p = meta["tone"][name]
        curve = Eetf(p["src_min"], p["src_max"], p["dst_min"], p["dst_max"])
        assert curve.min_lum > 0.0
        # The peak escapes the nominal [0, 1] by exactly the lift term, which
        # is why neither rgb nor maxrgb can be gated on a hard 1.0 here.
        overshoot = (
            curve.min_lum * (1.0 - curve.max_lum) ** 4 * (curve.pq_lw - curve.pq_lb)
        )
        expected = float(pq_eotf(float(pq_inverse_eotf(p["dst_max"])) + overshoot))
        assert arrays[f"tm/{name}/rgb"].max() == pytest.approx(
            expected / p["dst_max"], rel=1e-9
        )
        assert arrays[f"tm/{name}/rgb"].max() > 1.0


def test_a_gamut_input_sits_outside_the_target_volume(archive):
    _, arrays = archive
    rows = arrays["gm/softclip_bt2020/in"]
    assert rows.max() > 1.0
    assert apply_matrix(RGB2020_TO_XYZ, rows)[..., 1].max() > 1.0
    assert np.all(arrays["gm/softclip_bt2020/out"] >= 0.0)
    assert np.all(arrays["gm/softclip_bt2020/out"] <= 1.0)
