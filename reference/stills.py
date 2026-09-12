"""Prepare the material for the Phase 4 visual comparison.

    uv run --project reference python reference/stills.py

Picks three frames from the test clip by scanning it, writes this plugin's
output for them as 8-bit PNG in BT.709 with BT.1886 gamma, and prints the mpv
command lines that produce the same frames through the hdr-toys shader. The
comparison itself is the user's to make; nothing here runs mpv.

Everything this touches is local-only, so it lives outside the test suite and
writes into .local/resources/stills/.
"""

import argparse
import json
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import vapoursynth as vs

import hdrtoys_port
from bt2390_ref import Eetf, bt2390, pq_inverse_eotf
from bt2407_ref import clip as bt2407_clip

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / ".local" / "resources"
STILLS = RESOURCES / "stills"
PLUGIN = ROOT / "build" / "tonemapper.dll"

core: Any = vs.core  # the wheel's stub has no plugin namespaces

# The clip's mastering metadata, which both chains are pinned to so that the
# only difference between them is the thing being compared.
SRC_MIN = 0.0001
SRC_MAX = 1000.0
DST_MAX = 203.0

# This plugin's default target black, which is a true zero.
DST_MIN = 0.0

# hdr-toys takes a contrast ratio rather than a target black, so its black is
# ob = PQ(reference_white / contrast_ratio) and never reaches zero. At its
# default of 1000 that is PQ(0.203), which lands on 8-bit code 14: a visibly
# grey letterbox. Its maximum of 1e8 puts the black at 2e-6 cd/m2, which
# encodes to code 0 like this plugin's.
#
# The curves cannot be made to agree exactly, because hdr-toys also guards
# ib = min(get_min_i(), ob - 1e-3), which forces ib negative once ob is below
# 0.001 whatever else is set. Measured over the whole range at these
# luminances, the residual is 0.047 cd/m2 out of 203, a twentieth of an 8-bit
# code step, against 1.53 cd/m2 if the contrast ratio is left at 1000.
HDR_TOYS_CONTRAST_RATIO = 1e8


def encode_png(path, rgb):
    """8-bit PNG from linear BT.709 in [0, 1], through the BT.1886 inverse EOTF.

    Written by hand because the alternative is a dependency for twenty lines.
    The encoding is gamma 2.4, which is what hdr-toys' bt1886.glsl ends with,
    so the two sets of stills are directly comparable.
    """
    encoded = np.clip(rgb, 0.0, 1.0) ** (1.0 / 2.4)
    data = np.rint(encoded * 255.0).astype(np.uint8)
    height, width, _ = data.shape

    raw = b"".join(b"\x00" + data[y].tobytes() for y in range(height))

    def chunk(tag, payload):
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def open_clip(path):
    for name, dll, call in (
        (
            "bs",
            "BestSource-R21-win64-msvc.dll",
            lambda p: core.bs.VideoSource(source=p),
        ),
        ("ffms2", "ffms2.dll", lambda p: core.ffms2.Source(source=p)),
        ("lsmas", "LSMASHSource.dll", lambda p: core.lsmas.LWLibavSource(source=p)),
    ):
        if not (RESOURCES / dll).exists():
            continue
        if not hasattr(core, name):
            core.std.LoadPlugin(path=str(RESOURCES / dll))
        return call(str(path)), name
    raise SystemExit("no source filter found in .local/resources")


def to_linear(clip, width=None, height=None):
    extra = {} if width is None else {"width": width, "height": height}
    return core.resize.Bicubic(
        clip,
        format=vs.RGBS,
        transfer_in_s="st2084",
        transfer_s="linear",
        primaries_in_s="2020",
        primaries_s="2020",
        nominal_luminance=100,
        **extra,
    )


LUMA = np.array([0.2627, 0.6780, 0.0593])


def scan(clip, step):
    """Score every step-th frame for darkness, highlights and saturation.

    Scored on a downscale, because the choice only needs to be representative
    and a full-resolution scan of a ten minute 4K clip is not worth the wait.
    Downscaling does average specular highlights down, which is why the
    highlight measure is a high percentile rather than a share above a fixed
    level: on this clip nothing survives a 203 nit threshold after the scale.
    """
    small = to_linear(clip, 480, 270)
    rows = []
    for n in range(0, small.num_frames, step):
        frame = small.get_frame(n)
        rgb = np.stack([np.asarray(frame[p]) for p in range(3)], axis=-1)
        luma = rgb @ LUMA
        peak = rgb.max(axis=-1)
        floor = rgb.min(axis=-1)

        # Saturation weighted by luminance. Unweighted, a near-black frame
        # wins on sensor noise, where the ratio is meaningless.
        excursion = (peak - floor) / np.maximum(peak, 1e-9)
        weight = np.maximum(luma, 0.0)
        total = float(weight.sum())
        rows.append(
            {
                "frame": n,
                "mean_nits": float(luma.mean() * 100.0),
                "p999_nits": float(np.percentile(luma, 99.9) * 100.0),
                "peak_nits": float(luma.max() * 100.0),
                "saturation": float((excursion * weight).sum() / total)
                if total > 0
                else 0.0,
            }
        )
    return rows


def choose(rows):
    """One dark frame, one highlight-heavy frame, one saturated frame.

    Each pick is restricted to frames with something in them. A fade to black
    would otherwise win the dark category outright and tell nobody anything,
    and the saturated pick has to come from a reasonably lit frame or it is
    measuring noise.
    """
    lit = [r for r in rows if r["p999_nits"] > 1.0]
    if not lit:
        lit = rows
    median_mean = float(np.median([r["mean_nits"] for r in lit]))
    bright = [r for r in lit if r["mean_nits"] >= median_mean] or lit
    return {
        "dark": min(lit, key=lambda r: r["mean_nits"]),
        "highlights": max(rows, key=lambda r: r["p999_nits"]),
        "saturated": max(bright, key=lambda r: r["saturation"]),
    }


def timestamp(frame, fps_num=24000, fps_den=1001):
    seconds = frame * fps_den / fps_num
    return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"


def render(linear, frame, method):
    """This plugin, through the compiled filter, at its documented defaults."""
    sdr = core.tonemapper.BT2407(
        core.tonemapper.BT2390(
            linear,
            src_min=SRC_MIN,
            src_max=SRC_MAX,
            dst_min=DST_MIN,
            dst_max=DST_MAX,
            nominal_luminance=100,
        ),
        method=method,
    )
    got = sdr.get_frame(frame)
    return np.stack([np.asarray(got[p]) for p in range(3)], axis=-1)


def linear_frame(linear, frame):
    got = linear.get_frame(frame)
    return np.stack([np.asarray(got[p]) for p in range(3)], axis=-1).astype(np.float64)


def in_bands(rgb, work, rows=256):
    """Apply work to row bands, because a 4K frame in float64 is 200 MB.

    A whole-frame numpy pass through either reference builds enough
    temporaries to matter; banding keeps it flat.
    """
    out = np.empty_like(rgb)
    for start in range(0, rgb.shape[0], rows):
        out[start : start + rows] = work(rgb[start : start + rows])
    return out


def render_hdrtoys(rgb):
    """The hdr-toys shader through its float64 port, then the same hard clip.

    Rendered rather than screenshotted. The port is a line-for-line
    translation, so this is what the shader computes, without a GPU, a display
    mode or a screenshot path in the way.
    """
    params = hdrtoys_port.Params(
        min_luma=SRC_MIN,
        max_pq_y=float(pq_inverse_eotf(SRC_MAX)),
        reference_white=DST_MAX,
        contrast_ratio=HDR_TOYS_CONTRAST_RATIO,
    )
    # The port's convention is 1.0 meaning reference_white; ours is 100 cd/m2.
    scale = 100.0 / DST_MAX
    return in_bands(
        rgb,
        lambda band: bt2407_clip(hdrtoys_port.tone_map(band * scale, "ictcp", params)),
    )


def libplacebo_curve(code, curve, knee_offset=0.5):
    """libplacebo's BT.2390 curve, transcribed from its bt2390() not copied.

    Three things differ from the report. The knee position is a parameter,
    KS = (1 + k) maxLum - k, and BT.2390 prints the k = 0.5 instance. The black
    lift takes the exponent min(1 / minLum, 4) rather than a fixed 4, which
    only bites for a target black above a quarter of the source span. And the
    lifted curve is then rescaled by a gain that puts the peak back on maxLum,
    which the report does not do. The lift applies whatever the sign of
    minLum, so black lands at PQ(Lmin) here exactly as it does in the report.

    This is the curve before pl_tone_map_generate clamps the finished lookup
    table to the output range, so what the stills compare is curve against
    curve. The same transcription is tested in tests/test_oracles.py.
    """
    span = curve.pq_lw - curve.pq_lb
    ks = (1.0 + knee_offset) * curve.max_lum - knee_offset
    bp = min(1.0 / curve.min_lum, 4.0) if curve.min_lum > 0.0 else 4.0
    gain_inv = 1.0 + curve.min_lum / curve.max_lum * (1.0 - curve.max_lum) ** bp
    gain = 1.0 / gain_inv if curve.max_lum < 1.0 else 1.0

    x = np.clip((np.asarray(code) - curve.pq_lb) / span, 0.0, 1.0)
    if ks < 1.0:
        t = np.clip((x - ks) / (1.0 - ks), 0.0, None)
        t2, t3 = t * t, t * t * t
        spline = (
            (2.0 * t3 - 3.0 * t2 + 1.0) * ks
            + (t3 - 2.0 * t2 + t) * (1.0 - ks)
            + (-2.0 * t3 + 3.0 * t2) * curve.max_lum
        )
        x = np.where(x >= ks, spline, x)
    lifted = x + curve.min_lum * (1.0 - x) ** bp
    x = np.where(x < 1.0, gain * (lifted - curve.min_lum) + curve.min_lum, x)
    return x * span + curve.pq_lb


def render_libplacebo(rgb):
    """libplacebo's curve in this plugin's colour handling, then the hard clip.

    Swapping only the curve is what makes this a comparison of curves. The
    rest of the path is identical to the plugin's, so anything visible here is
    the curve and nothing else.
    """
    curve = Eetf(SRC_MIN, SRC_MAX, DST_MIN, DST_MAX)
    return in_bands(
        rgb,
        lambda band: bt2407_clip(
            bt2390(
                band,
                SRC_MIN,
                SRC_MAX,
                dst_min=DST_MIN,
                dst_max=DST_MAX,
                nominal_luminance=100,
                curve=lambda code: libplacebo_curve(code, curve),
            )
        ),
    )


HDR_TOYS_COMMON = [
    "--glsl-shader=~~/shaders/hdr-toys/utils/clip_both.glsl",
    "--glsl-shader=~~/shaders/hdr-toys/transfer-function/pq_inv.glsl",
    "--glsl-shader=~~/shaders/hdr-toys/tone-mapping/bt2390.glsl",
]
HDR_TOYS_TAIL = ["--glsl-shader=~~/shaders/hdr-toys/transfer-function/bt1886.glsl"]
HDR_TOYS_OPTS = [
    # Pin every input the shader's get_max_i() and get_min_i() consult, so the
    # curve cannot move with dynamic metadata. max_pq_y wins outright, and the
    # rest are set consistently in case a build substitutes them.
    f"--glsl-shader-opts=max_pq_y={0.7518270962220163:.16f}",
    f"--glsl-shader-opts=min_luma={SRC_MIN}",
    f"--glsl-shader-opts=max_luma={SRC_MAX}",
    "--glsl-shader-opts=max_cll=0",
    "--glsl-shader-opts=scene_max_r=0",
    "--glsl-shader-opts=scene_max_g=0",
    "--glsl-shader-opts=scene_max_b=0",
    f"--glsl-shader-opts=reference_white={DST_MAX}",
    f"--glsl-shader-opts=contrast_ratio={HDR_TOYS_CONTRAST_RATIO:g}",
    "--glsl-shader-opts=representation=ictcp",
    "--glsl-shader-opts=chroma_correction_scaling=1.0",
]


def mpv_command(clip, gamut_shader):
    parts = [
        "mpv",
        "--vo=gpu-next",
        "--target-colorspace-hint=no",
        "--tone-mapping=clip",
        "--gamut-mapping-mode=clip",
        "--target-prim=bt.2020",
        "--target-trc=pq",
        "--no-config",
        *HDR_TOYS_COMMON,
        f"--glsl-shader=~~/shaders/hdr-toys/gamut-mapping/{gamut_shader}",
        *HDR_TOYS_TAIL,
        *HDR_TOYS_OPTS,
        '"<the test clip>"',
    ]
    return " \\\n    ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=24, help="scan every Nth frame")
    parser.add_argument("--clip", default=str(RESOURCES / "vs-tonemap-test-clip.mkv"))
    parser.add_argument("--rescan", action="store_true", help="ignore the cached scan")
    args = parser.parse_args()

    if not hasattr(core, "tonemapper"):
        core.std.LoadPlugin(path=str(PLUGIN))
    source, indexer = open_clip(Path(args.clip))
    print(
        f"opened with {indexer}: {source.num_frames} frames, {source.width}x{source.height}"
    )

    # The scan is the slow part and the selection is the part worth adjusting,
    # so the scores are kept and reused unless the clip or the step changes.
    cache = RESOURCES / "scan.json"
    key = {"clip": Path(args.clip).name, "step": args.step, "frames": source.num_frames}
    held = json.loads(cache.read_text()) if cache.exists() else {}
    if held.get("key") == key and not args.rescan:
        rows = held["rows"]
        print(f"reusing {len(rows)} scanned frames from {cache.name}")
    else:
        rows = scan(source, args.step)
        cache.write_text(json.dumps({"key": key, "rows": rows}))

    picked = choose(rows)
    STILLS.mkdir(parents=True, exist_ok=True)

    linear = to_linear(source)
    summary = {}
    for label, row in picked.items():
        n = row["frame"]
        summary[label] = dict(row, timestamp=timestamp(n))
        print(
            f"{label:<11} frame {n:>6}  {timestamp(n)}  "
            f"mean {row['mean_nits']:7.2f}  "
            f"99.9th {row['p999_nits']:8.2f}  peak {row['peak_nits']:8.2f} nits  "
            f"saturation {row['saturation']:.3f}"
        )
        # Three curves through the same colour handling and the same hard clip,
        # so the tone curve is the only thing that differs between them, plus
        # this plugin's full default output for the gamut mapper.
        rgb = linear_frame(linear, n)
        variants = {
            "ours-clip": render(linear, n, "clip"),
            "hdrtoys-clip": render_hdrtoys(rgb),
            "libplacebo-clip": render_libplacebo(rgb),
            "ours-softclip": render(linear, n, "softclip"),
        }
        for name, image in variants.items():
            encode_png(STILLS / f"{label}-{name}.png", image)

        # What the eye will be looking for, stated as numbers first. An 8-bit
        # code step is the unit that matters: differences below one are not
        # visible in these stills whatever they are in float.
        base = variants["ours-clip"]
        for name, image in variants.items():
            if name == "ours-clip":
                continue
            codes = np.abs(
                np.rint(np.clip(image, 0, 1) ** (1 / 2.4) * 255)
                - np.rint(np.clip(base, 0, 1) ** (1 / 2.4) * 255)
            )
            summary[label][f"vs_{name}_max_code"] = float(codes.max())
            print(
                f"{'':11} vs {name:<16} max {codes.max():4.0f} code, "
                f"mean {codes.mean():6.3f}, "
                f"differing pixels {float((codes > 0).mean()) * 100:5.2f}%"
            )

    (STILLS / "frames.json").write_text(json.dumps(summary, indent=2))

    print(f"\nwrote {4 * len(picked)} stills to {STILLS}")
    print("\n(a) curve only, the acceptance comparison. hdr-toys side:\n")
    print("    " + mpv_command(args.clip, "clip.glsl"))
    print("\n(b) full output, information only. hdr-toys side:\n")
    print("    " + mpv_command(args.clip, "bottosson.glsl"))
    print(
        "\nIn mpv, seek with a single quote to enter a time, then press s for a\n"
        "window screenshot, which is the mode that includes shader output.\n"
        "Timestamps are above. Switch Windows to SDR mode first: in HDR mode\n"
        "mpv's output goes through the display tone mapping path and the\n"
        "screenshots do not represent the shader's SDR result."
    )


if __name__ == "__main__":
    main()
