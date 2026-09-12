"""Timing for the compiled filter on a synthetic 4K clip.

    uv run --project reference python bench/benchmark.py            one table
    uv run --project reference python bench/benchmark.py --sweep    every path
    uv run --project reference python bench/benchmark.py --sweep --write
    uv run --project reference python bench/benchmark.py --chain    section 4.3

The filter's own cost is the difference between draining the source and
draining the source with the filter on it, so the cost of generating the
synthetic frames cancels out. Two numbers come out of that: nanoseconds per
pixel on one thread, which is the kernel, and frames per second on every
thread, which is what a script would see.

This measures the filters alone. The end-to-end figure the design targets
also carries two resize stages.

Two things this script learned the hard way. Nodes have to be rebuilt for
every pass, because VapourSynth caches frames per node and draining the same
node twice measures the cache; that first reported a 4K filter at 3675 fps.
And a run needs at least as many frames as threads, because the model is
frame-parallel and a short run cannot occupy the machine.
"""

import argparse
import ctypes
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import vapoursynth as vs

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "build" / "tonemapper.dll"
RESULTS = ROOT / "bench" / "results.md"
WIDTH, HEIGHT = 3840, 2160
PIXELS = WIDTH * HEIGHT

REPRESENTATIONS = ("ictcp", "ycbcr", "yrgb", "rgb", "maxrgb")
METHODS = ("clip", "softclip")


def synthetic_frame(seed=20260912):
    """Content that exercises the whole curve rather than one branch of it.

    A luminance ramp across the width takes every pixel column from black to
    ten times SDR white, a hue sweep down the height keeps the chroma ratio
    busy, and a little noise stops neighbouring pixels sharing a branch.
    """
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0.0, 10.0, WIDTH, dtype=np.float32)[None, :]
    vertical = np.linspace(0.0, 1.0, HEIGHT, dtype=np.float32)[:, None]
    base = ramp * (0.5 + 0.5 * vertical)

    frame = np.empty((3, HEIGHT, WIDTH), dtype=np.float32)
    for plane in range(3):
        phase = 2.0 * np.pi * (vertical + plane / 3.0)
        frame[plane] = base * (0.5 + 0.5 * np.cos(phase))
    frame += rng.random(frame.shape, dtype=np.float32) * 0.05
    return frame


def synthetic_clip(core, length, frame):
    blank = core.std.BlankClip(
        format=vs.RGBS, width=WIDTH, height=HEIGHT, length=length, keep=True
    )

    def fill(n, f):
        out = f.copy()
        for plane in range(3):
            np.asarray(out[plane])[:, :] = frame[plane]
        return out

    clip = core.std.ModifyFrame(blank, blank, fill)
    return core.std.SetFrameProps(clip, _Transfer=8, _Primaries=9, _Range=1)


def drain(clip, prefetch):
    start = time.perf_counter()
    for _ in clip.frames(prefetch=prefetch, close=True):
        pass
    return time.perf_counter() - start


def build_chain(core, frames, frame, stage, simd):
    """Freshly built nodes: source alone, or source with one filter on it."""
    clip = synthetic_clip(core, frames, frame)
    kind, name = stage
    if kind == "source":
        return clip
    if kind == "tone":
        return core.tonemapper.BT2390(
            clip, src_min=0.0, src_max=1000.0, representation=name, simd=simd
        )
    return core.tonemapper.BT2407(clip, method=name, simd=simd)


def measure(core, stage, frames, threads, frame, simd):
    core.num_threads = threads
    build_chain(core, 2, frame, stage, simd)  # warm the pages
    drain(build_chain(core, 2, frame, stage, simd), threads)
    base = drain(build_chain(core, frames, frame, ("source", None), simd), threads)
    full = drain(build_chain(core, frames, frame, stage, simd), threads)
    return base, full


def run_case(core, stage, frame, frames, threads, simd):
    base, full = measure(core, stage, frames, threads, frame, simd)
    cost = max(full - base, 0.0) / frames
    return {"fps": frames / full, "ns_per_pixel": cost / PIXELS * 1e9}


def peak_memory_mb():
    """Peak working set of this process since it started.

    A high-water mark that cannot be reset, so it only means the chain when
    the process ran nothing else; --chain exists for that.
    """

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_uint32),
            ("PageFaultCount", ctypes.c_uint32),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        # The types have to be declared. GetCurrentProcess returns a pseudo
        # handle of -1, which ctypes truncates to a 32-bit int by default and
        # the call then fails with nothing to show for it.
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_uint32]
        query.restype = ctypes.c_int

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        if not query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return counters.PeakWorkingSetSize / (1024 * 1024)
    except (AttributeError, OSError):
        return None


def full_chain(core, clip, representation="ictcp"):
    """Section 4.3 end to end: PQ in, tone map, gamut map, BT.709 out."""
    lin = core.resize.Bicubic(
        clip,
        format=vs.RGBS,
        transfer_in_s="st2084",
        transfer_s="linear",
        primaries_in_s="2020",
        primaries_s="2020",
        nominal_luminance=100,
    )
    sdr = core.tonemapper.BT2407(
        core.tonemapper.BT2390(
            lin, representation=representation, nominal_luminance=100
        )
    )
    return core.resize.Bicubic(
        sdr, format=vs.YUV420P10, matrix_s="709", transfer_s="709", primaries_s="709"
    )


def pq_source(core, length):
    """A 4K PQ YUV clip, which is what a real chain starts from."""
    clip = core.std.BlankClip(
        format=vs.YUV420P10,
        width=WIDTH,
        height=HEIGHT,
        length=length,
        color=[700, 512, 600],
    )
    return core.std.SetFrameProps(
        clip,
        _Matrix=9,
        _Transfer=16,
        _Primaries=9,
        _Range=0,
        MasteringDisplayMinLuminance=0.0001,
        MasteringDisplayMaxLuminance=1000.0,
    )


def end_to_end(core, frames, threads):
    """Frames per second for the whole documented script, both resizes included."""
    core.num_threads = threads
    drain(full_chain(core, pq_source(core, 2)), threads)
    seconds = drain(full_chain(core, pq_source(core, frames)), threads)
    return frames / seconds


def chain_in_fresh_process(frames, dll=None):
    """The end-to-end pass on its own, so the peak working set is the chain's.

    Peak working set is a high-water mark for the life of the process, so a
    sweep that has already built 4K arrays and run every path would report its
    own peak rather than the chain's.
    """
    command = [sys.executable, __file__, "--chain", "--frames", str(frames)]
    if dll:
        command += ["--dll", dll]
    done = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(done.stdout.strip().splitlines()[-1])


def cpu_model():
    """The name the chip is sold under.

    platform.processor() gives the family and stepping on Windows, which does
    not say which part this is. The core count the registry name carries is
    dropped, because the line reports the thread count already.
    """
    try:
        import winreg

        path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
            name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
        return re.sub(r"\s*\d+-Core Processor\s*$", "", name).strip()
    except (ImportError, OSError):
        return platform.processor() or "unknown"


def environment(core):
    info = core.tonemapper.Info()
    return {
        "cpu": cpu_model(),
        "cores": os.cpu_count(),
        "vapoursynth": str(core.version_number()),
        "target": info["target"],
        "double_lanes": info["double_lanes"],
        "ictcp_float32_lanes": bool(info["ictcp_float32_lanes"]),
    }


def sweep(core, frame, frames_one, frames_many, threads):
    rows = []
    stages = [("tone", r) for r in REPRESENTATIONS] + [("gamut", m) for m in METHODS]
    for stage in stages:
        entry = {
            "filter": "BT2390" if stage[0] == "tone" else "BT2407",
            "path": stage[1],
        }
        for simd in (0, 1):
            one = run_case(core, stage, frame, frames_one, 1, simd)
            many = run_case(core, stage, frame, frames_many, threads, simd)
            label = "simd" if simd else "scalar"
            entry[f"{label}_ns"] = one["ns_per_pixel"]
            entry[f"{label}_fps"] = many["fps"]
        entry["speedup"] = entry["scalar_ns"] / entry["simd_ns"]
        rows.append(entry)
        print(
            f"  {entry['filter']:<7}{entry['path']:<10}"
            f"{entry['scalar_ns']:>9.1f}{entry['simd_ns']:>9.1f}"
            f"{entry['scalar_fps']:>10.2f}{entry['simd_fps']:>10.2f}{entry['speedup']:>9.2f}x"
        )
    return rows


def write_results(
    rows, env, frames_one, frames_many, threads, chain_fps, peak, float32_fps=None
):
    lines = [
        "# Benchmark results",
        "",
        "Synthetic 4K RGBS, filter only: the cost of generating the frames is",
        "removed by differencing a drain of the source against a drain of the",
        "source with the filter on it. Nanoseconds per pixel are measured on one",
        "thread and are the kernel; frames per second are measured on every",
        "thread and are what a script sees. Neither figure includes the two",
        "resize stages a real chain carries.",
        "",
        "## Machine",
        "",
        f"- CPU: {env['cpu']}, {env['cores']} logical cores",
        f"- VapourSynth core R{env['vapoursynth']}",
        f"- Highway target: {env['target']}, {env['double_lanes']} double lanes",
        f"- Frames: {frames_one} on one thread, {frames_many} on {threads}",
        "",
        "## Scalar against SIMD",
        "",
        "| filter | path | scalar ns/px | SIMD ns/px | scalar fps | SIMD fps | speedup |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['filter']} | {r['path']} | {r['scalar_ns']:.1f} | {r['simd_ns']:.1f} "
            f"| {r['scalar_fps']:.2f} | {r['simd_fps']:.2f} | {r['speedup']:.2f}x |"
        )
    lines += [
        "",
        "## End to end",
        "",
        "The whole script of section 4.3 over a synthetic 4K PQ source: the",
        "resize into linear RGBS, both filters, and the resize back out to",
        "10-bit YUV. This is what the design's target refers to.",
        "",
        f"- {chain_fps:.2f} frames per second, ictcp and softclip, on {threads} threads",
    ]
    if peak is not None:
        lines += [
            f"- Peak working set {peak / 1024:.1f} GB, measured in a process",
            "  that ran nothing but this chain",
            "",
            "A 4K RGBS frame is 100 MB and the model is frame-parallel, so the",
            "memory a chain needs scales with the thread count. Lower",
            "core.num_threads or core.max_cache_size to trade throughput for it.",
        ]
    if float32_fps is not None:
        double_fps = next(r["simd_fps"] for r in rows if r["path"] == "ictcp")
        lines += [
            "",
            "## What the precision costs",
            "",
            "The ictcp kernel built a second time with float lanes and SLEEF's",
            "float pow, everything else unchanged. It is not a shipped path and",
            "exists so the choice of double rests on a measurement.",
            "",
            f"- double lanes: {double_fps:.2f} fps",
            f"- float lanes: {float32_fps:.2f} fps",
            f"- ratio: {float32_fps / double_fps:.2f}x",
            "",
            "Against the float64 oracle the float kernel reaches 2.5e-04",
            "absolute and 5.9% relative, against frozen gates of 1.2e-07 for",
            "both. In output codes that is at worst 5 of 255 with 2.3% of",
            "samples moving by more than one, and at worst 20 of 1023 with 4.5%",
            "moving by more than one. The ICtCp matrices subtract numbers of",
            "similar size, so this path loses far more to float32 than a bare",
            "PQ round trip does.",
        ]
    lines.append("")
    RESULTS.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {RESULTS.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=4, help="frames on one thread")
    parser.add_argument("--representation", default="ictcp")
    parser.add_argument("--simd", type=int, default=1)
    parser.add_argument("--sweep", action="store_true", help="every path, both ways")
    parser.add_argument(
        "--chain",
        action="store_true",
        help="only the end-to-end pass, as one JSON line",
    )
    parser.add_argument("--write", action="store_true", help="write bench/results.md")
    parser.add_argument("--dll", default=None, help="load a different build")
    parser.add_argument(
        "--float32-fps",
        type=float,
        default=None,
        help="the float32 ictcp figure from a --dll run, to record alongside",
    )
    args = parser.parse_args()

    core = vs.core
    if not hasattr(core, "tonemapper"):
        core.std.LoadPlugin(path=str(Path(args.dll) if args.dll else PLUGIN))
    threads = os.cpu_count()
    frames_many = max(args.frames, threads * 2)

    if args.chain:
        # Nothing else runs here, and in particular no synthetic 4K array is
        # built, so the peak working set is the chain's.
        fps = end_to_end(core, frames_many, threads)
        print(json.dumps({"fps": fps, "peak_mb": peak_memory_mb()}))
        return

    frame = synthetic_frame()
    env = environment(core)
    print(f"{WIDTH}x{HEIGHT} RGBS, {env['target']}, {env['double_lanes']} double lanes")

    if args.sweep:
        print(
            f"  {'filter':<7}{'path':<10}{'scalar ns':>9}{'simd ns':>9}"
            f"{'scalar fps':>10}{'simd fps':>10}{'speedup':>10}"
        )
        rows = sweep(core, frame, args.frames, frames_many, threads)
        chain = chain_in_fresh_process(args.frames, args.dll)
        print(
            f"\n  section 4.3 end to end, synthetic 4K PQ source: {chain['fps']:.2f} fps"
        )
        if chain["peak_mb"] is not None:
            print(
                f"  peak working set of that chain alone: "
                f"{chain['peak_mb'] / 1024:.1f} GB at {threads} threads"
            )
        if args.write:
            write_results(
                rows,
                env,
                args.frames,
                frames_many,
                threads,
                chain["fps"],
                chain["peak_mb"],
                args.float32_fps,
            )
        return

    stage = ("tone", args.representation)
    one = run_case(core, stage, frame, args.frames, 1, args.simd)
    many = run_case(core, stage, frame, frames_many, threads, args.simd)
    print(
        f"  {args.representation} simd={args.simd}: "
        f"{one['ns_per_pixel']:.1f} ns/pixel on 1 thread, "
        f"{many['fps']:.2f} fps on {threads}"
    )


if __name__ == "__main__":
    main()
