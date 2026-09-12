"""Timing for the compiled filter on a synthetic 4K clip.

    uv run --project reference python bench/benchmark.py

The filter's own cost is the difference between draining the source and
draining the source with the filter on it, so the cost of generating the
synthetic frames cancels out. Two numbers come out of that: nanoseconds per
pixel on one thread, which is the kernel, and frames per second on every
thread, which is what a script would see.

This measures the filter alone. The end-to-end figure the design targets
also carries two resize stages and comes from vspipe over the test clip.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import vapoursynth as vs

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "build" / "tonemapper.dll"
WIDTH, HEIGHT = 3840, 2160
PIXELS = WIDTH * HEIGHT


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


def timed(core, frames, threads, frame, representation=None):
    """One drain over freshly built nodes.

    The nodes have to be new each time. VapourSynth caches frames per node, so
    draining the same node twice measures the cache on the second pass, which
    is how this script first reported a 4K filter running at 3675 fps.
    """
    clip = synthetic_clip(core, frames, frame)
    if representation is not None:
        clip = core.tonemapper.BT2390(
            clip, src_min=0.0, src_max=1000.0, representation=representation
        )
    return drain(clip, threads)


def measure(core, representation, frames, threads, frame):
    core.num_threads = threads
    timed(core, 2, threads, frame, representation)  # warm up the pages
    base = timed(core, frames, threads, frame)
    full = timed(core, frames, threads, frame, representation)
    return base, full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--representation", default="ictcp")
    args = parser.parse_args()

    core = vs.core
    if not hasattr(core, "tonemapper"):
        core.std.LoadPlugin(path=str(PLUGIN))
    frame = synthetic_frame()

    print(f"{WIDTH}x{HEIGHT} RGBS, {args.frames} frames, {args.representation}, scalar")
    print(
        f"  {'threads':>8}{'frames':>8}{'source s':>11}{'filtered s':>12}"
        f"{'fps':>9}{'ns/pixel':>11}"
    )
    for threads in (1, os.cpu_count()):
        # The model is frame-parallel, so a run of N frames can only occupy N
        # threads however many the core has.
        frames = args.frames if threads == 1 else max(args.frames, threads * 2)
        base, full = measure(core, args.representation, frames, threads, frame)
        cost = (full - base) / frames
        print(
            f"  {threads:>8}{frames:>8}{base:>11.3f}{full:>12.3f}"
            f"{frames / full:>9.2f}{cost / PIXELS * 1e9:>11.1f}"
        )


if __name__ == "__main__":
    main()
