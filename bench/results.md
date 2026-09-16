# Benchmark results

Synthetic 4K RGBS, filter only: the cost of generating the frames is
removed by differencing a drain of the source against a drain of the
source with the filter on it. Nanoseconds per pixel are measured on one
thread and are the kernel; frames per second are measured on every
thread and are what a script sees. Neither figure includes the two
resize stages a real chain carries.

Written by `bench/benchmark.py --sweep --write`; hand edits here do not
survive the next run.

## Machine

- CPU: AMD Ryzen 9 9950X3D, 32 logical cores
- VapourSynth core R79
- Highway target: AVX3_ZEN4, 8 double lanes
- Frames: 4 on one thread, 64 on 32

## Scalar against SIMD

| filter | path | scalar ns/px | SIMD ns/px | scalar fps | SIMD fps | speedup |
|---|---|---|---|---|---|---|
| BT2390 | ictcp | 156.4 | 48.3 | 14.22 | 47.84 | 3.24x |
| BT2390 | ycbcr | 152.0 | 48.5 | 15.03 | 48.05 | 3.14x |
| BT2390 | yrgb | 72.9 | 18.2 | 33.90 | 78.56 | 4.01x |
| BT2390 | rgb | 191.5 | 52.3 | 13.28 | 46.74 | 3.66x |
| BT2390 | maxrgb | 71.9 | 18.1 | 34.16 | 78.99 | 3.97x |
| BT2407 | clip | 0.9 | 0.7 | 83.87 | 84.45 | 1.35x |
| BT2407 | softclip | 8.4 | 4.2 | 78.85 | 82.44 | 1.99x |
| HLG | - | 13.5 | 6.9 | 82.80 | 79.82 | 1.96x |

## End to end

The whole script of section 4.3 over a synthetic 4K PQ source: the
resize into linear RGBS, both filters, and the resize back out to
10-bit YUV. This is what the design's target refers to.

- 35.18 frames per second, ictcp and softclip, on 32 threads
- Peak working set 7.9 GB, measured in a process
  that ran nothing but this chain

A 4K RGBS frame is 100 MB and the model is frame-parallel, so the
memory a chain needs scales with the thread count. Lower
core.num_threads or core.max_cache_size to trade throughput for it.

The same shape over a synthetic 4K HLG source instead: resize into
HLG-tagged RGBS, decode, both filters, and the resize back out. The
source carries no mastering display metadata, so BT2390 reads its
src_max from what HLG itself writes rather than from the clip.

- 30.44 frames per second, ictcp and softclip, on 32 threads
- Peak working set 7.7 GB, measured in a process
  that ran nothing but this chain

## What the precision costs

The ictcp kernel built a second time with float lanes and SLEEF's
float pow, everything else unchanged. It is not a shipped path and
exists so the choice of double rests on a measurement.

- double lanes: 47.84 fps
- float lanes: 63.20 fps
- ratio: 1.32x

The double-lane figure is this sweep's own ictcp row; the float-lane
figure is from a separate --dll run of the float32 build, so the two
numbers are never from the same invocation of the process.

Against the float64 oracle the float kernel reaches 2.5e-04
absolute and 9.9% relative, against frozen gates of 1.2e-07 for
both, measured over the tone fixtures against the oracle re-run
on the float32 the clip carries, so it is the kernel's own
arithmetic.

Encoded for an SDR display as
round(clip(v, 0, 1) ** (1 / 2.4) * (levels - 1)), the largest
difference is 5 codes of 255 and 22 of 1023. More than one code
of movement reaches 1.1% of channel values and 3.2% of pixels at
8 bits, and 3.0% and 7.9% at 10 bits; a pixel counts when any of
its three channels moves. The ICtCp matrices subtract numbers of
similar size, so this path loses far more to float32 than a bare
PQ round trip does.

