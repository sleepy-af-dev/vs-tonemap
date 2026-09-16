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
| BT2390 | ictcp | 159.6 | 48.2 | 13.86 | 46.74 | 3.31x |
| BT2390 | ycbcr | 153.7 | 48.4 | 14.82 | 46.18 | 3.18x |
| BT2390 | yrgb | 73.5 | 18.1 | 33.99 | 74.16 | 4.07x |
| BT2390 | rgb | 192.0 | 52.0 | 13.24 | 45.47 | 3.69x |
| BT2390 | maxrgb | 71.8 | 18.0 | 34.13 | 77.72 | 4.00x |
| BT2407 | clip | 1.0 | 0.7 | 87.51 | 87.49 | 1.39x |
| BT2407 | softclip | 8.2 | 4.0 | 76.53 | 81.29 | 2.04x |
| HLG | - | 13.7 | 6.9 | 74.90 | 78.80 | 2.00x |

## End to end

The whole script of section 4.3 over a synthetic 4K PQ source: the
resize into linear RGBS, both filters, and the resize back out to
10-bit YUV. This is what the design's target refers to.

- 34.22 frames per second, ictcp and softclip, on 32 threads
- Peak working set 7.3 GB, measured in a process
  that ran nothing but this chain

A 4K RGBS frame is 100 MB and the model is frame-parallel, so the
memory a chain needs scales with the thread count. Lower
core.num_threads or core.max_cache_size to trade throughput for it.

The same shape over a synthetic 4K HLG source instead: resize into
HLG-tagged RGBS, decode, both filters, and the resize back out. The
source carries no mastering display metadata, so BT2390 reads its
src_max from what HLG itself writes rather than from the clip.

- 29.84 frames per second, ictcp and softclip, on 32 threads
- Peak working set 7.4 GB, measured in a process
  that ran nothing but this chain

## What the precision costs

The ictcp kernel built a second time with float lanes and SLEEF's
float pow, everything else unchanged. It is not a shipped path and
exists so the choice of double rests on a measurement.

- double lanes: 46.74 fps
- float lanes: 63.20 fps
- ratio: 1.35x

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

