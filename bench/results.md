# Benchmark results

Synthetic 4K RGBS, filter only: the cost of generating the frames is
removed by differencing a drain of the source against a drain of the
source with the filter on it. Nanoseconds per pixel are measured on one
thread and are the kernel; frames per second are measured on every
thread and are what a script sees. Neither figure includes the two
resize stages a real chain carries.

## Machine

- CPU: AMD Ryzen 9 9950X3D, 32 logical cores
- VapourSynth core R79
- Highway target: AVX3_ZEN4, 8 double lanes
- Frames: 4 on one thread, 64 on 32

## Scalar against SIMD

| filter | path | scalar ns/px | SIMD ns/px | scalar fps | SIMD fps | speedup |
|---|---|---|---|---|---|---|
| BT2390 | ictcp | 158.5 | 48.0 | 13.94 | 47.59 | 3.30x |
| BT2390 | ycbcr | 153.1 | 48.3 | 14.88 | 48.23 | 3.17x |
| BT2390 | yrgb | 73.5 | 18.2 | 33.91 | 73.97 | 4.04x |
| BT2390 | rgb | 194.4 | 52.4 | 13.13 | 46.15 | 3.71x |
| BT2390 | maxrgb | 72.9 | 18.1 | 33.96 | 74.80 | 4.04x |
| BT2407 | clip | 1.0 | 0.7 | 84.83 | 85.02 | 1.47x |
| BT2407 | softclip | 8.4 | 4.1 | 76.07 | 81.90 | 2.04x |
| HLG | - | 13.6 | 6.8 | 76.76 | 79.91 | 1.99x |

## End to end

The whole script of section 4.3 over a synthetic 4K PQ source: the
resize into linear RGBS, both filters, and the resize back out to
10-bit YUV. This is what the design's target refers to.

- 34.38 frames per second, ictcp and softclip, on 32 threads
- Peak working set 7.4 GB, measured in a process
  that ran nothing but this chain

A 4K RGBS frame is 100 MB and the model is frame-parallel, so the
memory a chain needs scales with the thread count. Lower
core.num_threads or core.max_cache_size to trade throughput for it.

## What the precision costs

The ictcp kernel built a second time with float lanes and SLEEF's
float pow, everything else unchanged. It is not a shipped path and
exists so the choice of double rests on a measurement.

- double lanes: 47.56 fps
- float lanes: 63.20 fps
- ratio: 1.33x

These figures are from an earlier run and were not re-measured in the sweep
above, so the double-lanes number will drift slightly from the ictcp SIMD fps
in the table.

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

