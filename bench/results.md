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
| BT2390 | ictcp | 158.6 | 48.2 | 13.93 | 47.56 | 3.29x |
| BT2390 | ycbcr | 154.9 | 48.1 | 14.88 | 47.89 | 3.22x |
| BT2390 | yrgb | 74.0 | 18.1 | 34.02 | 74.40 | 4.09x |
| BT2390 | rgb | 192.9 | 52.0 | 13.18 | 46.51 | 3.71x |
| BT2390 | maxrgb | 72.3 | 17.9 | 33.92 | 74.23 | 4.03x |
| BT2407 | clip | 1.0 | 0.8 | 83.96 | 83.86 | 1.34x |
| BT2407 | softclip | 8.5 | 4.1 | 76.42 | 82.88 | 2.06x |
| HLG | - | 13.9 | 12.9 | 75.04 | 74.97 | 1.07x |

## End to end

The whole script of section 4.3 over a synthetic 4K PQ source: the
resize into linear RGBS, both filters, and the resize back out to
10-bit YUV. This is what the design's target refers to.

- 34.54 frames per second, ictcp and softclip, on 32 threads
- Peak working set 7.3 GB, measured in a process
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

