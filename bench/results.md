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
| BT2390 | ictcp | 192.4 | 48.3 | 12.87 | 48.30 | 3.98x |
| BT2390 | ycbcr | 183.4 | 48.5 | 13.59 | 48.46 | 3.78x |
| BT2390 | yrgb | 85.9 | 18.2 | 31.84 | 78.52 | 4.71x |
| BT2390 | rgb | 230.7 | 52.5 | 12.35 | 46.46 | 4.40x |
| BT2390 | maxrgb | 85.6 | 18.1 | 31.93 | 78.08 | 4.72x |
| BT2407 | clip | 1.0 | 0.7 | 88.45 | 88.69 | 1.48x |
| BT2407 | softclip | 8.2 | 3.9 | 79.70 | 83.30 | 2.10x |

## End to end

The whole script of section 4.3 over a synthetic 4K PQ source: the
resize into linear RGBS, both filters, and the resize back out to
10-bit YUV. This is what the design's target refers to.

- 34.24 frames per second, ictcp and softclip, on 32 threads
- Peak working set 7.3 GB, measured in a process
  that ran nothing but this chain

A 4K RGBS frame is 100 MB and the model is frame-parallel, so the
memory a chain needs scales with the thread count. Lower
core.num_threads or core.max_cache_size to trade throughput for it.

## What the precision costs

The ictcp kernel built a second time with float lanes and SLEEF's
float pow, everything else unchanged. It is not a shipped path and
exists so the choice of double rests on a measurement.

- double lanes: 48.30 fps
- float lanes: 63.53 fps
- ratio: 1.32x

Against the float64 oracle the float kernel reaches 2.5e-04
absolute and 5.9% relative, against frozen gates of 1.2e-07 for
both. In output codes that is at worst 5 of 255 with 2.3% of
samples moving by more than one, and at worst 20 of 1023 with 4.5%
moving by more than one. The ICtCp matrices subtract numbers of
similar size, so this path loses far more to float32 than a bare
PQ round trip does.

