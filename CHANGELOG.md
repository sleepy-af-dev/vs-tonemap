# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the tags follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
VapourSynth packs a plugin version into one int as `(major << 16) | minor`, so
the `PluginVersion` the core reports carries no patch component and a 0.1.0 and
a 0.1.1 look alike there. `tonemapper.Info()` reports the release in full, and
is what a bug report should quote.

## [Unreleased]

## [0.1.0] - 2026-09-13

### Added

- `BT2390`, applying the ITU-R BT.2390 tone curve as BT.2408 Annex 5 specifies
  it, in the `ictcp`, `ycbcr`, `yrgb`, `rgb` and `maxrgb` representations.
- `BT2407`, converting BT.2020 to BT.709 by the BT.2407 Annex 5 projection or by
  a hard clip.
- Source luminance taken from the `MasteringDisplayMinLuminance` and
  `MasteringDisplayMaxLuminance` frame properties when the arguments are absent.
  A clip with mastering metadata needs neither argument.
- A kernel for every x86-64 instruction set in one DLL, chosen at run time, on
  Google Highway with SLEEF's `pow`. Every pixel is computed in float64 and
  stored as float32.
- `Info()`, reporting the release in full, the target dispatch chose, the
  targets the build carries and the lane width, and `simd=0` on both filters to
  run the scalar reference.

[Unreleased]: https://github.com/sleepy-af-dev/vs-tonemap/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/sleepy-af-dev/vs-tonemap/releases/tag/v0.1.0
