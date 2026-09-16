# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the tags follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
VapourSynth packs a plugin version into one int as `(major << 16) | minor`, so
the `PluginVersion` the core reports carries no patch component and a 0.1.0 and
a 0.1.1 look alike there. `tonemap.Info()` reports the release in full, and
is what a bug report should quote.

## [Unreleased]

### Added

- `HLG`, decoding Hybrid Log-Gamma to display-referred linear light through
  the OOTF of ITU-R BT.2100-3 Table 5, so an HLG source can feed straight
  into the existing `BT2390` and `BT2407` stages. `lw` and `lb` come from
  `MasteringDisplayMaxLuminance` and `MasteringDisplayMinLuminance` when not
  given as arguments, defaulting to 1000 and 0 cd/m2 since most HLG content
  carries no mastering metadata. Feed it the HLG signal rather than linear
  light: the OOTF needs all three channels at once, so a resizer's transfer
  function cannot apply it. Asking `resize` for `transfer_s="linear"`
  substitutes the per-channel approximation of Note 5e, which measures 76%
  too bright on a fully saturated blue against Table 5.

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
  stored as float32. The runtimes are linked statically, so the DLL needs
  nothing installed beside it.
- `Info()`, reporting the release in full, the target dispatch chose, the
  targets the build carries and the lane width, and `simd=0` on both filters to
  run the scalar reference.

[Unreleased]: https://github.com/sleepy-af-dev/vs-tonemap/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/sleepy-af-dev/vs-tonemap/releases/tag/v0.1.0
