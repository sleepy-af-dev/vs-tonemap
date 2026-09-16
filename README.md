# vs-tonemap

A CPU-only VapourSynth plugin that converts PQ or HLG HDR to SDR. Three
filters:

- `HLG` decodes Hybrid Log-Gamma to display-referred linear light with the
  luminance-driven OOTF of ITU-R BT.2100-3 Table 5. A resizer's transfer
  function works one channel at a time and cannot reproduce it.
- `BT2390` applies the ITU-R BT.2390 tone mapping curve as ITU-R BT.2408
  Annex 5 specifies it, in any of the five colour representations that annex
  describes.
- `BT2407` converts BT.2020 to BT.709, either with the luminance-preserving
  gamut projection of ITU-R BT.2407 Annex 5 or with the hard clip of its
  section 2.

The maths follows the ITU text as written; no equation is approximated to
make it faster. Every pixel is computed in float64 and stored as float32,
and the speed comes from explicit SIMD. A float64 reference implementation
of the same equations, together with the test suite that compares the two,
is in `reference/`.

## Requirements

- VapourSynth R55 or later. The plugin is built against API 4.0, so newer
  cores load it as well; it is tested on R79.
- Windows x64. Nothing in the code is platform-specific, but this release is
  built and tested only there.
- An x86-64 CPU. The DLL carries kernels for SSE2 through AVX-512 and picks
  the best one at load time, so one binary serves every machine.

## Installing

Download the archive from the releases,
<https://github.com/sleepy-af-dev/vs-tonemap/releases>, or build it yourself;
see "Building" below. It holds `vs-tonemap.dll` together with the `LICENSE` and
`NOTICE` that travel with it. A `.sha256` ships beside the archive, so the
download can be checked before it is opened:

```
sha256sum -c vs-tonemap-v0.1.0.zip.sha256
```

Put the DLL in a directory VapourSynth autoloads plugins from, or load it from
the script:

```python
core.std.LoadPlugin(path="/path/to/vs-tonemap.dll")
```

It registers the namespace `tonemap` and the identifier
`com.vstonemap.plugin`.

## Usage

All three filters take RGBS, that is 32-bit float RGB with a constant format
and constant dimensions. `BT2390` and `BT2407` take linear light in BT.2020
primaries; `HLG` takes the HLG signal itself, also in BT.2020 primaries, and
produces the linear light the other two expect. `BT2390` produces BT.2020;
`BT2407` expects the output of `BT2390` and produces BT.709. A full chain
for PQ input converts into linear RGBS, tone maps, gamut maps, and converts
back out:

```python
lin = core.resize.Bicubic(src, format=vs.RGBS,
                          transfer_in_s="st2084", transfer_s="linear",
                          primaries_in_s="2020", primaries_s="2020",
                          nominal_luminance=100)
sdr = core.tonemap.BT2390(lin, nominal_luminance=100)
sdr = core.tonemap.BT2407(sdr)
out = core.resize.Bicubic(sdr, format=vs.YUV420P10, matrix_s="709",
                          transfer_s="709", primaries_s="709")
```

The source filter has to attach the standard mastering display properties for
the defaults to work; BestSource, L-SMASH-Works and ffms2 all do.

HLG input needs one extra stage first, decoding the signal to linear light
before the same two filters run:

```python
sig = core.resize.Bicubic(src, format=vs.RGBS,
                          transfer_in_s="std-b67", transfer_s="std-b67",
                          primaries_in_s="2020", primaries_s="2020")
lin = core.tonemap.HLG(sig, nominal_luminance=100)
sdr = core.tonemap.BT2390(lin, nominal_luminance=100)
sdr = core.tonemap.BT2407(sdr)
out = core.resize.Bicubic(sdr, format=vs.YUV420P10, matrix_s="709",
                          transfer_s="709", primaries_s="709")
```

### Units

Every luminance is in cd/m2. `nominal_luminance` says how many cd/m2 the
value 1.0 stands for on the linear-light side of the filter that takes it:
the input for `BT2390`, so the same number can go on both that argument and
the `resize` parameter of the same name, and the output for `HLG`, since
`HLG`'s input is the HLG signal itself, not light. `BT2407` takes no
`nominal_luminance`; the linear light it reads and writes carries whatever
scale came before it, `BT2390`'s `dst_max` in the documented chain. `BT2390`
scales its own output so that `dst_max` becomes 1.0, which is what the SDR
transfer function on the way out expects.

### Frame properties

A property that is present and contradicts the contract is an error. A
property that is absent is not, so untagged clips are accepted. A tag of the
wrong type is an error naming the property rather than something to skip over.
The mastering metadata is read as either a float or an integer, since a
hand-written `SetFrameProps` call turns a whole number into an integer
property.

`_Transfer` must be 8 (linear) for `BT2390` and `BT2407`, and 18 (ARIB
STD-B67) for `HLG`. `_Primaries` must be 9 (BT.2020) for all three. For
range, cores from R74 on carry `_Range`, which must be 1 (full); older cores
carry `_ColorRange`, which must be 0, the convention of the day. Both
spellings are read, so range validation works on every supported core.

## tonemap.HLG

```
HLG(clip clip, [float lw, float lb, float nominal_luminance=100.0,
    int simd=1])
```

| parameter | default | meaning |
|---|---|---|
| `lw` | `MasteringDisplayMaxLuminance`, else 1000 | nominal display peak, LW |
| `lb` | `MasteringDisplayMinLuminance`, else 0 | display black, LB |
| `nominal_luminance` | 100 | cd/m2 that 1.0 means on the output |
| `simd` | 1 | 0 runs the scalar reference instead |

`lw` and `lb` are read from the frame properties when they are not given, the
same as `src_max` and `src_min` on `BT2390`. Unlike those, they default
rather than error when neither the argument nor a usable property is there:
HLG is display-independent by design and most HLG content carries no
mastering metadata at all. 1000 cd/m2, the reference display both BT.2100
and BT.2408 are written around, is the default `lw`; 0 is the default `lb`.

It is an error if `lw` is not a positive, finite luminance, or if `lb` is
negative, non-finite, or at or above `lw`. `lw` is also capped at 10000
cd/m2, since the filter reuses the same luminance check `BT2390` does and
that bound is the PQ peak; Table 5 does not itself imply a ceiling, so a
mastering peak above 10000 fails with "lw must be a luminance from 0 to
10000 cd/m2" rather than with anything about HLG. The cap is defensible all
the same, since a display above the PQ peak would produce a frame `BT2390`
could not accept. Every check whose inputs are all arguments runs when the
filter is created, so a bad parameter fails at script evaluation; a check
that needs a value from the properties runs on the first frame, and its
message names the property the value came from. The split matches what
`BT2390` uses.

This filter implements the OOTF of BT.2100-3 Table 5: one scalar derived
from scene luminance, applied to all three channels together, which is what
holds chromaticity steady as luminance changes. It is checked against a
float64 implementation of the same equations to the bounds in Accuracy
below.

A resizer cannot do that, because a transfer function works on one channel
at a time. zimg's `std-b67` raises each channel separately, which Note 5e of
BT.2100-3 calls the approximation some legacy displays use. Measured against
Table 5, the two agree on neutral greys and separate as colour saturates:
zimg's result is 76% too bright on a fully saturated blue and 31% too bright
on a fully saturated red.

So do not ask `resize` to convert HLG to linear light. Ask for
`transfer_s="std-b67"`, which leaves the signal unconverted, and let this
filter decode it.

### Output

Linear BT.2020 RGBS, scaled so 1.0 means `nominal_luminance` cd/m2. Channels
can exceed 1.0 whenever `lw` is above `nominal_luminance`, which is the
normal case: a 1000 cd/m2 peak at the default nominal of 100 reaches 10.0.
`_Transfer` becomes 8, `_Primaries` stays 9, and `_Range` is written as 1 on
the same terms as the other two filters.

`MasteringDisplayMaxLuminance` and `MasteringDisplayMinLuminance` are
written as `lw` and `lb`, whichever way they were sourced, so the frame
describes the display it now renders for. That is also what lets
`BT2390(HLG(clip))` run with no arguments of its own. `ContentLightLevelMax`
and the other content-describing properties are left alone; `BT2390` is
what removes them.

## tonemap.BT2390

```
BT2390(clip clip, [float src_min, float src_max, float dst_min=0.0,
       float dst_max=203.0, float nominal_luminance=100.0,
       data representation="ictcp", int simd=1])
```

| parameter | default | meaning |
|---|---|---|
| `src_min` | `MasteringDisplayMinLuminance` | mastering display black, LB |
| `src_max` | `MasteringDisplayMaxLuminance` | mastering display peak, LW |
| `dst_min` | 0 | target black, Lmin |
| `dst_max` | 203 | target peak, Lmax. BT.2408 HDR reference white |
| `nominal_luminance` | 100 | cd/m2 that 1.0 in the input stands for |
| `representation` | `ictcp` | `ictcp`, `ycbcr`, `yrgb`, `rgb`, `maxrgb` |
| `simd` | 1 | 0 runs the scalar reference instead |

`src_min` and `src_max` are read from the frame properties when they are not
given, so a clip with mastering metadata needs neither. A max property that is
absent, not finite, or not positive counts as absent, while a min property of
0 is a valid black. If neither the argument nor a usable property is there,
the filter raises an error rather than guessing; BT.2408 names 0 and 10000 as
the fallbacks, and you can pass those explicitly if that is what you want.

Every check whose inputs are all arguments runs when the filter is created, so
a bad parameter fails at script evaluation. The checks that need a value from
the properties run on the first frame, and their messages name the property
the value came from. It is an error if LW or Lmax is not positive, if LB is at
or above LW, if Lmin is at or above Lmax, if any luminance falls outside the
PQ range of 0 to 10000, or if the parameters put the curve outside the range
where it is monotone (a black lift above 0.25 in PQ terms, or a knee point
below 0, which needs a target peak of about 5 cd/m2 against a 1000 cd/m2
master).

### Representations

Annex 5 lists five ways to drive the curve, and the plugin implements all
five. None of them takes a strength parameter; the annex's formulas decide
the result.

| value | Annex 5 option | what goes through the curve |
|---|---|---|
| `ictcp` | 1 | I of ICtCp; CT and CP scaled by the ratio |
| `ycbcr` | 2 | Y' of Y'CbCr; Cb and Cr scaled by the ratio |
| `yrgb` | 3 | luminance Y; RGB scaled by the linear ratio |
| `rgb` | 4 | each of R', G', B' on its own |
| `maxrgb` | 5 | max(R, G, B); RGB scaled by the linear ratio |

`ictcp` is the default: it compresses the intensity axis of the space BT.2100
defines for exactly that separation, and scales the two chroma axes to follow.
`yrgb` and `maxrgb` run about 2.6 times faster, 18 ns per pixel against 48,
because one value goes through PQ rather than three. `rgb` desaturates bright
colours by construction, since each channel is compressed on its own and the
largest one is compressed most.

### Output

Linear BT.2020 RGBS, scaled so `dst_max` is 1.0. `_Transfer` stays 8 and
`_Primaries` stays 9. `_Range` is written as 1 on every core: cores from R74
read it, and older ones ignore a key they do not know. `_ColorRange` is never
written and never deleted, since on a new core it is an alias of `_Range`
inside the map and deleting it would delete the tag just written.

The properties that described the HDR content are removed, because it no
longer exists: `MasteringDisplayMinLuminance`, `MasteringDisplayMaxLuminance`,
`ContentLightLevelMax`, `ContentLightLevelAverage`, `DolbyVisionRPU` and
`HDR10Plus`. The mastering primaries and white point stay, because `BT2407`
reads them.

That last part matters for a chain that stops here rather than going on to
`BT2407`: the four mastering primaries properties survive onto SDR frames,
where a downstream tool could read them as a claim about the output. Remove
them if that would mislead:

```python
sdr = core.std.RemoveFrameProps(sdr, props=[
    "MasteringDisplayPrimariesX", "MasteringDisplayPrimariesY",
    "MasteringDisplayWhitePointX", "MasteringDisplayWhitePointY",
])
```

Channels can exceed 1.0. In `ictcp`, `ycbcr` and `yrgb` the chroma scaling can
put a channel several times above SDR white on saturated input, which is what
Annex 5 means by a result outside the target colour volume, and `BT2407`
accepts such input. All five representations also overshoot when `dst_min` is
above `src_min`; see the next section.

## tonemap.BT2407

```
BT2407(clip clip, [data method="softclip", float beta=0.2,
       data src_gamut="auto", int simd=1])
```

| parameter | default | meaning |
|---|---|---|
| `method` | `softclip` | `softclip` is the Annex 5 projection, `clip` is the matrix and hard clamp of section 2 |
| `beta` | 0.2 | where the roll-off starts, in [0, 1). Annex 5's margin. Only used by `softclip` |
| `src_gamut` | `auto` | `auto`, `bt2020` or `p3d65`. Only used by `softclip` |
| `simd` | 1 | 0 runs the scalar reference instead |

`softclip` takes each colour along the ray from the D65 white point through
its own chromaticity, finds where that ray leaves the source gamut and where
it leaves BT.709 at the same luminance, and rolls the distance off with the
quadratic Bezier of Annex 5. Luminance is preserved, and so is the direction
from white in u'v'. A colour less than `1 - beta` of the way from white to the
BT.709 boundary comes through untouched, which is about a third of random
in-gamut colours.

`clip` applies the BT.2020 to BT.709 matrix and then clamps each channel to
[0, 1] on its own, which is the conversion BT.2407 section 2 describes.
Clamping channels independently changes the ratios between them, so it holds
neither hue nor luminance where it bites. `beta` and `src_gamut` do not apply
to it.

Equation (5-4) of the report prints the bracket in the roll-off unsquared. At
r = 1 + alpha, where the function has to be 1, the printed form gives 3.17
with alpha 0.5 and beta 0.2, and since it divides by (beta - alpha) squared it
also divides by zero whenever the two are equal. The squared form is used
instead, which is what the report's own construction, a quadratic Bezier
extension, gives, in the arrangement that divides by
sqrt(beta^2 + (alpha - beta)(r + beta - 1)) + beta rather than by
alpha - beta. The two are algebraically the same and only the second is
defined when alpha equals beta.

### Source gamut

BT.2407 section 3 notes that the content gamut is often much smaller than
BT.2020, and that using it reduces how much compression is needed. So
`src_gamut="auto"` reads the mastering display gamut from
`MasteringDisplayPrimariesX`, `MasteringDisplayPrimariesY`,
`MasteringDisplayWhitePointX` and `MasteringDisplayWhitePointY`, and falls
back to BT.2020 when they are absent or fail validation. Most HDR is mastered
on a P3 display, so this is not a rare path: it lets P3 boundary colours reach
the BT.709 boundary instead of stopping short of it.

The fallback is silent. The output property `TonemapSourceGamut` records what
was actually used for the frame, `bt2020`, `p3d65` or `mastering`, so a script
can tell.

The projection white point is D65 in every case, and the source matrix is
derived with D65 as its white, because both the container and the target are
D65. The declared white point is validated and otherwise unused. Deriving with
a declared non-D65 white would put D65 itself outside the source cube and
collapse the roll-off into a hard clip.

### Output

Linear BT.709 RGBS with every channel in [0, 1]. `_Primaries` is set to 1,
`_Transfer` stays 8, and `_Range` is written as 1 on the same terms as above.
The mastering primaries and white point properties are removed, and
`TonemapSourceGamut` is added.

## Behaviour at the edges

These are the plugin's choices where the specifications stop short. The
specifications do not say what to expect in these cases, so each choice is
written down here.

- Input above the mastering peak is treated as the peak, and input below the
  mastering black as black. Annex 5 defines the curve on [0, 1] only, and the
  cubic is not monotone outside it.
- A negative input channel is treated as black. Sign-preserving handling of
  negative linear light is possible and is not implemented.
- NaN and infinite samples are not supported input. The output for such a
  pixel is unspecified and neither the filter nor the reference checks for
  them, because a check per sample would cost every valid pixel. Measured on
  the pixel `[NaN, 0.5, 0.5]` at `lw=1000, nominal_luminance=1`: the scalar
  kernel of `HLG` gives `[0, 0, 0]` and the vector kernel gives
  `[0, 47.699226, 47.699226]`. Neither path produces NaN, and the two
  disagree on more than the NaN channel. The scalar clamp is a pair of
  comparisons, both false for NaN, so the NaN reaches the luminance sum,
  which comes out NaN too; the black guard, `!(ys > 0)`, is the NaN-safe
  spelling of that test, so it fires and the whole pixel goes black. The
  vector clamp is `Min`/`Max`, which on x86 return the second operand when
  either input is NaN, so only the NaN lane flushes to 0 before the
  luminance sum runs; the other two channels then render from a luminance
  computed as if that channel were 0, which is why they come out at 47.70
  rather than the 50.70 an unpolluted `[0.5, 0.5, 0.5]` pixel gives. Both
  stay within "unspecified" as stated above; it is written down because "the
  two paths are bit-identical" is otherwise true everywhere else.
- With `dst_min` above `src_min` the black lift of step 4 raises the whole
  curve, including its top. The output then exceeds `dst_max` by a factor of
  b(1 - maxLum)^4, for instance 0.39% for a 1000 cd/m2 master, a 1 cd/m2
  target black and a 203 cd/m2 target peak. Annex 5 has no output clamp and
  the plugin adds none.
- In `yrgb` and `maxrgb`, a pixel whose driving value is zero or negative has
  no ratio to apply and takes the curve's own black, which is the same value
  an exactly black input gets in the other three representations.
- In `BT2407`, a pixel with luminance at or below 0 is black, a pixel at or
  above 1 is white, and a pixel whose chromaticity is undefined, which needs a
  large negative input channel, takes the hard clip. Those three apply in that
  order. Luminance at or above 1 has to go somewhere: the effective gamut
  there is the white point alone, and the projection already converges on
  white as luminance approaches 1, so white is the continuous choice. Pick the
  `maxrgb` representation if you want chromaticity preserved instead; it never
  produces luminance above 1.
- A chromaticity beyond the effective source gamut lands on the BT.709
  boundary. The report does not contemplate that input.
- In `HLG`, an input signal outside [0, 1] is clamped to it. HLG is defined
  on that domain only; values outside it arrive from chroma upsampling
  ringing and from limited-range codes below 64 or above 940.
- In `HLG`, a pixel whose driving luminance YS is zero or negative comes out
  black on all three channels instead of going through the OOTF. Below an
  `lw` of about 301 cd/m2 the system gamma falls under 1, so the OOTF's
  exponent, gamma minus 1, is negative, and YS to that power at exact black
  is infinity; multiplied by a channel of zero, that is NaN. The clamp above
  keeps YS non-negative, so this guard only ever fires at exact black.
- The system gamma is not monotone in `lw`. Note 5f switches formula at 400
  cd/m2, and the two disagree there by 0.011, so an `lw` of 399 gets a
  system gamma of 1.044 while 400 gets 1.033. The step at 2000 cd/m2 goes
  the other way, up by 0.007. Both are the recommendation's own artefacts,
  and neither is smoothed over.

## Accuracy

The filters are tested against a float64 reference implementation of the same
equations over a fixture set that includes saturated and out-of-volume
colours. Measured over every fixture, the compiled filter is within one
float32 ULP of the reference: at most 1.2e-7 absolute where the reference
value is at or below SDR white, and at most 1.2e-7 relative where it is above
1e-3. Those are the bounds the test suite gates on. The absolute bound stops
at SDR white because half a float32 ULP passes it at about 17 times that
value, so above there storage alone would decide the result.

The SIMD kernels are compared against the scalar ones on every instruction set
the test machine can run, seven of the eight in the DLL; the Sapphire Rapids
kernel is compiled but untested. Four of the five representations are
bit-identical to the scalar path. `ictcp` differs by about 7e-12 in double,
which is the fused multiply-add contraction its matrix chain allows, and that
difference flips at most one ULP of the float32 the frame stores.

One place those bounds do not apply is the discontinuities of the policies
above, where the output jumps rather than varying smoothly. An input within
rounding distance of one of those boundaries can land on either side of it,
because the scalar path, the vector path and the reference sum in different
orders. No fixture and no real content sits there.

## Speed and memory

Measured on a 16-core desktop at 4K, 32 threads. The full chain above from a
PQ source, including both resize stages, runs at about 35 frames per second.
The same chain from an HLG source, with the extra decode stage, runs at
about 30 frames per second. The tone mapping filter alone costs 48 ns per
pixel in `ictcp` and 18 ns per pixel in `yrgb` or `maxrgb`; the gamut filter
costs 4 ns per pixel. The HLG decode costs 7 ns per pixel. `bench/results.md`
carries the current numbers and the machine they came from.

In an encode the filter is usually not what sets the pace. Piped into x265 at
`medium` and CRF 18, the whole chain ran at 14.8 frames per second, so the
tone mapping was a share of CPU time rather than the limit. Against a faster
encoder, NVENC or a fast software preset, it becomes the limit.

VapourSynth runs frames in parallel and a 4K RGBS frame is 100 MB, so the
memory a chain needs scales with the thread count. The PQ chain above peaked
at 7.9 GB with 32 threads, the HLG chain at 7.7 GB. Lower `core.num_threads`
or `core.max_cache_size` to trade throughput for memory.

## Diagnostics

`simd=0` on any of the three filters runs the scalar reference path instead
of the vector kernel. It exists so the test suite can compare the two, and
as a way out if a machine ever disagrees with its own vector unit. It is not
a tuning knob: the scalar path computes values within one float32 ULP of the
vector path and is up to five times slower.

`tonemap.Info()` reports what the plugin chose, which is worth including in
a bug report:

```python
>>> core.tonemap.Info()
{'available_targets': ['AVX3_ZEN4', 'AVX3_DL', 'AVX3', 'AVX2', 'SSE4',
 'SSSE3', 'SSE2'], 'double_lanes': 8, 'ictcp_float32_lanes': 0,
 'target': 'AVX3_ZEN4', 'version': '0.1.0'}
```

`version` is the release in full, all three components. VapourSynth's own
plugin version packs a major and a minor into one int and has nowhere to put a
patch, so two releases differing only in it report the same `PluginVersion`;
this key is how a 0.1.1 tells itself apart.
`available_targets` lists the kernels compiled into the DLL that this CPU can
run, best first. `Info(target="AVX2")` restricts dispatch to one of them for
the rest of the process and an empty string restores the automatic choice;
that is a test hook, not something a script should need.
`ictcp_float32_lanes` is 0 in every release build.

Both `simd` and `Info(target=...)` are diagnostics rather than part of the
stable interface, and may change between releases.

## Known limitations

- The BT.2407 projection shifts the hue of extremely saturated bright yellows.
  The report states this weakness itself; it is not worked around.
- The curve is static. Dynamic metadata, HDR10+ and Dolby Vision are not read,
  and no scene or frame peak detection is done.
- HLG input is decoded by the display-light conversion of BT.2408-9 Table 9.
  HLG output, and the table's scene-light conversion for matching an HLG
  camera against a live SDR feed, are not implemented.
- BT.709 output only. Of BT.2407 the plugin implements the Annex 5 projection
  and the section 2 hard clip; Annexes 2 and 4, and perceptual gamut mappers
  of other kinds, are not implemented.

## Building

A C++20 compiler, CMake 3.24 or later, Ninja, and a network connection for the
first configure, which fetches Highway and SLEEF. The release is built with
clang 22 targeting the MSVC ABI; MSVC and GCC should work too, and neither is
tested.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The result is `build/vs-tonemap.dll`. The tests need the `uv` Python package
manager and bring their own VapourSynth from PyPI, so no system install is
involved:

```sh
uv run --project reference pytest
```

They skip themselves when the DLL is not built.

## Sources

The specifications, which are the authority for everything here:

- ITU-R BT.2408: <https://www.itu.int/rec/R-REC-BT.2408/en> (Annex 5, the
  EETF and the five representations; section 2 and Table 4, the HDR
  reference white levels)
- ITU-R BT.2390: <https://www.itu.int/rec/R-REC-BT.2390/en> (the same curve,
  with the reasoning behind it)
- ITU-R BT.2100: <https://www.itu.int/rec/R-REC-BT.2100/en> (PQ, the
  primaries, the ICtCp and Y'CbCr matrices, and Table 5, the HLG reference
  EOTF)
- ITU-R BT.2407: <https://www.itu.int/rec/R-REC-BT.2407/en> (gamut
  conversion)
- ITU-R BT.2087: <https://www.itu.int/rec/R-REC-BT.2087/en> (deriving the
  matrices from primaries)

Two independent implementations were used as numeric second opinions during
development, by comparing outputs against them:

- hdr-toys: <https://github.com/natural-harmonia-gropius/hdr-toys>
- libplacebo: <https://code.videolan.org/videolan/libplacebo>

## Licence

MIT; see `LICENSE`. `NOTICE` lists the third-party components and their
licences: Google Highway and SLEEF are statically linked into the DLL, and the
VapourSynth API headers are vendored under `include/vapoursynth/`.

## Development

Development of this project is supported by Claude Code.
