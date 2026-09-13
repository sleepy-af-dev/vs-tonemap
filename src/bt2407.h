// The BT.2407 gamut conversion from BT.2020 to BT.709.

#ifndef TONEMAP_BT2407_H
#define TONEMAP_BT2407_H

#include <cstddef>

#include "colour.h"

namespace tonemap {

enum class GamutMethod {
    Clip,     // section 2: matrix, then clamp each channel
    Softclip  // Annex 5: luminance-preserving projection with a roll-off
};

enum class SourceGamut {
    Auto,    // mastering display primaries from the properties, else BT.2020
    Bt2020,  // ignore the properties
    P3D65,   // force P3 with D65
};

bool parseGamutMethod(const char* name, GamutMethod* out);
bool parseSourceGamut(const char* name, SourceGamut* out);
const char* gamutMethodNames();
const char* sourceGamutNames();

// True when three primaries and a white point can define a gamut. Every
// chromaticity has to be physically meaningful (x >= 0, y > 0, x + y at most
// 1 within a small margin, so z is not negative), the primaries have to form
// a triangle with a non-zero area, and the white point has to lie strictly
// inside it. Anything else counts as absent metadata.
//
// The bound on x + y is not strict: BT.2020 red (0.708, 0.292) and P3 red
// (0.680, 0.320) both sit exactly on z = 0, and a strict test would reject
// the two most common mastering gamuts there are. The margin covers a source
// filter whose arithmetic lands a bit high.
bool validGamut(const Primaries& primaries, const Chromaticity& white);

struct GamutParams {
    GamutMethod method;
    double beta;
    // XYZ to the source gamut's RGB, always derived with D65 whatever white
    // the metadata declares: the projection white is D65 at both ends of the
    // chain, and deriving with a declared non-D65 white would put D65 outside
    // the source cube and collapse the roll-off at high luminance.
    Mat3 xyzToSource;
    const char* label;  // what goes in TonemapSourceGamut
};

void gamutMapRow(const float* srcR, const float* srcG, const float* srcB, float* dstR,
                 float* dstG, float* dstB, size_t width, const GamutParams& params);

}  // namespace tonemap

#endif  // TONEMAP_BT2407_H
