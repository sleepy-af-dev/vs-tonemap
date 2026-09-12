// The BT.2390 tone mapper: the EETF applied in one colour representation.

#ifndef TONEMAPPER_BT2390_H
#define TONEMAPPER_BT2390_H

#include <cstddef>

#include "colour.h"

namespace tonemapper {

enum class Representation {
    Ictcp,
};

// Names the plugin accepts, in the order the error message lists them. Only
// representations that are implemented appear here, so the filter never
// advertises a mode it cannot run.
const char* representationNames();
bool parseRepresentation(const char* name, Representation* out);

struct FrameParams {
    Eetf curve;
    double nominal;  // cd/m2 that 1.0 means on the input
    double dstMax;   // cd/m2 that 1.0 means on the output
};

// One row of an RGBS frame. The three input pointers and the three output
// pointers address the same row of the R, G and B planes; in-place is fine.
void toneMapRow(const float* srcR, const float* srcG, const float* srcB,
                float* dstR, float* dstG, float* dstB, size_t width,
                Representation rep, const FrameParams& params);

}  // namespace tonemapper

#endif  // TONEMAPPER_BT2390_H
