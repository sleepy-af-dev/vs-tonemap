#include "bt2407.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace tonemap {

namespace {

struct NamedMethod {
    const char* name;
    GamutMethod value;
};
constexpr NamedMethod kMethods[] = {
    {"clip", GamutMethod::Clip},
    {"softclip", GamutMethod::Softclip},
};

struct NamedGamut {
    const char* name;
    SourceGamut value;
};
constexpr NamedGamut kGamuts[] = {
    {"auto", SourceGamut::Auto},
    {"bt2020", SourceGamut::Bt2020},
    {"p3d65", SourceGamut::P3D65},
};

// How far outside the CIE diagram a declared primary may land and still be
// accepted. Container metadata arrives as counts of 0.00002 and the standard
// primaries sit exactly on x + y = 1, so the last bit is not worth rejecting.
constexpr double kChromaticityTolerance = 1e-9;

constexpr double signedArea(const Chromaticity& a, const Chromaticity& b,
                            const Chromaticity& c) {
    return 0.5 * ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y));
}

bool insideDiagram(const Chromaticity& c) {
    return std::isfinite(c.x) && std::isfinite(c.y) && c.x >= 0.0 && c.y > 0.0 &&
           c.x + c.y <= 1.0 + kChromaticityTolerance;
}

}  // namespace

const char* gamutMethodNames() { return "clip, softclip"; }
const char* sourceGamutNames() { return "auto, bt2020, p3d65"; }

bool parseGamutMethod(const char* name, GamutMethod* out) {
    if (name == nullptr) return false;
    for (const NamedMethod& m : kMethods) {
        if (std::strcmp(name, m.name) == 0) {
            *out = m.value;
            return true;
        }
    }
    return false;
}

bool parseSourceGamut(const char* name, SourceGamut* out) {
    if (name == nullptr) return false;
    for (const NamedGamut& g : kGamuts) {
        if (std::strcmp(name, g.name) == 0) {
            *out = g.value;
            return true;
        }
    }
    return false;
}

bool validGamut(const Primaries& p, const Chromaticity& white) {
    const Chromaticity corners[3] = {p.r, p.g, p.b};
    for (const Chromaticity& c : corners) {
        if (!insideDiagram(c)) return false;
    }
    if (!insideDiagram(white)) return false;

    const double area = signedArea(p.r, p.g, p.b);
    if (!(area != 0.0) || !std::isfinite(area)) return false;

    // The white point has to be on the same side of all three edges as the
    // triangle's own winding, which is what puts it strictly inside.
    const double orientation = area > 0.0 ? 1.0 : -1.0;
    for (int i = 0; i < 3; ++i) {
        const Chromaticity& a = corners[i];
        const Chromaticity& b = corners[(i + 1) % 3];
        const double edge = (b.x - a.x) * (white.y - a.y) - (b.y - a.y) * (white.x - a.x);
        if (!(edge * orientation > 0.0)) return false;
    }
    return true;
}

namespace {

// Where the ray from the white point leaves a gamut, as a multiple of
// (du, dv). Along c(t) = w + t (du, dv) at fixed luminance y, multiplying
// through by 4 v'(t) makes both the XYZ triple and the denominator affine in
// t, so each of the six constraints 0 <= channel <= 1 is one linear
// inequality. The answer is the smallest upper bound among them, and infinity
// when the colour is achromatic and the ray does not move.
template <typename T>
T boundaryT(const Mat3& xyzToRgb, T y, T du, T dv) {
    const T uw = static_cast<T>(kWhiteUv.x);
    const T vw = static_cast<T>(kWhiteUv.y);

    T a0, a1, a2;
    applyMatrix(xyzToRgb, T(9) * y * uw, T(4) * y * vw,
                y * (T(12) - T(3) * uw - T(20) * vw), &a0, &a1, &a2);
    T b0, b1, b2;
    applyMatrix(xyzToRgb, T(9) * y * du, T(4) * y * dv, y * (T(-3) * du - T(20) * dv),
                &b0, &b1, &b2);

    const T d0 = T(4) * vw;
    const T d1 = T(4) * dv;
    const T a[3] = {a0, a1, a2};
    const T b[3] = {b0, b1, b2};

    T t = std::numeric_limits<T>::infinity();
    for (int i = 0; i < 3; ++i) {
        if (b[i] < T(0)) {  // channel >= 0
            const T bound = -a[i] / b[i];
            if (bound < t) t = bound;
        }
        const T upper = b[i] - d1;
        if (upper > T(0)) {  // channel <= 1
            const T bound = -(a[i] - d0) / upper;
            if (bound < t) t = bound;
        }
    }
    return t;
}

// The Annex 5 roll-off: identity below 1 - beta, flat at 1 above 1 + alpha.
// The quadratic Bezier through (1 - beta, 1 - beta), control point (1, 1) and
// end (1 + alpha, 1). With s the Bezier parameter the x coordinate is
// (1 - beta) + 2 beta s + (alpha - beta) s^2, so s solves a quadratic in r and
// the y coordinate reduces to r - alpha s^2.
//
// Equation (5-4) of the report prints the bracket unsquared, which gives 3.17
// rather than 1 at r = 1 + alpha. The squared form here is what the report's
// own Bezier construction produces.
//
// s is taken as q / (sqrt(beta^2 + k q) + beta) rather than
// (sqrt(beta^2 + k q) - beta) / k, which is the same number but stays finite
// as k = alpha - beta approaches zero.
template <typename T>
T softClipCurve(T r, T alpha, T beta) {
    const T q = r + beta - T(1);
    if (!(q > T(0))) return r;
    if (r > T(1) + alpha) return T(1);

    const T k = alpha - beta;
    T disc = beta * beta + k * q;
    if (disc < T(0)) disc = T(0);
    const T den = std::sqrt(disc) + beta;
    const T s = den > T(0) ? q / den : T(0);
    return r - alpha * s * s;
}

template <typename T>
void clipRow(const float* srcR, const float* srcG, const float* srcB, float* dstR,
             float* dstG, float* dstB, size_t width) {
    for (size_t x = 0; x < width; ++x) {
        T r, g, b;
        applyMatrix(kRgb2020ToRgb709, static_cast<T>(srcR[x]), static_cast<T>(srcG[x]),
                    static_cast<T>(srcB[x]), &r, &g, &b);
        dstR[x] = static_cast<float>(r < T(0) ? T(0) : (r > T(1) ? T(1) : r));
        dstG[x] = static_cast<float>(g < T(0) ? T(0) : (g > T(1) ? T(1) : g));
        dstB[x] = static_cast<float>(b < T(0) ? T(0) : (b > T(1) ? T(1) : b));
    }
}

template <typename T>
void softclipRow(const float* srcR, const float* srcG, const float* srcB, float* dstR,
                 float* dstG, float* dstB, size_t width, const GamutParams& params) {
    const T uw = static_cast<T>(kWhiteUv.x);
    const T vw = static_cast<T>(kWhiteUv.y);
    const T beta = static_cast<T>(params.beta);

    for (size_t x = 0; x < width; ++x) {
        const T in0 = static_cast<T>(srcR[x]);
        const T in1 = static_cast<T>(srcG[x]);
        const T in2 = static_cast<T>(srcB[x]);

        T bigX, y, bigZ;
        applyMatrix(kRgb2020ToXyz, in0, in1, in2, &bigX, &y, &bigZ);
        const T denom = bigX + T(15) * y + T(3) * bigZ;

        // The three input policies, in the order they apply. They overlap, so
        // the order is the answer and not a detail of how the lines are
        // written. Y at or below 0 is black. Y at or above the target peak is
        // white, which is the projection's own limit, since the effective
        // gamut shrinks to the white point as Y approaches 1. Only then does
        // the denominator guard apply: a non-positive one means there is no
        // chromaticity to project along, which needs a large negative input
        // channel, and u'v' passes through infinity as it crosses zero, so no
        // continuous answer exists and the pixel takes the hard clip.
        if (!(y > T(0))) {
            dstR[x] = dstG[x] = dstB[x] = 0.0f;
            continue;
        }
        if (y >= T(1)) {
            dstR[x] = dstG[x] = dstB[x] = 1.0f;
            continue;
        }
        if (!(denom > T(0))) {
            clipRow<T>(srcR + x, srcG + x, srcB + x, dstR + x, dstG + x, dstB + x, 1);
            continue;
        }

        const T du = T(4) * bigX / denom - uw;
        const T dv = T(9) * y / denom - vw;

        const T tSource = boundaryT(params.xyzToSource, y, du, dv);
        const T t709 = boundaryT(kXyzToRgb709, y, du, dv);

        // An achromatic ray leaves both gamuts at infinity, so the ratio is
        // 0/0. Those pixels have r = 0 and take the identity branch below,
        // where alpha is unused.
        T alpha = tSource / t709 - T(1);
        if (!std::isfinite(alpha) || alpha < T(0)) alpha = T(0);
        const T r = std::isfinite(t709) ? T(1) / t709 : T(0);

        const T scale = (r <= T(1) - beta) ? T(1) : t709 * softClipCurve(r, alpha, beta);
        const T u2 = uw + du * scale;
        const T v2 = vw + dv * scale;

        const T quarter = T(1) / (T(4) * v2);
        T out0, out1, out2;
        applyMatrix(kXyzToRgb709, T(9) * y * u2 * quarter, y,
                    y * (T(12) - T(3) * u2 - T(20) * v2) * quarter, &out0, &out1, &out2);

        dstR[x] = static_cast<float>(out0 < T(0) ? T(0) : (out0 > T(1) ? T(1) : out0));
        dstG[x] = static_cast<float>(out1 < T(0) ? T(0) : (out1 > T(1) ? T(1) : out1));
        dstB[x] = static_cast<float>(out2 < T(0) ? T(0) : (out2 > T(1) ? T(1) : out2));
    }
}

}  // namespace

void gamutMapRow(const float* srcR, const float* srcG, const float* srcB, float* dstR,
                 float* dstG, float* dstB, size_t width, const GamutParams& params) {
    switch (params.method) {
        case GamutMethod::Clip:
            clipRow<double>(srcR, srcG, srcB, dstR, dstG, dstB, width);
            return;
        case GamutMethod::Softclip:
            softclipRow<double>(srcR, srcG, srcB, dstR, dstG, dstB, width, params);
            return;
    }
}

}  // namespace tonemap
