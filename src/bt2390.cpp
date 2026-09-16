#include "bt2390.h"

#include <cstring>

namespace tonemap {

namespace {

// The order Annex 5 lists them in. Only what is implemented appears, so the
// error message and the accepted set can never disagree.
struct Named {
    const char* name;
    Representation rep;
};
constexpr Named kNamed[] = {
    {"ictcp", Representation::Ictcp},
    {"ycbcr", Representation::Ycbcr},
    {"yrgb", Representation::Yrgb},
    {"rgb", Representation::Rgb},
    {"maxrgb", Representation::Maxrgb},
};

}  // namespace

const char* representationNames() {
    static const std::string joined = [] {
        std::string all;
        for (const Named& n : kNamed) {
            if (!all.empty()) all += ", ";
            all += n.name;
        }
        return all;
    }();
    return joined.c_str();
}

bool parseRepresentation(const char* name, Representation* out) {
    if (name == nullptr) return false;
    for (const Named& n : kNamed) {
        if (std::strcmp(name, n.name) == 0) {
            *out = n.rep;
            return true;
        }
    }
    return false;
}

std::string checkLuminance(const char* name, double value) {
    if (!std::isfinite(value) || value < 0.0 || value > kPqPeak) {
        return std::string(name) + " must be a luminance from 0 to 10000 cd/m2, got " +
               describe(value);
    }
    return std::string();
}

std::string checkTargetRange(double dstMin, double dstMax) {
    std::string bad = checkLuminance("dst_min", dstMin);
    if (bad.empty()) bad = checkLuminance("dst_max", dstMax);
    if (!bad.empty()) return bad;
    if (dstMax <= 0.0) return "dst_max must be a positive luminance in cd/m2";
    if (dstMin >= dstMax) return "dst_min must be below dst_max";
    return std::string();
}

std::string makeEetf(double srcMin, double srcMax, double dstMin, double dstMax,
                     const char* minName, const char* maxName, Eetf* out) {
    // The domain check comes first: PQ clamps anything outside [0, 10000] to
    // its ends, so two values above the peak would both encode to 1 and leave
    // the span at zero.
    std::string bad = checkLuminance(minName, srcMin);
    if (bad.empty()) bad = checkLuminance(maxName, srcMax);
    if (bad.empty()) bad = checkTargetRange(dstMin, dstMax);
    if (!bad.empty()) return bad;

    if (srcMax <= 0.0) {
        return std::string(maxName) + " must be a positive luminance in cd/m2";
    }
    if (srcMin >= srcMax) {
        return std::string(minName) + " (" + describe(srcMin) + ") must be below " +
               maxName + " (" + describe(srcMax) + ")";
    }

    Eetf e{};
    e.pqLb = pqInverseEotf<double>(srcMin);
    e.pqLw = pqInverseEotf<double>(srcMax);
    e.span = e.pqLw - e.pqLb;

    // Two distinct doubles can encode to the same PQ code: the double just
    // below 1000 and 1000 do. A zero span makes minLum, maxLum and KS NaN,
    // and NaN passes every ordered comparison below, so without this the
    // filter would hand back a frame of NaN and report nothing wrong.
    const double targetSpan = pqInverseEotf<double>(dstMax) - pqInverseEotf<double>(dstMin);
    if (!(e.span > 0.0)) {
        return std::string(minName) + " (" + describe(srcMin) + ") and " + maxName +
               " (" + describe(srcMax) + ") encode to the same PQ code, "
               "so the source range has no width";
    }
    if (!(targetSpan > 0.0)) {
        return "dst_min (" + describe(dstMin) + ") and dst_max (" + describe(dstMax) +
               ") encode to the same PQ code, so the target range has no width";
    }

    e.minLum = (pqInverseEotf<double>(dstMin) - e.pqLb) / e.span;
    e.maxLum = (pqInverseEotf<double>(dstMax) - e.pqLb) / e.span;
    e.ks = 1.5 * e.maxLum - 0.5;
    e.degenerate = e.maxLum >= 1.0;

    // The black lift E3 = E2 + b (1 - E2)^4 has slope 1 - 4b at E2 = 0, so it
    // is monotone only for b <= 0.25. Not in the spec, derived.
    if (e.minLum > 0.25) {
        return "dst_min is too high for the black lift to stay monotone: minLum is " +
               describe(e.minLum) + ", the bound is 0.25";
    }
    // Below KS = 0 the whole domain is spline and P(0) is negative, so E2
    // leaves [0, maxLum] and the chroma ratio changes sign.
    if (e.ks < 0.0) {
        return "dst_max is too low for the tone curve to stay in range: maxLum is " +
               describe(e.maxLum) + " and KS is " + describe(e.ks) +
               ", which must not be below 0";
    }

    *out = e;
    return std::string();
}

namespace {

// PQ is defined on [0, 10000] only, so negative linear light from resampling
// ringing or sub-black codes reads as black and anything above the peak as
// the peak.
template <typename T>
inline T clampToPqDomain(T v) {
    return v < T(0) ? T(0) : (v > static_cast<T>(kPqPeak) ? static_cast<T>(kPqPeak) : v);
}

// Annex 5 option 1. I through the EETF, CT and CP scaled by the ratio, which
// is the same as scaling the whole PQ-encoded L'M'S' vector wherever the
// curve compresses.
template <typename T>
void toneMapIctcp(const float* srcR, const float* srcG, const float* srcB,
                  float* dstR, float* dstG, float* dstB, size_t width,
                  const FrameParams& params) {
    const T nominal = static_cast<T>(params.nominal);
    const T invDstMax = static_cast<T>(1.0 / params.dstMax);

    for (size_t x = 0; x < width; ++x) {
        T r = clampToPqDomain(static_cast<T>(srcR[x]) * nominal);
        T g = clampToPqDomain(static_cast<T>(srcG[x]) * nominal);
        T b = clampToPqDomain(static_cast<T>(srcB[x]) * nominal);

        T l, m, s;
        applyMatrix(kRgb2020ToLms, r, g, b, &l, &m, &s);
        l = pqInverseEotf(l);
        m = pqInverseEotf(m);
        s = pqInverseEotf(s);

        T i1, ct, cp;
        applyMatrix(kLmspToIctcp, l, m, s, &i1, &ct, &cp);

        const T i2 = params.curve.template apply<T>(i1);
        const T k = chromaRatio(i1, i2);

        applyMatrix(kIctcpToLmsp, i2, k * ct, k * cp, &l, &m, &s);
        l = pqEotf(l);
        m = pqEotf(m);
        s = pqEotf(s);

        applyMatrix(kLmsToRgb2020, l, m, s, &r, &g, &b);
        dstR[x] = static_cast<float>(r * invDstMax);
        dstG[x] = static_cast<float>(g * invDstMax);
        dstB[x] = static_cast<float>(b * invDstMax);
    }
}

// Annex 5 options 3 and 5. One driving value goes through the curve in PQ and
// the linear ratio it produces scales all three channels, which leaves the
// chromaticity alone. The two differ only in what drives them, so `driving`
// is the whole difference between yrgb and maxrgb.
template <typename T, typename Driving>
void toneMapRatio(const float* srcR, const float* srcG, const float* srcB,
                  float* dstR, float* dstG, float* dstB, size_t width,
                  const FrameParams& params, Driving driving) {
    const T nominal = static_cast<T>(params.nominal);
    const T invDstMax = static_cast<T>(1.0 / params.dstMax);
    // What an exact-black input becomes, which is dst_min. A driving value of
    // zero leaves no ratio to apply, and the other three representations put
    // exact black at Lmin, so these have to as well: otherwise a black lift
    // leaves exact black as the one unlifted pixel in the frame.
    const T black = static_cast<T>(params.curve.black()) * invDstMax;

    for (size_t x = 0; x < width; ++x) {
        const T r = clampToPqDomain(static_cast<T>(srcR[x]) * nominal);
        const T g = clampToPqDomain(static_cast<T>(srcG[x]) * nominal);
        const T b = clampToPqDomain(static_cast<T>(srcB[x]) * nominal);

        const T v1 = driving(r, g, b);
        if (!(v1 > T(0))) {
            dstR[x] = static_cast<float>(black);
            dstG[x] = static_cast<float>(black);
            dstB[x] = static_cast<float>(black);
            continue;
        }

        const T v2 = pqEotf(params.curve.template apply<T>(pqInverseEotf(v1)));
        const T k = (v2 / v1) * invDstMax;
        dstR[x] = static_cast<float>(r * k);
        dstG[x] = static_cast<float>(g * k);
        dstB[x] = static_cast<float>(b * k);
    }
}

// Annex 5 option 2. Y' through the EETF, Cb and Cr scaled by the ratio, which
// is the same as scaling the whole PQ-encoded R'G'B' triple wherever the curve
// compresses. BT.2100 Table 6 non-constant luminance, on the printed
// coefficients rather than the colorimetric ones.
template <typename T>
void toneMapYcbcr(const float* srcR, const float* srcG, const float* srcB, float* dstR,
                  float* dstG, float* dstB, size_t width, const FrameParams& params) {
    const T nominal = static_cast<T>(params.nominal);
    const T invDstMax = static_cast<T>(1.0 / params.dstMax);

    for (size_t x = 0; x < width; ++x) {
        const T rp = pqInverseEotf(static_cast<T>(srcR[x]) * nominal);
        const T gp = pqInverseEotf(static_cast<T>(srcG[x]) * nominal);
        const T bp = pqInverseEotf(static_cast<T>(srcB[x]) * nominal);

        const T y1 = static_cast<T>(kKr) * rp + static_cast<T>(kKg) * gp +
                     static_cast<T>(kKb) * bp;
        const T cb = (bp - y1) * static_cast<T>(1.0 / kCbDivisor);
        const T cr = (rp - y1) * static_cast<T>(1.0 / kCrDivisor);

        const T y2 = params.curve.template apply<T>(y1);
        const T k = chromaRatio(y1, y2);

        const T outR = y2 + static_cast<T>(kCrDivisor) * k * cr;
        const T outB = y2 + static_cast<T>(kCbDivisor) * k * cb;
        const T outG = (y2 - static_cast<T>(kKr) * outR - static_cast<T>(kKb) * outB) *
                       static_cast<T>(1.0 / kKg);

        dstR[x] = static_cast<float>(pqEotf(outR) * invDstMax);
        dstG[x] = static_cast<float>(pqEotf(outG) * invDstMax);
        dstB[x] = static_cast<float>(pqEotf(outB) * invDstMax);
    }
}

// Annex 5 option 4. Each of R', G' and B' through the EETF on its own, which
// is the only representation with no ratio and no cross-channel term.
template <typename T>
void toneMapRgb(const float* srcR, const float* srcG, const float* srcB, float* dstR,
                float* dstG, float* dstB, size_t width, const FrameParams& params) {
    const T nominal = static_cast<T>(params.nominal);
    const T invDstMax = static_cast<T>(1.0 / params.dstMax);
    const float* src[3] = {srcR, srcG, srcB};
    float* dst[3] = {dstR, dstG, dstB};

    for (int plane = 0; plane < 3; ++plane) {
        for (size_t x = 0; x < width; ++x) {
            const T v = static_cast<T>(src[plane][x]) * nominal;
            const T mapped = pqEotf(params.curve.template apply<T>(pqInverseEotf(v)));
            dst[plane][x] = static_cast<float>(mapped * invDstMax);
        }
    }
}

}  // namespace

void toneMapRow(const float* srcR, const float* srcG, const float* srcB,
                float* dstR, float* dstG, float* dstB, size_t width,
                Representation rep, const FrameParams& params) {
    switch (rep) {
        case Representation::Ictcp:
            toneMapIctcp<double>(srcR, srcG, srcB, dstR, dstG, dstB, width, params);
            return;
        case Representation::Ycbcr:
            toneMapYcbcr<double>(srcR, srcG, srcB, dstR, dstG, dstB, width, params);
            return;
        case Representation::Yrgb:
            toneMapRatio<double>(
                srcR, srcG, srcB, dstR, dstG, dstB, width, params,
                [](double r, double g, double b) { return kKr * r + kKg * g + kKb * b; });
            return;
        case Representation::Rgb:
            toneMapRgb<double>(srcR, srcG, srcB, dstR, dstG, dstB, width, params);
            return;
        case Representation::Maxrgb:
            toneMapRatio<double>(
                srcR, srcG, srcB, dstR, dstG, dstB, width, params,
                [](double r, double g, double b) {
                    return r > g ? (r > b ? r : b) : (g > b ? g : b);
                });
            return;
    }
}

}  // namespace tonemap
