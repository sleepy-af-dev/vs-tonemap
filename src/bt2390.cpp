#include "bt2390.h"

#include <cstring>

namespace tonemapper {

const char* representationNames() { return "ictcp"; }

bool parseRepresentation(const char* name, Representation* out) {
    if (name == nullptr) return false;
    if (std::strcmp(name, "ictcp") == 0) {
        *out = Representation::Ictcp;
        return true;
    }
    return false;
}

std::string makeEetf(double srcMin, double srcMax, double dstMin, double dstMax,
                     Eetf* out) {
    // The domain check comes first: PQ clamps anything outside [0, 10000] to
    // its ends, so two values above the peak would both encode to 1 and leave
    // the span at zero.
    const struct {
        const char* name;
        double value;
    } all[] = {{"src_min", srcMin},
               {"src_max", srcMax},
               {"dst_min", dstMin},
               {"dst_max", dstMax}};
    for (const auto& p : all) {
        if (!std::isfinite(p.value) || p.value < 0.0 || p.value > kPqPeak) {
            return std::string(p.name) +
                   " must be a luminance from 0 to 10000 cd/m2, got " +
                   std::to_string(p.value);
        }
    }
    if (srcMax <= 0.0) return "src_max must be a positive luminance in cd/m2";
    if (dstMax <= 0.0) return "dst_max must be a positive luminance in cd/m2";
    if (srcMin >= srcMax) return "src_min must be below src_max";
    if (dstMin >= dstMax) return "dst_min must be below dst_max";

    Eetf e{};
    e.pqLb = pqInverseEotf<double>(srcMin);
    e.pqLw = pqInverseEotf<double>(srcMax);
    e.span = e.pqLw - e.pqLb;
    e.minLum = (pqInverseEotf<double>(dstMin) - e.pqLb) / e.span;
    e.maxLum = (pqInverseEotf<double>(dstMax) - e.pqLb) / e.span;
    e.ks = 1.5 * e.maxLum - 0.5;
    e.degenerate = e.maxLum >= 1.0;

    // The black lift E3 = E2 + b (1 - E2)^4 has slope 1 - 4b at E2 = 0, so it
    // is monotone only for b <= 0.25. Not in the spec, derived.
    if (e.minLum > 0.25) {
        return "dst_min is too high for the black lift to stay monotone: minLum is " +
               std::to_string(e.minLum) + ", the bound is 0.25";
    }
    // Below KS = 0 the whole domain is spline and P(0) is negative, so E2
    // leaves [0, maxLum] and the chroma ratio changes sign.
    if (e.ks < 0.0) {
        return "dst_max is too low for the tone curve to stay in range: maxLum is " +
               std::to_string(e.maxLum) + " and KS is " + std::to_string(e.ks) +
               ", which must not be below 0";
    }

    *out = e;
    return std::string();
}

namespace {

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
        T r = static_cast<T>(srcR[x]) * nominal;
        T g = static_cast<T>(srcG[x]) * nominal;
        T b = static_cast<T>(srcB[x]) * nominal;
        r = r < T(0) ? T(0) : (r > static_cast<T>(kPqPeak) ? static_cast<T>(kPqPeak) : r);
        g = g < T(0) ? T(0) : (g > static_cast<T>(kPqPeak) ? static_cast<T>(kPqPeak) : g);
        b = b < T(0) ? T(0) : (b > static_cast<T>(kPqPeak) ? static_cast<T>(kPqPeak) : b);

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

}  // namespace

void toneMapRow(const float* srcR, const float* srcG, const float* srcB,
                float* dstR, float* dstG, float* dstB, size_t width,
                Representation rep, const FrameParams& params) {
    switch (rep) {
        case Representation::Ictcp:
            toneMapIctcp<double>(srcR, srcG, srcB, dstR, dstG, dstB, width, params);
            return;
    }
}

}  // namespace tonemapper
