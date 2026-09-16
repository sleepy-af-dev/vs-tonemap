#include "hlg.h"

#include <cmath>

#include "colour.h"  // checkLuminance, and kKr/kKg/kKb for the OOTF luminance

namespace tonemap {

double hlgSystemGamma(double lw) {
    if (lw >= kHlgGammaLo && lw <= kHlgGammaHi) {
        return 1.2 + 0.42 * std::log10(lw / 1000.0);
    }
    return 1.2 * std::pow(kHlgKappa, std::log2(lw / 1000.0));
}

double hlgBlackLift(double lw, double lb, double gamma) {
    return std::sqrt(3.0 * std::pow(lb / lw, 1.0 / gamma));
}

std::string makeHlgParams(double lw, double lb, double nominal, const char* lwName,
                          const char* lbName, HlgParams* out) {
    std::string bad = checkLuminance(lwName, lw);
    if (bad.empty()) bad = checkLuminance(lbName, lb);
    if (!bad.empty()) return bad;
    if (lw <= 0.0) {
        return std::string(lwName) + " must be a positive luminance in cd/m2";
    }
    if (lb >= lw) {
        return std::string(lbName) + " must be below " + lwName;
    }
    if (!std::isfinite(nominal) || nominal <= 0.0) {
        return "nominal_luminance must be a positive number of cd/m2";
    }

    HlgParams p{};
    p.gamma = hlgSystemGamma(lw);
    p.beta = hlgBlackLift(lw, lb, p.gamma);
    p.lw = lw;
    p.lb = lb;
    p.invNominal = 1.0 / nominal;
    *out = p;
    return std::string();
}

namespace {

// Note 5a, the inverse of the HLG OETF. The input is clamped to [0, 1]
// because that is the whole domain HLG defines; chroma upsampling ringing
// and limited-range codes outside 64 to 940 both put samples beyond it.
template <typename T>
inline T hlgInverseOetf(T ep) {
    ep = ep < T(0) ? T(0) : (ep > T(1) ? T(1) : ep);
    if (ep <= T(0.5)) return ep * ep * static_cast<T>(1.0 / 3.0);
    return (std::exp((ep - static_cast<T>(kHlgC)) * static_cast<T>(1.0 / kHlgA)) +
            static_cast<T>(kHlgB)) *
           static_cast<T>(1.0 / 12.0);
}

}  // namespace

void hlgRow(const float* srcR, const float* srcG, const float* srcB, float* dstR,
            float* dstG, float* dstB, size_t width, const HlgParams& params) {
    using T = double;
    const T beta = params.beta;
    const T oneMinusBeta = T(1) - beta;
    // Table 5 applies max(0, .) to the lifted signal rather than to E', which
    // is what hlgInverseOetf's own clamp does on the way past.
    const T exponent = params.gamma - T(1);
    const T scale = params.lw * params.invNominal;

    for (size_t x = 0; x < width; ++x) {
        const T r = hlgInverseOetf<T>(oneMinusBeta * static_cast<T>(srcR[x]) + beta);
        const T g = hlgInverseOetf<T>(oneMinusBeta * static_cast<T>(srcG[x]) + beta);
        const T b = hlgInverseOetf<T>(oneMinusBeta * static_cast<T>(srcB[x]) + beta);

        const T ys = kKr * r + kKg * g + kKb * b;
        // Below a peak of about 301 cd/m2 the system gamma falls under 1, so
        // the exponent is negative and ys^exponent at black is infinity;
        // infinity times a channel of zero is NaN. The clamp above leaves ys
        // non-negative, so only exact black reaches this.
        if (!(ys > T(0))) {
            dstR[x] = 0.0f;
            dstG[x] = 0.0f;
            dstB[x] = 0.0f;
            continue;
        }

        const T k = std::pow(ys, exponent) * scale;
        dstR[x] = static_cast<float>(r * k);
        dstG[x] = static_cast<float>(g * k);
        dstB[x] = static_cast<float>(b * k);
    }
}

}  // namespace tonemap
