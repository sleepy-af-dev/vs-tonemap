// Highway kernels for both filters, one source compiled for every target.
//
// The scalar kernels stay the reference: these are tested against them, and
// they were tested against the float64 oracle. Nothing here reworks the maths,
// it only rewrites the same steps in vector ops.

#include "simd.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "simd.cpp"

// Without this sleef.h marks every symbol dllimport on Windows, and the
// static library it is linked against then fails to resolve.
#define SLEEF_STATIC_LIBS 1
#include <sleef.h>

#include "hwy/foreach_target.h"  // must come before highway.h
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace tonemapper {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// --- pow ------------------------------------------------------------------
//
// SLEEF's u10 variants, documented to 1.0 ULP. Highway has no Pow, and
// composing Exp(y * Log(x)) would put the error of two 1 to 4 ULP functions
// through an outer exponent of 78.84 (section 3.1). The bridge is the `raw`
// member of Highway's vector wrapper, which is how Highway reaches the
// intrinsics itself, and the native width is picked per target so that one
// SLEEF call covers one whole vector.

// sleef.h guards each instruction set's declarations behind __SSE2__,
// __AVX__ or __AVX512F__. Highway never defines those: it compiles the whole
// file at the baseline and marks each target's code with a function
// attribute, so only the SSE2 declarations come through. The rest are
// declared here, under the same macros the header uses, so exactly one
// declaration exists whatever the baseline is. The definitions come from
// SLEEF's static library, which built every instruction set with its own
// flags.

#undef TONEMAPPER_HAVE_SLEEF

#if HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_DL || \
    HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX3_SPR
#define TONEMAPPER_HAVE_SLEEF 1
#ifndef __AVX512F__
extern "C" __m512d Sleef_powd8_u10avx512f(__m512d, __m512d);
extern "C" __m512 Sleef_powf16_u10avx512f(__m512, __m512);
#endif
HWY_INLINE hn::Vec512<double> SleefPow(hn::Vec512<double> x, hn::Vec512<double> y) {
    return hn::Vec512<double>{Sleef_powd8_u10avx512f(x.raw, y.raw)};
}
HWY_INLINE hn::Vec512<float> SleefPow(hn::Vec512<float> x, hn::Vec512<float> y) {
    return hn::Vec512<float>{Sleef_powf16_u10avx512f(x.raw, y.raw)};
}
#elif HWY_TARGET == HWY_AVX2
#define TONEMAPPER_HAVE_SLEEF 1
#ifndef __AVX__
extern "C" __m256d Sleef_powd4_u10avx2(__m256d, __m256d);
extern "C" __m256 Sleef_powf8_u10avx2(__m256, __m256);
#endif
HWY_INLINE hn::Vec256<double> SleefPow(hn::Vec256<double> x, hn::Vec256<double> y) {
    return hn::Vec256<double>{Sleef_powd4_u10avx2(x.raw, y.raw)};
}
HWY_INLINE hn::Vec256<float> SleefPow(hn::Vec256<float> x, hn::Vec256<float> y) {
    return hn::Vec256<float>{Sleef_powf8_u10avx2(x.raw, y.raw)};
}
#elif HWY_TARGET == HWY_SSE4
#define TONEMAPPER_HAVE_SLEEF 1
HWY_INLINE hn::Vec128<double> SleefPow(hn::Vec128<double> x, hn::Vec128<double> y) {
    return hn::Vec128<double>{Sleef_powd2_u10sse4(x.raw, y.raw)};
}
HWY_INLINE hn::Vec128<float> SleefPow(hn::Vec128<float> x, hn::Vec128<float> y) {
    return hn::Vec128<float>{Sleef_powf4_u10sse4(x.raw, y.raw)};
}
#elif HWY_TARGET == HWY_SSE2 || HWY_TARGET == HWY_SSSE3
#define TONEMAPPER_HAVE_SLEEF 1
HWY_INLINE hn::Vec128<double> SleefPow(hn::Vec128<double> x, hn::Vec128<double> y) {
    return hn::Vec128<double>{Sleef_powd2_u10sse2(x.raw, y.raw)};
}
HWY_INLINE hn::Vec128<float> SleefPow(hn::Vec128<float> x, hn::Vec128<float> y) {
    return hn::Vec128<float>{Sleef_powf4_u10sse2(x.raw, y.raw)};
}
#endif

// Every target SLEEF does not cover here, which on this release's only
// platform means the emulated ones. Correct, not fast.
template <class D, class V>
HWY_INLINE V PowPerLane(D d, V x, V y) {
    using T = hn::TFromD<D>;
    HWY_ALIGN T bx[HWY_MAX_BYTES / sizeof(T)];
    HWY_ALIGN T by[HWY_MAX_BYTES / sizeof(T)];
    hn::Store(x, d, bx);
    hn::Store(y, d, by);
    for (size_t i = 0; i < hn::Lanes(d); ++i) bx[i] = std::pow(bx[i], by[i]);
    return hn::Load(d, bx);
}

template <class D, class V>
HWY_INLINE V Pow(D d, V x, V y) {
#ifdef TONEMAPPER_HAVE_SLEEF
    (void)d;
    return SleefPow(x, y);
#else
    return PowPerLane(d, x, y);
#endif
}

// --- PQ, BT.2100-3 Table 4 -------------------------------------------------

template <class D, class V>
HWY_INLINE V PqInverseEotf(D d, V luminance) {
    using T = hn::TFromD<D>;
    const V zero = hn::Zero(d);
    const V one = hn::Set(d, T(1));
    V y = hn::Mul(luminance, hn::Set(d, static_cast<T>(1.0 / kPqPeak)));
    y = hn::Min(hn::Max(y, zero), one);
    const V ym = Pow(d, y, hn::Set(d, static_cast<T>(kM1)));
    const V num = hn::MulAdd(hn::Set(d, static_cast<T>(kC2)), ym,
                             hn::Set(d, static_cast<T>(kC1)));
    const V den = hn::MulAdd(hn::Set(d, static_cast<T>(kC3)), ym, one);
    return Pow(d, hn::Div(num, den), hn::Set(d, static_cast<T>(kM2)));
}

template <class D, class V>
HWY_INLINE V PqEotf(D d, V code) {
    using T = hn::TFromD<D>;
    const V zero = hn::Zero(d);
    const V t = Pow(d, hn::Max(code, zero), hn::Set(d, static_cast<T>(1.0 / kM2)));
    const V num = hn::Max(hn::Sub(t, hn::Set(d, static_cast<T>(kC1))), zero);
    const V den = hn::NegMulAdd(hn::Set(d, static_cast<T>(kC3)), t,
                                hn::Set(d, static_cast<T>(kC2)));
    return hn::Mul(Pow(d, hn::Div(num, den), hn::Set(d, static_cast<T>(1.0 / kM1))),
                   hn::Set(d, static_cast<T>(kPqPeak)));
}

// --- The EETF --------------------------------------------------------------

template <class D, class V>
HWY_INLINE V ApplyEetf(D d, V code, const Eetf& curve) {
    using T = hn::TFromD<D>;
    const V zero = hn::Zero(d);
    const V one = hn::Set(d, T(1));
    const V pqLb = hn::Set(d, static_cast<T>(curve.pqLb));
    const V span = hn::Set(d, static_cast<T>(curve.span));

    V e1 = hn::Mul(hn::Sub(code, pqLb), hn::Set(d, static_cast<T>(1.0 / curve.span)));
    e1 = hn::Min(hn::Max(e1, zero), one);

    V e2 = e1;
    if (!curve.degenerate) {
        const V ks = hn::Set(d, static_cast<T>(curve.ks));
        const V t = hn::Mul(hn::Sub(e1, ks), hn::Set(d, static_cast<T>(1.0 / (1.0 - curve.ks))));
        const V t2 = hn::Mul(t, t);
        const V t3 = hn::Mul(t2, t);
        const V a = hn::MulAdd(hn::Set(d, T(2)), t3,
                               hn::MulAdd(hn::Set(d, T(-3)), t2, one));
        const V b = hn::Add(hn::Sub(t3, hn::Mul(hn::Set(d, T(2)), t2)), t);
        const V c = hn::MulAdd(hn::Set(d, T(-2)), t3, hn::Mul(hn::Set(d, T(3)), t2));
        const V spline = hn::MulAdd(
            a, ks,
            hn::MulAdd(b, hn::Set(d, static_cast<T>(1.0 - curve.ks)),
                       hn::Mul(c, hn::Set(d, static_cast<T>(curve.maxLum)))));
        // The scalar path branches; here both sides are evaluated and the
        // identity one is selected, which is the same partition.
        e2 = hn::IfThenElse(hn::Lt(e1, ks), e1, spline);
    }

    const V lift = hn::Sub(one, e2);
    const V lift2 = hn::Mul(lift, lift);
    const V e3 = hn::MulAdd(hn::Set(d, static_cast<T>(curve.minLum)),
                            hn::Mul(lift2, lift2), e2);
    return hn::MulAdd(e3, span, pqLb);
}

// min(v1/v2, v2/v1), taken as 1 where either value is zero.
template <class D, class V>
HWY_INLINE V ChromaRatio(D d, V v1, V v2) {
    using T = hn::TFromD<D>;
    const V zero = hn::Zero(d);
    const V one = hn::Set(d, T(1));
    const auto ok = hn::And(hn::Ne(v1, zero), hn::Ne(v2, zero));
    const V a = hn::Div(v1, hn::IfThenElse(ok, v2, one));
    const V b = hn::Div(v2, hn::IfThenElse(ok, v1, one));
    return hn::IfThenElse(ok, hn::Min(a, b), one);
}

template <class D, class V>
HWY_INLINE void ApplyMatrix(D d, const Mat3& m, V x, V y, V z, V* o0, V* o1, V* o2) {
    using T = hn::TFromD<D>;
    const V r0 = hn::MulAdd(hn::Set(d, static_cast<T>(m.r[0][0])), x,
                            hn::MulAdd(hn::Set(d, static_cast<T>(m.r[0][1])), y,
                                       hn::Mul(hn::Set(d, static_cast<T>(m.r[0][2])), z)));
    const V r1 = hn::MulAdd(hn::Set(d, static_cast<T>(m.r[1][0])), x,
                            hn::MulAdd(hn::Set(d, static_cast<T>(m.r[1][1])), y,
                                       hn::Mul(hn::Set(d, static_cast<T>(m.r[1][2])), z)));
    const V r2 = hn::MulAdd(hn::Set(d, static_cast<T>(m.r[2][0])), x,
                            hn::MulAdd(hn::Set(d, static_cast<T>(m.r[2][1])), y,
                                       hn::Mul(hn::Set(d, static_cast<T>(m.r[2][2])), z)));
    *o0 = r0;
    *o1 = r1;
    *o2 = r2;
}

template <class D, class V>
HWY_INLINE V ClampToPqDomain(D d, V v) {
    using T = hn::TFromD<D>;
    return hn::Min(hn::Max(v, hn::Zero(d)), hn::Set(d, static_cast<T>(kPqPeak)));
}

// --- The representations ---------------------------------------------------

// The frame is float32 and the lanes are usually double, so every load
// widens and every store narrows. Rebind keeps the lane count the same and
// changes only the type, which is what PromoteTo and DemoteTo need. When the
// kernel is built with float lanes there is nothing to convert.
template <class D>
HWY_INLINE hn::Vec<D> LoadFloats(D d, const float* p, size_t count) {
    const size_t n = hn::Lanes(d);
    if constexpr (sizeof(hn::TFromD<D>) == sizeof(float)) {
        const auto* q = reinterpret_cast<const hn::TFromD<D>*>(p);
        return count >= n ? hn::LoadU(d, q) : hn::LoadN(d, q, count);
    } else {
        const hn::Rebind<float, D> df;
        return hn::PromoteTo(d, count >= n ? hn::LoadU(df, p) : hn::LoadN(df, p, count));
    }
}

template <class D, class V>
HWY_INLINE void StoreFloats(D d, V v, float* p, size_t count) {
    const size_t n = hn::Lanes(d);
    if constexpr (sizeof(hn::TFromD<D>) == sizeof(float)) {
        auto* q = reinterpret_cast<hn::TFromD<D>*>(p);
        if (count >= n) {
            hn::StoreU(v, d, q);
        } else {
            hn::StoreN(v, d, q, count);
        }
    } else {
        const hn::Rebind<float, D> df;
        const auto narrowed = hn::DemoteTo(df, v);
        if (count >= n) {
            hn::StoreU(narrowed, df, p);
        } else {
            hn::StoreN(narrowed, df, p, count);
        }
    }
}

template <typename Lane>
void ToneMapIctcp(const float* srcR, const float* srcG, const float* srcB, float* dstR,
                  float* dstG, float* dstB, size_t width, const FrameParams& params) {
    const hn::ScalableTag<Lane> d;
    using V = hn::Vec<decltype(d)>;
    const size_t N = hn::Lanes(d);

    const V nominal = hn::Set(d, static_cast<Lane>(params.nominal));
    const V invDstMax = hn::Set(d, static_cast<Lane>(1.0 / params.dstMax));

    for (size_t x = 0; x < width; x += N) {
        const size_t count = HWY_MIN(N, width - x);
        V r = LoadFloats(d, srcR + x, count);
        V g = LoadFloats(d, srcG + x, count);
        V b = LoadFloats(d, srcB + x, count);

        r = ClampToPqDomain(d, hn::Mul(r, nominal));
        g = ClampToPqDomain(d, hn::Mul(g, nominal));
        b = ClampToPqDomain(d, hn::Mul(b, nominal));

        V l, m, s;
        ApplyMatrix(d, kRgb2020ToLms, r, g, b, &l, &m, &s);
        l = PqInverseEotf(d, l);
        m = PqInverseEotf(d, m);
        s = PqInverseEotf(d, s);

        V i1, ct, cp;
        ApplyMatrix(d, kLmspToIctcp, l, m, s, &i1, &ct, &cp);

        const V i2 = ApplyEetf(d, i1, params.curve);
        const V k = ChromaRatio(d, i1, i2);

        ApplyMatrix(d, kIctcpToLmsp, i2, hn::Mul(k, ct), hn::Mul(k, cp), &l, &m, &s);
        l = PqEotf(d, l);
        m = PqEotf(d, m);
        s = PqEotf(d, s);

        ApplyMatrix(d, kLmsToRgb2020, l, m, s, &r, &g, &b);
        StoreFloats(d, hn::Mul(r, invDstMax), dstR + x, count);
        StoreFloats(d, hn::Mul(g, invDstMax), dstG + x, count);
        StoreFloats(d, hn::Mul(b, invDstMax), dstB + x, count);
    }
}

// Annex 5 option 2. Y' through the curve, Cb and Cr scaled by the ratio.
template <typename Lane>
void ToneMapYcbcr(const float* srcR, const float* srcG, const float* srcB, float* dstR,
                  float* dstG, float* dstB, size_t width, const FrameParams& params) {
    const hn::ScalableTag<Lane> d;
    using V = hn::Vec<decltype(d)>;
    const size_t N = hn::Lanes(d);

    const V nominal = hn::Set(d, static_cast<Lane>(params.nominal));
    const V invDstMax = hn::Set(d, static_cast<Lane>(1.0 / params.dstMax));
    const V kr = hn::Set(d, static_cast<Lane>(kKr));
    const V kb = hn::Set(d, static_cast<Lane>(kKb));
    const V cbDiv = hn::Set(d, static_cast<Lane>(kCbDivisor));
    const V crDiv = hn::Set(d, static_cast<Lane>(kCrDivisor));

    for (size_t x = 0; x < width; x += N) {
        const size_t count = HWY_MIN(N, width - x);
        const V rp = PqInverseEotf(d, hn::Mul(LoadFloats(d, srcR + x, count), nominal));
        const V gp = PqInverseEotf(d, hn::Mul(LoadFloats(d, srcG + x, count), nominal));
        const V bp = PqInverseEotf(d, hn::Mul(LoadFloats(d, srcB + x, count), nominal));

        const V y1 = hn::MulAdd(
            kr, rp,
            hn::MulAdd(hn::Set(d, static_cast<Lane>(kKg)), gp, hn::Mul(kb, bp)));
        const V cb = hn::Div(hn::Sub(bp, y1), cbDiv);
        const V cr = hn::Div(hn::Sub(rp, y1), crDiv);

        const V y2 = ApplyEetf(d, y1, params.curve);
        const V k = ChromaRatio(d, y1, y2);

        const V outR = hn::MulAdd(crDiv, hn::Mul(k, cr), y2);
        const V outB = hn::MulAdd(cbDiv, hn::Mul(k, cb), y2);
        const V outG =
            hn::Mul(hn::NegMulAdd(kb, outB, hn::NegMulAdd(kr, outR, y2)),
                    hn::Set(d, static_cast<Lane>(1.0 / kKg)));

        StoreFloats(d, hn::Mul(PqEotf(d, outR), invDstMax), dstR + x, count);
        StoreFloats(d, hn::Mul(PqEotf(d, outG), invDstMax), dstG + x, count);
        StoreFloats(d, hn::Mul(PqEotf(d, outB), invDstMax), dstB + x, count);
    }
}

// Annex 5 options 3 and 5. One driving value through the curve, its linear
// ratio applied to all three channels. Where the driving value is not
// positive there is no ratio, and the pixel takes the curve's own black.
template <typename Lane, bool kMaxRgb>
void ToneMapRatio(const float* srcR, const float* srcG, const float* srcB, float* dstR,
                  float* dstG, float* dstB, size_t width, const FrameParams& params) {
    const hn::ScalableTag<Lane> d;
    using V = hn::Vec<decltype(d)>;
    const size_t N = hn::Lanes(d);

    const V nominal = hn::Set(d, static_cast<Lane>(params.nominal));
    const V invDstMax = hn::Set(d, static_cast<Lane>(1.0 / params.dstMax));
    const V zero = hn::Zero(d);
    const V one = hn::Set(d, Lane(1));
    const V black = hn::Set(d, static_cast<Lane>(params.curve.black() / params.dstMax));

    for (size_t x = 0; x < width; x += N) {
        const size_t count = HWY_MIN(N, width - x);
        const V r = ClampToPqDomain(d, hn::Mul(LoadFloats(d, srcR + x, count), nominal));
        const V g = ClampToPqDomain(d, hn::Mul(LoadFloats(d, srcG + x, count), nominal));
        const V b = ClampToPqDomain(d, hn::Mul(LoadFloats(d, srcB + x, count), nominal));

        V v1;
        if constexpr (kMaxRgb) {
            v1 = hn::Max(r, hn::Max(g, b));
        } else {
            v1 = hn::MulAdd(hn::Set(d, static_cast<Lane>(kKr)), r,
                            hn::MulAdd(hn::Set(d, static_cast<Lane>(kKg)), g,
                                       hn::Mul(hn::Set(d, static_cast<Lane>(kKb)), b)));
        }

        const auto lit = hn::Gt(v1, zero);
        const V v2 = PqEotf(d, ApplyEetf(d, PqInverseEotf(d, v1), params.curve));
        const V k = hn::Mul(hn::Div(v2, hn::IfThenElse(lit, v1, one)), invDstMax);

        StoreFloats(d, hn::IfThenElse(lit, hn::Mul(r, k), black), dstR + x, count);
        StoreFloats(d, hn::IfThenElse(lit, hn::Mul(g, k), black), dstG + x, count);
        StoreFloats(d, hn::IfThenElse(lit, hn::Mul(b, k), black), dstB + x, count);
    }
}

// Annex 5 option 4. Each channel through the curve on its own.
template <typename Lane>
void ToneMapRgb(const float* srcR, const float* srcG, const float* srcB, float* dstR,
                float* dstG, float* dstB, size_t width, const FrameParams& params) {
    const hn::ScalableTag<Lane> d;
    using V = hn::Vec<decltype(d)>;
    const size_t N = hn::Lanes(d);

    const V nominal = hn::Set(d, static_cast<Lane>(params.nominal));
    const V invDstMax = hn::Set(d, static_cast<Lane>(1.0 / params.dstMax));
    const float* src[3] = {srcR, srcG, srcB};
    float* dst[3] = {dstR, dstG, dstB};

    for (int plane = 0; plane < 3; ++plane) {
        for (size_t x = 0; x < width; x += N) {
            const size_t count = HWY_MIN(N, width - x);
            const V v = hn::Mul(LoadFloats(d, src[plane] + x, count), nominal);
            const V mapped = PqEotf(d, ApplyEetf(d, PqInverseEotf(d, v), params.curve));
            StoreFloats(d, hn::Mul(mapped, invDstMax), dst[plane] + x, count);
        }
    }
}

void ToneMapRow(const float* srcR, const float* srcG, const float* srcB, float* dstR,
                float* dstG, float* dstB, size_t width, Representation rep,
                const FrameParams& params) {
    switch (rep) {
        case Representation::Ictcp:
#if TONEMAPPER_FLOAT32_ICTCP
            ToneMapIctcp<float>(srcR, srcG, srcB, dstR, dstG, dstB, width, params);
#else
            ToneMapIctcp<double>(srcR, srcG, srcB, dstR, dstG, dstB, width, params);
#endif
            return;
        case Representation::Ycbcr:
            ToneMapYcbcr<double>(srcR, srcG, srcB, dstR, dstG, dstB, width, params);
            return;
        case Representation::Yrgb:
            ToneMapRatio<double, false>(srcR, srcG, srcB, dstR, dstG, dstB, width, params);
            return;
        case Representation::Rgb:
            ToneMapRgb<double>(srcR, srcG, srcB, dstR, dstG, dstB, width, params);
            return;
        case Representation::Maxrgb:
            ToneMapRatio<double, true>(srcR, srcG, srcB, dstR, dstG, dstB, width, params);
            return;
    }
}

const char* TargetName() { return hwy::TargetName(HWY_TARGET); }

size_t DoubleLanes() { return hn::Lanes(hn::ScalableTag<double>()); }

// --- BT.2407 ---------------------------------------------------------------

template <class D, class V>
HWY_INLINE V ClampUnit(D d, V v) {
    using T = hn::TFromD<D>;
    return hn::Min(hn::Max(v, hn::Zero(d)), hn::Set(d, T(1)));
}

// Where the ray from the white point leaves a gamut. Same six linear
// inequalities as the scalar path; each bound is taken only where its
// condition holds, and the divisor is made safe first so the inactive lanes
// cannot raise anything.
template <class D, class V>
HWY_INLINE V BoundaryT(D d, const Mat3& xyzToRgb, V y, V du, V dv) {
    using T = hn::TFromD<D>;
    const V one = hn::Set(d, T(1));
    const V zero = hn::Zero(d);
    const V uw = hn::Set(d, static_cast<T>(kWhiteUv.x));
    const V vw = hn::Set(d, static_cast<T>(kWhiteUv.y));
    const V infinity = hn::Set(d, std::numeric_limits<T>::infinity());

    V a[3];
    ApplyMatrix(d, xyzToRgb, hn::Mul(hn::Mul(hn::Set(d, T(9)), y), uw),
                hn::Mul(hn::Mul(hn::Set(d, T(4)), y), vw),
                hn::Mul(y, hn::Set(d, static_cast<T>(12.0 - 3.0 * kWhiteUv.x -
                                                     20.0 * kWhiteUv.y))),
                &a[0], &a[1], &a[2]);
    V b[3];
    ApplyMatrix(d, xyzToRgb, hn::Mul(hn::Mul(hn::Set(d, T(9)), y), du),
                hn::Mul(hn::Mul(hn::Set(d, T(4)), y), dv),
                hn::Mul(y, hn::NegMulAdd(hn::Set(d, T(20)), dv,
                                         hn::Mul(hn::Set(d, T(-3)), du))),
                &b[0], &b[1], &b[2]);

    const V d0 = hn::Mul(hn::Set(d, T(4)), vw);
    const V d1 = hn::Mul(hn::Set(d, T(4)), dv);

    V t = infinity;
    for (int i = 0; i < 3; ++i) {
        const auto negative = hn::Lt(b[i], zero);
        const V lower = hn::Div(hn::Neg(a[i]), hn::IfThenElse(negative, b[i], one));
        t = hn::Min(t, hn::IfThenElse(negative, lower, infinity));

        const V den = hn::Sub(b[i], d1);
        const auto rising = hn::Gt(den, zero);
        const V upper =
            hn::Div(hn::Neg(hn::Sub(a[i], d0)), hn::IfThenElse(rising, den, one));
        t = hn::Min(t, hn::IfThenElse(rising, upper, infinity));
    }
    return t;
}

// The Annex 5 roll-off, as a select rather than a branch.
template <class D, class V>
HWY_INLINE V SoftClipCurve(D d, V r, V alpha, V beta) {
    using T = hn::TFromD<D>;
    const V zero = hn::Zero(d);
    const V one = hn::Set(d, T(1));
    const V q = hn::Sub(hn::Add(r, beta), one);
    const V k = hn::Sub(alpha, beta);

    const V disc = hn::Max(hn::MulAdd(k, q, hn::Mul(beta, beta)), zero);
    const V den = hn::Add(hn::Sqrt(disc), beta);
    const V s = hn::Div(q, hn::IfThenElse(hn::Gt(den, zero), den, one));
    const V rolled = hn::NegMulAdd(alpha, hn::Mul(s, s), r);

    const V flat = hn::IfThenElse(hn::Gt(r, hn::Add(one, alpha)), one, rolled);
    return hn::IfThenElse(hn::Gt(q, zero), flat, r);
}

template <typename Lane>
void GamutSoftclip(const float* srcR, const float* srcG, const float* srcB, float* dstR,
                   float* dstG, float* dstB, size_t width, const GamutParams& params) {
    const hn::ScalableTag<Lane> d;
    using V = hn::Vec<decltype(d)>;
    const size_t N = hn::Lanes(d);

    const V zero = hn::Zero(d);
    const V one = hn::Set(d, Lane(1));
    const V uw = hn::Set(d, static_cast<Lane>(kWhiteUv.x));
    const V vw = hn::Set(d, static_cast<Lane>(kWhiteUv.y));
    const V beta = hn::Set(d, static_cast<Lane>(params.beta));

    for (size_t x = 0; x < width; x += N) {
        const size_t count = HWY_MIN(N, width - x);
        const V in0 = LoadFloats(d, srcR + x, count);
        const V in1 = LoadFloats(d, srcG + x, count);
        const V in2 = LoadFloats(d, srcB + x, count);

        V bigX, y, bigZ;
        ApplyMatrix(d, kRgb2020ToXyz, in0, in1, in2, &bigX, &y, &bigZ);
        const V denom = hn::MulAdd(hn::Set(d, Lane(15)), y,
                                   hn::MulAdd(hn::Set(d, Lane(3)), bigZ, bigX));

        const auto hasChroma = hn::Gt(denom, zero);
        const auto projectable = hn::And(hn::And(hn::Gt(y, zero), hn::Lt(y, one)), hasChroma);
        const V safeDenom = hn::IfThenElse(hasChroma, denom, one);
        const V safeY = hn::IfThenElse(projectable, y, hn::Set(d, Lane(0.5)));

        const V du = hn::Sub(
            hn::IfThenElse(projectable,
                           hn::Div(hn::Mul(hn::Set(d, Lane(4)), bigX), safeDenom), uw),
            uw);
        const V dv = hn::Sub(
            hn::IfThenElse(projectable,
                           hn::Div(hn::Mul(hn::Set(d, Lane(9)), y), safeDenom), vw),
            vw);

        const V tSource = BoundaryT(d, params.xyzToSource, safeY, du, dv);
        const V t709 = BoundaryT(d, kXyzToRgb709, safeY, du, dv);

        // An achromatic ray leaves both gamuts at infinity, so the ratio is
        // 0/0 and r is 0, which takes the identity branch where alpha is
        // unused. Anything not finite reads as no headroom.
        const V finite709 = hn::IfThenElse(hn::IsFinite(t709), t709, one);
        V alpha = hn::Sub(hn::Div(tSource, finite709), one);
        alpha = hn::Max(hn::IfThenElse(hn::IsFinite(alpha), alpha, zero), zero);
        const V r = hn::IfThenElse(hn::IsFinite(t709), hn::Div(one, finite709), zero);

        const V scale = hn::IfThenElse(hn::Le(r, hn::Sub(one, beta)), one,
                                       hn::Mul(t709, SoftClipCurve(d, r, alpha, beta)));
        const V u2 = hn::MulAdd(du, scale, uw);
        const V v2 = hn::MulAdd(dv, scale, vw);
        const V quarter = hn::Div(one, hn::Mul(hn::Set(d, Lane(4)), v2));

        V out0, out1, out2;
        ApplyMatrix(d, kXyzToRgb709,
                    hn::Mul(hn::Mul(hn::Mul(hn::Set(d, Lane(9)), safeY), u2), quarter),
                    safeY,
                    hn::Mul(hn::Mul(safeY, hn::NegMulAdd(hn::Set(d, Lane(20)), v2,
                                                         hn::NegMulAdd(hn::Set(d, Lane(3)),
                                                                       u2, hn::Set(d, Lane(12))))),
                            quarter),
                    &out0, &out1, &out2);

        V hard0, hard1, hard2;
        ApplyMatrix(d, kRgb2020ToRgb709, in0, in1, in2, &hard0, &hard1, &hard2);

        // The three input policies in the order they apply: Y at or below 0 is
        // black, Y at or above the peak is white, and only then does the
        // missing-chromaticity guard take the hard clip.
        const auto dark = hn::Le(y, zero);
        const auto bright = hn::Ge(y, one);
        V result[3] = {ClampUnit(d, out0), ClampUnit(d, out1), ClampUnit(d, out2)};
        const V hard[3] = {ClampUnit(d, hard0), ClampUnit(d, hard1), ClampUnit(d, hard2)};
        float* dst[3] = {dstR, dstG, dstB};
        for (int i = 0; i < 3; ++i) {
            V v = hn::IfThenElse(hasChroma, result[i], hard[i]);
            v = hn::IfThenElse(bright, one, v);
            v = hn::IfThenElse(dark, zero, v);
            StoreFloats(d, v, dst[i] + x, count);
        }
    }
}

template <typename Lane>
void GamutClip(const float* srcR, const float* srcG, const float* srcB, float* dstR,
               float* dstG, float* dstB, size_t width) {
    const hn::ScalableTag<Lane> d;
    using V = hn::Vec<decltype(d)>;
    const size_t N = hn::Lanes(d);

    for (size_t x = 0; x < width; x += N) {
        const size_t count = HWY_MIN(N, width - x);
        V r, g, b;
        ApplyMatrix(d, kRgb2020ToRgb709, LoadFloats(d, srcR + x, count),
                    LoadFloats(d, srcG + x, count), LoadFloats(d, srcB + x, count), &r,
                    &g, &b);
        StoreFloats(d, ClampUnit(d, r), dstR + x, count);
        StoreFloats(d, ClampUnit(d, g), dstG + x, count);
        StoreFloats(d, ClampUnit(d, b), dstB + x, count);
    }
}

void GamutMapRow(const float* srcR, const float* srcG, const float* srcB, float* dstR,
                 float* dstG, float* dstB, size_t width, const GamutParams& params) {
    switch (params.method) {
        case GamutMethod::Clip:
            GamutClip<double>(srcR, srcG, srcB, dstR, dstG, dstB, width);
            return;
        case GamutMethod::Softclip:
            GamutSoftclip<double>(srcR, srcG, srcB, dstR, dstG, dstB, width, params);
            return;
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace tonemapper
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace tonemapper {

HWY_EXPORT(ToneMapRow);
HWY_EXPORT(GamutMapRow);
HWY_EXPORT(TargetName);
HWY_EXPORT(DoubleLanes);

void toneMapRowSimd(const float* srcR, const float* srcG, const float* srcB,
                    float* dstR, float* dstG, float* dstB, size_t width,
                    Representation rep, const FrameParams& params) {
    HWY_DYNAMIC_DISPATCH(ToneMapRow)
    (srcR, srcG, srcB, dstR, dstG, dstB, width, rep, params);
}

void gamutMapRowSimd(const float* srcR, const float* srcG, const float* srcB,
                     float* dstR, float* dstG, float* dstB, size_t width,
                     const GamutParams& params) {
    HWY_DYNAMIC_DISPATCH(GamutMapRow)
    (srcR, srcG, srcB, dstR, dstG, dstB, width, params);
}

const char* simdTargetName() { return HWY_DYNAMIC_DISPATCH(TargetName)(); }

size_t simdDoubleLanes() { return HWY_DYNAMIC_DISPATCH(DoubleLanes)(); }

bool ictcpUsesFloatLanes() {
#if TONEMAPPER_FLOAT32_ICTCP
    return true;
#else
    return false;
#endif
}

namespace {

// Highway has no getter for its test override, and the full target list can
// only be read with the override off, so what it was last given is kept here.
int64_t forcedTarget = 0;

}  // namespace

std::vector<const char*> simdTargets() {
    hwy::SetSupportedTargetsForTest(0);
    std::vector<const char*> names;
    for (const int64_t target : hwy::SupportedAndGeneratedTargets()) {
        names.push_back(hwy::TargetName(target));
    }
    hwy::SetSupportedTargetsForTest(forcedTarget);
    return names;
}

bool simdForceTarget(const char* name) {
    const bool restore = name == nullptr || name[0] == '\0';
    int64_t wanted = 0;
    if (!restore) {
        hwy::SetSupportedTargetsForTest(0);
        for (const int64_t target : hwy::SupportedAndGeneratedTargets()) {
            if (std::strcmp(hwy::TargetName(target), name) == 0) wanted = target;
        }
        if (wanted == 0) {
            hwy::SetSupportedTargetsForTest(forcedTarget);
            return false;
        }
    }
    forcedTarget = wanted;
    hwy::SetSupportedTargetsForTest(wanted);
    return true;
}

}  // namespace tonemapper
#endif  // HWY_ONCE
