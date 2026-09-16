// vs-tonemap: BT.2390 tone mapping and BT.2407 gamut conversion for VapourSynth.

#include <VSConstants4.h>
#include <VSHelper4.h>
#include <VapourSynth4.h>

#include <cmath>
#include <cstring>
#include <string>

#include "bt2390.h"
#include "bt2407.h"
#include "hlg.h"
#include "simd.h"

namespace {

using tonemap::FrameParams;
using tonemap::Representation;

// The version, in one place. VapourSynth packs a plugin version into one int
// as (major << 16) | minor, so the patch component cannot reach configPlugin
// and two releases that differ only in it report the same PluginVersion.
// Info() is therefore where a release names itself in full, and Info() is what
// a bug report quotes.
constexpr int kVersionMajor = 0;
constexpr int kVersionMinor = 1;
constexpr int kVersionPatch = 0;

// VapourSynth renamed the range property to `_Range`, where 1 is full range.
// The core translates the deprecated `_ColorRange` spelling into `_Range` and
// inverts the value as it does so, in the map itself, so a frame never holds
// both and reading `_Range` covers either spelling. That translation also
// means `_ColorRange` must never be touched: deleting it deletes `_Range`.
//
// VSC_RANGE_FULL cannot be used here. In the API 4.0 headers this plugin pins
// to it is 0, the old `_ColorRange` convention; only the API 4.2 headers
// redefine it to 1 to match `_Range`.
constexpr int64_t kRangeFull = 1;

// Pins the reading above. VSC_RANGE_FULL is 0 only while the headers are at
// API 4.0 or 4.1; defining VS_USE_API_42 flips it to 1 and would silently
// turn the `_ColorRange` fallback into a test for limited range. Keep
// VS_USE_API_42 undefined, as section 4 of the design requires anyway.
static_assert(VSC_RANGE_FULL == 0,
              "the _ColorRange fallback assumes the pre-R74 convention");

struct FilterData {
    VSNode* node;
    VSVideoInfo vi;
    Representation rep;
    bool simd;
    double nominal;
    double dstMin;
    double dstMax;
    // Absent means read the value from the frame properties instead.
    bool haveSrcMin;
    bool haveSrcMax;
    double srcMin;
    double srcMax;
};

bool isRgbs(const VSVideoInfo* vi) {
    return vsh::isConstantVideoFormat(vi) && vi->format.colorFamily == cfRGB &&
           vi->format.sampleType == stFloat && vi->format.bitsPerSample == 32;
}

// A property that is present and contradicts the contract is an error. A
// property that is absent is not: tagging is not demanded, only consistency.
// Only peUnset counts as absent. A tag of the wrong type is a mistake worth
// reporting rather than something to skip over, which is what treating every
// error as absent used to do.
std::string checkTag(const VSMap* props, const VSAPI* vsapi, const char* key,
                     int64_t expected, const char* what) {
    int err = 0;
    const int64_t value = vsapi->mapGetInt(props, key, 0, &err);
    if (err == peUnset) return std::string();
    if (err != 0) {
        return std::string(key) + " is not an integer property, so this filter cannot "
               "read it; it has to be the integer tag for " + what;
    }
    if (value == expected) return std::string();
    return std::string(key) + " says " + std::to_string(value) + ", but this filter " +
           "needs " + what + " (" + std::to_string(expected) + ")";
}

// `transfer` and `transferName` are the transfer characteristic this caller
// requires and the words its error message uses. The primaries and range
// checks are the same for every filter here.
std::string checkFrameTags(const VSMap* props, const VSAPI* vsapi, int64_t transfer,
                           const char* transferName) {
    std::string bad = checkTag(props, vsapi, "_Transfer", transfer, transferName);
    if (!bad.empty()) return bad;
    bad = checkTag(props, vsapi, "_Primaries", VSC_PRIMARIES_BT2020, "BT.2020 primaries");
    if (!bad.empty()) return bad;

    // `_Range` arrived in R74. Before that the key was `_ColorRange` with the
    // opposite convention, which is what VSC_RANGE_FULL names in the API 4.0
    // headers. From R74 on the core rewrites `_ColorRange` into `_Range`
    // inside the map, so a frame never carries the old key and this fallback
    // only ever fires on R55 to R73.
    if (vsapi->mapNumElements(props, "_Range") >= 0) {
        return checkTag(props, vsapi, "_Range", kRangeFull, "full range");
    }
    return checkTag(props, vsapi, "_ColorRange", VSC_RANGE_FULL, "full range");
}

// A mastering luminance from the frame properties. A max that is absent,
// non-finite or not positive counts as absent; a min of 0 is a valid black.
// A source filter writes these as floats, but a hand-written SetFrameProps
// call produces integers, so both are read.
bool readMasteringLuminance(const VSMap* props, const VSAPI* vsapi, const char* key,
                            bool requirePositive, double* out) {
    int err = 0;
    double value = vsapi->mapGetFloat(props, key, 0, &err);
    if (err == peType) {
        value = static_cast<double>(vsapi->mapGetInt(props, key, 0, &err));
    }
    if (err != 0 || !std::isfinite(value)) return false;
    if (requirePositive && value <= 0.0) return false;
    *out = value;
    return true;
}

std::string resolveFrameParams(const FilterData* d, const VSMap* props,
                               const VSAPI* vsapi, FrameParams* out) {
    std::string bad = checkFrameTags(props, vsapi, VSC_TRANSFER_LINEAR, "linear light");
    if (!bad.empty()) return bad;

    // The peak is reported first because it is the one that shapes the curve.
    double srcMin = d->srcMin;
    double srcMax = d->srcMax;
    const char* minName = "src_min";
    const char* maxName = "src_max";
    if (!d->haveSrcMax) {
        maxName = "MasteringDisplayMaxLuminance";
        if (!readMasteringLuminance(props, vsapi, maxName, true, &srcMax)) {
            return "src_max was not given and MasteringDisplayMaxLuminance is not "
                   "usable. Pass src_max explicitly; BT.2408 names 10000 as the "
                   "fallback when the mastering display peak is unknown";
        }
    }
    if (!d->haveSrcMin) {
        minName = "MasteringDisplayMinLuminance";
        if (!readMasteringLuminance(props, vsapi, minName, false, &srcMin)) {
            return "src_min was not given and MasteringDisplayMinLuminance is not "
                   "usable. Pass src_min explicitly; BT.2408 names 0 as the fallback "
                   "when the mastering display black is unknown";
        }
    }

    out->nominal = d->nominal;
    out->dstMax = d->dstMax;
    // Whichever name the value came in under is the name the message uses, so
    // a bad mastering tag does not read as a bad argument.
    return tonemap::makeEetf(srcMin, srcMax, d->dstMin, d->dstMax, minName, maxName,
                                &out->curve);
}

// Metadata that described content which no longer exists after tone mapping.
// The mastering primaries and white point stay, because BT2407 reads them.
void stripHdrMetadata(VSMap* props, const VSAPI* vsapi) {
    static const char* const keys[] = {
        "MasteringDisplayMinLuminance", "MasteringDisplayMaxLuminance",
        "ContentLightLevelMax",         "ContentLightLevelAverage",
        "DolbyVisionRPU",               "HDR10Plus",
    };
    for (const char* key : keys) vsapi->mapDeleteKey(props, key);
}

const VSFrame* VS_CC getFrame(int n, int activationReason, void* instanceData, void**,
                              VSFrameContext* frameCtx, VSCore* core,
                              const VSAPI* vsapi) {
    auto* d = static_cast<FilterData*>(instanceData);
    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        return nullptr;
    }
    if (activationReason != arAllFramesReady) return nullptr;

    const VSFrame* src = vsapi->getFrameFilter(n, d->node, frameCtx);

    FrameParams params{};
    const std::string bad =
        resolveFrameParams(d, vsapi->getFramePropertiesRO(src), vsapi, &params);
    if (!bad.empty()) {
        vsapi->setFilterError(("BT2390: " + bad).c_str(), frameCtx);
        vsapi->freeFrame(src);
        return nullptr;
    }

    const int width = vsapi->getFrameWidth(src, 0);
    const int height = vsapi->getFrameHeight(src, 0);
    VSFrame* dst = vsapi->newVideoFrame(&d->vi.format, width, height, src, core);

    const float* srcPlane[3];
    float* dstPlane[3];
    ptrdiff_t srcStride[3];
    ptrdiff_t dstStride[3];
    for (int p = 0; p < 3; ++p) {
        srcPlane[p] = reinterpret_cast<const float*>(vsapi->getReadPtr(src, p));
        dstPlane[p] = reinterpret_cast<float*>(vsapi->getWritePtr(dst, p));
        srcStride[p] = vsapi->getStride(src, p) / static_cast<ptrdiff_t>(sizeof(float));
        dstStride[p] = vsapi->getStride(dst, p) / static_cast<ptrdiff_t>(sizeof(float));
    }
    for (int y = 0; y < height; ++y) {
        const auto row = d->simd ? tonemap::toneMapRowSimd : tonemap::toneMapRow;
        row(srcPlane[0] + y * srcStride[0], srcPlane[1] + y * srcStride[1],
            srcPlane[2] + y * srcStride[2], dstPlane[0] + y * dstStride[0],
            dstPlane[1] + y * dstStride[1], dstPlane[2] + y * dstStride[2],
            static_cast<size_t>(width), d->rep, params);
    }

    VSMap* outProps = vsapi->getFramePropertiesRW(dst);
    stripHdrMetadata(outProps, vsapi);
    vsapi->mapSetInt(outProps, "_Transfer", VSC_TRANSFER_LINEAR, maReplace);
    vsapi->mapSetInt(outProps, "_Primaries", VSC_PRIMARIES_BT2020, maReplace);
    vsapi->mapSetInt(outProps, "_Range", kRangeFull, maReplace);

    vsapi->freeFrame(src);
    return dst;
}

void VS_CC freeFilter(void* instanceData, VSCore*, const VSAPI* vsapi) {
    auto* d = static_cast<FilterData*>(instanceData);
    vsapi->freeNode(d->node);
    delete d;
}

// The SIMD kernels are checked against the scalar ones, which are the
// reference, so both have to stay reachable. It doubles as a way out if a
// machine ever disagrees with its own vector unit.
bool optionalBool(const VSMap* in, const VSAPI* vsapi, const char* key, bool fallback) {
    int err = 0;
    const int64_t value = vsapi->mapGetInt(in, key, 0, &err);
    return err == 0 ? value != 0 : fallback;
}

double optionalFloat(const VSMap* in, const VSAPI* vsapi, const char* key,
                     double fallback, bool* present) {
    int err = 0;
    const double value = vsapi->mapGetFloat(in, key, 0, &err);
    if (present != nullptr) *present = err == 0;
    return err == 0 ? value : fallback;
}

// --- HLG -------------------------------------------------------------------

using tonemap::HlgParams;

struct HlgFilterData {
    VSNode* node;
    VSVideoInfo vi;
    bool simd;
    double nominal;
    // Absent means read the value from the frame properties instead.
    bool haveLw;
    bool haveLb;
    double lw;
    double lb;
};

std::string resolveHlgParams(const HlgFilterData* d, const VSMap* props,
                             const VSAPI* vsapi, HlgParams* out) {
    std::string bad =
        checkFrameTags(props, vsapi, VSC_TRANSFER_ARIB_B67, "HLG (ARIB STD-B67)");
    if (!bad.empty()) return bad;

    // Unlike src_max on BT2390 these have a default rather than an error.
    // HLG is display-independent by design and most HLG content carries no
    // mastering metadata at all; 1000 cd/m2 is the reference display both
    // BT.2100 and BT.2408 are written around.
    double lw = d->lw;
    double lb = d->lb;
    const char* lwName = "lw";
    const char* lbName = "lb";
    if (!d->haveLw) {
        lwName = "MasteringDisplayMaxLuminance";
        if (!readMasteringLuminance(props, vsapi, lwName, true, &lw)) {
            lw = 1000.0;
            lwName = "lw";
        }
    }
    if (!d->haveLb) {
        lbName = "MasteringDisplayMinLuminance";
        if (!readMasteringLuminance(props, vsapi, lbName, false, &lb)) {
            lb = 0.0;
            lbName = "lb";
        }
    }

    return tonemap::makeHlgParams(lw, lb, d->nominal, lwName, lbName, out);
}

const VSFrame* VS_CC hlgGetFrame(int n, int activationReason, void* instanceData, void**,
                                 VSFrameContext* frameCtx, VSCore* core,
                                 const VSAPI* vsapi) {
    auto* d = static_cast<HlgFilterData*>(instanceData);
    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        return nullptr;
    }
    if (activationReason != arAllFramesReady) return nullptr;

    const VSFrame* src = vsapi->getFrameFilter(n, d->node, frameCtx);

    HlgParams params{};
    const std::string bad =
        resolveHlgParams(d, vsapi->getFramePropertiesRO(src), vsapi, &params);
    if (!bad.empty()) {
        vsapi->setFilterError(("HLG: " + bad).c_str(), frameCtx);
        vsapi->freeFrame(src);
        return nullptr;
    }

    const int width = vsapi->getFrameWidth(src, 0);
    const int height = vsapi->getFrameHeight(src, 0);
    VSFrame* dst = vsapi->newVideoFrame(&d->vi.format, width, height, src, core);

    for (int y = 0; y < height; ++y) {
        const float* srcPlane[3];
        float* dstPlane[3];
        for (int p = 0; p < 3; ++p) {
            const ptrdiff_t srcStride =
                vsapi->getStride(src, p) / static_cast<ptrdiff_t>(sizeof(float));
            const ptrdiff_t dstStride =
                vsapi->getStride(dst, p) / static_cast<ptrdiff_t>(sizeof(float));
            srcPlane[p] =
                reinterpret_cast<const float*>(vsapi->getReadPtr(src, p)) + y * srcStride;
            dstPlane[p] =
                reinterpret_cast<float*>(vsapi->getWritePtr(dst, p)) + y * dstStride;
        }
        tonemap::hlgRow(srcPlane[0], srcPlane[1], srcPlane[2], dstPlane[0], dstPlane[1],
                        dstPlane[2], static_cast<size_t>(width), params);
    }

    VSMap* outProps = vsapi->getFramePropertiesRW(dst);
    vsapi->mapSetInt(outProps, "_Transfer", VSC_TRANSFER_LINEAR, maReplace);
    vsapi->mapSetInt(outProps, "_Primaries", VSC_PRIMARIES_BT2020, maReplace);
    vsapi->mapSetInt(outProps, "_Range", kRangeFull, maReplace);
    // After the OOTF the frame is a display-referred rendering for a display
    // of peak lw, so these describe the frame they are attached to. They are
    // also what lets BT2390 run on the result with no arguments.
    // Both come from params rather than from d, because either may have been
    // read from a frame property rather than given as an argument.
    vsapi->mapSetFloat(outProps, "MasteringDisplayMaxLuminance", params.lw, maReplace);
    vsapi->mapSetFloat(outProps, "MasteringDisplayMinLuminance", params.lb, maReplace);

    vsapi->freeFrame(src);
    return dst;
}

void VS_CC hlgFreeFilter(void* instanceData, VSCore*, const VSAPI* vsapi) {
    auto* d = static_cast<HlgFilterData*>(instanceData);
    vsapi->freeNode(d->node);
    delete d;
}

void VS_CC hlgCreate(const VSMap* in, VSMap* out, void*, VSCore* core,
                     const VSAPI* vsapi) {
    VSNode* node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    auto fail = [&](const std::string& message) {
        vsapi->mapSetError(out, ("HLG: " + message).c_str());
        vsapi->freeNode(node);
    };

    const VSVideoInfo* vi = vsapi->getVideoInfo(node);
    if (!isRgbs(vi)) {
        fail("clip must be RGBS, that is 32-bit float RGB with a constant format "
             "and constant dimensions");
        return;
    }

    auto* d = new HlgFilterData{};
    d->node = node;
    d->vi = *vi;
    d->lw = optionalFloat(in, vsapi, "lw", 1000.0, &d->haveLw);
    d->lb = optionalFloat(in, vsapi, "lb", 0.0, &d->haveLb);
    d->nominal = optionalFloat(in, vsapi, "nominal_luminance", 100.0, nullptr);
    d->simd = optionalBool(in, vsapi, "simd", true);

    // Every check whose inputs are all arguments runs here, so the script
    // fails at evaluation rather than on the first frame. The checks that
    // need a property-derived value wait for resolveHlgParams.
    HlgParams unused{};
    const std::string bad = tonemap::makeHlgParams(
        d->haveLw ? d->lw : 1000.0, d->haveLb ? d->lb : 0.0, d->nominal, "lw", "lb",
        &unused);
    if (!bad.empty()) {
        delete d;
        fail(bad);
        return;
    }

    VSFilterDependency deps[] = {{node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "HLG", &d->vi, hlgGetFrame, hlgFreeFilter, fmParallel,
                             deps, 1, d, core);
}

// --- BT2407 ----------------------------------------------------------------

using tonemap::Chromaticity;
using tonemap::GamutMethod;
using tonemap::GamutParams;
using tonemap::Primaries;
using tonemap::SourceGamut;

struct GamutFilterData {
    VSNode* node;
    VSVideoInfo vi;
    GamutMethod method;
    bool simd;
    double beta;
    SourceGamut srcGamut;
};

// One element of a property, as a float or as an integer. Source filters write
// chromaticities as floats, but a hand-written SetFrameProps call turns a
// whole number into an integer property.
bool readNumber(const VSMap* props, const VSAPI* vsapi, const char* key, int index,
                double* out) {
    int err = 0;
    double value = vsapi->mapGetFloat(props, key, index, &err);
    if (err == peType) {
        value = static_cast<double>(vsapi->mapGetInt(props, key, index, &err));
    }
    if (err != 0) return false;
    *out = value;
    return true;
}

// The mastering display gamut from the frame properties, or false. Anything
// missing, unreadable or failing validation counts as absent, and the caller
// falls back to BT.2020; TonemapSourceGamut is the only signal that happened.
bool masteringPrimaries(const VSMap* props, const VSAPI* vsapi, Primaries* out) {
    if (vsapi->mapNumElements(props, "MasteringDisplayPrimariesX") != 3) return false;
    if (vsapi->mapNumElements(props, "MasteringDisplayPrimariesY") != 3) return false;

    double xs[3];
    double ys[3];
    for (int i = 0; i < 3; ++i) {
        if (!readNumber(props, vsapi, "MasteringDisplayPrimariesX", i, &xs[i])) return false;
        if (!readNumber(props, vsapi, "MasteringDisplayPrimariesY", i, &ys[i])) return false;
    }
    Chromaticity white{};
    if (!readNumber(props, vsapi, "MasteringDisplayWhitePointX", 0, &white.x)) return false;
    if (!readNumber(props, vsapi, "MasteringDisplayWhitePointY", 0, &white.y)) return false;

    const Primaries found = {{xs[0], ys[0]}, {xs[1], ys[1]}, {xs[2], ys[2]}};
    if (!tonemap::validGamut(found, white)) return false;
    *out = found;
    return true;
}

std::string resolveGamutParams(const GamutFilterData* d, const VSMap* props,
                               const VSAPI* vsapi, GamutParams* out) {
    const std::string bad =
        checkFrameTags(props, vsapi, VSC_TRANSFER_LINEAR, "linear light");
    if (!bad.empty()) return bad;

    out->method = d->method;
    out->beta = d->beta;

    Primaries primaries = tonemap::kPrimariesBt2020;
    out->label = "bt2020";
    if (d->srcGamut == SourceGamut::P3D65) {
        primaries = tonemap::kPrimariesP3D65;
        out->label = "p3d65";
    } else if (d->srcGamut == SourceGamut::Auto &&
               masteringPrimaries(props, vsapi, &primaries)) {
        out->label = "mastering";
    }
    // The hard clip has no source gamut, so it reports the default whatever
    // was asked for. Deriving the matrix costs a 3x3 inverse per frame against
    // millions of pixels, so it is not cached: a cache would be shared mutable
    // state in a filter declared parallel, for no measurable gain.
    if (d->method == GamutMethod::Clip) out->label = "bt2020";
    out->xyzToSource = tonemap::inverse(tonemap::rgbToXyz(primaries, tonemap::kD65));
    return std::string();
}

const VSFrame* VS_CC gamutGetFrame(int n, int activationReason, void* instanceData,
                                   void**, VSFrameContext* frameCtx, VSCore* core,
                                   const VSAPI* vsapi) {
    auto* d = static_cast<GamutFilterData*>(instanceData);
    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        return nullptr;
    }
    if (activationReason != arAllFramesReady) return nullptr;

    const VSFrame* src = vsapi->getFrameFilter(n, d->node, frameCtx);

    GamutParams params{};
    const std::string bad =
        resolveGamutParams(d, vsapi->getFramePropertiesRO(src), vsapi, &params);
    if (!bad.empty()) {
        vsapi->setFilterError(("BT2407: " + bad).c_str(), frameCtx);
        vsapi->freeFrame(src);
        return nullptr;
    }

    const int width = vsapi->getFrameWidth(src, 0);
    const int height = vsapi->getFrameHeight(src, 0);
    VSFrame* dst = vsapi->newVideoFrame(&d->vi.format, width, height, src, core);

    for (int y = 0; y < height; ++y) {
        const float* srcPlane[3];
        float* dstPlane[3];
        for (int p = 0; p < 3; ++p) {
            const ptrdiff_t srcStride =
                vsapi->getStride(src, p) / static_cast<ptrdiff_t>(sizeof(float));
            const ptrdiff_t dstStride =
                vsapi->getStride(dst, p) / static_cast<ptrdiff_t>(sizeof(float));
            srcPlane[p] =
                reinterpret_cast<const float*>(vsapi->getReadPtr(src, p)) + y * srcStride;
            dstPlane[p] =
                reinterpret_cast<float*>(vsapi->getWritePtr(dst, p)) + y * dstStride;
        }
        const auto row = d->simd ? tonemap::gamutMapRowSimd : tonemap::gamutMapRow;
        row(srcPlane[0], srcPlane[1], srcPlane[2], dstPlane[0], dstPlane[1], dstPlane[2],
            static_cast<size_t>(width), params);
    }

    VSMap* outProps = vsapi->getFramePropertiesRW(dst);
    // The primaries and white point described the source gamut, which the
    // output no longer has.
    for (const char* key : {"MasteringDisplayPrimariesX", "MasteringDisplayPrimariesY",
                            "MasteringDisplayWhitePointX", "MasteringDisplayWhitePointY"}) {
        vsapi->mapDeleteKey(outProps, key);
    }
    vsapi->mapSetInt(outProps, "_Primaries", VSC_PRIMARIES_BT709, maReplace);
    vsapi->mapSetInt(outProps, "_Transfer", VSC_TRANSFER_LINEAR, maReplace);
    vsapi->mapSetInt(outProps, "_Range", kRangeFull, maReplace);
    vsapi->mapSetData(outProps, "TonemapSourceGamut", params.label,
                      static_cast<int>(std::strlen(params.label)), dtUtf8, maReplace);

    vsapi->freeFrame(src);
    return dst;
}

void VS_CC gamutFreeFilter(void* instanceData, VSCore*, const VSAPI* vsapi) {
    auto* d = static_cast<GamutFilterData*>(instanceData);
    vsapi->freeNode(d->node);
    delete d;
}

// What the plugin chose for this machine. The benchmark log records it, the
// tests check dispatch landed where it should, and a bug report can say which
// kernel ran without anyone having to guess.
//
// `target` restricts dispatch to one of the compiled targets for the rest of
// the process, and an empty string restores the automatic choice. It is there
// so the tests can compare every target the DLL ships against the scalar
// reference, not only the one this machine picks.
void VS_CC infoCreate(const VSMap* in, VSMap* out, void*, VSCore*, const VSAPI* vsapi) {
    int err = 0;
    const char* wanted = vsapi->mapGetData(in, "target", 0, &err);
    if (err == 0 && !tonemap::simdForceTarget(wanted)) {
        vsapi->mapSetError(out, ("Info: this build has no target named " +
                                 std::string(wanted) + " that this machine can run")
                                    .c_str());
        return;
    }

    const std::string version = std::to_string(kVersionMajor) + "." +
                                std::to_string(kVersionMinor) + "." +
                                std::to_string(kVersionPatch);
    vsapi->mapSetData(out, "version", version.c_str(), static_cast<int>(version.size()),
                      dtUtf8, maReplace);

    const char* target = tonemap::simdTargetName();
    vsapi->mapSetData(out, "target", target, static_cast<int>(std::strlen(target)),
                      dtUtf8, maReplace);
    vsapi->mapSetInt(out, "ictcp_float32_lanes", tonemap::ictcpUsesFloatLanes() ? 1 : 0,
                     maReplace);
    vsapi->mapSetInt(out, "double_lanes",
                     static_cast<int64_t>(tonemap::simdDoubleLanes()), maReplace);
    for (const char* name : tonemap::simdTargets()) {
        vsapi->mapSetData(out, "available_targets", name,
                          static_cast<int>(std::strlen(name)), dtUtf8, maAppend);
    }
}

void VS_CC bt2407Create(const VSMap* in, VSMap* out, void*, VSCore* core,
                        const VSAPI* vsapi) {
    VSNode* node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    auto fail = [&](const std::string& message) {
        vsapi->mapSetError(out, ("BT2407: " + message).c_str());
        vsapi->freeNode(node);
    };

    const VSVideoInfo* vi = vsapi->getVideoInfo(node);
    if (!isRgbs(vi)) {
        fail("clip must be RGBS, that is 32-bit float RGB with a constant format "
             "and constant dimensions");
        return;
    }

    auto* d = new GamutFilterData{};
    d->node = node;
    d->vi = *vi;
    d->beta = optionalFloat(in, vsapi, "beta", 0.2, nullptr);
    d->simd = optionalBool(in, vsapi, "simd", true);

    int err = 0;
    const char* method = vsapi->mapGetData(in, "method", 0, &err);
    if (err != 0) method = "softclip";
    const char* gamut = vsapi->mapGetData(in, "src_gamut", 0, &err);
    if (err != 0) gamut = "auto";

    // Both are checked whatever the method, so a typo in one is not swallowed
    // by the other choosing a path that ignores it.
    std::string bad;
    if (!tonemap::parseGamutMethod(method, &d->method)) {
        bad = "method must be one of " + std::string(tonemap::gamutMethodNames()) +
              ", got " + method;
    } else if (!tonemap::parseSourceGamut(gamut, &d->srcGamut)) {
        bad = "src_gamut must be one of " + std::string(tonemap::sourceGamutNames()) +
              ", got " + gamut;
    } else if (!std::isfinite(d->beta) || d->beta < 0.0 || d->beta >= 1.0) {
        bad = "beta must be in [0, 1)";
    }
    if (!bad.empty()) {
        delete d;
        fail(bad);
        return;
    }

    VSFilterDependency deps[] = {{node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "BT2407", &d->vi, gamutGetFrame, gamutFreeFilter,
                             fmParallel, deps, 1, d, core);
}

void VS_CC bt2390Create(const VSMap* in, VSMap* out, void*, VSCore* core,
                        const VSAPI* vsapi) {
    VSNode* node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    auto fail = [&](const std::string& message) {
        vsapi->mapSetError(out, ("BT2390: " + message).c_str());
        vsapi->freeNode(node);
    };

    const VSVideoInfo* vi = vsapi->getVideoInfo(node);
    if (!isRgbs(vi)) {
        fail("clip must be RGBS, that is 32-bit float RGB with a constant format "
             "and constant dimensions");
        return;
    }

    auto* d = new FilterData{};
    d->node = node;
    d->vi = *vi;
    d->srcMin = optionalFloat(in, vsapi, "src_min", 0.0, &d->haveSrcMin);
    d->srcMax = optionalFloat(in, vsapi, "src_max", 0.0, &d->haveSrcMax);
    d->dstMin = optionalFloat(in, vsapi, "dst_min", 0.0, nullptr);
    d->dstMax = optionalFloat(in, vsapi, "dst_max", 203.0, nullptr);
    d->nominal = optionalFloat(in, vsapi, "nominal_luminance", 100.0, nullptr);
    d->simd = optionalBool(in, vsapi, "simd", true);

    int err = 0;
    const char* rep = vsapi->mapGetData(in, "representation", 0, &err);
    if (err != 0) rep = "ictcp";
    if (!tonemap::parseRepresentation(rep, &d->rep)) {
        const std::string message = "representation must be one of " +
                                    std::string(tonemap::representationNames()) +
                                    ", got " + rep;
        delete d;
        fail(message);
        return;
    }

    // Every check whose inputs are all arguments runs here, so the script
    // fails at evaluation rather than on the first frame. Only the checks that
    // need a property-derived value wait, and resolveFrameParams repeats these
    // for free on the way past.
    std::string bad;
    if (!std::isfinite(d->nominal) || d->nominal <= 0.0) {
        bad = "nominal_luminance must be a positive number of cd/m2";
    }
    if (bad.empty()) bad = tonemap::checkTargetRange(d->dstMin, d->dstMax);
    if (bad.empty() && d->haveSrcMin) bad = tonemap::checkLuminance("src_min", d->srcMin);
    if (bad.empty() && d->haveSrcMax) bad = tonemap::checkLuminance("src_max", d->srcMax);
    if (!bad.empty()) {
        delete d;
        fail(bad);
        return;
    }

    // The source pair is only complete here when both came in as arguments.
    if (d->haveSrcMin && d->haveSrcMax) {
        tonemap::Eetf unused{};
        bad = tonemap::makeEetf(d->srcMin, d->srcMax, d->dstMin, d->dstMax, "src_min",
                                   "src_max", &unused);
        if (!bad.empty()) {
            delete d;
            fail(bad);
            return;
        }
    }

    VSFilterDependency deps[] = {{node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "BT2390", &d->vi, getFrame, freeFilter, fmParallel,
                             deps, 1, d, core);
}

}  // namespace

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.vstonemap.plugin", "tonemap",
                         "BT.2390 tone mapping and BT.2407 gamut conversion",
                         VS_MAKE_VERSION(kVersionMajor, kVersionMinor),
                         VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("Info", "target:data:opt;",
                             "version:data;target:data;ictcp_float32_lanes:int;"
                             "double_lanes:int;available_targets:data[];",
                             infoCreate, nullptr, plugin);
    vspapi->registerFunction("BT2407",
                             "clip:vnode;method:data:opt;beta:float:opt;"
                             "src_gamut:data:opt;simd:int:opt;",
                             "clip:vnode;", bt2407Create, nullptr, plugin);
    vspapi->registerFunction("BT2390",
                             "clip:vnode;src_min:float:opt;src_max:float:opt;"
                             "dst_min:float:opt;dst_max:float:opt;"
                             "nominal_luminance:float:opt;representation:data:opt;simd:int:opt;",
                             "clip:vnode;", bt2390Create, nullptr, plugin);
    vspapi->registerFunction("HLG",
                             "clip:vnode;lw:float:opt;lb:float:opt;"
                             "nominal_luminance:float:opt;simd:int:opt;",
                             "clip:vnode;", hlgCreate, nullptr, plugin);
}
