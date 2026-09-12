// vs-tonemapper: BT.2390 tone mapping and BT.2407 gamut conversion for VapourSynth.

#include <VSConstants4.h>
#include <VSHelper4.h>
#include <VapourSynth4.h>

#include <cmath>
#include <string>

#include "bt2390.h"

namespace {

using tonemapper::FrameParams;
using tonemapper::Representation;

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

std::string checkFrameTags(const VSMap* props, const VSAPI* vsapi) {
    std::string bad = checkTag(props, vsapi, "_Transfer", VSC_TRANSFER_LINEAR, "linear light");
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
    std::string bad = checkFrameTags(props, vsapi);
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
    return tonemapper::makeEetf(srcMin, srcMax, d->dstMin, d->dstMax, minName, maxName,
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
        tonemapper::toneMapRow(srcPlane[0] + y * srcStride[0], srcPlane[1] + y * srcStride[1],
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

double optionalFloat(const VSMap* in, const VSAPI* vsapi, const char* key,
                     double fallback, bool* present) {
    int err = 0;
    const double value = vsapi->mapGetFloat(in, key, 0, &err);
    if (present != nullptr) *present = err == 0;
    return err == 0 ? value : fallback;
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

    int err = 0;
    const char* rep = vsapi->mapGetData(in, "representation", 0, &err);
    if (err != 0) rep = "ictcp";
    if (!tonemapper::parseRepresentation(rep, &d->rep)) {
        const std::string message = "representation must be one of " +
                                    std::string(tonemapper::representationNames()) +
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
    if (bad.empty()) bad = tonemapper::checkTargetRange(d->dstMin, d->dstMax);
    if (bad.empty() && d->haveSrcMin) bad = tonemapper::checkLuminance("src_min", d->srcMin);
    if (bad.empty() && d->haveSrcMax) bad = tonemapper::checkLuminance("src_max", d->srcMax);
    if (!bad.empty()) {
        delete d;
        fail(bad);
        return;
    }

    // The source pair is only complete here when both came in as arguments.
    if (d->haveSrcMin && d->haveSrcMax) {
        tonemapper::Eetf unused{};
        bad = tonemapper::makeEetf(d->srcMin, d->srcMax, d->dstMin, d->dstMax, "src_min",
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
    vspapi->configPlugin("com.vstonemapper.plugin", "tonemapper",
                         "BT.2390 tone mapping and BT.2407 gamut conversion",
                         VS_MAKE_VERSION(0, 1), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("BT2390",
                             "clip:vnode;src_min:float:opt;src_max:float:opt;"
                             "dst_min:float:opt;dst_max:float:opt;"
                             "nominal_luminance:float:opt;representation:data:opt;",
                             "clip:vnode;", bt2390Create, nullptr, plugin);
}
