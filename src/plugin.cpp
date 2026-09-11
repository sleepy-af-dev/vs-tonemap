// vs-tonemapper: BT.2390 tone mapping and BT.2407 gamut conversion for VapourSynth.

#include <VapourSynth4.h>
#include <VSHelper4.h>

namespace {

struct FilterData {
    VSNode *node;
    VSVideoInfo vi;
};

const VSFrame *VS_CC getFrame(int n, int activationReason, void *instanceData, void **,
                              VSFrameContext *frameCtx, VSCore *, const VSAPI *vsapi) {
    auto *d = static_cast<FilterData *>(instanceData);
    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        // Phase 0 passthrough: the source frame is returned unchanged.
        return vsapi->getFrameFilter(n, d->node, frameCtx);
    }
    return nullptr;
}

void VS_CC freeFilter(void *instanceData, VSCore *, const VSAPI *vsapi) {
    auto *d = static_cast<FilterData *>(instanceData);
    vsapi->freeNode(d->node);
    delete d;
}

bool isRGBS(const VSVideoInfo *vi) {
    return vsh::isConstantVideoFormat(vi) && vi->format.colorFamily == cfRGB &&
           vi->format.sampleType == stFloat && vi->format.bitsPerSample == 32;
}

void VS_CC bt2390Create(const VSMap *in, VSMap *out, void *, VSCore *core, const VSAPI *vsapi) {
    VSNode *node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    const VSVideoInfo *vi = vsapi->getVideoInfo(node);
    if (!isRGBS(vi)) {
        vsapi->mapSetError(out, "BT2390: clip must be RGBS (32-bit float RGB, constant format)");
        vsapi->freeNode(node);
        return;
    }
    auto *d = new FilterData{node, *vi};
    VSFilterDependency deps[] = {{node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "BT2390", &d->vi, getFrame, freeFilter, fmParallel, deps, 1, d, core);
}

} // namespace

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin("com.vstonemapper.plugin", "tonemapper",
                         "BT.2390 tone mapping and BT.2407 gamut conversion",
                         VS_MAKE_VERSION(0, 1), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("BT2390", "clip:vnode;", "clip:vnode;", bt2390Create, nullptr, plugin);
}
