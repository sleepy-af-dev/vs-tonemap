// Runtime-dispatched SIMD kernels for both filters.
//
// Same contract and same results as the scalar kernels in bt2390.cpp and
// bt2407.cpp, which stay as the reference the SIMD path is tested against.

#ifndef TONEMAP_SIMD_H
#define TONEMAP_SIMD_H

#include <cstddef>
#include <vector>

#include "bt2390.h"
#include "bt2407.h"

namespace tonemap {

void toneMapRowSimd(const float* srcR, const float* srcG, const float* srcB,
                    float* dstR, float* dstG, float* dstB, size_t width,
                    Representation rep, const FrameParams& params);

void gamutMapRowSimd(const float* srcR, const float* srcG, const float* srcB,
                     float* dstR, float* dstG, float* dstB, size_t width,
                     const GamutParams& params);

// The Highway target chosen for this machine, for the benchmark log and for
// the test that checks dispatch landed where it should.
const char* simdTargetName();

// The number of double lanes the dispatched kernel runs at.
size_t simdDoubleLanes();

// Whether the ictcp kernel was built with float lanes. Off in the shipped
// build; the float build exists only to measure what double costs.
bool ictcpUsesFloatLanes();

// Every target compiled into this binary that this CPU can run, best first.
// Dispatch takes the first of them.
std::vector<const char*> simdTargets();

// Restrict dispatch to one of those targets, or restore the automatic choice
// when the name is empty. False means this build has no such target. Only the
// tests use it, so that the kernels for the targets this machine does not
// pick are exercised too; it is not thread safe and nothing may be in flight
// when it is called.
bool simdForceTarget(const char* name);

}  // namespace tonemap

#endif  // TONEMAP_SIMD_H
