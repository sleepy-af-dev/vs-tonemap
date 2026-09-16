// The HLG decode of ITU-R BT.2100-3 Table 5: the black level lift, the
// inverse OETF and the luminance-driven OOTF, which together are the HLG
// Reference EOTF.
//
// HLG is scene-referred, so the signal states a fraction of the light that
// fell on the sensor rather than a display luminance. The OOTF is what turns
// one into the other, and it is driven by scene luminance, so it touches all
// three channels at once. Note 5e records that some legacy displays raise
// each channel separately instead and calls that an approximation of this
// reference; a per-channel transfer function cannot express what is here.
//
// Source: ITU-R BT.2100-3 Table 5 and Notes 5a to 5i.

#ifndef TONEMAP_HLG_H
#define TONEMAP_HLG_H

#include <cstddef>
#include <string>

// colour.h is deliberately NOT included here. Nothing this header declares
// uses it: the params are plain doubles and the row kernel takes floats.
// hlg.cpp includes it directly for checkLuminance and the kKr/kKg/kKb
// coefficients. That differs from bt2390.h, whose colour.h include is
// earned, since FrameParams holds an Eetf.

namespace tonemap {

// Note 5b defines b = 1 - 4a and c = 0.5 - a ln(4a); Note 5c prints both as
// the decimals below. The printed decimals are used so this and the float64
// reference agree bit for bit.
inline constexpr double kHlgA = 0.17883277;
inline constexpr double kHlgB = 0.28466892;
inline constexpr double kHlgC = 0.55991073;

// Note 5f scopes its first gamma formula to the usual production monitoring
// range and directs the extended formula outside it.
inline constexpr double kHlgGammaLo = 400.0;
inline constexpr double kHlgGammaHi = 2000.0;
inline constexpr double kHlgKappa = 1.111;

// Note 5f. The two formulae agree at 1000 and differ by about 1% at the ends
// of the production range, so the switch is a small discontinuity. It is the
// spec's own and is not smoothed over.
double hlgSystemGamma(double lw);

// Table 5's beta, the user black level lift. The radical covers the whole
// expression: beta = sqrt(3 (LB/LW)^(1/gamma)). That is the reading under
// which the EOTF returns exactly L_B for a signal of 0.
double hlgBlackLift(double lw, double lb, double gamma);

struct HlgParams {
    double gamma;       // the system gamma, Note 5f
    double beta;        // the black level lift, Table 5
    double lw;          // alpha of Table 5, the display peak in cd/m2
    double lb;          // L_B, echoed onto the output for BT2390's src_min
    double invNominal;  // 1 / nominal_luminance, applied on the way out
};

// Fills out and returns an empty string, or leaves it alone and returns the
// reason. No exceptions cross the VapourSynth C boundary. lwName and lbName
// are what the message calls the two luminances, so a value that came from a
// frame property is reported under the property's name rather than under the
// argument it stood in for.
std::string makeHlgParams(double lw, double lb, double nominal, const char* lwName,
                          const char* lbName, HlgParams* out);

// One row of an RGBS frame. The three input pointers and the three output
// pointers address the same row of the R, G and B planes; in-place is fine.
void hlgRow(const float* srcR, const float* srcG, const float* srcB, float* dstR,
            float* dstG, float* dstB, size_t width, const HlgParams& params);

}  // namespace tonemap

#endif  // TONEMAP_HLG_H
