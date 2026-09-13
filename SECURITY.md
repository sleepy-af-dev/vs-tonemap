# Security policy

## Supported versions

The plugin is pre-release, so `main` is the only supported code and fixes land
there. Once releases begin, the latest release only.

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue. Use
GitHub's [private vulnerability reporting](https://github.com/sleepy-af-dev/vs-tonemap/security/advisories/new)
to open a confidential advisory, and you'll get a response as soon as possible.

The plugin is a DLL loaded into a VapourSynth process, and it parses no
container or bitstream of its own: it reads decoded frames a source filter has
already produced. The parts worth looking at are the frame and format handling
in `src/plugin.cpp`, where a malformed or unexpected clip property reaches the
filter, and the SIMD kernels, where a frame dimension or stride could drive a
read or write past the end of a plane.
