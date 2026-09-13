cmake_minimum_required(VERSION 3.24)
project(vs-tonemap LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# The ictcp kernel can be built with float lanes instead of double, to
# measure what the precision decision costs. It is off in the shipped build
# and exists only so the decision rests on a number rather than a guess.
option(TONEMAP_FLOAT32_ICTCP "build the ictcp kernel with float lanes" OFF)

# Link the C and C++ runtimes statically. The plugin is distributed as one DLL
# that someone drops into an autoload directory, and a dynamic CRT makes that
# fail on any machine without the Microsoft Visual C++ redistributable, with an
# error that says nothing useful. Set before FetchContent so Highway and SLEEF
# are built the same way; mixing runtimes across static libraries does not link.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

include(FetchContent)

# Google Highway: portable SIMD with one binary per instruction set and
# runtime dispatch. Contrib, tests and examples are not needed.
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(highway
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG 1.4.0
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(highway)

# SLEEF: vector math. Only its pow is used, from the static library, whose
# per-instruction-set objects carry their own compiler flags. src/simd.cpp
# declares the entry points it needs, because the header hides them behind
# macros Highway never sets. Nothing else SLEEF builds is wanted.
set(SLEEF_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(SLEEF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SLEEF_BUILD_DFT OFF CACHE BOOL "" FORCE)
set(SLEEF_BUILD_GNUABI_LIBS OFF CACHE BOOL "" FORCE)
set(SLEEF_BUILD_QUAD OFF CACHE BOOL "" FORCE)
set(SLEEF_BUILD_SCALAR_LIB OFF CACHE BOOL "" FORCE)
FetchContent_Declare(sleef
  GIT_REPOSITORY https://github.com/shibatch/sleef.git
  GIT_TAG 3.9.0
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(sleef)

add_library(vs-tonemap MODULE src/plugin.cpp src/bt2390.cpp src/bt2407.cpp src/simd.cpp)
set_target_properties(vs-tonemap PROPERTIES
  PREFIX ""
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
# src is on the include path because Highway re-includes simd.cpp once per
# target from inside its own header; without it the include only resolves
# through Microsoft's non-standard search rules, which clang warns about.
target_include_directories(vs-tonemap PRIVATE
  src include/vapoursynth ${sleef_BINARY_DIR}/include)
target_link_libraries(vs-tonemap PRIVATE hwy sleef)
if(TONEMAP_FLOAT32_ICTCP)
  target_compile_definitions(vs-tonemap PRIVATE TONEMAP_FLOAT32_ICTCP=1)
endif()
target_compile_options(vs-tonemap PRIVATE
  $<$<CXX_COMPILER_ID:Clang,AppleClang,GNU>:-Wall -Wextra -Wconversion -Wno-unused-parameter>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>)
