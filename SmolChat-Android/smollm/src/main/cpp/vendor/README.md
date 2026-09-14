# Vendored build-time dependencies

`vulkan-hpp/` and `spirv-headers/` are used only by the Vulkan variant
(`build_library_vulkan()` in `CMakeLists.txt`, gated behind
`-DSMOLLM_BUILD_VULKAN_VARIANT=ON`). `opencl-headers/` and
`opencl-icd-loader/` are used only by the OpenCL variant (gated behind
`-DSMOLLM_BUILD_OPENCL_VARIANT=ON`). None of these are needed by, or
referenced from, the default CPU-only build.

## vulkan-hpp/

The Khronos C++ bindings for Vulkan (`vulkan/vulkan.hpp`), which
`ggml-vulkan.cpp` includes directly but the Android NDK does not ship (only
the plain C `vulkan/vulkan.h` is NDK-provided).

- Source: https://github.com/KhronosGroup/Vulkan-Hpp
- Pinned commit: `1a24b015830c116632a0723f3ccfd1f06009ce12` (tag `v1.3.275`)
- **Must match this project's NDK's `VK_HEADER_VERSION`** (see
  `<ndk>/toolchains/llvm/prebuilt/<host>/sysroot/usr/include/vulkan/vulkan_core.h`).
  A newer Vulkan-Hpp targets a newer core spec than an older NDK's headers
  provide, which fails to compile (types promoted from `*_EXT`/`*_KHR` to
  core between versions). If the NDK version changes, re-check this pin.
- Pruned to the exact transitive `#include <vulkan/...>` closure of
  `vulkan.hpp` (traced directly, not guessed): `vulkan.hpp`,
  `vulkan_hpp_macros.hpp`, `vulkan_enums.hpp`, `vulkan_to_string.hpp`,
  `vulkan_handles.hpp`, `vulkan_structs.hpp`, `vulkan_funcs.hpp`. The full
  upstream `vulkan/` directory also includes unused RAII wrappers, hashing,
  format-traits, and the unrelated Vulkan SC variant (~18 MiB total vs. ~9.5
  MiB for this subset).

## spirv-headers/

Provides `SPIRV-Headers::SPIRV-Headers`, which `ggml-vulkan/CMakeLists.txt`
requires via `find_package(SPIRV-Headers CONFIG REQUIRED)`.

- Source: https://github.com/KhronosGroup/SPIRV-Headers
- Pinned commit: `496543121ce6419f23d6fa5d7194ba66c36212d2`
- `include/spirv/unified1/spirv.hpp` is the one header `ggml-vulkan.cpp`
  actually includes (self-contained; verified it needs no other file in
  this repo).
- `share/cmake/SPIRV-Headers/*.cmake` are the two files
  `cmake --install` normally generates for this package (upstream's own
  `CMakeLists.txt` never registers a build-tree package, so `find_package`
  only works against an installed prefix). Pre-generated once and checked in
  here instead of adding a nested install step to this project's build;
  they're self-contained and locate their own `include/` via a path
  relative to their own location, so this directory can be freely moved/
  copied as a unit.

## opencl-headers/

Provides `CL/*.h` (`ggml-opencl.cpp` includes `CL/cl.h` etc.), which the NDK
does not ship. `ggml-opencl/CMakeLists.txt` finds these via
`find_package(OpenCL REQUIRED)`, pointed at this directory's `CL/` by
`OpenCL_INCLUDE_DIR` (see `CMakeLists.txt`).

- Source: https://github.com/KhronosGroup/OpenCL-Headers
- Pinned commit: `c4c8fd6f9556c92b212308880854e6294d61b314`
- Only `CL/` is vendored (exactly the directory upstream's own Android build
  instructions in `docs/backend/OPENCL.md` say to copy).

## opencl-icd-loader/

Android provides no NDK-bundled `libOpenCL.so` (unlike Vulkan, which has an
NDK stub from API 28) -- a device's actual OpenCL driver is a vendor/OEM
library, discovered and dlopen'd at runtime by an ICD (Installable Client
Driver) loader. `libOpenCL.so` here is that loader (linked by
`ggml-opencl.cpp` via `find_package(OpenCL REQUIRED)`, pointed at this file
by `OpenCL_LIBRARY` -- see `CMakeLists.txt`) -- it's the loader shim, not a
vendor driver; it still needs a real OpenCL driver present on-device at
runtime to actually do anything.

- Source: https://github.com/KhronosGroup/OpenCL-ICD-Loader
- Built from pinned commit `f27c925e782499eebc4df20e121144358ccd5ac6`,
  arm64-v8a, `ANDROID_PLATFORM=28`, `ANDROID_STL=c++_shared`,
  `OPENCL_ICD_LOADER_HEADERS_DIR` pointed at `opencl-headers/` above --
  exactly the invocation `docs/backend/OPENCL.md` documents for Android,
  run once in a standalone scratch build (see conversation) and the
  resulting `libOpenCL.so` checked in here as a prebuilt binary, the same
  way `spirv-headers/share/cmake/` above vendors pre-generated output
  instead of adding a nested build step to this project's own CMake
  configure pass -- source-vendoring it like `opencl-headers/` would need
  its own `find_package(OpenCL)` call satisfied to build itself, a
  circularity not worth taking on for a loader shim that never changes
  unless this pin is bumped.
