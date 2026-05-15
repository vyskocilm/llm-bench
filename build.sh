#!/usr/bin/env bash
# Build llama.cpp with Vulkan backend inside distrobox Tumbleweed container.
# Outputs binaries to ./dist/ for staging into runtime Docker image.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${ROOT}/llama.cpp"
BUILD="${SRC}/build"
DIST="${ROOT}/dist"
LLAMA_REF="${LLAMA_REF:-master}"

# Required openSUSE Tumbleweed packages.
#   cmake/ninja/gcc-c++ - toolchain
#   glslang  - glslangValidator (SPIR-V compiler used by ggml-vulkan)
#   shaderc  - glslc (fallback shader compiler)
#   vulkan-headers / vulkan-tools / libvulkan1 - Vulkan SDK + loader + vulkaninfo
#   libcurl-devel - LLAMA_CURL=ON for model downloads
#   pkgconf-pkg-config - CMake find_package for curl/vulkan
#   libgomp1 - OpenMP runtime (header comes with gcc-c++)
REQUIRED_PKGS=(
  cmake
  ninja
  gcc-c++
  git
  glslang-devel
  shaderc
  spirv-headers
  vulkan-headers
  vulkan-devel
  vulkan-tools
  libvulkan1
  libcurl-devel
  pkgconf-pkg-config
  libgomp1
)

# Sanity: refuse to run outside an openSUSE container so we don't wreck a host.
if ! command -v rpm >/dev/null 2>&1; then
  echo "FATAL: no rpm — run inside the openSUSE distrobox container." >&2
  exit 2
fi

missing=()
for pkg in "${REQUIRED_PKGS[@]}"; do
  rpm -q --whatprovides "${pkg}" >/dev/null 2>&1 || missing+=("${pkg}")
done

if (( ${#missing[@]} > 0 )); then
  echo "Missing packages:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  echo >&2
  echo "Install on the host (Tumbleweed), then re-enter distrobox:" >&2
  echo "  sudo zypper in ${missing[*]}" >&2
  exit 1
fi

# Tooling sanity — binary names openSUSE ships under.
for bin in cmake ninja g++ git glslangValidator glslc vulkaninfo pkg-config; do
  command -v "${bin}" >/dev/null 2>&1 || {
    echo "FATAL: ${bin} not on PATH despite package present." >&2
    exit 1
  }
done

# Vulkan device probe — warn, do not fail (build is host-portable artifact prep).
if vulkaninfo --summary 2>/dev/null | grep -q 'GFX115'; then
  echo "Vulkan: gfx115x device visible — runtime should work."
else
  echo "WARN: vulkaninfo did not show a gfx115x device. Build proceeds; runtime needs /dev/dri + render group." >&2
fi

# Fetch source.
if [[ ! -d "${SRC}/.git" ]]; then
  echo "Cloning llama.cpp..."
  git clone --depth=1 --branch "${LLAMA_REF}" https://github.com/ggml-org/llama.cpp "${SRC}"
else
  echo "Updating llama.cpp (ref=${LLAMA_REF})..."
  git -C "${SRC}" fetch --depth=1 origin "${LLAMA_REF}"
  git -C "${SRC}" checkout FETCH_HEAD
fi

# Configure.
# Notes:
#  - GGML_NATIVE=ON enables -march=native; binary becomes non-portable (intentional, single-host).
#  - libcurl-devel present → curl is auto-enabled (LLAMA_CURL flag is deprecated/ignored).
#  - BUILD_SHARED_LIBS=OFF keeps ggml/llama as static libs inside the binary;
#    only vulkan/curl/system libs remain dynamic — easier Docker staging.
cmake -S "${SRC}" -B "${BUILD}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DGGML_NATIVE=ON \
  -DGGML_LTO=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DBUILD_SHARED_LIBS=OFF

cmake --build "${BUILD}" -j "$(nproc)" --target llama-server llama-cli llama-bench

# Stage artifacts.
rm -rf "${DIST}"
mkdir -p "${DIST}/bin"
for b in llama-server llama-cli llama-bench; do
  install -Dm0755 "${BUILD}/bin/${b}" "${DIST}/bin/${b}"
done

# Record what we built for Dockerfile + debugging.
{
  echo "llama.cpp commit: $(git -C "${SRC}" rev-parse HEAD)"
  echo "built: $(date -Iseconds)"
  echo "host:  $(uname -srvm)"
  echo "cpu:   $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
} > "${DIST}/BUILDINFO"

echo
echo "OK. Artifacts:"
ls -la "${DIST}/bin"
echo
echo "Next: build the runtime Docker image with these binaries."
