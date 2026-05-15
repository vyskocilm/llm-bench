# Optimized llama.cpp for Radeon 860M (gfx1151)

## Goal

Build optimized `llama.cpp` (server + CLI) for **localhost**: Ryzen AI 7 PRO 350,
Radeon 860M iGPU (gfx1151 / Krackan Point), 64 GiB RAM.

Backend: **Vulkan via RADV**. ROCm is broken on this GPU — see memory
`ollama_gpu_stack`: rocBLAS bundled in upstream images has no Tensile kernels
for gfx1150/1151.

Repo name is `ollama` because the prior setup wrapped upstream Ollama. This is
not a concern and can be ignored.

## Host constraints

- **Immutable Tumbleweed.** No host package install. Build + run inside an
  artifact (Docker image preferred — see decision below).
- **Dev env:** distrobox rootless podman. `/dev/dri` visible. Only `runc` is
  available; it ignores `--group-add=keep-groups`, so the supplementary
  `render`/`video` gids cannot be propagated via that flag. Work around it
  with `--userns=host` on the container — the container process then *is* the
  host user (uid + all supplementary gids), so `/dev/dri/renderD*` is
  accessible naturally. Tradeoff: no uid remapping, less isolation; fine for
  a localhost-only iGPU service.

## CPU features available (relevant subset)

`avx512f avx512_bf16 avx_vnni avx512_vnni avx512_vbmi avx512_vbmi2
avx512_bitalg avx512_vpopcntdq sha_ni`

Build with `-DGGML_NATIVE=ON` (or explicit `-march=znver5` once cmake/clang
knows it; fallback `-march=native` inside the container — caveat: image then
non-portable, which is fine here, single-host artifact).

## Decision: Docker image, not static binary

Vulkan path needs:
- `libvulkan.so` (loader, dynamic — by design)
- RADV ICD JSON + `libvulkan_radeon.so` from host mesa
- libdrm

Fully static binary impossible without shipping a mesa userspace stack
ourselves. Docker image bundles mesa + RADV from a known distro, mounts
`/dev/dri` from host. The `llama.container` quadlet wires this together for
rootless systemd-user.

Static `llama-cli` for **CPU-only** is feasible (musl + `-DGGML_VULKAN=OFF`)
but defeats the goal of using the iGPU. Skip unless explicitly needed.

## Components (as built)

| Path                | Role                                                                                                             |
|---------------------|------------------------------------------------------------------------------------------------------------------|
| `build.sh`          | Phase 1. Inside distrobox: probe required zypper pkgs, clone llama.cpp, `cmake -G Ninja` w/ `-DGGML_VULKAN=ON -DGGML_NATIVE=ON -DGGML_LTO=ON -DBUILD_SHARED_LIBS=OFF`, stage binaries into `dist/bin/`. `LLAMA_CURL` is auto-detected via `libcurl-devel` — the explicit flag is deprecated/ignored. |
| `Dockerfile`        | Phase 2. Runtime image *only*; no compilation. `FROM registry.opensuse.org/opensuse/tumbleweed:latest`. Installs `libvulkan1`, `libvulkan_radeon`, `libgomp1`, `libssl60`, `libcrypto57`, `libcurl4`, `ca-certificates`. Copies `dist/bin/*`. Entrypoint `llama-server`, exposes 8080. |
| `llama.container`   | Phase 3. Rootless user quadlet (`~/.config/containers/systemd/`). `PublishPort=127.0.0.1:8080:8080`, `AddDevice=/dev/dri`, `Environment=AMD_VULKAN_ICD=RADV`, `PodmanArgs=--userns=host --security-opt=label=disable`. Bind-mounts `models/` read-only. |
| `pull-model.sh`     | Phase 4. Fetches GGUF into `./models/`, verifies sha256 against the HF LFS `X-Linked-Etag` header. Default: **Qwen2.5-3B-Instruct Q4_K_M** (~1.9 GiB). Override via `MODEL=` / `FILE=` env. |
| `README.md`         | Phase 5. Build / image / model / quadlet / smoke-test / clients / benchmark walk-through.                        |
| `bench/bench.py`    | Phase 6. 5-task reproducible benchmark across OSS-licensed (Apache-2.0 / MIT) models — see `bench/README.md`. Spawns `dist/bin/llama-server` per model on port 18080, scores via `/v1/chat/completions`. |

## Gotchas learned during build

- **Vulkan_LIBRARY not found**: openSUSE splits the loader — `libvulkan1`
  ships only the SONAME, `vulkan-devel` ships the `libvulkan.so` symlink
  CMake's FindVulkan needs. Both required at build time.
- **`spirv/unified1/spirv.hpp` missing**: needs the upstream `spirv-headers`
  package, *not* `libLLVMSPIRVLib-devel` (LLVM's SPIRV translator — different
  thing, wrong include layout).
- **LibreSSL on Tumbleweed**: current `libcurl4` links against LibreSSL 4.x
  with SONAMEs `libssl.so.60` / `libcrypto.so.57`, *not* OpenSSL 3
  (`libssl.so.3`). Runtime image needs `libssl60` + `libcrypto57`; do not
  substitute `libopenssl3`.
- **SELinux on bind mounts**: rootless podman labels project directories
  with the container SELinux context unless `:Z` (relabel) is given or
  `--security-opt=label=disable` is set. The quadlet uses the latter.
- **Device string**: at runtime RADV reports `AMD Radeon 860M Graphics
  (RADV KRACKAN1)` — *not* "GFX1152" as it appears in some older memory
  notes. Search journal for `RADV KRACKAN1`.

## Success criteria

- [x] llama-server starts, reports `Vulkan0 (AMD Radeon 860M Graphics (RADV
      KRACKAN1))`, loads Qwen2.5-3B Q4_K_M with all layers on GPU.
- [x] `curl :8080/completion` and `:8080/v1/chat/completions` return tokens.
      Measured Qwen2.5-3B Q4_K_M baseline: **gen ≈ 37.6 tok/s, prompt ≈ 29.3
      tok/s** (2026-05-11, llama.cpp commit `838374375c`).
- [ ] Quadlet auto-starts on boot; survives reboot. (User-installable; not
      yet activated in this session.)

## Out of scope

- Multi-GPU.
- ROCm path (revisit only if upstream rocBLAS ships gfx1151 kernels).
- Training / fine-tuning.
- Quantization tooling (download pre-quantized GGUF only).

## LLM Benchmark Suite (GGUF / llama.cpp) - May 2026

| Model Family | License | Size | Suggested Quants | Notes / Benchmark Target |
| :--- | :--- | :--- | :--- | :--- |
| **Gemma 4** | Apache 2.0 | 4B | `Q8_0`, `FP16` | Max speed, iGPU cache efficiency |
| **Gemma 4** | Apache 2.0 | 9B | `Q4_K_M`, `Q6_K` | General reasoning baseline |
| **Gemma 4** | Apache 2.0 | 27B | `IQ3_M`, `Q4_K_S` | Memory bandwidth stress test |
| **DeepSeek-V4** | MIT | 7B | `Q4_K_M`, `Q8_0` | Logic & Coding (Dense model) |
| **DeepSeek-V4** | MIT | 14B (MoE) | `Q4_K_M`, `Q5_K_M` | Mixture-of-Experts (NPU/CPU test) |
| **Qwen 3.5** | Apache 2.0 | 7B | `Q4_K_M`, `Q6_K` | Balanced all-rounder |
| **Qwen 3.5** | Apache 2.0 | 32B | `IQ2_M`, `Q3_K_L` | VRAM / LPDDR5x bottleneck test |
| **Mistral NeMo v2**| Apache 2.0 | 12B | `Q4_K_M`, `Q8_0` | Context window (128k+) scalability |
| **Phi-4 Med** | MIT | 14B | `Q4_K_M`, `FP16` | High density benchmark |
| **Llama-3.3-Open**| Apache 2.0* | 8B | `Q4_K_M`, `Q8_0` | Standard industry comparison |
| **Llama-3.3-Open**| Apache 2.0* | 70B | `IQ2_XS`, `Q2_K` | Extreme swap/offload test |

Tests with `-ngl 99` and variable threading `-t 8` or `t 10` and batching `-b 512` and `-ub 1024`

### Empirically resolved against HuggingFace (May 2026)

The table above is aspirational; what HF actually has + how the strict-OSI
gate resolved each row:

| Suite row | HF resolution | In bench.py |
|-----------|---------------|-------------|
| Gemma 4 4B/9B/27B | `google/gemma-4-E2B/E4B/26B-A4B/31B-it`; `cardData.license: apache-2.0`, `license_link` = Google docs page containing **verbatim** Apache 2.0 text (no AUP). Genuine OSS relicence. | `gemma-4-e2b` (E2B-it Q4_K_M, 2.9 GiB) added |
| DeepSeek-V4 7B/14B | Repo doesn't exist (`deepseek-ai/DeepSeek-V4*` → 401). DeepSeek-V3 (the closest released sibling) ships dual `LICENSE-CODE` (MIT) + `LICENSE-MODEL` (DeepSeek License with use-based restrictions) — the **weights** licence has AUP, not OSS. | skipped (model not released) |
| Qwen 3.5 7B/32B | Repo doesn't exist (`Qwen/Qwen3.5-*` → 401). Qwen3 (without .5) is genuinely Apache 2.0. | `qwen3-8b` substituted |
| Mistral NeMo v2 12B | `mistralai/Mistral-Nemo-v2-*` 401; closest released sibling `mistralai/Mistral-Nemo-Instruct-2407` is Apache 2.0. | `mistral-nemo-2407` substituted |
| Phi-4 Med 14B | `microsoft/Phi-4-medium*` 401; base `microsoft/phi-4` (14B) is MIT. | `phi-4` substituted |
| Llama-3.3-Open 8B/70B | Llama Community License has 700M-MAU clause → OSD violation. The `*` in the suite table flagged this. | skipped per strict-OSI |

Note: Gemma 1/2/3 carry HF metadata `license: gemma` (Gemma Terms of Use,
with AUP). Only **Gemma 4** has been relicensed to genuine Apache 2.0 —
don't backport the assumption.
