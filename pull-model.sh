#!/usr/bin/env bash
# Fetch GGUF model into ./models/ with integrity check.
#
# Default: Qwen2.5-3B-Instruct Q4_K_M (~1.9 GiB), small enough for fast smoke tests
# of the Vulkan/RADV stack while still exercising real attention/matmul kernels.
#
# Override by env, e.g.:
#   MODEL=Qwen/Qwen2.5-7B-Instruct-GGUF FILE=qwen2.5-7b-instruct-q4_k_m.gguf ./pull-model.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${ROOT}/models"

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct-GGUF}"
FILE="${FILE:-qwen2.5-3b-instruct-q4_k_m.gguf}"
URL="https://huggingface.co/${MODEL}/resolve/main/${FILE}?download=true"

mkdir -p "${MODELS_DIR}"
DEST="${MODELS_DIR}/${FILE}"

# HF doesn't put a checksum in a sidecar file, but it does expose one in the
# JSON metadata endpoint. We compute sha256 locally after download and compare
# against the value HF returns from the LFS pointer (X-Linked-Etag header on
# the resolve URL is the LFS sha256 hex).
echo "Fetching HF LFS metadata for ${MODEL}/${FILE}..."
EXPECTED_SHA="$(curl -fsSL -I "${URL}" | awk -F'"' '/^[Xx]-[Ll]inked-[Ee]tag/ {print $2; exit}')"
if [[ -z "${EXPECTED_SHA}" ]]; then
  echo "WARN: could not extract expected sha256 from HF headers; proceeding without verification." >&2
fi

if [[ -f "${DEST}" ]]; then
  if [[ -n "${EXPECTED_SHA}" ]]; then
    HAVE_SHA="$(sha256sum "${DEST}" | awk '{print $1}')"
    if [[ "${HAVE_SHA}" == "${EXPECTED_SHA}" ]]; then
      echo "OK: ${DEST} already present and matches sha256."
      exit 0
    fi
    echo "Existing file checksum mismatch — re-downloading." >&2
  else
    echo "OK: ${DEST} present (no checksum to verify against)."
    exit 0
  fi
fi

echo "Downloading ${FILE} -> ${DEST} ..."
# -C - resumes if partial.
curl -fL --retry 5 --retry-delay 2 -C - -o "${DEST}.part" "${URL}"
mv "${DEST}.part" "${DEST}"

if [[ -n "${EXPECTED_SHA}" ]]; then
  echo "Verifying sha256..."
  HAVE_SHA="$(sha256sum "${DEST}" | awk '{print $1}')"
  if [[ "${HAVE_SHA}" != "${EXPECTED_SHA}" ]]; then
    echo "FATAL: sha256 mismatch" >&2
    echo "  expected: ${EXPECTED_SHA}" >&2
    echo "  got:      ${HAVE_SHA}" >&2
    exit 2
  fi
  echo "OK: sha256 verified (${HAVE_SHA})."
fi

ls -lh "${DEST}"
