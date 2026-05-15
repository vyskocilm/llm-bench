# Runtime image for prebuilt llama.cpp binaries (Vulkan/RADV backend).
# NEVER compile in this stage — phase 1 (build.sh, inside distrobox) produces
# ./dist/bin/* on the host; this image only stages runtime deps + binaries.
FROM registry.opensuse.org/opensuse/tumbleweed:latest

# Runtime deps:
#   libvulkan1         - Vulkan loader (provides libvulkan.so.1)
#   libvulkan_radeon   - RADV ICD for gfx1151 (provides JSON + libvulkan_radeon.so)
#   libgomp1           - OpenMP runtime used by ggml-cpu
#   libssl60 / libcrypto57 - LibreSSL 4.x; current Tumbleweed libcurl links
#                            against this, *not* libopenssl3 (SONAMEs are
#                            libssl.so.60 / libcrypto.so.57, not .so.3).
#   libcurl4               - HTTPS model fetch from HF (`-hf` / `--hf-repo`)
#   ca-certificates        - HTTPS trust roots
RUN zypper --non-interactive --gpg-auto-import-keys ref \
 && zypper --non-interactive in --no-recommends \
        libvulkan1 \
        libvulkan_radeon \
        libgomp1 \
        libssl60 \
        libcrypto57 \
        libcurl4 \
        ca-certificates \
 && zypper clean -a \
 && rm -rf /var/cache/zypp/* /var/log/zypp/*

COPY dist/bin/llama-server /usr/local/bin/llama-server
COPY dist/bin/llama-cli    /usr/local/bin/llama-cli
COPY dist/bin/llama-bench  /usr/local/bin/llama-bench
COPY dist/BUILDINFO        /etc/llama.buildinfo

# Models mounted from host; keep mountpoint stable for the quadlet.
VOLUME ["/models"]

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/llama-server"]
CMD ["--host", "0.0.0.0", "--port", "8080", "--alias", "qwen2.5-3b"]
