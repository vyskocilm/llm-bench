# Models used in this bench

The `models/` directory is **not in the repo** (combined size ~110 GiB). Use this manifest to re-download what you need.

All models are **Apache-2.0 or MIT** at the time of selection — see `bench/REPORT.md` "Empirically resolved against HuggingFace" for the licence resolution per family.

## Download

For each row below, the GGUF is at `https://huggingface.co/<source-repo>/resolve/main/<filename>`.

The included `pull-model.sh` handles download + SHA verification:

```bash
MODEL=<source-repo> FILE=<filename> ./pull-model.sh
```

It downloads to `./models/<filename>` and verifies sha256 against HuggingFace's `X-Linked-Etag` (LFS pointer).

## Manifest

| filename | source repo | size | sha256 |
|----------|-------------|------|--------|
| SmolLM2-360M-Instruct-Q4_K_M.gguf | bartowski/SmolLM2-360M-Instruct-GGUF | 259M | `2fa3f013dcdd7b99f9b237717fa0b12d75bbb89984cc1274be1471a465bac9c2` |
| Qwen_Qwen3-0.6B-Q4_K_M.gguf | bartowski/Qwen_Qwen3-0.6B-GGUF | 462M | `9acfc1e001311f34b4252001b626f2e466d592a42065f66571bff3790d4e1b14` |
| moondream2-mmproj-f16.gguf | vikhyatk/moondream2 (gguf branch) | 868M | `4cc1cb3660d87ff56432ebeb7884ad35d67c48c7b9f6b2856f305e39c38eed8f` |
| smollm2-1.7b-instruct-q4_k_m.gguf | HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF | 1.0G | `decd2598bc2c8ed08c19adc3c8fdd461ee19ed5708679d1c54ef54a5a30d4f33` |
| Qwen_Qwen3-1.7B-Q4_K_M.gguf | bartowski/Qwen_Qwen3-1.7B-GGUF | 1.2G | `72c5c3cb38fa32d5256e2fe30d03e7a64c6c79e668ad84057e3bd66e250b24fb` |
| SmolLM3-3B-Q4_K_M.gguf | bartowski/HuggingFaceTB_SmolLM3-3B-GGUF | 1.8G | `4de907d2d388a5508fb7cb443a06effe14cce3518b0a78d3bdd9e74d9edce989` |
| qwen2.5-3b-instruct-q4_k_m.gguf | Qwen/Qwen2.5-3B-Instruct-GGUF | 2.0G | `626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d` |
| microsoft_Phi-4-mini-instruct-Q4_K_M.gguf | bartowski/microsoft_Phi-4-mini-instruct-GGUF | 2.4G | `01999f17c39cc3074afae5e9c539bc82d45f2dd7faa3917c66cbef76fce8c0c2` |
| moondream2-text-model-f16.gguf | vikhyatk/moondream2 (gguf branch) | 2.7G | `4e17e9107fb8781629b3c8ce177de57ffeae90fe14adcf7b99f0eef025889696` |
| gemma-4-E2B-it-Q4_K_M.gguf | google/gemma-4-E2B-it-GGUF (or bartowski mirror) | 2.9G | `9378bc471710229ef165709b62e34bfb62231420ddaf6d729e727305b5b8672d` |
| gemma-4-E2B-it-UD-Q4_K_XL.gguf | unsloth/gemma-4-E2B-it-GGUF | 3.0G | `b8906b8c5e05e57b657646bbc657bd35814a269b2c20f0a2579047fafa1a67dd` |
| granite-4.0-h-tiny-UD-Q4_K_XL.gguf | unsloth/granite-4.0-h-tiny-GGUF | 3.8G | `517b4f5cbff45c35090f70e8b1b06dc40f0df33077fb05c25deab98abc5c294f` |
| granite-4.0-h-tiny-Q4_K_M.gguf | bartowski/ibm-granite_granite-4.0-h-tiny-GGUF | 4.0G | `d9baffe0c4c061e699ddeb8a51271f1e1bab29579a27ad3f213bf621a30b24ee` |
| Mistral-7B-Instruct-v0.3-Q4_K_M.gguf | bartowski/Mistral-7B-Instruct-v0.3-GGUF | 4.1G | `1270d22c0fbb3d092fb725d4d96c457b7b687a5f5a715abe1e818da303e562b6` |
| olmo-2-1124-7B-instruct-Q4_K_M.gguf | bartowski/allenai_OLMo-2-1124-7B-Instruct-GGUF | 4.2G | `e08112e5f84aab7c05fa6e713c58e5214cd5d8e32ed773ff3354b006eed41b95` |
| Qwen3-8B-Q4_K_M.gguf | Qwen/Qwen3-8B-GGUF | 4.7G | `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785` |
| Mistral-Nemo-Instruct-2407-Q4_K_M.gguf | bartowski/Mistral-Nemo-Instruct-2407-GGUF | 7.0G | `7c1a10d202d8788dbe5628dc962254d10654c853cae6aaeca0618f05490d4a46` |
| phi-4-Q4_K_M.gguf | bartowski/phi-4-GGUF | 8.5G | `009aba717c09d4a35890c7d35eb59d54e1dba884c7c526e7197d9c13ab5911d9` |
| gemma-4-26B-A4B-it-UD-Q4_K_M.gguf | unsloth/gemma-4-26B-A4B-it-GGUF | 16G | `34c746b1d50ab813e29cd46c4796e3f43c741901a582f93a67b55b9fc9687b35` |
| Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf | Qwen/Qwen3-Coder-30B-A3B-Instruct-GGUF | 18G | `fadc3e5f8d42bf7e894a785b05082e47daee4df26680389817e2093056f088ad` |
| Qwen_Qwen3.6-35B-A3B-Q5_K_M.gguf | Qwen/Qwen3.6-35B-A3B-GGUF (or bartowski mirror) | 24G | `194a9e9024b4196a9e6c9a2ba6224eafb656572aa80045a1a4c9ab12f1c83e2a` |

## Licence notes

Each weights file inherits its repo's licence. This bench only included models with **explicit Apache-2.0 or MIT** at the time of the run (May 2026). Strict gate is documented in `bench/REPORT.md` ("Empirically resolved against HuggingFace"). Notable:

- **Gemma 4** is genuine Apache-2.0 (verified via Google's `license_link` page). Gemma 1/2/3 carry HF metadata `license: gemma` (Gemma Terms of Use with AUP) — do not backport.
- **Llama-3.x** carries a 700M-MAU clause → OSD violation, excluded.
- **DeepSeek-V3** dual-licenses code (MIT) and weights (DeepSeek License with AUP) — weights are not OSS, excluded.
- **Mistral-7B-v0.3** is Apache-2.0; "v0.1" / "instruct" derivatives vary, check per-repo.

The bench data (gsm8k, humaneval, mmlu_pro, hendrycks_math500, ifeval, mbpp) is public and used under each dataset's licence (CC-BY-4.0 / MIT — see HuggingFace dataset cards).
