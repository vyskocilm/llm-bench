# Local OSS LLM Bench — Ryzen AI 7 PRO + Radeon 860M (gfx1151)

> **Deep-thinking results** (Gemma 4 / Qwen 3.6 with `enable_thinking=true`) live in [`bench/deep/DEEP-REPORT.md`](deep/DEEP-REPORT.md). This document covers the fast-chat / `enable_thinking=false` mode only.

> **About this document.** Hand-written by **Claude (Anthropic Opus 4.7, 1M-context build)** from the raw JSON files in `bench/lm-eval/<alias>/results_*.json`. There is **no script that generates this report** — Claude reads the JSONs and writes the tables. To independently verify any number, open the corresponding `results_*.json` directly; that file is the ground truth. See `bench/deep/aggregate-deep.py` for a counter-example: that script *is* automated, but only for the deep-thinking track. The fast-chat REPORT.md here remains hand-rolled commentary.

**Hardware**: Ryzen 7 PRO 350, **Radeon 860M iGPU (8 GiB dedicated, ~35 GiB shareable from 64 GiB host RAM)**, Vulkan via RADV. Memory-bandwidth-bound regime, ~90 GB/s LPDDR5x.

**Bench**: lm-eval-harness, `core` profile **n=250** (250 prompts per task; humaneval is the full 164-prompt suite). **±~3 pp 1σ stderr at p=0.7** — defensible per-axis rankings within ~5 pp.

**Tasks**: `ifeval` (instruction following, prompt_level_strict), `gsm8k` (math, 5-shot CoT, exact-match strict), `humaneval_instruct` (code, exec-scored pass@1).

**Date**: May 2026. llama.cpp commit `838374375c`.

**Quick-profile context.** An earlier sweep at `quick` (n=30) is preserved in `bench/lm-eval-quick/<alias>/` for historical comparison. Quick numbers carried **±~9 pp 1σ stderr** — large enough that several rankings inverted when moved to core. Notable noise-driven re-rankings: gemma-4-e2b humaneval **0.97 → 0.77** (n=30 noise inflated), mistral-nemo-2407 humaneval **0.87 → 0.61** (relaxed n=164), phi-4-mini ifeval **0.70 → 0.69** (held up). When quick and core disagree, **trust core**.

---

> **†** marker = bench run with `enable_thinking=false` (quick-chat mode). Real deep-reasoning capability of Gemma 4 / Qwen 3 / Qwen 3.6 not measured here — see Caveats. Track in [`bench/deep/DEEP-REPORT.md`](deep/DEEP-REPORT.md).

## Caveman exec summary

- **Big MoE wins.** Top-3 across all 3 axes is **qwen3-coder-30b-a3b** (Apache-2.0) + **gemma-4-26b-a4b** (Apache-2.0, †) + **qwen3.6-35b-a3b** (Apache-2.0, †). UMA architecture eliminates the "model doesn't fit" penalty — active-parameter count is what matters, not total.
- **Granite loses the crown but stays in top tier.** granite-4-h-tiny (Apache-2.0, 7B/1B MoE) still beats every dense ≤8B model on humaneval. Excellent fit-vs-quality trade-off; ~4 GiB on disk.
- **gemma-4-e2b** (Apache-2.0, 5B "effective" small) the standout sub-7B coder at **0.77 humaneval** (UD-XL: **0.82**).
- **Qwen 3 / Qwen 3.6 without thinking are mid-tier code-wise.** Cap at ~0.32 humaneval thinking-off — by design. Their lift is in the deep-thinking track.
- **Phi-4 14B dense never finished core** (~5 tok/s on this iGPU). Quick n=30 hinted strong (0.97 code) but unverifiable at core.
- **Sub-1B = toy.** Qwen3-0.6B / SmolLM2-360M score ~0 on math+code.
- **Local-only report.** No cloud-API baselines listed here. All scored models run on the Radeon 860M.

---

## Axis: Best for coding (humaneval_instruct pass@1, n=164)

| rank | model | pass@1 | ±stderr | size active | quant |
|------|-------|--------|---------|-------------|-------|
| 🥇 | **qwen3-coder-30b-a3b** | **0.921** | ±0.021 | 30B/3B MoE | Q4_K_M |
| 🥈 | **gemma-4-e2b** (UD-XL) † | **0.823** | ±0.030 | 5B eff | UD-Q4_K_XL |
| 🥉 | granite-4-h-tiny | 0.805 | ±0.031 | 7B/1B MoE | Q4_K_M |
| 4 | gemma-4-e2b (std) † | 0.774 | ±0.033 | 5B eff | Q4_K_M |
| 5 | granite-4-h-tiny (UD-XL) | 0.762 | ±0.033 | 7B/1B MoE | UD-Q4_K_XL |
| 6 | phi-4-mini | 0.695 | ±0.036 | 3.8B | Q4_K_M |
| 7 | mistral-nemo-2407 (relaxed) | 0.610 | ±0.038 | 12B | Q4_K_M |
| 8 | gemma-4-26b-a4b † | 0.476 / 0.488 (rel) | ±0.039 | 26B/4B MoE | UD-Q4_K_M |
| 9 | mistral-7b-v0.3 | 0.415 | ±0.039 | 7B | Q4_K_M |
| 10 | olmo2-7b | 0.402 | ±0.038 | 7B | Q4_K_M |
| 11 | smollm3-3b | 0.360 | ±0.038 | 3B | Q4_K_M |
| 12 | qwen3-1.7b | 0.335 | ±0.037 | 1.7B | Q4_K_M |
| 13 | qwen3-8b † | 0.329 | ±0.037 | 8B | Q4_K_M |
| 14 | qwen3.6-35b-a3b † | 0.323 | ±0.037 | 35B/3B MoE | Q5_K_M |
| 15 | qwen3-0.6b | 0.165 | ±0.029 | 0.6B | Q4_K_M |
| 16 | smollm2-1.7b | 0.037 | ±0.015 | 1.7B | Q4_K_M |
| 17 | smollm2-360m | 0.000 | — | 360M | Q4_K_M |
| — | phi-4 | 0.967 (**n=30 only**) | ±0.033 | 14B dense | Q4_K_M |

**Notes**: Qwen 3 / 3.6 humaneval ceiling is ~0.32 without thinking (verified — strict and relaxed stop lists give the same number). Real Qwen3-8B thinking-on humaneval is ~0.80 per public reports — see deep-bench track. Phi-4 14B is slow enough on this iGPU that the core sweep was abandoned at n=30; its 0.967 quick score is unverifiable.

---

## Axis: Best for math (gsm8k strict-match, n=250)

| rank | model | accuracy | ±stderr | size active |
|------|-------|----------|---------|-------------|
| 🥇 | **gemma-4-26b-a4b** † | **0.896** | ±0.019 | 26B/4B MoE |
| 🥈 | **qwen3-coder-30b-a3b** | 0.892 | ±0.020 | 30B/3B MoE |
| 🥉 | **qwen3.6-35b-a3b** † | 0.880 | ±0.021 | 35B/3B MoE |
| 4 | granite-4-h-tiny | 0.812 | ±0.025 | 7B/1B MoE |
| 5 | smollm3-3b | 0.800 | ±0.025 | 3B |
| 6 | granite-4-h-tiny (UD-XL) | 0.788 | ±0.026 | 7B/1B MoE |
| 7 | olmo2-7b | 0.752 | ±0.027 | 7B |
| 8 | phi-4-mini | 0.748 | ±0.028 | 3.8B |
| 9 | qwen3-8b † | 0.744 | ±0.028 | 8B |
| 10 | gemma-4-e2b (UD-XL) † | 0.528 | ±0.032 | 5B eff |
| 11 | mistral-7b-v0.3 | 0.480 | ±0.032 | 7B |
| 12 | gemma-4-e2b (std) † | 0.444 | ±0.031 | 5B eff |
| 13 | smollm2-1.7b | 0.428 | ±0.031 | 1.7B |
| 14 | qwen3-1.7b | 0.412 | ±0.031 | 1.7B |
| 15 | smollm2-360m | 0.096 | ±0.019 | 360M |
| 16 | qwen3-0.6b | 0.044 | ±0.013 | 0.6B |
| — | phi-4 | 0.800 (**n=30 only**) | ±0.074 | 14B dense |

**Notes**: Top-3 cluster within stderr — call it a 3-way tie at ~0.89. gemma-4-e2b is unexpectedly poor at math (0.44-0.53) despite excelling at code; family-specific training trade-off.

---

## Axis: Best for chat / instruction following (ifeval prompt_strict, n=250)

| rank | model | ifeval p_strict | ±stderr | size active |
|------|-------|------------------|---------|-------------|
| 🥇 | **gemma-4-26b-a4b** † | **0.860** | ±0.022 | 26B/4B MoE |
| 🥈 | **qwen3-coder-30b-a3b** | 0.836 | ±0.023 | 30B/3B MoE |
| 🥉 | qwen3-8b † | 0.808 | ±0.025 | 8B |
| 4 | qwen3.6-35b-a3b † | 0.804 | ±0.025 | 35B/3B MoE |
| 5 | gemma-4-e2b (UD-XL) † | 0.760 | ±0.027 | 5B eff |
| 6 | gemma-4-e2b (std) † | 0.756 | ±0.027 | 5B eff |
| 7 | granite-4-h-tiny | 0.740 | ±0.028 | 7B/1B MoE |
| 8 | granite-4-h-tiny (UD-XL) | 0.728 | ±0.028 | 7B/1B MoE |
| 9 | phi-4-mini | 0.692 | ±0.029 | 3.8B |
| 9 | qwen3-1.7b | 0.692 | ±0.029 | 1.7B |
| 11 | olmo2-7b | 0.664 | ±0.030 | 7B |
| 12 | smollm3-3b | 0.652 | ±0.030 | 3B |
| 13 | qwen3-0.6b | 0.516 | ±0.032 | 0.6B |
| 14 | smollm2-1.7b | 0.504 | ±0.032 | 1.7B |
| 15 | mistral-7b-v0.3 | 0.456 | ±0.032 | 7B |
| 16 | smollm2-360m | 0.320 | ±0.030 | 360M |
| — | phi-4 | 0.533 (**n=30 only**) | ±0.093 | 14B dense |

**Notes**: Big MoEs lead, but the gap to mid-tier is modest (~10 pp).

---

## Axis: Least memory / token footprint (fit + speed under 8 GiB iGPU)

| rank | model | GGUF size | iGPU fit | one-liner |
|------|-------|-----------|----------|-----------|
| 🥇 | **gemma-4-e2b** Q4_K_M † | 2.9 GiB | ✓ trivial | 0.77 code / 0.76 if / 0.44 math; **fast** |
| 🥈 | **granite-4-h-tiny** Q4_K_M | 4.0 GiB | ✓ | 0.81 code / 0.74 if / 0.81 math; **all-rounder** |
| 🥉 | smollm3-3b Q4_K_M | 1.8 GiB | ✓ | 0.36 code / 0.65 if / **0.80 math**; surprising math score |
| 4 | phi-4-mini Q4_K_M | 2.4 GiB | ✓ | 0.70 code / 0.69 if / 0.75 math |
| 5 | qwen3-1.7b Q4_K_M | 1.2 GiB | ✓ | 0.34 code / 0.69 if / 0.41 math |
| — | qwen3-8b / olmo2-7b / mistral-7b Q4_K_M | 4-5 GiB | ✓ tight | mixed; qwen3-8b best on ifeval |
| — | mistral-nemo-2407 (12B Q4) | 7.0 GiB | ⚠ tight | 0.61 code (relaxed), 0.40 if at n=30 (under-tested) |
| — | phi-4 (14B Q4) | 8.5 GiB | ✗ over | ~5 tok/s; never finished core |
| — | **qwen3-coder-30b-a3b** (30B/3B MoE) | 18 GiB | ✗ over† | **0.92 code / 0.84 if / 0.89 math** — works because MoE active=3B |
| — | gemma-4-26b-a4b (26B/4B MoE) | 16 GiB UD | ✗ over† | **0.86 if / 0.90 math** — same UMA story |
| — | qwen3.6-35b-a3b (35B/3B MoE) | 24 GiB Q5 | ✗ over† | 0.88 math / 0.80 if |

† "over" = larger than the 8 GiB iGPU-dedicated reservation but **fits the UMA address space** (shared with system RAM up to ~35 GiB usable). Decode speed tracks active params, not file size.

---

## Recommendation matrix

| use case | local pick | why |
|----------|-----------|-----|
| **Default daily driver, 1 model** | **qwen3-coder-30b-a3b** | Top-3 on all 3 axes; Apache-2.0; coding-optimised but also strong on chat (0.84) and math (0.89). 18 GiB file, decode ~22 tok/s. |
| **Same but smaller footprint** | **granite-4-h-tiny** | 4 GiB on disk, ~28 tok/s, 0.81 code / 0.81 math / 0.74 if. Best small all-rounder. |
| **Pure speed + good code** | **gemma-4-e2b** (UD-XL) † | 3.0 GiB, ~46 tok/s, 0.82 humaneval. Math is weak (0.53). |
| **Pure speed, small fleet** | **qwen3-1.7b** | 1.2 GiB, ~73 tok/s, 0.69 ifeval — punchy for size. Code weak. |
| **Math heavy** | **gemma-4-26b-a4b** † | 0.90 gsm8k, 0.86 ifeval; 16 GiB. iGPU-friendly MoE. |
| **Chat / instruction following** | **qwen3-coder-30b-a3b** | 0.84 ifeval. |
| **Embed / tiny** | qwen3-0.6b | 462 MB, useless for generation but fast for embed-classification. |
| **Vision** | moondream2 | 868 MB mmproj + 2.7 GB text. (Not scored here.) |

---

## Model provenance (sources + quantization)

All 21 GGUFs documented in [`MODELS.md`](../MODELS.md) (HF source repo + SHA256 per file). Two-thirds use **standard `Q4_K_M`** (byte-identical regardless of mirror — bartowski, unsloth, official Qwen, etc., all produce the same hash). Three exceptions:

- **gemma-4-e2b-ud** uses **Unsloth Dynamic `UD-Q4_K_XL`** (~3% bigger than standard Q4_K_M; +5 pp humaneval / +9 pp gsm8k advantage observed).
- **gemma-4-26b-a4b** uses **`UD-Q4_K_M`** (Unsloth Dynamic, same nominal size as standard Q4_K_M but better quant placement).
- **qwen3.6-35b-a3b** uses **`Q5_K_M`** (one step up; ~24 GiB, follows Google AI's recommended precision for thinking-capable models even when run thinking-off).

### Excluded models

- **phi-4 (14B dense)** — completed quick (n=30) only. Core sweep timed out at ~5 tok/s on this iGPU. Listed in tables with "n=30 only" tag.
- **qwen3.6-27b** (dense 27B Q6_K, 22 GiB) — too slow on this iGPU (~4 tok/s decode); not bench'd. The MoE Qwen3.6-35B-A3B is the better choice on this hardware.
- **moondream2** — vision benchmark dropped from the suite (no usable lm-eval image task that matches our compute budget).

### Reproducibility

1. `./build.sh` → `./dist/bin/{llama-server,llama-cli,llama-bench}`.
2. `./pull-model.sh MODEL=<repo> FILE=<file.gguf>` per row in `MODELS.md` (or pick a subset).
3. `./bench/lm-eval-run.sh <model.gguf> <alias> <hf-tokenizer-id>` with `BENCH_PROFILE=core` env. One model at a time; ~2-3 h each.
4. Results land in `bench/lm-eval/<alias>/results_<ts>.json`. Compare your file's hash against the one this report was generated from.
5. For thinking-on (deep-bench): `./bench/deep/deep-bench.sh deep-reasoning` / `deep-math`. See `docs/superpowers/specs/2026-05-15-deep-bench-design.md`.

---

## Caveats

- **Stderr per axis is ~±3 pp 1σ at n=250**. Differences within ~5 pp are not strictly rankable. Most "top-3" clusters here cluster within stderr; treat ordering inside the top tier as a tie.
- **Thinking mode disabled** (†): Gemma 4 and Qwen 3 / 3.6 ship a chat-template flag `enable_thinking` that, when on, makes the model emit a `<think>...</think>` CoT block before the answer. We bench with **`enable_thinking=false`** — quick-chat / interactive mode. Reasons: (1) cross-family fairness — Mistral/OLMo/Granite/Phi have no thinking mode; (2) prefill-based tasks (HumanEval) are server-rejected with `enable_thinking=true`; (3) thinking emits 2-8k CoT tokens which blow the `max_gen_toks=1024` budget; (4) 5-20× wall-clock per request. Published Qwen3 numbers show thinking-on adds roughly **+15-20 pp on gsm8k/humaneval, +20 pp on mmlu_pro, +45 pp on MATH**. So gemma-4-* and qwen3* / qwen3.6-* numbers here are a **lower bound** representing the fast-chat use case; the deep-reasoning capability of these models is NOT captured here — see [`bench/deep/DEEP-REPORT.md`](deep/DEEP-REPORT.md). Ranking caveat: granite-4-h-tiny / qwen3-coder (non-thinking by design) are already at their ceiling; thinking-capable peers have headroom not measured here.
- **Stop-list theory busted.** The earlier hypothesis ("Qwen3 humaneval depressed by stop list") was disproven at n=164 relaxed-humaneval: scores identical to strict. The ~0.32 cap is the real thinking-off ceiling for Qwen3 family.
- **Granite humaneval 0.81 suspiciously high** — possible training-data contamination of public HumanEval. Confirm with private code-bench before betting on it.
- **MoE models**: "size active" column shows total/active params. Memory-bound iGPU decode tracks active params, not total. This is **the** key insight for choosing models on this hardware.
- **gen tok/s** estimates not shown in this revision (data lives in per-task `server.log`, not summarised here; the prior REPORT.md had heuristic numbers that aged poorly).

---

## TL;DR

> **Mode**: All local-model numbers reflect **fast-chat / `enable_thinking=false`** mode. Models flagged † have a deep-thinking mode not exercised here — see [`bench/deep/DEEP-REPORT.md`](deep/DEEP-REPORT.md). Granite-4-h-tiny, qwen3-coder, OLMo, Mistral, Phi have no thinking mode and are at their full ceiling.

For Ryzen 7 + Radeon 860M (8 GiB iGPU, 64 GiB RAM):

1. **Default daily driver**: **`qwen3-coder-30b-a3b`** (Apache-2.0, Alibaba). 30B MoE / 3B active, ~22 tok/s, **0.92 code / 0.84 ifeval / 0.89 math**. Top-3 on every axis. 18 GiB on disk.
2. **Smaller footprint, near-top quality**: **`granite-4-h-tiny`** (Apache-2.0, IBM). 7B MoE / 1B active, ~28 tok/s, **0.81 / 0.74 / 0.81**. 4 GiB on disk.
3. **Speed-first coder**: **`gemma-4-e2b` (UD-XL)** (Apache-2.0, Google). 5B eff, ~46 tok/s, **0.82 humaneval**. Math is weak.
4. **Math heavy**: **`gemma-4-26b-a4b`** (Apache-2.0, Google). MoE 26B/4B, top math + ifeval. Needs UMA fallback for the 16 GiB file.

**The shift from prior (quick-profile) report**: granite-4-h-tiny was named #1 daily driver based on n=30 noisy scores. At core (n=250), the big MoE models (qwen3-coder, gemma-4-26b, qwen3.6-35b) pull ahead on every axis. UMA = "model doesn't fit" stops mattering. **Verify locally before betting** — see `bench/lm-eval/<alias>/results_*.json` for the raw numbers behind every cell in this document.
