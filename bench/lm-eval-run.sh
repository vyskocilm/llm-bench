#!/usr/bin/env bash
# bench/lm-eval-run.sh — full lm-eval-harness sweep against llama-server,
# with all fixes 1-7 from our review baked in:
#
#   1. Completion endpoint (no chat wrap) for code/QA tasks; chat endpoint
#      only for chat-shaped tasks (ifeval).
#   2. --parallel 8 server-side + num_concurrent=4 client-side → 4-8× faster.
#   3. tokenized_requests=True + tokenizer=<HF id> → prompts pre-checked
#      against the 4096 ctx, no silent truncation.
#   4. Profile-driven task set + sample caps. BENCH_PROFILE=quick|core|full
#      picks a curated 5-task suite (instruction/math/code/MC/common-sense)
#      with light/medium caps, or a 12-task wide suite at full splits.
#      Default `core` targets ~45 min/model with ±2-3% stderr.
#   5. lm-eval's openai_completions.parse_logprobs patched in-place to
#      accept llama-server's modern OpenAI logprobs shape (logprobs.content[].
#      {token,logprob,top_logprobs[]}), unlocking MC tasks.
#   6. --seed 42,42,42,42 → fewshot example selection reproducible across
#      runs and aligned with server --seed 42.
#   7. One server lifecycle across all tasks; tasks batched into one
#      lm-eval call per group (chat / completion / mc).
#
# Usage:
#   bench/lm-eval-run.sh <model.gguf> <alias> <hf-tokenizer-id>
#
# Example:
#   bench/lm-eval-run.sh models/smollm2-1.7b-instruct-q4_k_m.gguf \
#                        smollm2-1.7b HuggingFaceTB/SmolLM2-1.7B-Instruct
set -uo pipefail
# (no -e: lm-eval may exit non-zero on transient errors; we want the next
#  group to still run. Errors land in the group log files.)

MODEL_GGUF="${1:?usage: lm-eval-run.sh <model.gguf> <alias> <hf-tokenizer-id> [draft.gguf]}"
MODEL_ALIAS="${2:?missing alias}"
TOKENIZER="${3:?missing HF tokenizer id}"
DRAFT_GGUF="${4:-${DRAFT_GGUF:-}}"   # optional speculative-decoding draft model
PORT="${PORT:-18080}"
PARALLEL="${PARALLEL:-4}"

# Speculative-decoding tuning (only used when DRAFT_GGUF is set).
# Defaults are llama.cpp's: draft up to 16 tokens per step, accept only when
# the draft's top-token probability exceeds 0.75 (greedy regime). Tune via
# env: DRAFT_N_MAX=8 DRAFT_P_MIN=0.6 ./lm-eval-run.sh ...
DRAFT_N_MAX="${DRAFT_N_MAX:-16}"
DRAFT_N_MIN="${DRAFT_N_MIN:-0}"
DRAFT_P_MIN="${DRAFT_P_MIN:-0.75}"
DRAFT_NGL="${DRAFT_NGL:-999}"

# BENCH_PROFILE selects task set + sample caps. Two axes to trade speed
# for coverage/statistical power:
#   - quick: 3 curated tasks, 30 samples each   → ~10 min, ±10% stderr
#   - core : 3 curated tasks, 250 samples each  → ~30 min, ±2-3% stderr  (DEFAULT)
#   - full : 6 wide tasks,    full splits       → 4-6 h,  ~±1% stderr
#
# All tasks routed through /v1/chat/completions with --apply_chat_template,
# because every model in our OSS registry is instruct-tuned: feeding the
# instruct model raw few-shot text via /v1/completions produces empty or
# random output (verified — arc/hellaswag scored below the 0.25 random
# floor when sent through the completions endpoint).
#
# Loglikelihood-based MC tasks (ARC, HellaSwag, WinoGrande, PIQA,
# TruthfulQA-MC1, MMLU) are excluded — the OpenAI chat-completions API
# can't return the logprob of an arbitrary completion string, only of
# tokens the model itself generated. Use a generation-based variant when
# available (e.g. `mmlu_pro` instead of `mmlu`).
BENCH_PROFILE="${BENCH_PROFILE:-core}"
case "${BENCH_PROFILE}" in
  quick)
    : "${TASKS:=ifeval,gsm8k,humaneval_instruct}"
    : "${LIMIT:=30}"
    ;;
  core)
    : "${TASKS:=ifeval,gsm8k,humaneval_instruct,mbpp_instruct}"
    : "${LIMIT:=250}"
    ;;
  full)
    : "${TASKS:=ifeval,gsm8k,humaneval_instruct,humaneval_plus,mbpp_instruct,mbpp_plus_instruct,drop,truthfulqa_gen}"
    : "${LIMIT:=99999}"
    ;;
  *)
    echo "FATAL: unknown BENCH_PROFILE=${BENCH_PROFILE}; expected quick|core|full" >&2
    exit 2
    ;;
esac
echo "profile=${BENCH_PROFILE} limit=${LIMIT}"
echo "  tasks: ${TASKS}"

# Per-model total ctx-size (KV cache). With --parallel N each slot gets
# CTX/N; values below give each slot up to ~native model context so even
# long-context benchmarks (LongBench, RULER) work without RoPE stretching.
# KV memory scales linearly with ctx — at 131K total + Q4 KV the largest
# entry costs ~4 GiB on the iGPU (35 GiB pool, room to spare).
declare -A MODEL_CTX=(
  [smollm2-1.7b]=32768       # native 8K  → 8K/slot at parallel=4
  [olmo2-7b]=16384           # native 4K  → 4K/slot
  [mistral-7b-v0.3]=131072   # native 32K → 32K/slot
  [qwen3-8b]=131072          # native 32K (RoPE-128K) → 32K/slot
  [mistral-nemo-2407]=131072 # native 128K — cap here to keep KV < 4 GiB
  [phi-4]=65536              # native 16K → 16K/slot
  [gemma-4-e2b]=131072       # native 32K → 32K/slot
  [gemma-4-26b-a4b]=131072   # native 32K, MoE (4B active) — 32K/slot OK
  [qwen3.6-35b-a3b]=131072   # MoE 3B active
  # qwen3.6-27b excluded: Q6_K (22 GiB) dense → ~4 tok/s on this iGPU,
  # ifeval alone would take 20+ h at core profile. See REPORT.md.
  [qwen3-coder-30b-a3b]=131072 # MoE 3B active, code-tuned
  [smollm2-360m]=8192          # native 8K
  [smollm3-3b]=32768            # native 32K? bench-safe 8K/slot
  [qwen3-0.6b]=131072
  [qwen3-1.7b]=131072
  [granite-4-h-tiny]=131072    # MoE
  [phi-4-mini]=65536            # native 128K but bench-safe
  # A/B variants — Unsloth Dynamic Q4_K_XL (different bits-per-tensor
  # vs the standard Q4_K_M counterpart; ~2-5% quality lift expected).
  [gemma-4-e2b-ud]=131072
  [granite-4-h-tiny-ud]=131072
)
CTX_SIZE="${CTX_SIZE:-${MODEL_CTX[${MODEL_ALIAS}]:-65536}}"
echo "ctx-size=${CTX_SIZE} (per-slot: $((CTX_SIZE / PARALLEL))) parallel=${PARALLEL}"

# Per-model extras injected into the lm-eval request body via the patched
# openai_completions.py. Used to flip thinking/reasoning gates on models
# that default to it (Gemma 4, Qwen 3). Override with EXTRA_BODY env var.
declare -A MODEL_EXTRA_BODY=(
  [gemma-4-e2b]='{"chat_template_kwargs":{"enable_thinking":false}}'
  [gemma-4-26b-a4b]='{"chat_template_kwargs":{"enable_thinking":false}}'
  [qwen3-8b]='{"chat_template_kwargs":{"enable_thinking":false}}'
  [qwen3.6-35b-a3b]='{"chat_template_kwargs":{"enable_thinking":false}}'
  [qwen3.6-27b]='{"chat_template_kwargs":{"enable_thinking":false}}'
  [qwen3-coder-30b-a3b]='{"chat_template_kwargs":{"enable_thinking":false}}'
  [qwen3-0.6b]='{"chat_template_kwargs":{"enable_thinking":false}}'
  [qwen3-1.7b]='{"chat_template_kwargs":{"enable_thinking":false}}'
  [smollm3-3b]='{"chat_template_kwargs":{"enable_thinking":false}}'
  [gemma-4-e2b-ud]='{"chat_template_kwargs":{"enable_thinking":false}}'
)
export LM_EVAL_EXTRA_BODY="${EXTRA_BODY:-${MODEL_EXTRA_BODY[${MODEL_ALIAS}]:-}}"
if [[ -n "${LM_EVAL_EXTRA_BODY}" ]]; then
  echo "extra_body: ${LM_EVAL_EXTRA_BODY}"
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/bench/lm-eval/${MODEL_ALIAS}"
mkdir -p "${OUT}"
echo "writing results to: ${OUT}"

# Start llama-server with parallelism + larger ctx.
#
# IMPORTANT: --ctx-size is the *total* KV cache size shared across all
# --parallel slots. Per-request ctx = ctx-size / parallel. lm-eval's
# few-shot prompts can hit ~2-3K tokens (GSM8K-5shot ~1500, MMLU-5shot
# similar, DROP-3shot longer). With ctx=32768 and parallel=4 each slot
# gets 8K, plenty of headroom. 32K KV @ Q4_K_M is ~600 MiB — trivial on
# the 35 GiB iGPU pool.
# Compose optional speculative-decoding args. The draft model runs on the
# same GPU; for memory-bound iGPUs the speedup comes from amortising the
# target-model forward across multiple accepted draft tokens.
SPEC_ARGS=()
if [[ -n "${DRAFT_GGUF}" ]]; then
  if [[ ! -f "${DRAFT_GGUF}" ]]; then
    echo "FATAL: draft model not found: ${DRAFT_GGUF}" >&2
    exit 2
  fi
  echo "speculative decoding enabled: draft=${DRAFT_GGUF}"
  SPEC_ARGS+=(
    --spec-draft-model "${DRAFT_GGUF}"
    --spec-draft-n-max "${DRAFT_N_MAX}"
    --spec-draft-n-min "${DRAFT_N_MIN}"
    --spec-draft-p-min "${DRAFT_P_MIN}"
    --spec-draft-ngl "${DRAFT_NGL}"
  )
fi

"${ROOT}/dist/bin/llama-server" \
  -m "${MODEL_GGUF}" \
  --host 127.0.0.1 --port "${PORT}" --alias "${MODEL_ALIAS}" \
  --n-gpu-layers 999 --ctx-size "${CTX_SIZE}" --threads 8 \
  -b 512 -ub 1024 \
  --jinja --seed 42 --no-warmup \
  --parallel "${PARALLEL}" \
  "${SPEC_ARGS[@]}" \
  > "${OUT}/server.log" 2>&1 &
SERVER_PID=$!
trap "kill ${SERVER_PID} 2>/dev/null; wait 2>/dev/null || true" EXIT INT TERM

# Wait for the model to be *fully* loaded, not just the HTTP listener up.
# /health flips to 200 as soon as the server binds the port, but
# /v1/chat/completions returns 503 until slot init + KV cache allocation
# completes. /v1/models only returns 200 once the model is ready to serve.
# Large/new architectures (Gemma 4 takes ~3 min) need a generous timeout.
for i in {1..600}; do
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null)
  if [[ "$code" == "200" ]]; then
    echo "server up (pid=${SERVER_PID}, model ready in ${i}s)"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "FATAL: llama-server exited during load; see ${OUT}/server.log" >&2
    exit 3
  fi
  sleep 1
done

# Tasks needing code execution require this env var in addition to the CLI flag.
export HF_ALLOW_CODE_EVAL=1

# tokenized_requests=False for the chat endpoint: chat models take a
# `messages` list of dicts and apply their template server-side. With True,
# lm-eval skips chat-template rendering and the assertion in
# LocalChatCompletion._create_payload fires.
CHAT_ARGS="model=${MODEL_ALIAS},num_concurrent=${PARALLEL},tokenized_requests=False,tokenizer_backend=huggingface,tokenizer=${TOKENIZER},timeout=1800"

# Single invocation: all tasks through chat-completions + apply_chat_template.
# Per-task gen_kwargs come from each task's YAML (max_gen_toks etc.), so
# bundling tasks in one lm-eval call is safe.
echo
echo "############# lm-eval (chat-completions) #############"
lm-eval run \
  --batch_size 1 \
  --seed 42,42,42,42 \
  --confirm_run_unsafe_code \
  --model local-chat-completions \
  --model_args "base_url=http://127.0.0.1:${PORT}/v1/chat/completions,${CHAT_ARGS}" \
  --tasks "${TASKS}" \
  --limit "${LIMIT}" \
  --apply_chat_template \
  --output_path "${OUT}/" 2>&1 | tee "${OUT}/lm-eval.log"

echo
echo "ALL_DONE — results under ${OUT}/"
