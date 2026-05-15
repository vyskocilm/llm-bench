#!/usr/bin/env bash
# Deep-thinking bench driver for Track A.
# Usage:
#   ./deep-bench.sh smoke            # n=2/task on smollm3-3b (~15 min)
#   ./deep-bench.sh deep-reasoning   # gsm8k + humaneval + mmlu_pro on both models (~11h)
#   ./deep-bench.sh deep-math        # hendrycks_math500 on both models (~10h)
set -uo pipefail  # no -e: lm-eval may exit non-zero on transient errors

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT=18080
EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":true}}'

# alias|gguf|tokenizer|ctx-size
declare -A MODELS=(
  [gemma-4-e2b]='models/gemma-4-E2B-it-UD-Q4_K_XL.gguf|google/gemma-4-E2B-it|131072'
  [qwen3.6-35b-a3b]='models/Qwen_Qwen3.6-35B-A3B-Q5_K_M.gguf|Qwen/Qwen3.6-35B-A3B|131072'
  [smollm3-3b]='models/SmolLM3-3B-Q4_K_M.gguf|HuggingFaceTB/SmolLM3-3B|65536'
)

# sub-profile|tasks_csv|limit_per_task_csv
declare -A PROFILES=(
  [deep-reasoning]='gsm8k_deep,humaneval_deep,mmlu_pro_deep|200,164,200'
  [deep-math]='math_deep|100'
  [smoke]='gsm8k_deep,humaneval_deep,mmlu_pro_deep,math_deep|2,2,2,2'
)

SUBPROFILE="${1:?usage: deep-bench.sh <smoke|deep-reasoning|deep-math>}"
PROFILE_SPEC="${PROFILES[$SUBPROFILE]:-}"
[[ -z "$PROFILE_SPEC" ]] && { echo "unknown sub-profile: $SUBPROFILE"; exit 2; }

IFS='|' read -r TASKS_CSV LIMITS_CSV <<< "$PROFILE_SPEC"
IFS=',' read -ra TASKS <<< "$TASKS_CSV"
IFS=',' read -ra LIMITS <<< "$LIMITS_CSV"

# pick model set
if [[ "$SUBPROFILE" == "smoke" ]]; then
  MODEL_LIST=(smollm3-3b)
else
  MODEL_LIST=(gemma-4-e2b qwen3.6-35b-a3b)
fi

OUT_BASE="$ROOT/bench/deep/$SUBPROFILE"
mkdir -p "$OUT_BASE"

for alias in "${MODEL_LIST[@]}"; do
  spec="${MODELS[$alias]}"
  IFS='|' read -r gguf tok ctx <<< "$spec"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] === MODEL $alias ==="

  log_dir="$OUT_BASE/$alias"
  mkdir -p "$log_dir"

  # start server
  "$ROOT/dist/bin/llama-server" \
    -m "$ROOT/$gguf" \
    --host 127.0.0.1 --port "$PORT" \
    --alias "$alias" \
    --n-gpu-layers 999 --ctx-size "$ctx" \
    --threads 8 -b 512 -ub 1024 \
    --jinja --seed 42 --no-warmup --parallel 4 \
    > "$log_dir/server.log" 2>&1 &
  SERVER_PID=$!
  trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

  # poll up to 300s
  ready=false
  for i in $(seq 1 300); do
    if curl -s "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
      echo "  server up after ${i}s"
      ready=true
      break
    fi
    sleep 1
  done
  if [[ "$ready" != true ]]; then
    echo "  server failed to start (last 30 server.log):"
    tail -30 "$log_dir/server.log"
    kill $SERVER_PID 2>/dev/null || true
    continue
  fi

  # run each task
  for idx in "${!TASKS[@]}"; do
    task="${TASKS[$idx]}"
    limit="${LIMITS[$idx]}"
    task_out="$log_dir/$task"
    mkdir -p "$task_out"

    if find "$task_out" -name 'results_*.json' -print -quit 2>/dev/null | grep -q .; then
      echo "  SKIP $task (results already present)"
      continue
    fi

    echo "  [$(date +%H:%M:%S)] task $task limit=$limit"
    LM_EVAL_EXTRA_BODY="$EXTRA_BODY" HF_ALLOW_CODE_EVAL=1 \
      lm-eval \
      --model local-chat-completions \
      --model_args "base_url=http://127.0.0.1:$PORT/v1/chat/completions,model=$alias,num_concurrent=4,tokenized_requests=False,tokenizer_backend=huggingface,tokenizer=$tok,timeout=3600" \
      --tasks "$task" \
      --limit "$limit" \
      --output_path "$task_out" \
      --log_samples \
      --apply_chat_template \
      --fewshot_as_multiturn \
      --confirm_run_unsafe_code \
      --seed 42 \
      > "$task_out/driver.log" 2>&1 \
      || echo "  TASK FAILED: $task (see $task_out/driver.log)"
  done

  # stop server
  kill $SERVER_PID 2>/dev/null || true
  wait $SERVER_PID 2>/dev/null || true
  trap - EXIT
  echo "[$(date +%H:%M:%S)] === MODEL $alias done ==="
done

echo "[$(date +%Y-%m-%d_%H:%M:%S)] === SUB-PROFILE $SUBPROFILE done ==="
