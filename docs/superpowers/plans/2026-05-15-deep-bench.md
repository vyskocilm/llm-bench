# Deep-thinking Bench (Track A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained "deep-bench" track that runs gemma-4-e2b and qwen3.6-35b-a3b in `enable_thinking=true` mode against gsm8k / humaneval / mmlu_pro / hendrycks_math, reports thinking-on lift in its own document, under 24h per sub-profile.

**Architecture:** New isolated tree under `bench/deep/`. Driver shell script orchestrates one model at a time. Task YAMLs override stop list + token budget + temperature. Harness patch makes `<think>` stripping robust to unclosed / multi-block / channel-tag variants. Aggregator script reads results.json files and regenerates DEEP-REPORT.md.

**Tech Stack:** bash, llama.cpp llama-server (Vulkan/RADV), lm-evaluation-harness via uv tool, Python 3.14 for unit tests + aggregator, lm-eval task YAML inheritance.

**Not a git repo.** Project has no `.git/`. Skip all `git add` / `git commit` steps the writing-plans skill normally inserts — they apply only to git-tracked projects. Use file presence as the checkpoint signal instead.

---

## File map

| Path | Created / Modified | Purpose |
|------|--------------------|---------|
| `bench/future/VIBES-GALLERY.md` | Create | Track B (qualitative bench) future-work stub |
| `bench/deep/strip_thinking.py` | Create | Importable strip function (4-pattern regex) |
| `bench/deep/test_strip.py` | Create | Unit tests for strip function |
| `~/.local/share/uv/tools/lm-eval/lib/python3.14/site-packages/lm_eval/models/openai_completions.py` | Modify | Swap leading-only strip for 4-pattern strip |
| `bench/deep/tasks/gsm8k_deep.yaml` | Create | thinking-on gsm8k override |
| `bench/deep/tasks/humaneval_deep.yaml` | Create | thinking-on humaneval override |
| `bench/deep/tasks/mmlu_pro_deep.yaml` | Create | thinking-on mmlu_pro override |
| `bench/deep/tasks/math_deep.yaml` | Create | thinking-on hendrycks_math override |
| `bench/deep/deep-bench.sh` | Create | Driver: `smoke` / `deep-reasoning` / `deep-math` sub-commands |
| `bench/deep/aggregate-deep.py` | Create | results.json → DEEP-REPORT.md |
| `bench/deep/DEEP-REPORT.md` | Create (then auto-regenerated) | Public results |
| `bench/REPORT.md` | Modify (1 line) | Pointer to DEEP-REPORT.md |

---

## Task 1: Track B future-work stub

**Files:**
- Create: `bench/future/VIBES-GALLERY.md`

- [ ] **Step 1: Create directory and file**

```bash
mkdir -p /home/michal/projects/vyskocilm/ollama/bench/future
```

- [ ] **Step 2: Write stub content**

```markdown
# Vibes Gallery — Track B (Future Work)

**Status:** not built; design captured here for later pickup.

## Goal

Capture model "personality" via curated qualitative prompts. Complement to the quant bench by exposing what numbers cannot — style, helpfulness, "feel", multi-turn coherence, refusal calibration.

## Structure

- ~20-30 prompts across 8 categories:
  1. **Constrained creative** — "SVG of a pelican on a monocycle" (Simon Willison's classic vibes-test)
  2. **Open creative** — "Write a 200-word noir story about a router"
  3. **Multi-turn coherence** — 3-turn dialogue with callbacks; catches "amnesia"
  4. **Style mimicry** — "Rewrite in Hemingway voice" / "PR statement voice"
  5. **Reasoning vibes** — riddles, lateral-thinking; overlap with track A but qualitative
  6. **Tool-use intent** — "Plan steps to refactor X"; structured output
  7. **Refusal / safety** — calibration probes
  8. **Locale / language** — non-English prompt

## Scoring

- **Gallery (always)**: markdown table with rendered SVGs + side-by-side text outputs. Durable artifact, eyeball-friendly.
- **LLM-judge (optional)**: a stronger local model (or any OpenAI-compat API endpoint of the user's choice) scores each response on a small rubric. Produces ranking when requested.

## Models

- Top 6-8 daily-driver candidates from `bench/REPORT.md` rankings.
- (Optional) any OpenAI-compatible cloud endpoint of the user's choice for an additional baseline column.

## Budget

- ~3-4h to run the gallery (20-30 prompts × 6-8 models × 1-3 min per response).
- ~30 min for judge pass.

## Headline prompt

Pelican on a monocycle (SVG, eyeball-renderable).

## Status

Not scheduled. Re-opens after Track A (deep-thinking bench) ships and its results are reviewed.
```

- [ ] **Step 3: Verify file exists and is non-empty**

```bash
test -s /home/michal/projects/vyskocilm/ollama/bench/future/VIBES-GALLERY.md && echo OK
```

Expected: `OK`

---

## Task 2: <think>-strip module (TDD)

**Files:**
- Create: `bench/deep/strip_thinking.py`
- Test: `bench/deep/test_strip.py`

- [ ] **Step 1: Write failing tests**

Create `bench/deep/test_strip.py`:

```python
"""Unit tests for strip_thinking.strip(text)."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from strip_thinking import strip


def test_no_thinking_passthrough():
    assert strip("def f(): return 1") == "def f(): return 1"


def test_single_closed_block():
    assert strip("<think>reasoning here</think>final answer") == "final answer"


def test_multiple_closed_blocks():
    assert strip(
        "<think>step1</think>middle<think>step2</think>end"
    ) == "middleend"


def test_closed_block_with_newlines():
    assert strip("<think>line1\nline2\nline3</think>answer") == "answer"


def test_unclosed_thinking_drops_to_eos():
    # Truncated at max_gen_toks: open tag, no close → drop to EOS
    assert strip("ok<think>started thinking but truncated") == "ok"


def test_channel_tag_variant():
    # Gemma 4 MoE emits <|channel|>...<|channel|> instead
    assert strip("<|channel|>thinking<|channel|>answer") == "answer"


def test_unclosed_channel_tag():
    assert strip("ok<|channel|>truncated thinking") == "ok"


def test_mixed_think_and_channel_in_same_response():
    text = "<think>think1</think>middle<|channel|>think2<|channel|>end"
    assert strip(text) == "middleend"


if __name__ == "__main__":
    import inspect
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and inspect.isfunction(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    sys.exit(0 if failed == 0 else 1)
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
mkdir -p /home/michal/projects/vyskocilm/ollama/bench/deep
cd /home/michal/projects/vyskocilm/ollama
python3.14 bench/deep/test_strip.py
```

Expected: `ModuleNotFoundError: No module named 'strip_thinking'` (because strip_thinking.py doesn't exist yet).

- [ ] **Step 3: Implement minimal strip_thinking.py**

Create `bench/deep/strip_thinking.py`:

```python
"""Robust <think>-block stripper for thinking-on model outputs.

Handles:
  * one or more closed <think>...</think> blocks
  * Gemma 4 MoE channel tags <|channel|>...<|channel|>
  * unclosed <think> (truncated at max_gen_toks → drop to EOS, treat as fail)
"""
import re

_PATTERNS = (
    (re.compile(r"<think>.*?</think>", re.DOTALL), ""),
    (re.compile(r"<\|channel\|>.*?<\|channel\|>", re.DOTALL), ""),
    (re.compile(r"<think>.*", re.DOTALL), ""),
    (re.compile(r"<\|channel\|>.*", re.DOTALL), ""),
)


def strip(text: str) -> str:
    """Strip thinking blocks from `text`. Returns the user-facing remainder."""
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    return text
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
python3.14 bench/deep/test_strip.py
```

Expected: all 8 tests PASS, exit code 0.

---

## Task 3: Patch openai_completions.py to use robust strip

**Files:**
- Modify: `~/.local/share/uv/tools/lm-eval/lib/python3.14/site-packages/lm_eval/models/openai_completions.py`

- [ ] **Step 1: Locate existing strip code**

```bash
grep -n "think\|channel" ~/.local/share/uv/tools/lm-eval/lib/python3.14/site-packages/lm_eval/models/openai_completions.py
```

Expected: a `re.sub` call with `<think>` and a similar one with `<\|channel\|>`. Note the line numbers — they're the regex strip that runs on each generation.

- [ ] **Step 2: Replace with 4-pattern strip**

Find the existing 2-line strip code (single leading `<think>...</think>` + leading `<|channel|>...<|channel|>` strip). Replace it with the same 4-pattern block used in `strip_thinking.py`. Inline (not an import — the lm-eval site-packages path can't reliably import from our `bench/deep/`):

```python
# Robust <think>-block strip for thinking-on outputs:
# closed-block, channel-tag variant, unclosed (truncated) → drop to EOS.
text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
text = re.sub(r"<\|channel\|>.*?<\|channel\|>", "", text, flags=re.DOTALL)
text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
text = re.sub(r"<\|channel\|>.*", "", text, flags=re.DOTALL)
```

Use the Edit tool with the exact existing lines as `old_string` and the 4-line block as `new_string`.

- [ ] **Step 3: Verify patch with a hand-crafted curl smoke**

```bash
# Start any small local model briefly, smoke a fake thinking response via curl:
# (we'll exercise this properly in Task 11; this step just confirms the file parses)
python3.14 -c "import importlib.util, pathlib; \
  p = pathlib.Path.home() / '.local/share/uv/tools/lm-eval/lib/python3.14/site-packages/lm_eval/models/openai_completions.py'; \
  spec = importlib.util.spec_from_file_location('oc', p); \
  m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); \
  print('parse OK')"
```

Expected: `parse OK` (no SyntaxError).

---

## Task 4: gsm8k_deep task YAML

**Files:**
- Create: `bench/deep/tasks/gsm8k_deep.yaml`
- Create: `~/.local/share/uv/tools/lm-eval/lib/python3.14/site-packages/lm_eval/tasks/gsm8k/gsm8k_deep.yaml` (sibling-of-parent copy for `include:` resolution)

- [ ] **Step 1: Write source YAML**

Create `bench/deep/tasks/gsm8k_deep.yaml`:

```yaml
include: gsm8k.yaml
task: gsm8k_deep
generation_kwargs:
  until: []
  max_gen_toks: 8192
  temperature: 0.6
  do_sample: true
metadata:
  version: 3.0-deep
```

- [ ] **Step 2: Mirror to upstream package dir**

```bash
mkdir -p /home/michal/projects/vyskocilm/ollama/bench/deep/tasks
cp /home/michal/projects/vyskocilm/ollama/bench/deep/tasks/gsm8k_deep.yaml \
   ~/.local/share/uv/tools/lm-eval/lib/python3.14/site-packages/lm_eval/tasks/gsm8k/gsm8k_deep.yaml
```

- [ ] **Step 3: Verify lm-eval discovers the task**

```bash
cd /home/michal/projects/vyskocilm/ollama
lm-eval --tasks list 2>&1 | grep gsm8k_deep
```

Expected: line containing `gsm8k_deep`.

---

## Task 5: humaneval_deep task YAML

**Files:**
- Create: `bench/deep/tasks/humaneval_deep.yaml`
- Create: `~/.local/share/uv/tools/lm-eval/lib/python3.14/site-packages/lm_eval/tasks/humaneval/humaneval_deep.yaml`

- [ ] **Step 1: Write source YAML**

Inherits from `humaneval_instruct_relaxed` (already in upstream pkg dir from earlier work):

```yaml
include: humaneval_instruct_relaxed.yaml
task: humaneval_deep
generation_kwargs:
  until: []
  max_gen_toks: 8192
  temperature: 0.6
  do_sample: true
metadata:
  version: 4.0-deep
```

- [ ] **Step 2: Mirror to upstream package dir**

```bash
cp /home/michal/projects/vyskocilm/ollama/bench/deep/tasks/humaneval_deep.yaml \
   ~/.local/share/uv/tools/lm-eval/lib/python3.14/site-packages/lm_eval/tasks/humaneval/humaneval_deep.yaml
```

- [ ] **Step 3: Verify task registers**

```bash
lm-eval --tasks list 2>&1 | grep humaneval_deep
```

Expected: line containing `humaneval_deep`.

---

## Task 6: mmlu_pro_deep task YAML

**Files:**
- Create: `bench/deep/tasks/mmlu_pro_deep.yaml`
- Create: `~/.local/share/uv/tools/lm-eval/lib/python3.14/site-packages/lm_eval/tasks/mmlu_pro/mmlu_pro_deep.yaml`

- [ ] **Step 1: Inspect existing mmlu_pro task layout**

```bash
ls ~/.local/share/uv/tools/lm-eval/lib/python3.14/site-packages/lm_eval/tasks/mmlu_pro/
```

Expected: see the parent task file (likely `default.yaml` or `_default.yaml` or `mmlu_pro.yaml`). Note the exact filename — `include:` must point to it.

- [ ] **Step 2: Write source YAML**

Replace `<PARENT>` with the actual mmlu_pro parent YAML filename observed in Step 1:

```yaml
include: <PARENT>
task: mmlu_pro_deep
generation_kwargs:
  max_gen_toks: 8192
  temperature: 0.6
  do_sample: true
metadata:
  version: 1.0-deep
```

- [ ] **Step 3: Mirror and verify**

```bash
cp /home/michal/projects/vyskocilm/ollama/bench/deep/tasks/mmlu_pro_deep.yaml \
   ~/.local/share/uv/tools/lm-eval/lib/python3.14/site-packages/lm_eval/tasks/mmlu_pro/mmlu_pro_deep.yaml
lm-eval --tasks list 2>&1 | grep mmlu_pro_deep
```

Expected: line containing `mmlu_pro_deep`.

---

## Task 7: math_deep task YAML

**Files:**
- Create: `bench/deep/tasks/math_deep.yaml`
- Create: corresponding copy in upstream `lm_eval/tasks/hendrycks_math/` (or wherever `hendrycks_math` lives)

- [ ] **Step 1: Locate hendrycks_math parent task**

```bash
find ~/.local/share/uv/tools/lm-eval/lib/python3.14/site-packages/lm_eval/tasks -type d -name "*math*"
```

Expected: a directory like `hendrycks_math` containing per-topic YAMLs.

- [ ] **Step 2: Write source YAML**

Use the topic-aggregate parent (likely `hendrycks_math.yaml` group). Replace `<PARENT>` with the observed parent YAML path:

```yaml
include: <PARENT>
task: math_deep
generation_kwargs:
  until: []
  max_gen_toks: 8192
  temperature: 0.6
  do_sample: true
num_fewshot: 0
metadata:
  version: 1.0-deep
```

- [ ] **Step 3: Mirror and verify**

```bash
cp /home/michal/projects/vyskocilm/ollama/bench/deep/tasks/math_deep.yaml \
   ~/.local/share/uv/tools/lm-eval/lib/python3.14/site-packages/lm_eval/tasks/hendrycks_math/math_deep.yaml
lm-eval --tasks list 2>&1 | grep math_deep
```

Expected: line containing `math_deep`.

---

## Task 8: Driver script `deep-bench.sh`

**Files:**
- Create: `bench/deep/deep-bench.sh`

- [ ] **Step 1: Write driver**

Create `bench/deep/deep-bench.sh`:

```bash
#!/usr/bin/env bash
# Deep-thinking bench driver for Track A.
# Usage:
#   ./deep-bench.sh smoke            # n=2/task on smollm3-3b (~10 min)
#   ./deep-bench.sh deep-reasoning   # gsm8k + humaneval + mmlu_pro on both models (~11h)
#   ./deep-bench.sh deep-math        # hendrycks_math on both models (~10h)
set -euo pipefail

ROOT=/home/michal/projects/vyskocilm/ollama
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

# pick model set: smoke = smollm3-3b only; real runs = both deep-bench models
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

  # start server
  log_dir="$OUT_BASE/$alias"
  mkdir -p "$log_dir"
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
  for i in $(seq 1 300); do
    if curl -s "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
      echo "  server up after ${i}s"
      break
    fi
    sleep 1
    [[ $i -eq 300 ]] && { echo "  server failed to start"; cat "$log_dir/server.log" | tail -30; exit 3; }
  done

  # run each task
  for idx in "${!TASKS[@]}"; do
    task="${TASKS[$idx]}"
    limit="${LIMITS[$idx]}"
    task_out="$log_dir/$task"
    mkdir -p "$task_out"

    # skip-existing: if results.json already there, skip
    if ls "$task_out"/results_*.json >/dev/null 2>&1; then
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
```

- [ ] **Step 2: Make executable**

```bash
chmod +x /home/michal/projects/vyskocilm/ollama/bench/deep/deep-bench.sh
```

- [ ] **Step 3: Verify shellcheck (if installed) or bash -n**

```bash
bash -n /home/michal/projects/vyskocilm/ollama/bench/deep/deep-bench.sh && echo "syntax OK"
```

Expected: `syntax OK`.

---

## Task 9: Aggregator `aggregate-deep.py`

**Files:**
- Create: `bench/deep/aggregate-deep.py`

- [ ] **Step 1: Write aggregator**

Create `bench/deep/aggregate-deep.py`:

```python
#!/usr/bin/env python3.14
"""Aggregate bench/deep/ results into DEEP-REPORT.md.

Reads:
  bench/deep/<sub-profile>/<model>/<task>/results_*.json   (lm-eval output)
  bench/lm-eval/<model>/results_*.json                     (fast-chat baseline)

Writes:
  bench/deep/DEEP-REPORT.md                                (regenerated in place)
"""
import json, pathlib, glob, sys
from datetime import datetime

ROOT = pathlib.Path(__file__).parent.parent.parent  # /home/michal/projects/vyskocilm/ollama
DEEP = ROOT / "bench/deep"
FAST = ROOT / "bench/lm-eval"

TASK_TO_FASTCHAT_TASK = {
    "gsm8k_deep": "gsm8k",
    "humaneval_deep": "humaneval_instruct",  # closest fast-chat counterpart
    "mmlu_pro_deep": "mmlu_pro",
    "math_deep": None,  # no fast-chat baseline ran on math
}


def load_score(json_path: pathlib.Path, task_name: str) -> tuple[float | None, float | None]:
    """Return (pass@1 or accuracy, stderr) for task_name in lm-eval results JSON. None if missing."""
    try:
        data = json.loads(json_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None, None
    results = data.get("results", {})
    task = results.get(task_name)
    if not task:
        return None, None
    # lm-eval keys vary by task; try pass@1, acc, acc_norm in order
    for key in ("pass@1,create_test", "pass@1", "exact_match,strict-match", "acc,none", "acc"):
        if key in task:
            v = task[key]
            stderr_key = key.replace(",none", "_stderr,none") if ",none" in key else f"{key}_stderr"
            stderr_key = stderr_key if stderr_key in task else None
            return float(v), float(task[stderr_key]) if stderr_key else None
    return None, None


def find_latest(pattern: str) -> pathlib.Path | None:
    matches = sorted(glob.glob(pattern))
    return pathlib.Path(matches[-1]) if matches else None


def main():
    rows = []  # (sub_profile, model, task, deep_score, deep_stderr, fast_score)
    for sub_profile in ("deep-reasoning", "deep-math"):
        sub_dir = DEEP / sub_profile
        if not sub_dir.exists():
            continue
        for model_dir in sorted(sub_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            for task_dir in sorted(model_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                task = task_dir.name
                deep_json = find_latest(str(task_dir / "results_*.json"))
                if not deep_json:
                    continue
                deep_score, deep_stderr = load_score(deep_json, task)
                fast_task = TASK_TO_FASTCHAT_TASK.get(task)
                fast_score = None
                if fast_task:
                    fast_json = find_latest(str(FAST / model / "results_*.json"))
                    if fast_json:
                        fast_score, _ = load_score(fast_json, fast_task)
                rows.append((sub_profile, model, task, deep_score, deep_stderr, fast_score))

    out = ["# Deep-thinking Bench Results", "",
           f"Generated: {datetime.now().isoformat(timespec='seconds')}", "",
           "Models: gemma-4-e2b, qwen3.6-35b-a3b. Mode: `enable_thinking=true`. See `docs/superpowers/specs/2026-05-15-deep-bench-design.md`.", ""]
    for sp in ("deep-reasoning", "deep-math"):
        sp_rows = [r for r in rows if r[0] == sp]
        if not sp_rows:
            continue
        out.append(f"## {sp}")
        out.append("")
        out.append("| model | task | thinking-on | stderr | fast-chat baseline | Δ pp |")
        out.append("|-------|------|-------------|--------|--------------------|------|")
        for _, model, task, deep_s, deep_se, fast_s in sp_rows:
            ds = f"{deep_s:.3f}" if deep_s is not None else "—"
            dse = f"±{deep_se:.3f}" if deep_se is not None else ""
            fs = f"{fast_s:.3f}" if fast_s is not None else "—"
            delta = f"{(deep_s - fast_s) * 100:+.1f}" if (deep_s is not None and fast_s is not None) else "—"
            out.append(f"| {model} | {task} | {ds} | {dse} | {fs} | {delta} |")
        out.append("")

    target = DEEP / "DEEP-REPORT.md"
    target.write_text("\n".join(out) + "\n")
    print(f"wrote {target} ({len(rows)} task-results)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script parses**

```bash
python3.14 -m py_compile /home/michal/projects/vyskocilm/ollama/bench/deep/aggregate-deep.py && echo "parse OK"
```

Expected: `parse OK`.

- [ ] **Step 3: Run on empty state (no results yet)**

```bash
cd /home/michal/projects/vyskocilm/ollama
python3.14 bench/deep/aggregate-deep.py
```

Expected: writes a near-empty `bench/deep/DEEP-REPORT.md` with header but no rows, exits 0.

---

## Task 10: REPORT.md pointer

**Files:**
- Modify: `bench/REPORT.md` (add 1 line near top)

- [ ] **Step 1: Add pointer line below the title**

Edit `bench/REPORT.md` to insert this line directly after the H1 title:

```markdown
> **Deep-thinking results** (Gemma 4 / Qwen 3.6 with `enable_thinking=true`) live in [`bench/deep/DEEP-REPORT.md`](deep/DEEP-REPORT.md). This document covers the fast-chat / `enable_thinking=false` mode only.
```

Use the Edit tool with the H1 `# Local OSS LLM Bench — Ryzen AI 7 PRO + Radeon 860M (gfx1151)` line as the unique anchor.

- [ ] **Step 2: Verify the line is present**

```bash
grep -n "DEEP-REPORT" /home/michal/projects/vyskocilm/ollama/bench/REPORT.md
```

Expected: line number near the top + matching content.

---

## Task 11: Smoke run (validation stages 0-5)

This task gates the overnight runs. Cheap insurance.

- [ ] **Step 1: Tokenizer pre-flight (spec §9 stage 0)**

```bash
python3.14 -c "
from transformers import AutoTokenizer
for ref in ['google/gemma-4-E2B-it', 'Qwen/Qwen3.6-35B-A3B', 'HuggingFaceTB/SmolLM3-3B']:
    t = AutoTokenizer.from_pretrained(ref)
    print(f'{ref}: vocab_size={t.vocab_size}')
"
```

Expected: 3 lines, each ending in `vocab_size=<number>`. If any fails with sentencepiece/tiktoken error, run:
```bash
uv tool install lm-eval --force --with sentencepiece --with tiktoken
```
and retry.

- [ ] **Step 2: Strip-function unit tests (spec §9, complements stage 2)**

```bash
cd /home/michal/projects/vyskocilm/ollama
python3.14 bench/deep/test_strip.py
```

Expected: all 8 PASS, exit 0.

- [ ] **Step 3: `deep-bench.sh smoke` (spec §9 stages 2-3)**

```bash
cd /home/michal/projects/vyskocilm/ollama
./bench/deep/deep-bench.sh smoke 2>&1 | tee bench/deep/smoke.log
```

Expected wall-clock: < 15 min. At end, 4 result dirs exist under `bench/deep/smoke/smollm3-3b/{gsm8k_deep,humaneval_deep,mmlu_pro_deep,math_deep}/` each containing a `results_*.json`.

- [ ] **Step 4: Sanity score check (spec §9 stage 3)**

```bash
python3.14 bench/deep/aggregate-deep.py
cat bench/deep/DEEP-REPORT.md
```

Expected: smollm3-3b row shows gsm8k score > 0.0, humaneval > 0.0. (n=2 is too small for absolute thresholds; we only care that *something* non-zero comes out, proving the strip + plumbing work.)

If any score is 0.0 with empty completions, debug `<think>` strip — likely the strip regex consumed the answer. Inspect `bench/deep/smoke/smollm3-3b/<task>/lm-eval.log` for a sample request/response.

- [ ] **Step 5: One real model, one real task, n=10 (spec §9 stage 4)**

```bash
# Hand-craft a one-shot call to verify gemma-4-e2b thinking-on works end-to-end
cd /home/michal/projects/vyskocilm/ollama
# Temporary: override smoke list to gemma-4-e2b + gsm8k_deep + n=10
EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":true}}' \
PORT=18080 \
bash -c '
ROOT=/home/michal/projects/vyskocilm/ollama
"$ROOT/dist/bin/llama-server" -m "$ROOT/models/gemma-4-E2B-it-UD-Q4_K_XL.gguf" \
  --host 127.0.0.1 --port 18080 --alias gemma-4-e2b \
  --n-gpu-layers 999 --ctx-size 131072 --threads 8 -b 512 -ub 1024 \
  --jinja --seed 42 --no-warmup --parallel 4 > /tmp/g4-stage5.log 2>&1 &
SERVER_PID=$!
for i in $(seq 1 300); do curl -s http://127.0.0.1:18080/v1/models > /dev/null && break; sleep 1; done
mkdir -p bench/deep/stage5
LM_EVAL_EXTRA_BODY=$EXTRA_BODY lm-eval \
  --model local-chat-completions \
  --model_args "base_url=http://127.0.0.1:18080/v1/chat/completions,model=gemma-4-e2b,num_concurrent=4,tokenized_requests=False,tokenizer_backend=huggingface,tokenizer=google/gemma-4-E2B-it,timeout=3600" \
  --tasks gsm8k_deep --limit 10 --output_path bench/deep/stage5/ --log_samples --seed 42 \
  > bench/deep/stage5/driver.log 2>&1
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
grep "exact_match\|pass@1" bench/deep/stage5/driver.log | tail -3
'
```

Expected wall-clock: ~5-10 min. Score on gsm8k should be > 0.85 (thinking-on should be at least at the current fast-chat baseline of ~0.85; ideally higher).

If score is < 0.85, EXTRA_BODY is not being applied. Verify by grepping `bench/deep/stage5/lm-eval.log` (sample log) for `<think>` in any prompt response. If no `<think>` blocks visible, plumbing is broken — debug before queuing overnight.

---

## Task 12: Run `deep-reasoning` overnight

- [ ] **Step 1: Launch in background**

```bash
cd /home/michal/projects/vyskocilm/ollama
nohup ./bench/deep/deep-bench.sh deep-reasoning > bench/deep/deep-reasoning.run.log 2>&1 &
echo "deep-reasoning PID: $!"
```

Expected wall-clock: ~11h. Pid recorded so it can be killed if needed.

- [ ] **Step 2: Monitor first 20 min for fast-fail**

```bash
sleep 1200
tail -30 /home/michal/projects/vyskocilm/ollama/bench/deep/deep-reasoning.run.log
```

Expected: server-up message + at least one `task <name> limit=200` line + progress indicators. No FAILED lines.

If a task fast-failed (banner-to-banner < 5 min), abort and diagnose before continuing.

- [ ] **Step 3: Wait for completion**

Either monitor periodically or wait for the `SUB-PROFILE deep-reasoning done` line in the log. Total ~11h.

- [ ] **Step 4: Aggregate**

```bash
python3.14 /home/michal/projects/vyskocilm/ollama/bench/deep/aggregate-deep.py
cat /home/michal/projects/vyskocilm/ollama/bench/deep/DEEP-REPORT.md
```

Expected: 6 rows in the `deep-reasoning` table (2 models × 3 tasks), with non-zero scores and visible Δ pp values vs fast-chat baselines.

---

## Task 13: Run `deep-math` overnight

- [ ] **Step 1: Launch in background**

```bash
cd /home/michal/projects/vyskocilm/ollama
nohup ./bench/deep/deep-bench.sh deep-math > bench/deep/deep-math.run.log 2>&1 &
echo "deep-math PID: $!"
```

Expected wall-clock: ~10h.

- [ ] **Step 2: Monitor first 20 min**

```bash
sleep 1200
tail -30 /home/michal/projects/vyskocilm/ollama/bench/deep/deep-math.run.log
```

Expected: similar to Task 12 Step 2.

- [ ] **Step 3: Wait for completion**

Total ~10h.

- [ ] **Step 4: Final aggregate**

```bash
python3.14 /home/michal/projects/vyskocilm/ollama/bench/deep/aggregate-deep.py
cat /home/michal/projects/vyskocilm/ollama/bench/deep/DEEP-REPORT.md
```

Expected: `deep-math` table has 2 rows (one per model) with scores + (math has no fast-chat baseline → fast-chat / Δ columns show `—`).

---

## Self-review

Coverage:
- spec §4.1 models (2 models) → Task 8 MODELS array
- spec §4.2 sub-profiles + n → Task 8 PROFILES + Tasks 12/13
- spec §4.3 runtime config (enable_thinking, max_gen_toks, temperature) → Tasks 4-7 YAMLs + Task 8 EXTRA_BODY
- spec §5 architecture (dir layout, separation) → Tasks 1, 4-9
- spec §6.1 task YAMLs → Tasks 4-7
- spec §6.2 EXTRA_BODY → Task 8
- spec §6.3 strip patch → Tasks 2, 3
- spec §6.4 driver → Task 8
- spec §6.5 aggregator → Task 9
- spec §8 error handling → Task 8 (server poll + kill + skip-existing) and Task 9 (graceful JSON-missing handling)
- spec §9 validation stages → Task 11
- spec §11 Track B stub → Task 1

Placeholder scan: only `<PARENT>` placeholders in Tasks 6 + 7, which are resolved by Step 1 of each (inspecting the upstream task dir to find the actual filename). All other steps contain concrete commands and code.

Type / name consistency: `EXTRA_BODY` env name used identically in Task 8 + Task 11 Step 5. Task names `gsm8k_deep / humaneval_deep / mmlu_pro_deep / math_deep` consistent across Tasks 4-9. `MODEL_CTX` field in spec §6.4 is realized as `ctx-size` in the MODELS table — matches the `--ctx-size` llama-server flag.

No issues found. Plan is implementation-ready.
