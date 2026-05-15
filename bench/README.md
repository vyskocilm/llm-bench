# bench/

Reproducible 4-task benchmark across OSS-licensed models (Apache-2.0 / MIT
only — no Llama Community, DeepSeek, or Gemma-1/2/3 weights, which carry
field-of-use restrictions that fail the OSD. Gemma 4 is genuine Apache 2.0
and *is* included).

## Tasks

| Task         | What it tests                       | Scorer                       |
|--------------|--------------------------------------|------------------------------|
| `analyze`    | reading comprehension on a passage   | keyword match in answer      |
| `prose`      | short creative generation            | length + type/token rubric   |
| `codegen`    | write a Python function              | execute unit tests           |
| `codereview` | locate a planted off-by-one bug      | keyword match (range/index)  |

Image classification was dropped: `moondream/moondream2-gguf` + `--jinja`
emits a 1-token EOS instead of an answer, and no other small OSS VLM has
working llama.cpp plumbing yet. For VLM evaluation, use a dedicated tool
(see *Better-benchmark targets* below).

All prompts, sampler params, and the seed are baked into `bench.py`. Same
inputs → same outputs (greedy decoding for all tasks except `prose` which
uses `temperature=0.7`).

## Models

| ID                   | Origin               | License    | Size  | Modality |
|----------------------|----------------------|------------|-------|----------|
| `smollm2-1.7b`       | HuggingFaceTB        | Apache-2.0 | ~1.1G | text |
| `olmo2-7b`           | Allen AI             | Apache-2.0 | ~4.7G | text |
| `mistral-7b-v0.3`    | Mistral (Apache GGUF via bartowski) | Apache-2.0 | ~4.4G | text |
| `qwen3-8b`           | Alibaba/Qwen         | Apache-2.0 | ~4.7G | text |
| `mistral-nemo-2407`  | Mistral × NVIDIA     | Apache-2.0 | ~7.0G | text |
| `phi-4`              | Microsoft            | MIT        | ~8.5G | text |
| `gemma-4-e2b`        | Google               | Apache-2.0 | ~2.9G | text |

## Setup

1. Build `dist/bin/llama-server` (see top-level `README.md` Build section).
2. Download the model files into `models/` — the script prints the exact
   `pull-model.sh` invocations on first run:

   ```sh
   python3 bench/bench.py
   # → Missing model files; download into models/:
   #   MODEL=HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF FILE=smollm2-1.7b-instruct-q4_k_m.gguf ./pull-model.sh
   #   ...
   ```

3. (Optional) `Pillow` if you re-enable any image task. With `uv`:

   ```sh
   uv pip install --system Pillow      # or: uv tool install ...
   ```

## Run

```sh
python3 bench/bench.py
```

Optional flags:

- `--only-models smollm2-1.7b olmo2-7b` — restrict to a subset
- `--only-tasks analyze codegen` — restrict to a subset
- `--bin /path/to/llama-server` — override binary
- `--out bench/results.json` — output path (also dumps per-model server
  logs under `bench/logs/<model>.log`)

The script starts a fresh `llama-server` (port 18080, all-GPU offload) for
each model, runs every task, kills it, and moves on.

## HTML report

After `bench.py` writes `results.json`, render a static page:

```sh
python3 bench/render.py
xdg-open bench/results.html
```

Single-file HTML with mvp.css from CDN. Score cells link to per-prompt
detail blocks (full model output, score detail JSON, latency, token counts).
Cell colour: green ≥ 0.99, amber ≥ 0.5, orange > 0, red = 0.

## Validation

`results.json` includes:

- `schema` version + `seed`
- the exact model list (HF repo + filename)
- task names that ran
- per-(model, task, prompt) `{output, score, detail, latency_s}`
- the `dist/BUILDINFO` snapshot (llama.cpp commit, CPU, build host)

Anyone re-running on the same llama.cpp commit, same models, same hardware
should see identical outputs for the deterministic tasks (`analyze`,
`codegen`, `codereview`). `prose` is non-deterministic (temperature 0.7);
its rubric is the stable signal.

## Adding models / tasks

Add a `ModelSpec(...)` entry to `MODELS` in `bench.py`. Verify the upstream
license is OSI-style (MIT / Apache / BSD / similar without field-of-use
restrictions). For tasks, append to `TASKS` and `SCORERS`.

## Caveats

- Single run per prompt — no multi-sample averaging. Sufficient to
  rank-order strong vs. weak; weak signal between similarly-sized models.
- `prose` rubric (length + lexical diversity) is a coarse proxy for
  "readable poem" — substitute a stronger judge (e.g. another LLM scoring
  on a rubric) for a real comparison.

## Better-benchmark targets

`bench.py` here is a sanity harness — 4 hand-picked prompts, scored by
keyword/exec/rubric. For headline numbers you want established benchmarks
with real test sets and consensus scoring methods. Standards worth running
against llama-server on the same hardware:

### Standard text benchmarks

| Suite | Tests | Notes |
|-------|-------|-------|
| **lm-evaluation-harness** (EleutherAI) | wrapper for all the below; talks to llama-server's `/v1/completions` & `/v1/chat/completions` | The de-facto runner. `pip install lm-eval` + `lm-eval --model local-completions --tasks mmlu,gsm8k,humaneval --base_url http://127.0.0.1:8080/v1`. |
| **MMLU** | 14k multiple-choice, 57 subjects | Mainstream knowledge benchmark. |
| **MMLU-Pro** | harder, more reasoning | Modern successor; less saturated. |
| **GSM8K** | 8.5k grade-school math word problems | Exact-match on final numeric answer. |
| **MATH** | 12.5k competition math | Harder than GSM8K. |
| **HumanEval / HumanEval+** | 164 Python functions, unit-test scored | Industry standard for codegen. |
| **MBPP / MBPP+** | 974 basic Python problems | Lower-bar than HumanEval. |
| **BBH** (Big-Bench Hard) | 23 hard reasoning tasks | Strong signal vs. saturating benchmarks. |
| **IFEval** | 541 instruction-following checks | Verifiable constraints (programmatic), not LLM-judged. |
| **TruthfulQA** | factuality / misconceptions | Single-MC and generation variants. |
| **HellaSwag / WinoGrande / ARC** | common-sense + science MC | Cheap, run in seconds; often part of "OpenLLM Leaderboard v2" rollup. |
| **AGIEval** | reasoning + comprehension | Drawn from human exams (SAT, LSAT, etc.). |
| **GPQA** | graduate-level science MC | Designed to resist memorisation. |
| **LiveCodeBench** | contest problems, time-stamped | Less contaminated than HumanEval. |

### Open-ended / judge-based

| Suite | Notes |
|-------|-------|
| **MT-Bench** | 80 multi-turn prompts, GPT-4-judged in original; can use a local judge |
| **AlpacaEval 2** | 805 prompts, LC-judged; rank vs. reference |
| **Arena-Hard-Auto** | 500 prompts curated from Chatbot Arena, LLM-judged |

Judge-based suites need a strong judge model (typically a frontier model
via API). Don't bother unless you have a judge budget.

### Vision (for when a VLM is wired up)

| Suite | Notes |
|-------|-------|
| **MMMU** | Multimodal MMLU; expert-level vision-language reasoning |
| **MMBench / MMStar** | Comprehensive VLM eval |
| **TextVQA, DocVQA, ChartQA** | Domain-specific (OCR, documents, charts) |
| **POPE** | Object-hallucination probe |

### Profiles (BENCH_PROFILE)

`bench/lm-eval-run.sh` is profile-driven. Pick the trade-off between
wall-clock and statistical power via the `BENCH_PROFILE` env var:

| `BENCH_PROFILE` | tasks | samples/task | wall-clock @ 7B | typical stderr |
|-----------------|-------|--------------|------------------|----------------|
| `quick` | 5 curated | 30 | ~10–15 min | ±~10% |
| `core` (default) | 5 curated | 250 | ~30–45 min | ±~2–3% |
| `full` | 12 wide | full splits | 8–12 h | ±~1% |

The curated 5 cover orthogonal capabilities so models can be rank-ordered
without redundant signal:

| capability | task |
|------------|------|
| instruction-following | IFEval (prompt_strict / inst_strict) |
| math reasoning | GSM8K (CoT, exact-match on final number) |
| code generation | HumanEval (pass@1 via unit tests) |
| MC reasoning | ARC-Challenge |
| common-sense MC | HellaSwag |

`full` adds MBPP, DROP, TruthfulQA-gen, ARC-Easy, WinoGrande, PIQA,
TruthfulQA-MC1 for breadth. Override per-group via `TASKS_CHAT`,
`TASKS_GEN`, `TASKS_MC`, `GEN_LIMIT`, `MC_LIMIT` env vars if you want
a specific mix.

```sh
# default (core): 5 tasks, ~250 samples each, ~45 min
bench/lm-eval-run.sh models/smollm2-1.7b-instruct-q4_k_m.gguf smollm2-1.7b HuggingFaceTB/SmolLM2-1.7B-Instruct

# quick sanity check (~15 min)
BENCH_PROFILE=quick bench/lm-eval-run.sh ...

# headline numbers (~8-12 h)
BENCH_PROFILE=full  bench/lm-eval-run.sh ...

# custom: only math+code, 500 samples each
TASKS_CHAT='' TASKS_GEN=gsm8k,humaneval GEN_LIMIT=500 \
  bench/lm-eval-run.sh ...
```

### Speculative decoding (optional)

`bench/lm-eval-run.sh` takes an optional 4th positional arg: a draft-model
GGUF. When set, llama-server runs with `--spec-draft-model …` and the
generation path becomes speculative: the draft model proposes up to
`DRAFT_N_MAX` tokens per step, the target verifies them in one forward
pass, accepting all that exceed `DRAFT_P_MIN`.

```sh
bench/lm-eval-run.sh \
  models/phi-4-Q4_K_M.gguf phi-4 microsoft/phi-4 \
  models/phi-4-mini-instruct-Q4_K_M.gguf       # 3.8B draft for the 14B target
```

Tuning env vars (defaults match llama.cpp's):

| var | default | meaning |
|-----|---------|---------|
| `DRAFT_N_MAX` | 16 | max draft tokens per step |
| `DRAFT_N_MIN` | 0 | min draft tokens (0 = don't enforce) |
| `DRAFT_P_MIN` | 0.75 | min draft-token probability for acceptance (greedy) |
| `DRAFT_NGL` | 999 | GPU layers for the draft (all of them by default) |

Speedup depends on **family match** and **size ratio**:

| target | draft | typical speedup |
|--------|-------|-----------------|
| Phi-4 14B | Phi-4-mini 3.8B | 1.5–2× (same family) |
| Qwen3-8B | Qwen3-0.6B | 1.4–1.8× |
| Mistral-Nemo 12B | Mistral-7B-v0.3 | 1.2–1.4× (related-family, smaller jump) |
| SmolLM2-1.7B | SmolLM2-360M | 1.1–1.3× (target already memory-bound at 1.7B) |
| any | unrelated family | usually slower than no draft |

Acceptance rate drops sharply across architectures or tokenizers, so the
draft *must* share the same tokenizer as the target. Mixing
families gives <30% acceptance and net regression.

### Practical next step for this repo

Wire `lm-evaluation-harness` to `llama-server`. Use the distro's `uv`
package (do *not* use the upstream `curl | sh` installer — package-managed
binary keeps host trust boundary intact and survives Tumbleweed snapshots).

```sh
# install once on the host (immutable Tumbleweed):
sudo zypper in uv

# run lm-eval in an ephemeral uv-managed env — no global install
uv tool run --from lm-eval lm-eval \
    --model local-completions \
    --model_args base_url=http://127.0.0.1:8080/v1,model=qwen3-8b \
    --tasks mmlu,gsm8k,humaneval,ifeval,arc_challenge \
    --batch_size 1 \
    --output_path bench/lm-eval/

# or: persistent install as a tool (still isolated)
uv tool install lm-eval
lm-eval --model local-completions --model_args ... --tasks ...
```

Replace `qwen3-8b` with each entry from our `MODELS` registry; the harness
handles few-shot prompts, chain-of-thought variants, and proper exact-match
scoring per benchmark. Output JSON merges trivially with our existing
`results.json` layout for the same `render.py` to consume.

Why `uv` over `pip` in 2026: 10–100× faster, doesn't require system Python
to have `pip`/`ensurepip` (matters on immutable distros), per-tool isolation
without manual venv management, and `uv tool run` lets you invoke
one-shot without installing anything globally.
- `image-class` synthetic image is a 50/50 red/blue split — trivial for any
  competent VLM. Replace with a labelled image dataset for harder evaluation.
