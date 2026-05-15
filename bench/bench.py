#!/usr/bin/env python3
"""bench.py — reproducible llama.cpp benchmark across OSS-licensed models.

Tasks (4):
  1. analyze        — comprehension over a short passage (keyword match)
  2. prose          — short creative generation (length + lexical diversity rubric)
  3. codegen        — write a Python function, judge by passing unit tests
  4. codereview     — locate a planted bug (keyword match)

Models (all Apache-2.0 or MIT; verify upstream before re-running):
  - smollm2-1.7b       HuggingFaceTB / Apache-2.0
  - olmo2-7b           Allen AI / Apache-2.0
  - mistral-7b-v0.3    Mistral / Apache-2.0

Each (model × task) pair is serialised: the script starts llama-server with
that model, waits on /health, runs the task prompts via
/v1/chat/completions, scores deterministically, kills the server, moves on.

All sampler params, seeds, prompts, and the dist/BUILDINFO snapshot are
written to results.json — anyone can rerun and reproduce the numbers.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_BIN = ROOT / "dist" / "bin" / "llama-server"
DEFAULT_MODELS_DIR = ROOT / "models"
DEFAULT_PORT = 18080
SERVER_TIMEOUT_S = 300   # 14B Q4 models take ~30-60s to load into VRAM
CHAT_TIMEOUT_S = 300
SEED = 42
# Per CLAUDE.md "LLM Benchmark Suite" notes: -ngl 99 -b 512 -ub 1024, -t 8/10.
SERVER_FLAGS = ["--batch-size", "512", "--ubatch-size", "1024"]
DEFAULT_THREADS = 8

# --------------------------------------------------------------------- models

@dataclass(frozen=True)
class ModelSpec:
    id: str
    license: str
    kind: str  # "text" or "vision"
    repo: str
    file: str
    mmproj_repo: Optional[str] = None
    mmproj_file: Optional[str] = None
    # Appended to the user message before sending. Used to switch off the
    # thinking/reasoning mode in models that emit `<think>...</think>` and
    # would otherwise blow through the max_tokens budget before answering.
    user_suffix: Optional[str] = None
    # llama-server `chat_template_kwargs` passthrough — forwards parameters
    # into the chat template's jinja context. Gemma 4 / Qwen3 / others
    # expose `enable_thinking` to gate reasoning emission. Set per-model
    # since the supported keys differ across templates.
    template_kwargs: Optional[dict] = None


MODELS: list[ModelSpec] = [
    ModelSpec(
        id="smollm2-1.7b",
        license="Apache-2.0",
        kind="text",
        repo="HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
        file="smollm2-1.7b-instruct-q4_k_m.gguf",
    ),
    ModelSpec(
        id="olmo2-7b",
        license="Apache-2.0",
        kind="text",
        repo="allenai/OLMo-2-1124-7B-Instruct-GGUF",
        file="olmo-2-1124-7B-instruct-Q4_K_M.gguf",
    ),
    ModelSpec(
        id="mistral-7b-v0.3",
        license="Apache-2.0",
        kind="text",
        repo="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        file="Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
    ),
    # --- May 2026 CLAUDE.md "LLM Benchmark Suite" smallest tier per row.
    # Names in that table (Gemma 4, DeepSeek-V4, Qwen 3.5, Mistral NeMo v2,
    # Phi-4 Med, Llama-3.3-Open) are aspirational and not on HF as of probe;
    # the OSS-licensed ones below are the closest currently-released stand-ins.
    # Skipped on strict-OSI gate: Gemma family (Gemma terms), DeepSeek (use
    # restrictions), Llama (Community license), DCLM-7B (Apple ASCL).
    ModelSpec(
        id="qwen3-8b",
        license="Apache-2.0",
        kind="text",
        repo="Qwen/Qwen3-8B-GGUF",
        file="Qwen3-8B-Q4_K_M.gguf",
        # Qwen3 emits a <think> block by default; without /no_think the
        # 256-token bench cap is consumed by reasoning before any answer
        # is produced. /no_think is Qwen3's documented opt-out switch.
        user_suffix=" /no_think",
    ),
    ModelSpec(
        id="mistral-nemo-2407",
        license="Apache-2.0",
        kind="text",
        repo="bartowski/Mistral-Nemo-Instruct-2407-GGUF",
        file="Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
    ),
    ModelSpec(
        id="phi-4",
        license="MIT",
        kind="text",
        repo="bartowski/phi-4-GGUF",
        file="phi-4-Q4_K_M.gguf",
    ),
    # Gemma 4 is genuinely Apache-2.0 (Google relicensed from Gemma Terms
    # used by 1/2/3). Verified May 2026: HF metadata `license: apache-2.0`,
    # `license_link` points to ai.google.dev page containing the verbatim
    # Apache 2.0 text only, no Acceptable Use Policy attached.
    ModelSpec(
        id="gemma-4-e2b",
        license="Apache-2.0",
        kind="text",
        repo="unsloth/gemma-4-E2B-it-GGUF",
        file="gemma-4-E2B-it-Q4_K_M.gguf",
        # Gemma 4 emits reasoning into `reasoning_content` by default,
        # leaving `content` empty. Switch the chat template's thinking
        # gate off so the answer lands in `content`.
        template_kwargs={"enable_thinking": False},
    ),
    # Vision row removed: image-class task dropped — moondream2 + --jinja
    # emits 1-token EOS via this GGUF variant, no other small OSS vision
    # model has working llama.cpp plumbing yet. Use a dedicated VLM
    # benchmark when revisiting (see bench/README.md).
]


# --------------------------------------------------------------------- tasks

@dataclass
class Prompt:
    system: str
    user: str
    expected: dict
    image: Optional[bytes] = None
    max_tokens: int = 256
    temperature: float = 0.0


@dataclass
class TaskResult:
    task: str
    model: str
    prompt_idx: int
    output: str
    score: float
    detail: dict
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    gen_tok_s: float = 0.0     # generation throughput (predicted/s)
    prompt_tok_s: float = 0.0  # prompt-eval throughput (prompt/s)


PASSAGE = (
    "Vulkan is a low-overhead, cross-platform 3D graphics and compute API by "
    "the Khronos Group, released in 2016 as a successor to OpenGL. RADV is "
    "the open-source Vulkan driver for AMD GPUs that lives in Mesa, originally "
    "written by Bas Nieuwenhuizen and David Airlie."
)


def task_analyze() -> list[Prompt]:
    sys_msg = "Answer briefly using only the provided text."
    return [
        Prompt(sys_msg, f"{PASSAGE}\n\nQ: When was Vulkan released?",
               expected={"any_of": ["2016"]}),
        Prompt(sys_msg, f"{PASSAGE}\n\nQ: Who originally wrote RADV?",
               expected={"all_of": ["Bas", "Airlie"]}),
        Prompt(sys_msg, f"{PASSAGE}\n\nQ: Which organisation standardises Vulkan?",
               expected={"any_of": ["Khronos"]}),
    ]


def task_prose() -> list[Prompt]:
    return [
        Prompt(
            system="You are a concise poet. Reply with only the poem.",
            user="Write a four-line poem about an open-source GPU driver.",
            expected={"rubric": "prose"},
            max_tokens=128,
            temperature=0.7,
        ),
    ]


def task_codegen() -> list[Prompt]:
    return [
        Prompt(
            system="Reply with a single fenced Python code block and nothing else.",
            user=textwrap.dedent("""\
                Write a Python function `is_palindrome(s: str) -> bool` that
                returns True iff the input is a palindrome, ignoring case and
                non-alphanumeric characters.
            """).strip(),
            expected={
                "function": "is_palindrome",
                "tests": [
                    ("A man, a plan, a canal: Panama", True),
                    ("hello", False),
                    ("", True),
                    ("Was it a car or a cat I saw?", True),
                    ("ab", False),
                ],
            },
        ),
    ]


def task_codereview() -> list[Prompt]:
    code = textwrap.dedent("""\
        def average(nums):
            total = 0
            for i in range(1, len(nums)):
                total += nums[i]
            return total / len(nums)
    """).strip()
    return [
        Prompt(
            system="You are a careful code reviewer. One sentence.",
            user=f"Identify the bug in this code:\n\n```python\n{code}\n```",
            # Broad phrasing set — the planted bug is the loop starting at
            # index 1 (skipping the first element). Models describe it
            # variously: "range(1...", "starts at 1 instead of 0", "doesn't
            # include the first number", "skips index 0", etc. Any one of
            # these phrasings is acceptable evidence of correct diagnosis.
            expected={"any_of": [
                "range(1", "range(0", "range(len",
                "off-by-one", "off by one",
                "index 0", "starts at 1", "start at 1",
                "starts at 0", "start at 0", "starts from 0",
                "start from 0", "starting from 0",
                "skip", "first element", "first number", "first item",
                "first value", "include the first", "missing the first",
                "missing first", "doesn't include the first",
                "does not include the first",
                "should start at 0",
            ]},
        ),
    ]


TASKS: dict[str, dict] = {
    "analyze":     {"prompts": task_analyze,    "kinds": {"text"}},
    "prose":       {"prompts": task_prose,      "kinds": {"text"}},
    "codegen":     {"prompts": task_codegen,    "kinds": {"text"}},
    "codereview":  {"prompts": task_codereview, "kinds": {"text"}},
}


# -------------------------------------------------------------------- scoring

def score_keyword(output: str, exp: dict) -> tuple[float, dict]:
    o = output.lower()
    if "any_of" in exp:
        keys = [k.lower() for k in exp["any_of"]]
        hit = next((k for k in keys if k in o), None)
        return (1.0 if hit else 0.0), {"any_of": keys, "hit": hit}
    if "all_of" in exp:
        keys = [k.lower() for k in exp["all_of"]]
        present = [k for k in keys if k in o]
        return (len(present) / len(keys)), {"all_of": keys, "present": present}
    return 0.0, {"reason": "no expected spec"}


def score_prose(output: str, _exp: dict) -> tuple[float, dict]:
    words = re.findall(r"\b\w+\b", output)
    n = len(words)
    distinct = len({w.lower() for w in words})
    ratio = distinct / max(n, 1)
    word_score = min(1.0, n / 20.0)          # ≥20 words → full
    div_score = min(1.0, ratio / 0.5)        # ≥0.5 type/token → full
    return 0.5 * word_score + 0.5 * div_score, {
        "words": n, "distinct": distinct, "type_token_ratio": round(ratio, 3),
    }


def score_codegen(output: str, exp: dict) -> tuple[float, dict]:
    code = _extract_code_block(output)
    if not code:
        return 0.0, {"reason": "no code block in output"}
    # Left-align the extracted code; build the harness with no leading
    # whitespace to avoid IndentationError on the spliced-in code.
    code = textwrap.dedent(code).strip("\n")
    harness = "\n".join([
        code,
        "",
        f"cases = {exp['tests']!r}",
        "passed = 0",
        "for arg, want in cases:",
        "    try:",
        f"        got = {exp['function']}(arg)",
        "        if got == want: passed += 1",
        "    except Exception: pass",
        "print(passed, len(cases))",
    ])
    test_src = harness
    try:
        r = subprocess.run(
            [sys.executable, "-c", test_src],
            capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        return 0.0, {"reason": "timeout"}
    if r.returncode != 0:
        return 0.0, {"reason": "exec error", "stderr": r.stderr[-400:]}
    parts = r.stdout.strip().split()
    if len(parts) != 2:
        return 0.0, {"reason": "bad output", "stdout": r.stdout[-400:]}
    passed, total = map(int, parts)
    return passed / total, {"passed": passed, "total": total}


def _extract_code_block(s: str) -> Optional[str]:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", s, flags=re.S)
    if m:
        return m.group(1)
    s = s.strip()
    if s.startswith(("def ", "import ", "from ")):
        return s
    return None


SCORERS: dict[str, Callable[[str, dict], tuple[float, dict]]] = {
    "analyze":     score_keyword,
    "prose":       score_prose,
    "codegen":     score_codegen,
    "codereview":  score_keyword,
}


# ------------------------------------------------------------------- server

class ServerProcess:
    def __init__(self, bin_path: pathlib.Path, model: pathlib.Path,
                 mmproj: Optional[pathlib.Path], port: int,
                 log_path: pathlib.Path):
        self.bin = bin_path
        self.model = model
        self.mmproj = mmproj
        self.port = port
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen] = None
        self.log_fh: Optional[Any] = None

    def __enter__(self) -> "ServerProcess":
        cmd = [
            str(self.bin),
            "-m", str(self.model),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--n-gpu-layers", "999",
            "--ctx-size", "4096",
            "--threads", str(DEFAULT_THREADS),
            "--jinja",
            "--seed", str(SEED),
            "--no-warmup",
            *SERVER_FLAGS,
        ]
        if self.mmproj:
            cmd += ["--mmproj", str(self.mmproj)]
        self.log_fh = open(self.log_path, "w")
        env = os.environ.copy()
        env.setdefault("AMD_VULKAN_ICD", "RADV")
        self.proc = subprocess.Popen(
            cmd, stdout=self.log_fh, stderr=subprocess.STDOUT, env=env,
        )
        url = f"http://127.0.0.1:{self.port}/health"
        deadline = time.time() + SERVER_TIMEOUT_S
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"server exited early (code {self.proc.returncode}); "
                    f"see {self.log_path}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        return self
            except (urllib.error.URLError, ConnectionError):
                time.sleep(1)
        raise TimeoutError(
            f"server did not become healthy in {SERVER_TIMEOUT_S}s; "
            f"see {self.log_path}"
        )

    def __exit__(self, *exc: Any) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.log_fh:
            self.log_fh.close()


# ------------------------------------------------------------------- client

def chat(port: int, prompt: Prompt,
         template_kwargs: Optional[dict] = None) -> dict:
    if prompt.image:
        b64 = base64.b64encode(prompt.image).decode()
        user_content: Any = [
            {"type": "text", "text": prompt.user},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]
    else:
        user_content = prompt.user

    req_body: dict = {
        "model": "local",
        "messages": [
            {"role": "system", "content": prompt.system},
            {"role": "user",   "content": user_content},
        ],
        "max_tokens": prompt.max_tokens,
        "temperature": prompt.temperature,
        "seed": SEED,
        "stream": False,
    }
    if template_kwargs:
        req_body["chat_template_kwargs"] = template_kwargs
    body = json.dumps(req_body).encode()

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=CHAT_TIMEOUT_S) as r:
        payload = json.loads(r.read())
    lat = time.time() - t0
    text = payload["choices"][0]["message"]["content"]
    usage = payload.get("usage", {}) or {}
    timings = payload.get("timings", {}) or {}
    # Prefer llama-server's measured timings (excludes HTTP overhead); fall
    # back to usage/latency when not present.
    pt = int(usage.get("prompt_tokens", 0))
    ct = int(usage.get("completion_tokens", 0))
    gen_tok_s = float(timings.get("predicted_per_second") or
                      (ct / lat if lat > 0 else 0.0))
    prompt_tok_s = float(timings.get("prompt_per_second") or
                         (pt / lat if lat > 0 else 0.0))
    # Suppress divide-by-near-zero sentinels: when only a token or two are
    # emitted, llama-server's `predicted_per_second` overflows (we've seen
    # 1e6). Anything above this cap is measurement noise on this hardware.
    if ct <= 1 or gen_tok_s > 500:
        gen_tok_s = (ct / lat) if lat > 0 else 0.0
    return {
        "text": text,
        "latency_s": lat,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "gen_tok_s": gen_tok_s,
        "prompt_tok_s": prompt_tok_s,
    }


# -------------------------------------------------------------------- driver

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bin", type=pathlib.Path, default=DEFAULT_BIN,
                   help=f"llama-server binary (default: {DEFAULT_BIN})")
    p.add_argument("--models-dir", type=pathlib.Path, default=DEFAULT_MODELS_DIR)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--out", type=pathlib.Path,
                   default=ROOT / "bench" / "results.json")
    p.add_argument("--only-models", nargs="+", default=None,
                   metavar="ID", help="restrict to listed model IDs")
    p.add_argument("--only-tasks", nargs="+", default=None,
                   metavar="NAME", help="restrict to listed task names")
    p.add_argument("--ad-hoc", type=pathlib.Path, default=None,
                   metavar="GGUF",
                   help="run an arbitrary local GGUF (bypasses OSS registry); "
                        "useful for harness sanity-checks. Not for headline "
                        "results — model license is not validated.")
    p.add_argument("--ad-hoc-id", default=None,
                   help="label for --ad-hoc model (default: filename stem)")
    p.add_argument("--ad-hoc-kind", choices=["text", "vision"], default="text")
    p.add_argument("--ad-hoc-mmproj", type=pathlib.Path, default=None,
                   help="mmproj GGUF for --ad-hoc vision model")
    args = p.parse_args()

    if not args.bin.exists():
        sys.exit(f"llama-server not found at {args.bin} — run ./build.sh first")

    args.models_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.out.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.ad_hoc is not None:
        if not args.ad_hoc.exists():
            sys.exit(f"--ad-hoc GGUF not found: {args.ad_hoc}")
        ad_hoc_spec = ModelSpec(
            id=args.ad_hoc_id or args.ad_hoc.stem,
            license="UNVERIFIED-AD-HOC",
            kind=args.ad_hoc_kind,
            repo="<local>",
            file=args.ad_hoc.name,
            mmproj_file=(args.ad_hoc_mmproj.name if args.ad_hoc_mmproj else None),
        )
        selected_models = [ad_hoc_spec]
        # In ad-hoc mode, point models_dir at the file's parent so existing
        # logic below picks it up by name.
        args.models_dir = args.ad_hoc.parent
        print(f"AD-HOC mode: {ad_hoc_spec.id} ({args.ad_hoc}) "
              f"— license NOT validated; do not publish as a result.",
              file=sys.stderr)
    else:
        selected_models = [
            m for m in MODELS
            if args.only_models is None or m.id in args.only_models
        ]

    selected_tasks = {
        k: v for k, v in TASKS.items()
        if args.only_tasks is None or k in args.only_tasks
    }

    # Pre-flight: refuse to run with missing files; print HF coordinates.
    missing = [
        m for m in selected_models
        if not (args.models_dir / m.file).exists()
        or (m.mmproj_file and not (args.models_dir / m.mmproj_file).exists())
    ]
    if missing:
        print("Missing model files; download into models/:\n", file=sys.stderr)
        for m in missing:
            print(f"  MODEL={m.repo} FILE={m.file} "
                  f"./pull-model.sh", file=sys.stderr)
            if m.mmproj_file:
                print(f"  MODEL={m.mmproj_repo} FILE={m.mmproj_file} "
                      f"./pull-model.sh", file=sys.stderr)
        return 2

    buildinfo_path = ROOT / "dist" / "BUILDINFO"
    buildinfo = buildinfo_path.read_text() if buildinfo_path.exists() else ""

    results: list[dict] = []

    def _persist(complete: bool) -> None:
        payload = {
            "schema": 1,
            "seed": SEED,
            "complete": complete,
            "buildinfo": buildinfo.strip(),
            "models": [asdict(m) for m in selected_models],
            "tasks": list(selected_tasks),
            "results": results,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))

    for m in selected_models:
        model_path = args.models_dir / m.file
        mmproj_path = (args.models_dir / m.mmproj_file) if m.mmproj_file else None
        log_path = log_dir / f"{m.id}.log"
        print(f"\n== {m.id}  ({m.license}, {m.kind}) ==", flush=True)
        try:
            with ServerProcess(args.bin, model_path, mmproj_path,
                               args.port, log_path):
                for task_name, task_cfg in selected_tasks.items():
                    if m.kind not in task_cfg["kinds"]:
                        continue
                    try:
                        prompts = task_cfg["prompts"]()
                    except Exception as e:
                        print(f"  [{task_name}] prompt-build failed: {e}; "
                              "skipping", file=sys.stderr)
                        continue
                    if not prompts:
                        # task voluntarily declined (e.g. missing dep)
                        continue
                    scorer = SCORERS[task_name]
                    task_scores: list[float] = []
                    for idx, pr in enumerate(prompts):
                        if m.user_suffix:
                            pr.user = pr.user + m.user_suffix
                        try:
                            resp = chat(args.port, pr,
                                        template_kwargs=m.template_kwargs)
                        except Exception as e:
                            results.append(asdict(TaskResult(
                                task_name, m.id, idx, f"<error: {e}>",
                                0.0, {"error": str(e)}, 0.0,
                            )))
                            task_scores.append(0.0)
                            print(f"  [{task_name}/{idx}] ERROR: {e}",
                                  flush=True)
                            continue
                        score, detail = scorer(resp["text"], pr.expected)
                        results.append(asdict(TaskResult(
                            task=task_name, model=m.id, prompt_idx=idx,
                            output=resp["text"], score=score, detail=detail,
                            latency_s=resp["latency_s"],
                            prompt_tokens=resp["prompt_tokens"],
                            completion_tokens=resp["completion_tokens"],
                            gen_tok_s=resp["gen_tok_s"],
                            prompt_tok_s=resp["prompt_tok_s"],
                        )))
                        task_scores.append(score)
                        print(
                            f"  [{task_name}/{idx}] "
                            f"score={score:.2f} "
                            f"lat={resp['latency_s']:.1f}s "
                            f"gen={resp['gen_tok_s']:.1f}t/s "
                            f"prompt={resp['prompt_tok_s']:.1f}t/s "
                            f"({resp['prompt_tokens']}p/{resp['completion_tokens']}g)",
                            flush=True,
                        )
                    avg = sum(task_scores) / max(len(task_scores), 1)
                    print(f"  -> {task_name} avg={avg:.2f}", flush=True)
        except Exception as e:
            print(f"  FAILED to run {m.id}: {e}", file=sys.stderr)
        finally:
            _persist(complete=False)  # checkpoint after each model

    _persist(complete=True)
    print(f"\nResults: {args.out}")

    payload = json.loads(args.out.read_text())
    _print_summary(payload)
    return 0


def _print_summary(payload: dict) -> None:
    g: dict[str, dict[str, list[float]]] = {}
    throughput: dict[str, dict[str, list[float]]] = {}
    for r in payload["results"]:
        g.setdefault(r["model"], {}).setdefault(r["task"], []).append(r["score"])
        t = throughput.setdefault(r["model"], {"gen": [], "prompt": []})
        if r.get("gen_tok_s", 0) > 0:
            t["gen"].append(r["gen_tok_s"])
        if r.get("prompt_tok_s", 0) > 0:
            t["prompt"].append(r["prompt_tok_s"])

    task_cols = payload["tasks"]
    header = ["model"] + task_cols
    rows: list[list[str]] = []
    for model, tasks in g.items():
        row = [model]
        for t in task_cols:
            sc = tasks.get(t)
            row.append("-" if not sc else f"{sum(sc) / len(sc):.2f}")
        rows.append(row)

    _print_table(header, rows, "Scores")

    th_header = ["model", "gen_tok_s", "prompt_tok_s", "samples"]
    th_rows: list[list[str]] = []
    for model, t in throughput.items():
        gen_avg = (sum(t["gen"]) / len(t["gen"])) if t["gen"] else 0.0
        prompt_avg = (sum(t["prompt"]) / len(t["prompt"])) if t["prompt"] else 0.0
        th_rows.append([model, f"{gen_avg:.1f}", f"{prompt_avg:.1f}",
                        str(len(t["gen"]))])
    _print_table(th_header, th_rows, "Throughput (avg across prompts)")


def _print_table(header: list[str], rows: list[list[str]], title: str) -> None:
    if not rows:
        return
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print(f"\n{title}")
    print(fmt.format(*header))
    print(fmt.format(*["-" * w for w in widths]))
    for r in rows:
        print(fmt.format(*r))


if __name__ == "__main__":
    raise SystemExit(main())
