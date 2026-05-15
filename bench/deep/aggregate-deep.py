#!/usr/bin/env python3.14
"""Aggregate bench/deep/ results into DEEP-REPORT.md.

Reads:
  bench/deep/<sub-profile>/<model>/<task>/results_*.json   (lm-eval output)
  bench/lm-eval/<model>/results_*.json                     (fast-chat baseline)

Writes:
  bench/deep/DEEP-REPORT.md                                (regenerated in place)
"""
import json, pathlib, glob
from datetime import datetime

ROOT = pathlib.Path(__file__).parent.parent.parent
DEEP = ROOT / "bench/deep"
FAST = ROOT / "bench/lm-eval"

TASK_TO_FASTCHAT_TASK = {
    "gsm8k_deep": "gsm8k",
    "humaneval_deep": "humaneval_instruct",
    "mmlu_pro_deep": "mmlu_pro_philosophy",
    "math_deep": None,
}

SCORE_KEYS = (
    "pass@1,create_test",
    "pass@1",
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "acc,none",
    "acc",
)


def load_score(json_path: pathlib.Path, task_name: str):
    try:
        data = json.loads(json_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None, None
    task = data.get("results", {}).get(task_name)
    if not task:
        return None, None
    for key in SCORE_KEYS:
        if key in task:
            v = float(task[key])
            stderr_key = (
                key + "_stderr" if "," not in key
                else key.split(",")[0] + "_stderr," + key.split(",")[1]
            )
            se = task.get(stderr_key)
            return v, float(se) if se is not None else None
    return None, None


def find_latest(pattern: str):
    matches = sorted(glob.glob(pattern, recursive=True))
    return pathlib.Path(matches[-1]) if matches else None


def main():
    rows = []
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
                deep_json = find_latest(str(task_dir / "**/results_*.json"))
                if not deep_json:
                    continue
                deep_score, deep_stderr = load_score(deep_json, task)
                fast_task = TASK_TO_FASTCHAT_TASK.get(task)
                fast_score = None
                if fast_task:
                    fast_json = find_latest(str(FAST / model / "**/results_*.json"))
                    if fast_json:
                        fast_score, _ = load_score(fast_json, fast_task)
                rows.append((sub_profile, model, task, deep_score, deep_stderr, fast_score))

    out = [
        "# Deep-thinking Bench Results",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Models: gemma-4-e2b, qwen3.6-35b-a3b. Mode: `enable_thinking=true`.",
        "See spec: `docs/superpowers/specs/2026-05-15-deep-bench-design.md`",
        "See fast-chat counterpart: `bench/REPORT.md`",
        "",
    ]
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
