#!/usr/bin/env python3
"""render.py — turn bench/results.json into a static results.html (mvp.css).

Inputs:
  bench/results.json   produced by bench.py

Output (default):
  bench/results.html   single-file static page; opens in any browser.

The summary tables link into per-prompt detail blocks below so the cell
score is one click away from the full model output, score detail, and token
timing for that (model, task, prompt) triplet.
"""

from __future__ import annotations

import argparse
import html
import json
import pathlib
from collections import defaultdict
from datetime import datetime, timezone

MVP_CSS_URL = "https://unpkg.com/mvp.css"

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "bench" / "results.json"
DEFAULT_OUTPUT = ROOT / "bench" / "results.html"


def anchor(model: str, task: str, idx: int) -> str:
    # ids: a-Z, 0-9, '-' only; replace dots etc.
    safe = lambda s: "".join(c if c.isalnum() or c == "-" else "-" for c in s)
    return f"d-{safe(model)}-{safe(task)}-{idx}"


def cell(score: float, link: str | None) -> str:
    if link is None:
        return "—"
    color = "var(--color-bg-secondary)"
    if score >= 0.99: color = "#cdeacd"   # green
    elif score >= 0.5: color = "#fff1c2"  # amber
    elif score > 0:    color = "#ffd6cc"  # orange
    else:               color = "#f5c9c1"  # red
    return (f'<a href="#{link}" '
            f'style="background:{color};padding:.15rem .4rem;'
            f'border-radius:.25rem;text-decoration:none;">'
            f'{score:.2f}</a>')


def render(payload: dict) -> str:
    results = payload["results"]
    models = [m["id"] for m in payload["models"]]
    tasks = payload["tasks"]
    schema = payload.get("schema", 1)
    seed = payload.get("seed", "?")
    complete = payload.get("complete", True)
    buildinfo = payload.get("buildinfo", "").strip()

    # group: model -> task -> [results]
    by_model_task: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_model_task[r["model"]][r["task"]].append(r)

    # score grid (avg per task)
    score_rows = []
    for m in models:
        row = [m]
        for t in tasks:
            xs = by_model_task[m].get(t, [])
            if not xs:
                row.append(("—", None))
                continue
            avg = sum(x["score"] for x in xs) / len(xs)
            # link to first prompt of this (model, task)
            link = anchor(m, t, xs[0]["prompt_idx"])
            row.append((avg, link))
        score_rows.append(row)

    # throughput (filter sentinels)
    th_rows = []
    for m in models:
        gens, prompts = [], []
        for t in tasks:
            for x in by_model_task[m].get(t, []):
                if x.get("gen_tok_s", 0) > 0:    gens.append(x["gen_tok_s"])
                if x.get("prompt_tok_s", 0) > 0: prompts.append(x["prompt_tok_s"])
        if not gens and not prompts:
            continue
        th_rows.append((m,
                        sum(gens) / len(gens) if gens else 0.0,
                        sum(prompts) / len(prompts) if prompts else 0.0,
                        len(gens)))

    # render summary
    def score_cell(v, link):
        if isinstance(v, str): return v
        return cell(v, link)

    s_head = "<tr><th>model</th>" + "".join(f"<th>{t}</th>" for t in tasks) + "</tr>"
    s_body = ""
    for row in score_rows:
        m = row[0]
        s_body += f"<tr><td><code>{html.escape(m)}</code></td>"
        for v, link in row[1:]:
            s_body += f"<td>{score_cell(v, link)}</td>"
        s_body += "</tr>"

    t_body = ""
    for m, g, p, n in th_rows:
        t_body += (f"<tr><td><code>{html.escape(m)}</code></td>"
                   f"<td>{g:.1f}</td><td>{p:.1f}</td><td>{n}</td></tr>")

    # details sections
    detail_html = []
    for m in models:
        if m not in by_model_task: continue
        detail_html.append(f'<section><h2 id="m-{html.escape(m)}">'
                           f'<code>{html.escape(m)}</code></h2>')
        for t in tasks:
            for x in by_model_task[m].get(t, []):
                aid = anchor(m, t, x["prompt_idx"])
                lat = x.get("latency_s", 0)
                gt = x.get("gen_tok_s", 0)
                pt = x.get("prompt_tok_s", 0)
                ptk = x.get("prompt_tokens", 0)
                ctk = x.get("completion_tokens", 0)
                output = html.escape(x.get("output", ""))
                detail = html.escape(json.dumps(x.get("detail", {}),
                                                indent=2, ensure_ascii=False))
                detail_html.append(
                    f'<article id="{aid}" style="margin-bottom:1.5rem;'
                    f'border-left:4px solid #ccc;padding-left:1rem;">'
                    f'<h4>{html.escape(t)} / prompt {x["prompt_idx"]}'
                    f' &mdash; score <strong>{x["score"]:.2f}</strong></h4>'
                    f'<p><small>'
                    f'latency {lat:.2f}s &middot; gen {gt:.1f} tok/s &middot; '
                    f'prompt {pt:.1f} tok/s &middot; '
                    f'tokens p={ptk} g={ctk}'
                    f'</small></p>'
                    f'<details><summary>output</summary>'
                    f'<pre style="white-space:pre-wrap">{output}</pre></details>'
                    f'<details><summary>scoring detail</summary>'
                    f'<pre>{detail}</pre></details>'
                    f'</article>'
                )
        detail_html.append("</section>")

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    incomplete_banner = "" if complete else (
        '<aside style="background:#fff1c2;padding:.5rem 1rem;border-radius:.25rem;">'
        '<strong>Note:</strong> run did not complete — results may be partial.'
        "</aside>"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>llama.cpp / Radeon 860M — OSS bench</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="{MVP_CSS_URL}">
<style>
  :root {{ --line-height: 1.4; }}
  body {{ max-width: 1100px; }}
  table {{ font-variant-numeric: tabular-nums; }}
  pre {{ font-size: .85rem; }}
  article h4 {{ margin: 0 0 .25rem 0; }}
  section {{ margin-top: 2rem; }}
  .meta {{ color: var(--color-text-secondary); font-size: .85rem; }}
</style>
</head>
<body>
<header>
<h1>llama.cpp / Radeon 860M — OSS bench</h1>
<p class="meta">
Rendered {generated} &middot; seed={seed} &middot; schema={schema}<br>
{html.escape(buildinfo).replace(chr(10), '<br>')}
</p>
{incomplete_banner}
</header>

<main>
<h2>Scores</h2>
<p>Each cell links to the per-prompt output that produced it.
Colour: <span style="background:#cdeacd;padding:.1rem .3rem">≥ 0.99</span>
 <span style="background:#fff1c2;padding:.1rem .3rem">≥ 0.50</span>
 <span style="background:#ffd6cc;padding:.1rem .3rem">&gt; 0</span>
 <span style="background:#f5c9c1;padding:.1rem .3rem">= 0</span>.</p>
<table>
<thead>{s_head}</thead>
<tbody>{s_body}</tbody>
</table>

<h2>Throughput (avg tok/s)</h2>
<table>
<thead><tr><th>model</th><th>gen</th><th>prompt</th><th>samples</th></tr></thead>
<tbody>{t_body}</tbody>
</table>

<h2>Details</h2>
{''.join(detail_html)}
</main>

<footer>
<p class="meta">Generated by <code>bench/render.py</code> from
<code>bench/results.json</code>. CSS:
<a href="{MVP_CSS_URL}">mvp.css</a>.</p>
</footer>
</body>
</html>
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=pathlib.Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()

    if not args.input.exists():
        raise SystemExit(f"results.json not found at {args.input}")
    payload = json.loads(args.input.read_text())
    html_text = render(payload)
    args.output.write_text(html_text)
    print(f"wrote {args.output} ({len(html_text):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
