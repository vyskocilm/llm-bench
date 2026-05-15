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
- **LLM-judge (optional)**: a stronger local model (or any OpenAI-compatible API endpoint of the user's choice) scores each response on a small rubric. Produces ranking when requested.

## Models

- Top 6-8 daily-driver candidates from `bench/REPORT.md` rankings.
- (Optional) any OpenAI-compatible cloud endpoint for an additional baseline column.

## Budget

- ~3-4h to run the gallery (20-30 prompts × 6-8 models × 1-3 min per response).
- ~30 min for judge pass.

## Headline prompt

Pelican on a monocycle (SVG, eyeball-renderable).

## Status

Not scheduled. Re-opens after Track A (deep-thinking bench) ships and its results are reviewed.
