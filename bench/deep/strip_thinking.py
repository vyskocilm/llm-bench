"""Robust <think>-block stripper for thinking-on model outputs.

Handles:
  * one or more closed <think>...</think> blocks
  * Gemma 4 MoE channel tags <|channel|>...<|channel|>
  * unclosed <think> (truncated at max_gen_toks → drop to EOS, treat as fail)
"""
import re

_PATTERNS = (
    (re.compile(r"<think>.*?</think>", re.DOTALL), ""),
    (re.compile(r"<\|channel>.*?<channel\|>", re.DOTALL), ""),
    (re.compile(r"<think>.*", re.DOTALL), ""),
    (re.compile(r"<\|channel>.*", re.DOTALL), ""),
)


def strip(text: str) -> str:
    """Strip thinking blocks from `text`. Returns the user-facing remainder."""
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    return text
