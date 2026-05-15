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
    assert strip("ok<think>started thinking but truncated") == "ok"


def test_channel_tag_variant():
    # Gemma 4 MoE emits <|channel>...<channel|> (asymmetric pipes)
    assert strip("<|channel>thinking<channel|>answer") == "answer"


def test_unclosed_channel_tag():
    assert strip("ok<|channel>truncated thinking") == "ok"


def test_mixed_think_and_channel_in_same_response():
    text = "<think>think1</think>middle<|channel>think2<channel|>end"
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
