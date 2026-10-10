"""Regression tests for model_utils.format_chat_prompt - added after a live
Llama-3.1-8B-Instruct run showed a ~90% judge_fallback_rate (vs. ~20-27% for Qwen on
the identical code path) traced to generation and extraction both tokenizing raw
prompt text instead of applying the model's chat template."""


class _FakeTokenizerWithTemplate:
    chat_template = "some non-none template string"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        assert messages == [{"role": "user", "content": "hello"}]
        return "<|template_wrapped|>hello<|assistant|>"


class _FakeTokenizerNoTemplate:
    chat_template = None


def test_format_chat_prompt_wraps_when_template_exists():
    from model_utils import format_chat_prompt

    result = format_chat_prompt(_FakeTokenizerWithTemplate(), "hello")
    assert result == "<|template_wrapped|>hello<|assistant|>"


def test_format_chat_prompt_falls_back_to_raw_when_no_template():
    from model_utils import format_chat_prompt

    result = format_chat_prompt(_FakeTokenizerNoTemplate(), "hello")
    assert result == "hello"


def test_format_chat_prompt_falls_back_when_attribute_missing_entirely():
    from model_utils import format_chat_prompt

    class _NoAttrAtAll:
        pass

    result = format_chat_prompt(_NoAttrAtAll(), "hello")
    assert result == "hello"
