from hybrid_router.types import (
    CompletionRequest,
    CompletionResponse,
    Message,
    TextBlock,
    Tier,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
    estimate_tokens,
)


def test_message_accepts_plain_dicts_with_discriminated_blocks():
    m = Message.model_validate(
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "42", "is_error": True},
                {"type": "text", "text": "and now?"},
            ],
        }
    )
    blocks = m.blocks()
    assert isinstance(blocks[0], ToolResultBlock) and blocks[0].is_error
    assert isinstance(blocks[1], TextBlock)
    assert m.text() == "and now?"


def test_string_content_normalizes_to_one_text_block():
    assert Message(role="user", content="hi").blocks() == [TextBlock(text="hi")]
    assert Message(role="user", content="").blocks() == []


def test_last_user_text_skips_tool_result_only_turns():
    req = CompletionRequest(
        messages=[
            Message(role="user", content="find the bug"),
            Message(role="assistant", content=[ToolUseBlock(id="t1", name="grep", input={})]),
            Message(role="user", content=[ToolResultBlock(tool_use_id="t1", content="x")]),
        ]
    )
    assert req.last_user_text() == "find the bug"


def test_response_helpers():
    resp = CompletionResponse(
        content=[TextBlock(text="a"), ToolUseBlock(id="t", name="f"), TextBlock(text="b")],
        stop_reason="tool_use",
        usage=Usage(input_tokens=10, output_tokens=3, cache_read_tokens=90, cache_write_tokens=5),
        model="m",
        backend="local:ollama",
        tier=Tier.LOCAL,
        latency_ms=1.0,
        token_logprobs=[-1.0, -3.0],
    )
    assert resp.text == "ab"
    assert [c.name for c in resp.tool_calls] == ["f"]
    assert resp.mean_logprob == -2.0
    assert resp.usage.total_input_tokens == 105
    assert resp.as_message().role == "assistant"


def test_estimate_tokens_counts_text_tools_and_schema():
    small = CompletionRequest(messages=[Message(role="user", content="x" * 350)])
    big = small.model_copy(
        update={
            "tools": [ToolSpec(name="t", description="d" * 700)],
            "response_schema": {"type": "object"},
        }
    )
    assert estimate_tokens(small) == 100
    assert estimate_tokens(big) > estimate_tokens(small) + 190


def test_tier_order_helpers():
    assert not Tier.LOCAL.is_cloud
    assert Tier.SONNET.is_cloud and Tier.OPUS.is_cloud
