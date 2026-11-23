from __future__ import annotations

from workers.official_arenas.providers._message_compat import (
    coalesce_chat_system_messages,
    fold_responses_system_messages_into_instructions,
    normalize_qwen38_reasoning_effort,
    requires_qwen38_initial_system_message,
)


def test_qwen38_alias_detection_is_narrow() -> None:
    assert requires_qwen38_initial_system_message("qwen3.8-27b")
    assert requires_qwen38_initial_system_message("QWEN3.8_27B")
    assert not requires_qwen38_initial_system_message("qwen3-4b")
    assert not requires_qwen38_initial_system_message("gpt-oss-20b")
    assert not requires_qwen38_initial_system_message(None)


def test_qwen38_high_reasoning_effort_maps_to_supported_strongest_label() -> None:
    assert normalize_qwen38_reasoning_effort("qwen3.8-27b", "high") == "xhigh"
    assert normalize_qwen38_reasoning_effort("qwen3.8-27b", "medium") == "medium"
    assert normalize_qwen38_reasoning_effort("gpt-oss-20b", "high") == "high"


def test_chat_system_messages_are_coalesced_at_the_front() -> None:
    messages = [
        {"role": "user", "content": "first"},
        {"role": "system", "content": "policy"},
        {"role": "assistant", "content": "answer"},
        {"role": "system", "content": "visible intervention"},
        {"role": "user", "content": "latest"},
    ]

    normalized = coalesce_chat_system_messages(messages)

    assert normalized == [
        {"role": "system", "content": "policy\n\nvisible intervention"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "latest"},
    ]
    assert messages[1]["content"] == "policy"


def test_responses_system_items_are_folded_into_instructions() -> None:
    output_item = object()
    input_items = [
        {"role": "user", "content": "first"},
        {"role": "system", "content": "visible intervention"},
        output_item,
        {"type": "function_call_output", "call_id": "call-1", "output": "ok"},
    ]

    normalized, instructions = fold_responses_system_messages_into_instructions(
        input_items,
        "base policy",
    )

    assert instructions == "base policy\n\nvisible intervention"
    assert normalized == [
        {"role": "user", "content": "first"},
        output_item,
        {"type": "function_call_output", "call_id": "call-1", "output": "ok"},
    ]


def test_inputs_without_system_messages_are_unchanged() -> None:
    messages = [{"role": "user", "content": "hello"}]
    chat = coalesce_chat_system_messages(messages)
    responses, instructions = fold_responses_system_messages_into_instructions(
        messages,
        "base policy",
    )

    assert chat == messages
    assert responses == messages
    assert instructions == "base policy"
