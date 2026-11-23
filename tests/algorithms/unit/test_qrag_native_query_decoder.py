"""The opt-in IR codec cannot silently reinterpret QA answers or reasoning."""

import json

import pytest

from agentinputlab.algorithms.acquire.qrag import (
    QRAG_NATIVE_QUERY_DECODER_ID,
    QRAG_NATIVE_QUERY_DECODER_SHA256,
    parse_instruction_reader_answer,
    parse_native_query_reader_answer,
)
from agentinputlab.contracts import AlgorithmOutputError


@pytest.mark.parametrize("query", ["original query", "  query  ", "事实与证据", 'quoted "query"'])
@pytest.mark.parametrize("fenced", [False, True])
def test_native_query_is_returned_verbatim(query, fenced):
    text = json.dumps({"query": query}, ensure_ascii=False)
    if fenced:
        text = "```json\n" + text + "\n```"
    assert parse_native_query_reader_answer(text) == {"query": query}
    with pytest.raises(AlgorithmOutputError):
        parse_instruction_reader_answer(text)


@pytest.mark.parametrize("text", [
    None, 0, "", "not JSON", "[]", "null", '"query"',
    '{"query": null}', '{"query": true}', '{"query": 1}', '{"query": []}',
    '{"query": ""}', '{"query": "  "}', '{"query": "q", "extra": 1}',
    '{"query": "q", "query": "q"}',
    'reasoning then {"query": "q"}',
    '<think>reasoning</think>{"query": "q"}',
    '```\n{"query": "q"}\n```',
    '```json\n{"query": "q"}\n``` trailing',
    '{"query": "q"}{"query": "q2"}',
    json.dumps(dict(answer="Paris", support_ids=[], evidence_ids=[], evidence_texts=[])),
    json.dumps(dict(answer='{"query":"q"}', support_ids=[], evidence_ids=[], evidence_texts=[])),
])
def test_rejects_invalid_schema_and_answer_salvage(text):
    with pytest.raises(AlgorithmOutputError):
        parse_native_query_reader_answer(text)


def test_codec_is_separately_named_and_does_not_replace_QA():
    assert QRAG_NATIVE_QUERY_DECODER_ID == "agentinputlab.qrag.native_query_decoder.v1"
    assert len(QRAG_NATIVE_QUERY_DECODER_SHA256) == 64
    prediction = dict(answer="Paris", support_ids=[], evidence_ids=[], evidence_texts=[])
    assert parse_instruction_reader_answer(json.dumps(prediction)) == prediction
