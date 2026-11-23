import ast

import pytest

from workers.llmcompiler.export_prompts import literal_string


def test_literal_prompt_export_handles_only_literal_concatenation_and_known_constants():
    value = ast.parse('"Question\\n" + f"2. join(){END_OF_PLAN}"', mode="eval").body
    assert literal_string(value, {"END_OF_PLAN": "<END_OF_PLAN>"}) == "Question\n2. join()<END_OF_PLAN>"


@pytest.mark.parametrize("text", ['__import__("os").system("echo forbidden")',
    'open("secret").read()', 'UNKNOWN_CONSTANT', 'f"{END_OF_PLAN!r}"'])
def test_export_never_executes_prompt_file_code_or_unknown_expressions(text):
    with pytest.raises(ValueError):
        literal_string(ast.parse(text, mode="eval").body, {"END_OF_PLAN": "<END_OF_PLAN>"})
