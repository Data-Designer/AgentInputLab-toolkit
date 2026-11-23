"""Export source-bound HotpotQA prompts without executing config-file code."""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path

from . import driver


def literal_string(node, constants):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return literal_string(node.left, constants) + literal_string(node.right, constants)
    if isinstance(node, ast.JoinedStr):
        parts = []
        for part in node.values:
            if isinstance(part, ast.FormattedValue):
                if part.conversion != -1 or part.format_spec is not None:
                    raise ValueError("prompt formatting is outside the literal exporter")
                part = part.value
            parts.append(literal_string(part, constants))
        return "".join(parts)
    raise ValueError("prompt expression is not a literal string; code execution forbidden")


def export(checkout, output, template="gpt"):
    if template not in {"gpt", "llama"}:
        raise ValueError("template must be gpt or llama")
    driver.verify_checkout(checkout)
    relative = f"configs/hotpotqa/{template}_prompts.py"
    source = driver._read_bytes(checkout / relative, label="pinned prompt source")
    tracked = subprocess.run(("git", "-C", str(checkout), "show", f"HEAD:{relative}"),
        capture_output=True, check=True, timeout=30).stdout
    if tracked != source:
        raise ValueError("prompt file differs from its pinned Git blob")
    constants = {}
    for node in ast.parse((checkout / "src/llm_compiler/constants.py").read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    values = {node.targets[0].id: literal_string(node.value, constants)
        for node in ast.parse(source).body if isinstance(node, ast.Assign) and len(node.targets) == 1}
    prompts = {"planner_example_prompt": values["PLANNER_PROMPT"],
        "planner_example_prompt_replan": values["PLANNER_PROMPT"],
        "planner_stop": [constants["END_OF_PLAN"]], "joinner_prompt": values["OUTPUT_PROMPT"],
        "joinner_prompt_final": values["OUTPUT_PROMPT"]}
    bundle = {"schema_version": "agentinputlab.llmcompiler.prompts.v1",
        "upstream_commit": driver.UPSTREAM_COMMIT, "prompts": prompts}
    raw = json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=False).encode()
    if len(raw) > 64 * 1024:
        raise ValueError("exported prompt bundle exceeds64KiB")
    with output.open("xb") as handle:
        handle.write(raw)
    return {"status": "ok", "source_relative_path": relative,
        "source_sha256": driver._sha256_bytes(source), "prompt_hash": driver._canonical_hash(prompts),
        "upstream_commit": driver.UPSTREAM_COMMIT, "config_code_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", choices=("gpt", "llama"), default="gpt")
    args = parser.parse_args()
    print(json.dumps(export(args.checkout, args.output, args.template), sort_keys=True))


if __name__ == "__main__":
    main()
