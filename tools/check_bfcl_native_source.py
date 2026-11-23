"""Evaluator-only source checks. These are NOT model acceptance attempts."""

import argparse
import json

from agentinputlab.benchmarks import SourceVerifiedBFCLV3ASTBridge, load_bfcl_v3_simple_adapter
from agentinputlab.contracts.serialization import to_jsonable


def reference_literal(alternatives, schema):
    """Materialize one legal value from BFCL's nested alternative notation.

    This stays entirely evaluator-side and is never a model prediction.
    """
    nonempty = [v for v in alternatives if v != ""]
    value = nonempty[0] if nonempty else alternatives[0]
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        return {
            key: reference_literal(choices, properties.get(key, {}))
            for key, choices in value.items()
        }
    if isinstance(value, list) and schema.get("items", {}).get("type") == "dict":
        properties = schema["items"].get("properties", {})
        return [
            {
                key: reference_literal(choices, properties.get(key, {}))
                for key, choices in row.items()
            }
            for row in value
        ]
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-archive", required=True)
    p.add_argument("--prepared-root", required=True)
    a = p.parse_args()
    bridge = SourceVerifiedBFCLV3ASTBridge(
        source_archive=a.source_archive, model_id="scorer-selftest-only"
    )
    adapter = load_bfcl_v3_simple_adapter(a.prepared_root, evaluation_bridge=bridge)
    positives = []
    negative = []
    for example in adapter.examples("test"):
        function = to_jsonable(example.visible_metadata["function"])
        gold = to_jsonable(example.evaluator_gold["ground_truth"])
        _, parameters = next(iter(gold[0].items()))
        name = bridge.tools(function)[0]["function"]["name"]
        required = set(function["parameters"]["required"])
        arguments = {}
        for key, values in parameters.items():
            if any(v != "" for v in values) or key in required:
                arguments[key] = reference_literal(
                    values, function["parameters"]["properties"][key]
                )
        payload = {
            "language": "Python",
            "test_category": "simple",
            "function": function,
            "ground_truth": gold,
        }
        result = bridge.evaluate(payload, [{name: json.dumps(arguments)}])
        if result.ast_accuracy != 1:
            positives.append({"id": example.example_id, "diagnostic": result.diagnostic})
        if bridge.evaluate(payload, []).ast_accuracy != 0:
            negative.append(example.example_id)
    print(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "evaluator_only_not_a_model_test",
                "examples": 400,
                "source_verified": True,
                "positive_failures": positives,
                "empty_prediction_failures": negative,
                "real_model_calls": 0,
                "paper_results_changed": False,
            }
        )
    )
    if positives or negative:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
