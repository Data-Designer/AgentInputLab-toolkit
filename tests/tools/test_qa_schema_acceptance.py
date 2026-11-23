from run_qa_schema_acceptance import parameters


def test_schema_configuration_does_not_change_per_call_budget_or_seed_policy():
    config = parameters()
    assert config == {"schema_version": 1, "task": "qa", "qa": {
        "output_format": "structured_json", "structured_output_mode": "json_schema",
        "max_tokens": 512, "use_seed": True}}
