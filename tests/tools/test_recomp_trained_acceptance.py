import pytest
from run_recomp_trained_acceptance import parameters, protocol


def test_protocol_separates_frozen_cases_without_relabelling_encoder_evidence():
    assert protocol({}) == ("extractive", "curate.recomp.extractive", 208)
    assert protocol({"training_mode": "abstractive"}) == (
        "abstractive",
        "curate.recomp.abstractive",
        212,
    )
    with pytest.raises(ValueError):
        protocol({"training_mode": "not-a-native-mode"})


@pytest.mark.parametrize("field", ["sentence_splitter_revision", "input_template_hash"])
def test_helper_replay_uses_actual_mode_specific_request_identity(field):
    payload = dict(
        checkpoint_revision="c",
        tokenizer_revision="t",
        asset_manifest_hash="a",
        **{field: "native-identity"},
    )
    worker = type("CaptureFixture", (), {"requests": [dict(payload=payload)]})()
    result = parameters(worker)
    assert result[field] == "native-identity"
    assert result["asset_manifest_hash"] == "a"
    assert ("sentence_splitter_revision" in result) != ("input_template_hash" in result)
