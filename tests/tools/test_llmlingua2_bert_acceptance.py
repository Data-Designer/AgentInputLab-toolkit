"""Entry preflight fixtures, no checkpoints or provider calls."""
import run_llmlingua2_bert_acceptance as worker


def test_boundary_probes_have_valid_query_but_only_source_context_is_compressed():
    rows = list(worker.boundary_states())
    assert [language for language, _ in rows] == ["zh", "es"]
    assert all(state.query and state.state_hash for _, state in rows)
    assert rows[0][1].context[0].text.startswith("项目")
    assert rows[1][1].context[0].text.startswith("El comité")
