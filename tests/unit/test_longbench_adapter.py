"""Synthetic unit boundaries; these fixtures never certify real LongBench runs."""
import hashlib
import json
import zipfile

import pytest
from agentinputlab.benchmarks import longbench as lb
from agentinputlab.contracts import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import to_jsonable


def fixture_adapter(tmp_path, monkeypatch):
    archive=tmp_path/'synthetic-data.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('data/qasper.jsonl',json.dumps({'_id':'synthetic-unit-1','dataset':'qasper',
            'input':'Public synthetic question','context':'Public synthetic document',
            'answers':['gold-only-answer'],'all_classes':None})+'\n')
    scorer=tmp_path/'synthetic_metrics.py'
    scorer.write_text('def qa_f1_score(prediction,reference,**kwargs):\n    return float(prediction == reference)\n')
    prompts=tmp_path/'synthetic_prompts.json'
    prompts.write_text(json.dumps({'qasper':'Read {context}. Question: {input}'}))
    for name,path in [('LONG_BENCH_DATA_SHA256',archive),('LONG_BENCH_SCORER_SHA256',scorer),('LONG_BENCH_PROMPTS_SHA256',prompts)]:
        monkeypatch.setattr(lb,name,hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setitem(lb._SUBSETS,'longbench.qasper',('qasper','qa_f1_score','answer_f1',1))
    return lb.LongBenchAdapter(benchmark_id='longbench.qasper',data_zip=archive,
        scorer_path=scorer,prompts_path=prompts)


def test_references_are_only_available_to_scoring(tmp_path,monkeypatch):
    adapter=fixture_adapter(tmp_path,monkeypatch);example=adapter.examples('test')[0]
    state=example.visible_state()
    assert 'gold-only-answer' not in json.dumps(to_jsonable(state))
    assert adapter.render(state)=='Read Public synthetic document. Question: Public synthetic question'
    assert adapter.score(example,'gold-only-answer')=={'answer_f1':1.0}
    assert adapter.score(example,'wrong')=={'answer_f1':0.0}


def test_changed_native_source_is_rejected_before_import(tmp_path):
    path=tmp_path/'untrusted.py';path.write_text('raise RuntimeError("must not execute")')
    with pytest.raises(ArtifactError,match='checksum'):
        lb._checked(path,lb.LONG_BENCH_SCORER_SHA256)


def test_verified_extensionless_cache_scorer_can_load(tmp_path, monkeypatch):
    fixture_adapter(tmp_path, monkeypatch)
    scorer = tmp_path / 'payload'
    (tmp_path / 'synthetic_metrics.py').rename(scorer)
    adapter = lb.LongBenchAdapter(benchmark_id='longbench.qasper',
        data_zip=tmp_path/'synthetic-data.zip', scorer_path=scorer,
        prompts_path=tmp_path/'synthetic_prompts.json')
    assert adapter.score(adapter.examples('test')[0], 'gold-only-answer') == {'answer_f1': 1.0}


def test_other_splits_are_not_silently_substituted(tmp_path,monkeypatch):
    adapter=fixture_adapter(tmp_path,monkeypatch)
    with pytest.raises(ValidationError):adapter.examples('train')


def test_id_lookup_returns_the_native_example_and_same_score(tmp_path,monkeypatch):
    adapter=fixture_adapter(tmp_path,monkeypatch)
    original=adapter.examples('test')[0]
    observed=adapter.example('test',original.example_id)
    assert observed is original
    assert adapter.score(observed,'gold-only-answer')==adapter.score(original,'gold-only-answer')
    assert adapter.render(observed.visible_state())==adapter.render(original.visible_state())


def test_id_lookup_rejects_wrong_split_and_missing_id(tmp_path,monkeypatch):
    adapter=fixture_adapter(tmp_path,monkeypatch)
    with pytest.raises(ValidationError,match='test only'):
        adapter.example('train','synthetic-unit-1')
    with pytest.raises(ValidationError,match='unknown LongBench example ID'):
        adapter.example('test','missing-native-id')
    for invalid in ('', ' ', None, [], 1):
        with pytest.raises(ValidationError,match='non-empty text'):
            adapter.example('test',invalid)
