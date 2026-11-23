"""Public file binding for the original controlled Adaptive-RAG worker."""
from __future__ import annotations

import inspect
import json
import re
from dataclasses import asdict, fields, replace
from pathlib import Path

from agentinputlab.contracts.errors import ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable

RELEASE = 'acquire.adaptive_rag.controlled_reimplementation'
PATHS = ('source_root', 'router_snapshot_path', 'router_weights_path',
         'router_tokenizer_path', 'prompt_root')


def validate_runtime(config, *, prepared=True):
    from agentinputlab.configuration import _known
    value = config.runtime
    if config.release_id != RELEASE or value.get('kind') != 'adaptive_rag_controlled':
        raise ValidationError('adaptive_rag_controlled requires the controlled_reimplementation profile')
    _known(value, {'kind', *PATHS, 'deployment_path', 'deployment_sha256',
                   'device', 'input_token_upper_bound', 'max_input_tokens'}, 'runtime')
    for key in PATHS + (('deployment_path',) if prepared else ()):
        if not isinstance(value.get(key), str) or not value[key].strip() or '\x00' in value[key]:
            raise ValidationError(f'runtime.{key} requires an explicit local path')
    if prepared and (not isinstance(value.get('deployment_sha256'), str)
                     or not re.fullmatch(r'[0-9a-f]{64}', value['deployment_sha256'])):
        raise ValidationError('Adaptive-RAG deployment requires its trusted SHA-256')
    if (not isinstance(value.get('device', 'cpu'), str)
            or not re.fullmatch(r'cpu|cuda(?::[0-9]+)?', value.get('device', 'cpu'))):
        raise ValidationError('Adaptive-RAG device must be cpu or cuda with optional index')
    for key in ('input_token_upper_bound', 'max_input_tokens'):
        v = value.get(key)
        if key == 'max_input_tokens' and v is None:
            continue
        if type(v) is not int or v < 1:
            raise ValidationError(f'runtime.{key} must be a positive integer')
    if not config.backend or not config.retriever or config.vector_runtime:
        raise ValidationError('Adaptive-RAG requires explicit backend and retriever, not vector_runtime')


class _LazyRouter:
    """Keep preparation/source checks ahead of model loading or classifier calls."""
    def __init__(self, config, identity):
        self.config, self.identity, self.native = config, identity, None

    def component_manifest(self):
        i = self.identity
        return {'component_type': 'router', 'fixture_runtime': False,
                'implementation_class': f'{type(self).__module__}.{type(self).__qualname__}',
                'revision': i.router_revision, 'snapshot_sha256': i.router_snapshot_sha256,
                'weights_sha256': i.router_weights_sha256, 'tokenizer_sha256': i.router_tokenizer_sha256}

    def _load(self):
        if self.native is None:
            from workers.adaptive_rag.runtime import LocalTransformersRouter
            v = self.config.runtime
            self.native = LocalTransformersRouter(snapshot_root=Path(v['router_snapshot_path']),
                weights_path=Path(v['router_weights_path']), tokenizer_path=Path(v['router_tokenizer_path']),
                identity=self.identity, device=v.get('device', 'cpu'), max_input_tokens=v.get('max_input_tokens'))
        return self.native

    def estimate_input_tokens(self, question):
        return self._load().estimate_input_tokens(question)

    def first_decoder_step(self, question, *, seed):
        return self._load().first_decoder_step(question, seed=seed)


def _fingerprint(identity, backend, retriever):
    from workers.adaptive_rag import driver, runtime
    return {'worker_id': 'agentinputlab.adaptive.controlled-file.v1',
            'driver_sha256': driver._path_closure_sha256(Path(driver.__file__)),
            'runtime_sha256': driver._path_closure_sha256(Path(runtime.__file__)),
            'binding_sha256': driver._path_closure_sha256(Path(__file__)),
            'identity_sha256': content_hash(asdict(identity)),
            'backend': to_jsonable(backend.fingerprint()), 'retriever': to_jsonable(retriever.fingerprint()),
            'fixture_mode': False}


class FileAdaptiveRuntime:
    """Typed startup configuration; original driver owns routing and budgets."""
    worker_id = 'agentinputlab.adaptive.controlled-file.v1'

    def __init__(self, config, backend, retriever):
        from agentinputlab.artifacts import load_pinned_json_mapping
        from workers.adaptive_rag.driver import (
            AdaptiveRAGWorkerDriver,
            RuntimeBundle,
            RuntimeIdentity,
        )
        from workers.adaptive_rag.runtime import OpenAICompatibleGenerator, ToolkitRetriever

        validate_runtime(config)
        v = config.runtime
        saved = load_pinned_json_mapping(v['deployment_path'], expected_sha256=v['deployment_sha256'])
        if (saved.get('schema') != 'agentinputlab.adaptive.controlled-deployment.v1'
                or saved.get('runtime_settings') != to_jsonable({k: x for k, x in v.items()
                                                        if k not in {'deployment_path', 'deployment_sha256'}})
                or saved.get('backend') != to_jsonable(backend.fingerprint())
                or saved.get('retriever') != to_jsonable(retriever.fingerprint())
                or saved.get('retriever_configuration') != to_jsonable(config.retriever)):
            raise ValidationError('Adaptive-RAG deployment differs from this backend/retriever/runtime; prepare again')
        identity = RuntimeIdentity(**saved['identity'])
        if identity.fixture_runtime:
            raise ValidationError('Adaptive-RAG public runtime rejects fixture identity')
        self.identity = _fingerprint(identity, backend, retriever)
        self.driver = AdaptiveRAGWorkerDriver(
            source_root=Path(v['source_root']), router_snapshot_path=Path(v['router_snapshot_path']),
            router_weights_path=Path(v['router_weights_path']), router_tokenizer_path=Path(v['router_tokenizer_path']),
            prompt_root=Path(v['prompt_root']),
            runtime=RuntimeBundle(identity, _LazyRouter(config, identity),
                OpenAICompatibleGenerator(backend=backend, identity=identity,
                    input_token_upper_bound=v['input_token_upper_bound']),
                ToolkitRetriever(retriever=retriever, identity=identity)),
            **{key: Path(saved['component_paths'][key]) for key in (
                'generator_artifact_manifest_path', 'generator_config_path',
                'retriever_implementation_path', 'retriever_config_path',
                'corpus_manifest_path', 'index_manifest_path')})

    def fingerprint(self):
        return self.identity

    def call(self, request):
        return self.driver.handle(request)

    def call_controlled(self, request):
        return self.call(request.as_dict())


def prepare(config, output_dir, *, router_checkpoint):
    """Freeze actual file/deployment identities; no routing, generation or fitting."""
    from agentinputlab.algorithms.acquire.adaptive_rag import AdaptiveRAGArtifactIdentity
    from agentinputlab.configuration import _known
    from workers.adaptive_rag.driver import RuntimeIdentity, _file_sha256, _path_closure_sha256

    validate_runtime(config, prepared=False)
    _known(config.parameters, {'controls'}, 'Adaptive-RAG preparation parameters')
    if config.artifacts or any(k in config.runtime for k in ('deployment_path', 'deployment_sha256')):
        raise ValidationError('prepare-adaptive requires an unbound template')
    if not isinstance(router_checkpoint, str) or not router_checkpoint.strip():
        raise ValidationError('router_checkpoint must name the actual chosen router, not assumed upstream weights')
    backend, retriever = config._backend(), config._retriever()
    output = Path(output_dir).absolute()
    if output.exists():
        raise ValidationError('Adaptive-RAG preparation requires a new output directory')
    v = config.runtime
    snapshot = _path_closure_sha256(Path(v['router_snapshot_path']))
    weights = _path_closure_sha256(Path(v['router_weights_path']))
    tokenizer = _path_closure_sha256(Path(v['router_tokenizer_path']))
    # Built-in file-config retrievers only; no caller-imported Python factories.
    implementation = Path(inspect.getfile(type(retriever)))
    bf, rf = to_jsonable(backend.fingerprint()), to_jsonable(retriever.fingerprint())
    output.mkdir(parents=True)
    def write(name, value):
        path = output/name
        with path.open('x') as stream:
            json.dump(to_jsonable(value), stream, sort_keys=True, indent=2, allow_nan=False)
        return path
    paths = {
        'generator_artifact_manifest_path': write('generator-artifact.json', {'deployment': bf,
            'served_weights_remotely_attested': False}),
        'generator_config_path': write('generator-config.json', {'backend': bf,
            'input_token_upper_bound': v['input_token_upper_bound'], 'temperature': 0,
            'final_text_only': True, 'implicit_retry': False}),
        'retriever_implementation_path': implementation,
        'retriever_config_path': write('retriever-config.json', {'configuration': config.retriever, 'fingerprint': rf}),
        'corpus_manifest_path': write('corpus-manifest.json', {'retriever': rf, 'configuration': config.retriever}),
        'index_manifest_path': write('index-manifest.json', rf)}
    identity = RuntimeIdentity(router_revision=snapshot, router_snapshot_sha256=snapshot,
        router_weights_sha256=weights, router_tokenizer_sha256=tokenizer,
        base_model_revision=content_hash(bf), generator_artifact_sha256=_path_closure_sha256(paths['generator_artifact_manifest_path']),
        generator_config_sha256=_path_closure_sha256(paths['generator_config_path']),
        retriever_revision=content_hash(rf), retriever_implementation_sha256=_path_closure_sha256(implementation),
        retriever_config_sha256=_path_closure_sha256(paths['retriever_config_path']),
        corpus_revision=content_hash(config.retriever), corpus_manifest_sha256=_path_closure_sha256(paths['corpus_manifest_path']),
        index_revision=content_hash(rf), index_manifest_sha256=_path_closure_sha256(paths['index_manifest_path']))
    deployment = write('deployment.json', {'schema': 'agentinputlab.adaptive.controlled-deployment.v1',
        'identity': asdict(identity), 'component_paths': {k: str(p) for k, p in paths.items()},
        'backend': bf, 'retriever': rf, 'retriever_configuration': config.retriever, 'runtime_settings': v})
    runtime_settings = {**v, 'deployment_path': str(deployment), 'deployment_sha256': _file_sha256(deployment)}
    artifacts = AdaptiveRAGArtifactIdentity(
        **{f.name: getattr(identity, f.name) for f in fields(identity) if f.name != 'fixture_runtime'},
        router_checkpoint=router_checkpoint,
        worker_fingerprint_sha256=content_hash(_fingerprint(identity, backend, retriever)),
        **{name+'_prompt_sha256': _file_sha256(Path(v['prompt_root'])/(name+'.txt'))
           for name in ('no_retrieval', 'single_retrieval', 'ircot')})
    ready = replace(config, parameters={**config.parameters, 'artifacts': asdict(artifacts)}, runtime=runtime_settings)
    ready.validate()
    runtime = FileAdaptiveRuntime(ready, backend, retriever)
    runtime.driver.source_fingerprint()
    runtime.driver._verify_runtime_assets(artifacts.as_mapping())
    run = write('run.json', ready.to_dict())
    proof = {'release_id': RELEASE, 'run_path': str(run), 'run_sha256': _file_sha256(run),
             'artifact_identity_sha256': artifacts.identity_sha256,
             'scope': 'controlled_question_only_given_retriever', 'model_calls': 0,
             'retrieval_calls': 0, 'fit_performed': False, 'router_weights_loaded': False}
    write('prepared.json', proof)
    return proof
