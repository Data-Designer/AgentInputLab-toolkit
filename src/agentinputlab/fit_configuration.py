"""Explicit file entrypoints for existing text-space optimization recipes.

Fitting is separate from test-time inference. No automatic retries, implicit
split selection, SFT/RL training, or cache-miss fitting is performed here.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from agentinputlab.configuration import _json_mapping, _known, _object, _unique
from agentinputlab.contracts import ExternalDependencyError, InterventionContext, ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable


def _save(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(to_jsonable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or '\x00' in value:
        raise ValidationError(f'{label} must be a non-empty path')
    return value


def _reference(value: Any, label: str = 'native_config') -> dict[str, Any]:
    value = _object(value, label)
    if set(value) != {'path', 'sha256'}:
        raise ValidationError(f'{label} requires path and sha256')
    _path(value['path'], f'{label}.path')
    digest = value['sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValidationError(f'{label}.sha256 must be a lowercase SHA-256')
    return value


@dataclass(frozen=True)
class FitConfig:
    """A caller-selected native fit, with explicit controls and model identities."""

    value: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'value', _json_mapping(self.value, 'fit configuration'))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FitConfig:
        config = cls(value)
        config.validate()
        return config

    @classmethod
    def from_file(cls, path: str | Path) -> FitConfig:
        try:
            with Path(path).open('rb') as stream:
                raw = stream.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValidationError('fit configuration exceeds 1 MiB')
            value = _object(json.loads(raw, object_pairs_hook=_unique), 'fit configuration')
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
            raise ValidationError('cannot read fit configuration as UTF-8 JSON') from None
        base = Path(path).absolute().parent
        for key in ('checkout', 'python_executable'):
            if isinstance(value.get(key), str):
                value[key] = os.path.abspath(base / Path(value[key]).expanduser())
        for group, keys in (('native_config', ('path',)),
                            ('feedback', ('checkout',)),
                            ('environment', ('checkout', 'worker_python')),
                            ('environment_binding', ('factory_path', 'configuration_path'))):
            if isinstance(value.get(group), dict):
                for key in keys:
                    if isinstance(value[group].get(key), str):
                        value[group][key] = os.path.abspath(base / Path(value[group][key]).expanduser())
        for field in ('feedback', 'environment'):
            feedback = value.get(field)
            if isinstance(feedback, dict) and isinstance(feedback.get('source_input'), dict):
                pin = feedback['source_input']
                if isinstance(pin.get('path'), str):
                    pin['path'] = os.path.abspath(base / Path(pin['path']).expanduser())
        return cls.from_dict(value)

    def _ace_feedback(self):
        feedback = self.value.get('feedback')
        if feedback is None:
            return None
        feedback = _object(feedback, 'ACE feedback')
        kind = feedback.get('kind')
        if kind == 'retrieval':
            _known(feedback, {'kind', 'source_input'}, 'ACE retrieval feedback')
            _reference(feedback.get('source_input'), 'feedback.source_input')
            return feedback
        if kind == 'bfcl':
            _known(feedback, {'kind', 'source_input', 'model_id', 'partition_label'}, 'ACE BFCL feedback')
            _reference(feedback.get('source_input'), 'feedback.source_input')
            for field in ('model_id', 'partition_label'):
                _path(feedback.get(field), 'feedback.' + field)
            return feedback
        if kind not in ('statebench', 'toolsandbox', 'tau'):
            raise ValidationError('ACE feedback kind must be statebench, toolsandbox, tau, bfcl or retrieval')
        allowed = {'kind', 'checkout', 'source_input', 'environment_names'}
        if kind != 'toolsandbox':
            allowed.add('environment_base_url')
        _known(feedback, allowed, 'ACE feedback')
        _path(feedback.get('checkout'), 'feedback.checkout')
        _reference(feedback.get('source_input'), 'feedback.source_input')
        names = feedback.get('environment_names', ())
        if not isinstance(names, (list, tuple)) or any(not isinstance(n, str) or not n.isidentifier() for n in names) or len(set(names)) != len(names):
            raise ValidationError('feedback.environment_names must contain unique names, not values')
        if 'environment_base_url' in feedback:
            url = feedback['environment_base_url']
            if not isinstance(url, str):
                raise ValidationError('feedback.environment_base_url must be an HTTP(S) URL')
            try:
                parsed = urlsplit(url)
                valid = parsed.scheme in ('http', 'https') and parsed.hostname and not (
                    parsed.username or parsed.password or parsed.query or parsed.fragment)
            except ValueError:
                valid = False
            if not valid:
                raise ValidationError('feedback.environment_base_url must be HTTP(S) without credentials')
        return feedback

    def _feedback_environment_names(self, feedback, source):
        """Include credentials already named by pinned native ToolSandbox input.

        The input stores names, never credential values. Offline workers need
        these forwarded too; online must fail before creating an attempt if a
        required variable is absent. Do not inspect arbitrary source fields.
        """
        names = list(feedback.get('environment_names', ()))
        if feedback['kind'] == 'toolsandbox':
            native = source.get('native_configuration', {})
            if not isinstance(native, Mapping):
                raise ValidationError('ToolSandbox native_configuration must be an object')
            for field in ('agent_key_env', 'user_key_env'):
                if field not in native:
                    continue  # The native schema validator reports missing fields.
                name = native[field]
                if not isinstance(name, str) or not name.isidentifier():
                    raise ValidationError(f'ToolSandbox {field} must name an environment variable')
                if name not in names:
                    names.append(name)
        return names

    def _controls(self):
        from agentinputlab.recipes import ACEControls, SkillOptControls

        release = self.value.get('release_id')
        factory = {'recipe.ace.offline': ACEControls, 'recipe.skillopt.offline': SkillOptControls}.get(release)
        if factory is None:
            raise ValidationError('fit supports recipe.ace.offline or recipe.skillopt.offline')
        controls = _object(self.value.get('controls'), 'fit controls')
        if set(controls) != {field.name for field in fields(factory)}:
            raise ValidationError('fit controls must explicitly specify every native control, including seed')
        return factory(**controls)

    def _ace_examples(self):
        from agentinputlab.recipes.ace import ACEExample, validate_split_isolation

        groups = _object(self.value.get('examples'), 'examples')
        if set(groups) != {'train', 'validation'}:
            raise ValidationError('ACE examples must explicitly contain train and validation, never test')
        result = {}
        for split, values in groups.items():
            if not isinstance(values, (list, tuple)) or not values:
                raise ValidationError(f'ACE {split} examples must be a non-empty array')
            parsed = []
            for value in values:
                value = _object(value, 'ACE example')
                _known(value, {'example_id', 'input', 'expected_output', 'split', 'metadata'}, 'ACE example')
                if value.get('split') != split:
                    raise ValidationError(f'ACE example split must explicitly be {split}; no relabeling')
                parsed.append(ACEExample.from_value(value, split=split))
            result[split] = tuple(parsed)
        validate_split_isolation(result['train'], result['validation'])
        return result

    def validate(self) -> dict[str, Any]:
        controls = self._controls()
        value = self.value
        common = {'release_id', 'checkout', 'python_executable', 'controls'}
        for key in ('checkout', 'python_executable'):
            _path(value.get(key), key)
        if value['release_id'] == 'recipe.ace.offline':
            _known(value, common | {'examples', 'deployment', 'timeout_seconds', 'feedback'}, 'ACE fit')
            timeout = value.get('timeout_seconds')
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
                raise ValidationError('ACE timeout_seconds must be finite and positive')
            self._ace_examples()
            self._ace_feedback()
            from workers.ace.driver import LLMConfig, WorkerFailure

            try:
                deployment = LLMConfig.from_json(json.dumps(to_jsonable(value.get('deployment'))))
            except WorkerFailure as exc:
                raise ValidationError('ACE deployment must match the native three-role configuration') from exc
            if deployment.max_tokens_per_call > controls.max_tokens:
                raise ValidationError('ACE per-call tokens exceed total fit max_tokens')
            feedback = self._ace_feedback()
            if (feedback is not None and feedback['kind'] == 'bfcl'
                    and feedback['model_id'] != value['deployment']['roles']['target']['model']):
                raise ValidationError('BFCL scorer model_id must match the ACE target deployment')
        else:
            _known(value, common | {'native_config', 'model_fingerprints', 'environment_names',
                                   'environment_binding', 'allow_custom_environment', 'environment'}, 'SkillOpt fit')
            _reference(value.get('native_config'))
            from agentinputlab.recipes.skillopt import _model_fingerprints

            _model_fingerprints(InterventionContext(run_config={
                'skillopt_model_fingerprints': value.get('model_fingerprints')}))
            names = value.get('environment_names', ())
            if not isinstance(names, (list, tuple)) or any(not isinstance(n, str) or not n.isidentifier() for n in names) or len(set(names)) != len(names):
                raise ValidationError('environment_names must contain unique environment variable names, not values')
            if 'allow_custom_environment' in value and not isinstance(value['allow_custom_environment'], bool):
                raise ValidationError('allow_custom_environment must be boolean')
            if value.get('environment_binding') is not None:
                if value.get('allow_custom_environment') is not True:
                    raise ValidationError('caller-owned environment code requires allow_custom_environment=true')
                self._skill_binding()
            if value.get('environment') is not None:
                from agentinputlab.recipes.skillopt_builtin_environment import validate_environment

                if 'environment_binding' in value or 'allow_custom_environment' in value:
                    raise ValidationError('choose the built-in environment or a caller environment, not both')
                native, _ = validate_environment(value['environment'])
                if native['seed'] != controls.seed:
                    raise ValidationError('SkillOpt environment seed must match fit controls')
                if native['backend']['deployment_identity'] != value['model_fingerprints']['target']:
                    raise ValidationError('SkillOpt environment target identity must match model_fingerprints')
                if native['backend'].get('api_key_env', 'OPENAI_API_KEY') not in names:
                    raise ValidationError('declare the target api_key_env in environment_names')
                if (native.get('environment_backend') and
                        native['environment_backend'].get('api_key_env', 'OPENAI_API_KEY') not in names):
                    raise ValidationError('declare the environment api_key_env in environment_names')
        return dict(validated=True, executed=False, release_id=value['release_id'],
                    model_calls=0, runtime_ready=None, automatic_retry=False,
                    optimization_controls=to_jsonable(value['controls']))

    def _skill_binding(self):
        from agentinputlab.recipes.skillopt_environment import SkillOptEnvironmentBinding

        value = self.value.get('environment_binding')
        if value is None:
            return None
        try:
            return SkillOptEnvironmentBinding(**_object(value, 'environment_binding'))
        except TypeError:
            raise ValidationError('invalid SkillOpt environment_binding fields') from None

    def execute(self, output_dir: str | Path) -> dict[str, Any]:
        """Consume at most one native fit; all attempts get a create-only terminal."""
        self.validate()
        output = Path(output_dir).absolute()
        if output.exists() or output.is_symlink():
            raise ValidationError('fit output must be a new directory; reuse the saved artifact, not the attempt')
        value, controls = self.value, self._controls()
        environment = {}
        if value['release_id'] == 'recipe.ace.offline' and self._ace_feedback() is not None:
            from agentinputlab.artifacts import load_pinned_json_mapping

            feedback = self._ace_feedback()
            pin = feedback['source_input']
            source = load_pinned_json_mapping(pin['path'], expected_sha256=pin['sha256'], max_bytes=16 * 1024 * 1024)
            for name in self._feedback_environment_names(feedback, source):
                if not os.environ.get(name):
                    raise ExternalDependencyError(f'missing fit environment variable: {name}')
        if value['release_id'] == 'recipe.skillopt.offline':
            config = Path(value['native_config']['path'])
            if not config.is_file() or config.is_symlink() or config.stat().st_size > 1024 * 1024:
                raise ValidationError('native_config must be a bounded regular file')
            if _sha(config) != value['native_config']['sha256']:
                raise ValidationError('native_config checksum mismatch')
            binding = self._skill_binding()
            if binding is not None:
                binding.verify()
            if value.get('environment') is not None:
                from agentinputlab.artifacts import load_pinned_json_mapping

                pin = value['environment']['source_input']
                load_pinned_json_mapping(pin['path'], expected_sha256=pin['sha256'], max_bytes=8 * 1024 * 1024)
            for name in value.get('environment_names', ()):
                if name not in os.environ:
                    raise ExternalDependencyError(f'missing fit environment variable: {name}')
                environment[name] = os.environ[name]
        output.mkdir(parents=True)
        identity = dict(release_id=value['release_id'], configuration_sha256=content_hash(value),
                        automatic_retry=False,
                        test_evaluation_performed=False if value['release_id'] == 'recipe.ace.offline' else None,
                        skillopt_evaluation_splits_controlled_by_native_configuration=(value['release_id'] == 'recipe.skillopt.offline'))
        _save(output / 'started.json', dict(identity, status='started'))
        try:
            if value['release_id'] == 'recipe.ace.offline':
                artifact = self._fit_ace(output, controls)
            else:
                from agentinputlab.recipes import SkillOptOfflineRecipe

                if value.get('environment') is not None:
                    from agentinputlab.recipes import skillopt_builtin_environment
                    from agentinputlab.recipes.skillopt_environment import (
                        SkillOptEnvironmentBinding,
                    )

                    environment_config = output / 'environment.json'
                    _save(environment_config, value['environment'])
                    factory = Path(skillopt_builtin_environment.__file__).resolve()
                    binding = SkillOptEnvironmentBinding(factory, _sha(factory),
                        environment_config, _sha(environment_config))
                artifact = SkillOptOfflineRecipe(controls=controls,
                    python_executable=value['python_executable'], environment_binding=binding).fit(
                        checkout=value['checkout'], config_path=config, output_root=output / 'native',
                        context=InterventionContext(seed=controls.seed, run_config={
                            'skillopt_model_fingerprints': value['model_fingerprints']}),
                        environment=environment)
            _save(output / 'artifact.json', artifact.as_dict())
            reference = dict(path=str(output / 'artifact.json'), sha256=_sha(output / 'artifact.json'))
            evaluation_release = value['release_id'].replace('.offline', '.eval_only')
            result = dict(identity, status='fit_completed', artifact=reference,
                          artifact_id=artifact.artifact_id, evaluation_release_id=evaluation_release,
                          optimization_usage=to_jsonable(getattr(artifact, 'usage', None)))
            _save(output / 'terminal.json', result)
            return result
        except Exception as exc:
            # Native exception strings can contain provider credentials or prompts.
            _save(output / 'terminal.json', dict(identity, status='failed_fit',
                error_type=type(exc).__name__, usage_may_have_been_consumed=True,
                native_error_sha256=hashlib.sha256(str(exc).encode()).hexdigest()))
            raise ExternalDependencyError(
                f'fitting failed; inspect saved native evidence in {output}; no automatic retry'
            ) from None

    def _fit_ace(self, output: Path, controls):
        from agentinputlab.integrations.external_workers.base import JsonLineSubprocessWorker
        from agentinputlab.recipes import ACEOfflineRecipe
        from workers.ace import driver

        value = self.value
        deployment = to_jsonable(value['deployment'])
        key = deployment.get('api_key_env')
        if key is None:
            key = {'openai': 'OPENAI_API_KEY', 'sambanova': 'SAMBANOVA_API_KEY',
                   'together': 'TOGETHER_API_KEY', 'commonstack': 'COMMONSTACK_API_KEY'}.get(deployment['api_provider'])
        command = (value['python_executable'], str(Path(driver.__file__).resolve()),
                   '--checkout', value['checkout'], '--llm-config-json', json.dumps(deployment))
        environment = {'PYTHONDONTWRITEBYTECODE': '1'}
        names = [key] if key else []
        feedback = self._ace_feedback()
        if feedback is not None:
            kind = feedback['kind']
            configuration = {k: to_jsonable(v) for k, v in feedback.items()
                             if k not in ('kind', 'environment_names')}
            configuration.update(output_root=str(output / 'native-feedback'), mode='offline')
            command += ('--task-processor-factory',
                        f'agentinputlab.recipes.ace_{kind}:create_{kind}_feedback',
                        '--task-processor-config-json', json.dumps(configuration))
            # Load the same installed toolkit in the isolated native worker,
            # while its selected Python retains the native arena dependencies.
            environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
            from agentinputlab.artifacts import load_pinned_json_mapping

            pin = feedback['source_input']
            source = load_pinned_json_mapping(pin['path'], expected_sha256=pin['sha256'], max_bytes=16 * 1024 * 1024)
            names.extend(self._feedback_environment_names(feedback, source))
        delegate = JsonLineSubprocessWorker(
            command=command,
            worker_id='ace-configured-offline-fit', identity={'configuration_sha256': content_hash(value)},
            environment=environment,
            pass_environment=tuple(names), timeout_seconds=value['timeout_seconds'])

        class RecordedWorker:
            worker_id = delegate.worker_id

            def fingerprint(self):
                return delegate.fingerprint()

            @property
            def command(self):
                return delegate.command

            def call(self, request):
                _save(output / 'worker-request.json', request)
                try:
                    response = delegate.call(request)
                    _save(output / 'worker-response.json', response)
                    return response
                finally:
                    if delegate.last_resource_receipt is not None:
                        _save(output / 'worker-resource.json', delegate.last_resource_receipt)

        groups = self._ace_examples()
        return ACEOfflineRecipe(client=RecordedWorker(), controls=controls).fit(
            groups['train'], validation_examples=groups['validation'],
            context=InterventionContext(seed=controls.seed, run_config={'ace_model_fingerprints': {
                name: role['fingerprint'] for name, role in deployment['roles'].items()}}))
