"""Held-out native inference from a verified fitted SmartSearch training chain.

This is not the immutable released-3B profile or a training sampler. References
are never accepted here; an evaluator scores the returned prediction afterwards.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from agentinputlab.contracts import BudgetExceededError, ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from workers.smartsearch.driver import (
    MAX_RESPONSE_BYTES,
    MAX_TEXT_CHARS,
    BudgetGate,
    DriverError,
    LocalSnapshotOpenAIModel,
    ModelTurn,
    SearchRecorder,
    SmartSearchDriver,
    _completion_stop,
    _nonnegative,
    _NoRedirect,
)

from .qrag import _sha
from .smartsearch import SMARTSEARCH_TOOL_SCHEMA
from .smartsearch_curriculum import SmartSearchTrainingCase, _question_hash
from .smartsearch_grpo_tools import _new_output, _save
from .smartsearch_sampling import SmartSearchSamplingControls, verify_smartsearch_sampling_source
from .smartsearch_training import SmartSearchTrainingChainArtifact

TRAINED_INFERENCE_PROTOCOL = "agentinputlab.smartsearch.trained_native_inference.v1"


@dataclass(frozen=True, slots=True)
class SmartSearchInferenceControls(SmartSearchSamplingControls):
    """Native 8192-token/ten-turn caps; separately metered, not a512/4096 claim."""


class _TrainedSnapshotTransport(LocalSnapshotOpenAIModel):
    """Honor explicit API authentication on inventory checks as well as POSTs."""

    def _verify_served_root(self):
        if self.http.api_key and urllib.parse.urlsplit(self.base_url).scheme != "https":
            raise DriverError("ENDPOINT_INVALID", "API keys may only be sent over HTTPS")
        if urllib.parse.urlsplit(self.base_url).hostname in {"localhost", "127.0.0.1", "::1"}:
            # Local inference must not leak loopback requests/API credentials
            # into an inherited outbound proxy. This changes this client only.
            self.http.opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({}), _NoRedirect()
            )
        headers = {"Accept": "application/json"}
        if self.http.api_key:
            headers["Authorization"] = "Bearer " + self.http.api_key
        request = urllib.request.Request(self.base_url + "/v1/models", headers=headers)
        try:
            with self.http.opener.open(request, timeout=self.http.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError("model inventory exceeds bound")
            value = json.loads(raw)
            rows = value.get("data") if isinstance(value, dict) else None
            if not isinstance(rows, list):
                raise ValueError("model inventory has no rows")
            matches = [row for row in rows if isinstance(row, dict) and row.get("id") == self.model]
            if len(matches) != 1 or not isinstance(matches[0].get("root"), str):
                raise ValueError("served alias/root is ambiguous")
            if Path(matches[0]["root"]).expanduser().resolve() != self.model_root:
                raise DriverError("MODEL_IDENTITY_MISMATCH", "served model root differs")
        except DriverError:
            raise
        except Exception as exc:
            raise DriverError(
                "MODEL_IDENTITY_MISMATCH", "trained model inventory handshake failed"
            ) from exc


class SmartSearchTrainedLocalModel:
    """Concrete OpenAI-completions transport for the verified learned export.

    The caller starts its own model service with the fitted exported-policy
    root. The served alias/root handshake and manifest closure are checked;
    the secret API key is neither part of fingerprints nor written to journals.
    """

    def __init__(
        self, *, training_chain_root, base_url, served_model_name, timeout_seconds=300, api_key=""
    ):
        chain = SmartSearchTrainingChainArtifact.load(training_chain_root)
        record = json.loads((chain.root / "artifact.json").read_text())
        identity = record.pop("artifact_id", None)
        if identity != chain.artifact_id or content_hash(record) != identity:
            raise ValidationError("SmartSearch fitted service training-chain receipt changed")
        grpo = Path(record["references"]["grpo_root"]).resolve()
        manifest_path = grpo / "model-manifest.json"
        manifest_sha256 = record["chain"]["learned_grpo_manifest_sha256"]
        if _sha(manifest_path) != manifest_sha256:
            raise ValidationError("SmartSearch fitted service model manifest changed")
        manifest = json.loads(manifest_path.read_text())
        plan = grpo / "fit-plan.json"
        if _sha(plan) != record["chain"]["receipt_files"]["grpo"]["plan_sha256"]:
            raise ValidationError("SmartSearch fitted service training plan changed")
        self.seed = json.loads(plan.read_text())["controls"]["seed"]
        self.identity = dict(
            training_chain_artifact_id=chain.artifact_id,
            trained_policy_artifact_id=record["chain"]["artifacts"]["grpo"],
            model_manifest_sha256=manifest_sha256,
            model_root=str(grpo / "exported-policy"),
        )
        self.inner = _TrainedSnapshotTransport(
            dict(
                model_transport="openai_completions_local_snapshot",
                model_base_url=base_url,
                tokenizer_base_url=base_url,
                served_model_name=served_model_name,
                local_model_root=self.identity["model_root"],
                model_timeout_seconds=timeout_seconds,
                checkpoint_repository="agentinputlab-local-trained-smartsearch-policy",
                checkpoint_revision=manifest["revision"],
                tokenizer_revision=manifest["revision"],
                model_snapshot_tree_sha256=content_hash(manifest["files"]),
                context_admission_fingerprint_sha256=content_hash(
                    dict(model_manifest_sha256=manifest_sha256, native_tokenize_endpoint=True)
                ),
            ),
            api_key=api_key,
        )

    def fingerprint(self):
        self.inner._verify_served_root()
        return {**self.identity, "transport": self.inner.fingerprint(), "seed": self.seed}

    def count_prompt_tokens(self, prompt):
        return self.inner.count_prompt_tokens(prompt)

    def generate(self, prompt, *, temperature, max_new_tokens, stop, seed):
        if (
            type(temperature) not in (int, float)
            or temperature != 0.0
            or type(max_new_tokens) is not int
            or max_new_tokens != 8192
            or (not isinstance(stop, (tuple, list)) or tuple(stop) != ("</search>", "</answer>"))
            or (type(seed) is not int or seed != self.seed)
        ):
            raise ValidationError(
                "SmartSearch fitted transport requires its native cap/stops and fitted seed"
            )
        result = self.inner.http.post(
            self.inner.base_url + "/v1/completions",
            dict(
                model=self.inner.model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_new_tokens,
                stop=list(stop),
                seed=seed,
            ),
        )
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("usage"), dict)
            or (
                not isinstance(result.get("choices"), list)
                or not result["choices"]
                or not isinstance(result["choices"][0], dict)
            )
        ):
            raise DriverError(
                "MODEL_RESPONSE_INVALID", "trained completion lacks measured usage", consumed=True
            )
        choice, usage = result["choices"][0], result["usage"]
        text = choice.get("text")
        if not isinstance(text, str) or len(text) > MAX_TEXT_CHARS:
            raise DriverError(
                "MODEL_RESPONSE_INVALID", "trained completion text is invalid", consumed=True
            )
        text, matched = _completion_stop(text, choice)
        return ModelTurn(
            text,
            str(choice.get("finish_reason", "")),
            matched,
            _nonnegative(usage.get("prompt_tokens"), "prompt_tokens"),
            _nonnegative(usage.get("completion_tokens"), "completion_tokens"),
        )


def _surface(source, binding):
    loader = SimpleNamespace(checkout=source, manifest_sha256=content_hash(binding))
    recall, parser = SmartSearchDriver._load_upstream_surface(loader)
    return recall, parser, loader._upstream_module


class SmartSearchTrainedPolicyInference:
    """Run a fitted native policy on declared, disjoint test inputs once.

    Model seam: fingerprint(), count_prompt_tokens(prompt), and
    generate(prompt, temperature=..., max_new_tokens=..., stop=..., seed=...)
    returning ModelTurn with measured usage. Its fingerprint must bind the
    verified chain ID, trained GRPO artifact, exported model root and manifest.
    A caller-owned fingerprint is an identity contract, not remote attestation.
    """

    def __init__(self, *, training_chain_root, model, retriever, controls=None):
        self.controls = controls if controls is not None else SmartSearchInferenceControls()
        if not isinstance(self.controls, SmartSearchInferenceControls):
            raise ValidationError("SmartSearch fitted inference requires typed inference controls")
        self.chain = SmartSearchTrainingChainArtifact.load(training_chain_root)
        self.chain_record_path = self.chain.root / "artifact.json"
        self.chain_record_sha256 = _sha(self.chain_record_path)
        chain = json.loads(self.chain_record_path.read_text())
        identity = chain.pop("artifact_id", None)
        if (
            identity != self.chain.artifact_id
            or content_hash(chain) != identity
            or (_sha(self.chain_record_path) != self.chain_record_sha256)
        ):
            raise ValidationError("SmartSearch training-chain receipt changed during binding")
        self.references, self.chain_summary = chain["references"], chain["chain"]
        self.grpo_root = Path(self.references["grpo_root"]).resolve()
        self.model_manifest = self.grpo_root / "model-manifest.json"
        self.model_manifest_sha256 = self.chain_summary["learned_grpo_manifest_sha256"]
        if _sha(self.model_manifest) != self.model_manifest_sha256:
            raise ValidationError(
                "SmartSearch trained policy manifest changed after chain verification"
            )
        self.fit_plan = self.grpo_root / "fit-plan.json"
        self.fit_plan_sha256 = self.chain_summary["receipt_files"]["grpo"]["plan_sha256"]
        if _sha(self.fit_plan) != self.fit_plan_sha256:
            raise ValidationError("SmartSearch trained fit plan changed after chain verification")
        plan = json.loads(self.fit_plan.read_text())
        if self.controls.seed != plan["controls"]["seed"]:
            raise ValidationError(
                "SmartSearch trained inference seed differs from its fitted chain"
            )
        for value, methods in (
            (model, ("fingerprint", "count_prompt_tokens", "generate")),
            (retriever, ("fingerprint", "search")),
        ):
            if any(not callable(getattr(value, name, None)) for name in methods):
                raise ValidationError(
                    "SmartSearch trained inference needs explicit model/retrieval seams"
                )
        self.model, self.retriever = model, retriever
        self.model_identity = to_jsonable(model.fingerprint())
        required = dict(
            training_chain_artifact_id=self.chain.artifact_id,
            trained_policy_artifact_id=self.chain_summary["artifacts"]["grpo"],
            model_manifest_sha256=self.model_manifest_sha256,
            model_root=str(self.grpo_root / "exported-policy"),
        )
        if not isinstance(self.model_identity, dict) or any(
            self.model_identity.get(k) != v for k, v in required.items()
        ):
            raise ValidationError(
                "SmartSearch inference provider is not bound to the verified trained policy"
            )
        self.retrieval_identity = to_jsonable(retriever.fingerprint())
        for name in ("corpus_sha256", "index_sha256"):
            digest = (
                self.retrieval_identity.get(name)
                if isinstance(self.retrieval_identity, dict)
                else None
            )
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or set(digest) - set("0123456789abcdef")
            ):
                raise ValidationError(
                    "SmartSearch inference requires frozen test corpus/index identities"
                )
        split = self.chain_summary["split_identity"]
        self.fit_ids = set(split["train"]) | set(split["validation"])
        self.fit_questions = set(split["train"].values()) | set(split["validation"].values())

    def _unchanged(self):
        if _sha(self.chain_record_path) != self.chain_record_sha256 or (
            _sha(self.model_manifest) != self.model_manifest_sha256
            or _sha(self.fit_plan) != self.fit_plan_sha256
            or to_jsonable(self.model.fingerprint()) != self.model_identity
            or to_jsonable(self.retriever.fingerprint()) != self.retrieval_identity
        ):
            raise ExternalDependencyError(
                "SmartSearch trained inference source/model/retrieval binding changed"
            )

    def predict(self, *, case: SmartSearchTrainingCase, output_root):
        if not isinstance(case, SmartSearchTrainingCase) or case.split != "test":
            raise ValidationError(
                "SmartSearch trained inference accepts explicitly declared test cases"
            )
        if case.case_id in self.fit_ids or _question_hash(case.question) in self.fit_questions:
            raise ValidationError(
                "SmartSearch held-out test overlaps fitted training/development inputs"
            )
        self._unchanged()
        source, binding = verify_smartsearch_sampling_source(self.references["checkout"])
        recall, parser, module = _surface(source, binding)
        output = Path(output_root).resolve(strict=False)
        protected = [source, self.chain.root, self.grpo_root]
        for item in self.references.values():
            if isinstance(item, dict):
                protected.extend(
                    Path(value).resolve()
                    for key, value in item.items()
                    if key != "model_manifest_sha256"
                )
            else:
                protected.append(Path(item).resolve())
        if any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected):
            raise ValidationError(
                "SmartSearch prediction output overlaps protected training/source inputs"
            )
        root = _new_output(output_root)
        _save(
            root,
            "claim.json",
            dict(
                protocol=TRAINED_INFERENCE_PROTOCOL,
                case=case,
                training_chain_artifact_id=self.chain.artifact_id,
                source=binding,
                model=self.model_identity,
                retrieval=self.retrieval_identity,
                controls=self.controls,
                references_accepted=False,
                training_sampler_used=False,
                no_automatic_retry=True,
            ),
        )
        gate = BudgetGate(
            dict(
                model_calls=self.controls.max_model_calls,
                output_tokens=self.controls.max_output_tokens,
                retrieval_calls=self.controls.max_retrieval_calls,
                context_tokens=self.controls.max_input_tokens,
            )
        )
        recorder = SearchRecorder(self.retriever, gate, self.retrieval_identity)
        responses = 0
        owner = self

        class ModelSeam:
            def post(self, url, json):
                nonlocal responses
                sampling = dict(
                    temperature=0.0, max_new_tokens=8192, stop=["</search>", "</answer>"]
                )
                if (
                    url != "agentinputlab://model/generate"
                    or set(json) != {"text", "sampling_params"}
                    or json["sampling_params"] != sampling
                ):
                    raise ExternalDependencyError(
                        "SmartSearch native held-out generation contract changed"
                    )
                owner._unchanged()
                prompt = json["text"]
                if not isinstance(prompt, str) or len(prompt) > MAX_TEXT_CHARS:
                    raise ValidationError("SmartSearch held-out prompt must be bounded text")
                admitted = owner.model.count_prompt_tokens(prompt)
                if type(admitted) is not int or admitted < 1:
                    raise ValidationError(
                        "SmartSearch held-out inference needs exact token admission"
                    )
                owner._unchanged()
                gate.before_model(admitted)
                number = gate.usage.model_calls
                _save(
                    root,
                    f"call-{number:04d}-request.json",
                    dict(prompt=prompt, **sampling, seed=owner.controls.seed),
                )
                turn = owner.model.generate(
                    prompt,
                    temperature=0.0,
                    max_new_tokens=8192,
                    stop=tuple(sampling["stop"]),
                    seed=owner.controls.seed,
                )
                if (
                    not isinstance(turn, ModelTurn)
                    or any(
                        type(n) is not int or n < 0 for n in (turn.input_tokens, turn.output_tokens)
                    )
                    or (
                        any(
                            not isinstance(value, str)
                            for value in (turn.text, turn.finish_reason, turn.matched)
                        )
                    )
                ):
                    raise ExternalDependencyError(
                        "SmartSearch inference provider lacks typed measured usage"
                    )
                if len(turn.text) > MAX_TEXT_CHARS:
                    _save(
                        root,
                        f"call-{number:04d}-response.json",
                        dict(
                            text_sha256=content_hash(turn.text),
                            oversized_text_not_truncated_for_scoring=True,
                            input_tokens=turn.input_tokens,
                            output_tokens=turn.output_tokens,
                        ),
                    )
                else:
                    _save(root, f"call-{number:04d}-response.json", turn)
                responses += 1
                gate.after_model(
                    turn.input_tokens, turn.output_tokens, admitted_input_tokens=admitted
                )
                if len(turn.text) > MAX_TEXT_CHARS:
                    raise ExternalDependencyError(
                        "SmartSearch inference response exceeded text bound"
                    )
                if turn.output_tokens > 8192:
                    raise BudgetExceededError(
                        "SmartSearch inference provider exceeded native per-call token cap"
                    )
                text, matched = _completion_stop(
                    turn.text, dict(finish_reason=turn.finish_reason, stop_reason=turn.matched)
                )
                owner._unchanged()
                return SimpleNamespace(
                    json=lambda: dict(
                        text=text,
                        meta_info=dict(
                            finish_reason=dict(type=turn.finish_reason, matched=matched)
                        ),
                    )
                )

        class FixedReCall(recall):
            def execute_tool_calls(self, env, calls):
                if env != "agentinputlab:no-exec":
                    raise ValidationError(
                        "SmartSearch inference only exposes fixed Wikipedia retrieval"
                    )
                owner._unchanged()
                result = recorder.execute(calls)
                owner._unchanged()
                return result

        try:
            module.requests = ModelSeam()
            agent = FixedReCall("agentinputlab://model", "agentinputlab://disabled-executor")
            transcript = recall.run.__wrapped__(
                agent,
                "agentinputlab:no-exec",
                json.dumps(to_jsonable(SMARTSEARCH_TOOL_SCHEMA)),
                case.question,
            )
            prediction = parser(transcript)
            if not isinstance(prediction, str):
                raise ExternalDependencyError(
                    "SmartSearch native answer parser did not return text"
                )
            owner._unchanged()
            _, after = verify_smartsearch_sampling_source(self.references["checkout"])
            if after != binding:
                raise ExternalDependencyError(
                    "SmartSearch held-out native source changed during inference"
                )
            _save(
                root,
                "transcript.json",
                dict(raw_transcript=transcript, native_prediction=prediction),
            )
            result = dict(
                protocol=TRAINED_INFERENCE_PROTOCOL,
                case_id=case.case_id,
                split="test",
                training_chain_artifact_id=self.chain.artifact_id,
                model=self.model_identity,
                source=binding,
                retrieval=self.retrieval_identity,
                prediction=prediction,
                native_answer_present=bool(prediction.strip()),
                scored=False,
                benchmark_score=None,
                usage=gate.usage.as_mapping(),
                model_responses=responses,
                unknown_output_token_reservations=0,
                trajectory=recorder.trajectory,
                evidence=recorder.evidence,
                seed_adapter=self.controls.seed,
                transcript_sha256=_sha(root / "transcript.json"),
                native_loop_and_parser_called=True,
                training_sampler_used=False,
                test_reference_or_reward_used=False,
                optimizer_steps=0,
                empty_prediction_is_retained_not_success=True,
                no_automatic_retry=True,
            )
            _save(root, "manifest.json", result)
            return result
        except Exception as exc:
            _save(
                root,
                "failed.json",
                dict(
                    protocol=TRAINED_INFERENCE_PROTOCOL,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    usage=gate.usage.as_mapping(),
                    model_responses=responses,
                    unknown_output_token_reservations=max(0, gate.usage.model_calls - responses)
                    * 8192,
                    trajectory=recorder.trajectory,
                    evidence=recorder.evidence,
                    no_automatic_retry=True,
                    optimizer_steps=0,
                    scored=False,
                ),
            )
            raise
