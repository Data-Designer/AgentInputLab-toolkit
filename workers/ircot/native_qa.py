"""Complete pinned IRCoT-QA graph with explicitly substituted model and index.

No source is copied here: the native builder, all six participants, prompt
reader, best-first loop and prediction extractor execute from the mount.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import os
from dataclasses import replace
from pathlib import Path

from agentinputlab.backends import GenerationRequest
from agentinputlab.contracts.serialization import content_hash, to_jsonable

from .controlled import _NATIVE_LOCK, PinnedIRCoTControlledRuntime, _LiveBindings
from .driver import (
    DriverError,
    IRCoTDriver,
    _BoundedCapture,
    _load_config,
    _parse_request,
    _project_trajectory,
    _RetrievalCapture,
    _safe_file,
    _TracingController,
    _upstream_import_scope,
    _working_directory,
)

QA_PROFILE = "upstream.hotpotqa.codex_qa.v1"
QA_BINDING_MODE = "agentinputlab.ircot.native_like_qa.v1"
QA_MANIFEST = Path(__file__).with_name("qa_compatibility_manifest.json")
QA_ROLES = {
    "retrieve_and_reset_paragraphs",
    "step_by_step_cot_gen",
    "step_by_step_exit_controller",
    "copy_question",
    "llmqa",
    "answer_extractor",
}


def patch_native_qa_config(config, checkout, controls):
    """Change bounded limits only; all graph transitions/parsers stay native."""
    value = copy.deepcopy(dict(config))
    models = value.get("models")
    if (
        not isinstance(models, dict)
        or {v.get("name") for v in models.values()} != QA_ROLES
        or len(models) != 6
    ):
        raise DriverError(
            "CONFIG_INVALID", "native IRCoT-QA requires all six original participants"
        )
    for model in models.values():
        role = model["name"]
        if role == "retrieve_and_reset_paragraphs":
            model["retrieval_count"] = controls.per_step_top_k
            model["global_max_num_paras"] = controls.max_total_documents
            model["cumulate_titles"] = controls.reset_policy != "replace_each_step"
        elif role in ("step_by_step_cot_gen", "step_by_step_exit_controller"):
            model["max_num_sentences"] = controls.max_steps
            if role == "step_by_step_exit_controller":
                model["global_max_num_paras"] = controls.max_total_documents
        if role in ("step_by_step_cot_gen", "llmqa"):
            if _safe_file(checkout.root, model.get("prompt_file")) != checkout.prompt_path:
                raise DriverError("PROMPT_MISMATCH", "native IRCoT-QA uses an unpinned prompt")
            if model.get("gen_model") != "gpt3":
                raise DriverError("CONFIG_INVALID", "native IRCoT-QA generator class changed")
    if value.get("prediction_type") != "answer":
        raise DriverError("CONFIG_INVALID", "native IRCoT-QA must use answer prediction")
    return value


class PinnedIRCoTNativeQARuntime(PinnedIRCoTControlledRuntime):
    """Dedicated-process complete QA runtime; inherited startup checks unchanged."""

    worker_id = "agentinputlab.ircot.native_like_full_qa.v1"

    def _checkout(self):
        if Path(os.path.expanduser("~")).resolve() != self.cache_home:
            raise DriverError("CACHE_SCOPE_CHANGED", "dedicated worker HOME changed")
        checkout = IRCoTDriver(
            manifest_path=QA_MANIFEST,
            profile_id=QA_PROFILE,
            environment={"AGENTINPUTLAB_UPSTREAM_ROOT": str(self.source_root)},
        )._checkout()
        return replace(
            checkout,
            profile_id="native_like.hotpotqa.full_qa.v1",
            profile={
                **checkout.profile,
                "identity_kind": "native_like_substituted_qa",
                "corpus_revision": self.corpus_revision,
                "index_revision": self.index_revision,
            },
        )

    def fingerprint(self):
        value = dict(super().fingerprint())
        value.update(
            binding_mode=QA_BINDING_MODE,
            profile_id="native_like.hotpotqa.full_qa.v1",
            identity_kind="native_like_substituted_qa",
            source_profile=QA_PROFILE,
            runtime_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            controlled_binding_helpers_sha256=hashlib.sha256(
                Path(__file__).with_name("controlled.py").read_bytes()
            ).hexdigest(),
            full_six_participant_QA_graph=True,
            native_final_prediction_extraction=True,
            shared_final_call=True,
            native_generator_default_output_cap=300,
            caller_backend_sampling_is_explicit=True,
            published_score_parity=False,
        )
        return value

    def run(self, request):
        from agentinputlab.algorithms.acquire.ircot_native import IRCoTNativeQARequest

        if not isinstance(request, IRCoTNativeQARequest):
            raise TypeError("native IRCoT-QA requires its typed request")
        if not _NATIVE_LOCK.acquire(blocking=False):
            raise RuntimeError("one native IRCoT call per dedicated worker process")
        self.last_events, self.last_usage = [], {}
        bindings = None
        try:
            identity = self.fingerprint()
            legacy = _parse_request(request.legacy.as_dict())
            if (
                content_hash(identity) != request.expected_runtime_fingerprint_sha256
                or request.seed != self.seed
                or request.max_model_calls > self.max_model_calls
                or request.per_call_output_tokens != self.per_call_output_tokens
                or legacy.controls.max_steps + 1 > request.max_model_calls
                or legacy.controls.max_output_tokens > self.max_output_tokens
            ):
                raise DriverError(
                    "BINDING_MISMATCH", "native IRCoT-QA source/model/budget identity differs"
                )
            if (
                to_jsonable(legacy.generator_fingerprint) != self.generator_fingerprint
                or to_jsonable(self.backend.fingerprint()) != self.generator_fingerprint
                or to_jsonable(legacy.retriever_fingerprint) != self.retriever_fingerprint
                or to_jsonable(self.retriever.fingerprint()) != self.retriever_fingerprint
                or legacy.corpus_revision != self.corpus_revision
                or legacy.index_revision != self.index_revision
            ):
                raise DriverError("BINDING_MISMATCH", "native IRCoT-QA context bindings differ")
            checkout = self._checkout()
            if legacy.prompt_hash != checkout.profile["prompt_sha256"]:
                raise DriverError("PROMPT_MISMATCH", "native IRCoT-QA prompt changed")
            bindings = _QABindings(self, legacy, request)
            config = patch_native_qa_config(
                _load_config(
                    checkout,
                    {
                        "AGENTINPUTLAB_IRCOT_RETRIEVER_HOST": "http://127.0.0.1",
                        "AGENTINPUTLAB_IRCOT_RETRIEVER_PORT": "0",
                    },
                ),
                checkout,
                legacy.controls,
            )
            for model in config["models"].values():
                if model["name"] == "retrieve_and_reset_paragraphs":
                    model["source_corpus_name"] = self.corpus_revision
                prompt_arguments = model.get("prompt_reader_args", {})
                demonstration_ids = prompt_arguments.get("filter_by_key_values", {}).get("qid", [])
                if legacy.task_id in demonstration_ids:
                    raise DriverError(
                        "EVALUATION_ID_IS_DEMONSTRATION",
                        "native IRCoT-QA evaluation ID appears among the pinned prompt demonstrations",
                    )
            capture = _BoundedCapture()
            with (
                _working_directory(checkout.root),
                _upstream_import_scope(checkout.root, fixture_shims=False),
                contextlib.redirect_stdout(capture),
            ):
                configurable = importlib.import_module("commaqa.inference.configurable_inference")
                native = importlib.import_module("commaqa.inference.ircot")
                qa_model = importlib.import_module("commaqa.execution.llm_qa_model")
                prompts = importlib.import_module("commaqa.inference.prompt_reader")
                prompts.get_tokenizer = bindings.get_tokenizer
                bindings.fit_prompt = prompts.fit_prompt_into_given_limit
                native.GPT3Generator = bindings.generator_class("reasoning")
                qa_model.GPT3Generator = bindings.generator_class("final_answer")
                retrieval = _RetrievalCapture(bindings.retrieve)
                native.safe_post_request = retrieval
                decomposer, model_map = configurable.build_decomposer_and_models(
                    copy.deepcopy(config)
                )
                tracing = _TracingController(
                    decomposer.controller,
                    {k: v["name"] for k, v in config["models"].items()},
                    retrieval,
                )
                decomposer.controller = tracing
                # Observe find_answer_decomp without substituting its loop or
                # the source return_qid_prediction JSON/string extraction.
                native_find = decomposer.find_answer_decomp
                captured = []

                def observed_find(example, debug=False):
                    result = native_find(example, debug=debug)
                    captured.append(result[0])
                    return result

                decomposer.find_answer_decomp = observed_find
                example = dict(
                    qid=legacy.task_id or legacy.request_id,
                    query=legacy.question,
                    question=legacy.question,
                    metadata={"pinned_title": None, "pinned_para": None, "pin_position": None},
                    titles=[],
                    paras=[],
                )
                qid, answer, chain = decomposer.return_qid_prediction(example, silent=True)
                if len(captured) != 1 or captured[0] is None or qid != example["qid"]:
                    raise DriverError(
                        "NATIVE_QA_NO_FINAL_STATE",
                        "native full QA produced no final state",
                        pre_execution=False,
                    )
                if not isinstance(answer, str) or not answer.strip():
                    raise DriverError(
                        "NATIVE_QA_EMPTY_OR_NON_TEXT",
                        "native full QA returned an empty or non-text prediction",
                        pre_execution=False,
                    )
                result = _project_trajectory(
                    legacy,
                    checkout,
                    config,
                    tracing.events,
                    captured[0],
                    native,
                    model_map,
                    bindings,
                )
                result.update(
                    identity=identity,
                    answer=answer,
                    exit_reason="answer_detected",
                    usage=bindings.usage(),
                    native_chain=chain,
                    native_controller_events=to_jsonable(tracing.events),
                    native_participant_model_calls=result["usage"]["model_calls"],
                    native_model_classes={
                        k: type(v).__module__ + "." + type(v).__name__ for k, v in model_map.items()
                    },
                    controlled_runtime_events=copy.deepcopy(self.last_events),
                    native_configuration_sha256=content_hash(config),
                    unknown_token_usage_model_calls=bindings.unknown_usage_calls,
                    usage_is_complete=bindings.unknown_usage_calls == 0,
                    upstream_stdout_sha256=capture.hexdigest,
                )
                if self.fingerprint() != identity:
                    raise DriverError(
                        "BINDING_MISMATCH",
                        "native IRCoT-QA runtime identity changed",
                        pre_execution=False,
                    )
                return result
        except (Exception, SystemExit) as exc:
            usage = (
                bindings.usage()
                if bindings
                else dict(model_calls=0, retrieval_calls=0, input_tokens=0, output_tokens=0)
            )
            return dict(
                status="error",
                failure_code=getattr(exc, "code", "NATIVE_QA_EXECUTION_FAILED"),
                error_type=type(exc).__name__,
                message=getattr(exc, "public_message", "native IRCoT-QA execution failed"),
                consumed=bool(usage["model_calls"] or usage["retrieval_calls"]),
                usage=usage,
                controlled_runtime_events=copy.deepcopy(self.last_events),
                unknown_token_usage_model_calls=bindings.unknown_usage_calls if bindings else 0,
                usage_is_complete=not bindings or bindings.unknown_usage_calls == 0,
            )
        finally:
            self.last_usage = bindings.usage() if bindings else {}
            _NATIVE_LOCK.release()


class _QABindings(_LiveBindings):
    def __init__(self, owner, request, full_request):
        super().__init__(owner, request)
        self.full_request = full_request

    def generator_class(self, stage):
        binding = self
        owner = self.owner

        class Generator:
            def __init__(self, **kwargs):
                self.options = dict(kwargs)

            def generate_text_sequence(self, prompt):
                cap = min(owner.per_call_output_tokens, int(self.options.get("max_tokens", 300)))
                # Native default300 remains300 even though the common ceiling
                # is512. Reserve one native final call and its bounded output.
                reserved = min(owner.per_call_output_tokens, 300) if stage == "reasoning" else 0
                remaining = binding.request.controls.max_output_tokens - binding.output_tokens
                if (
                    len(binding.generations)
                    >= binding.full_request.max_model_calls - (1 if stage == "reasoning" else 0)
                    or remaining <= reserved
                ):
                    raise DriverError(
                        "MODEL_BUDGET_EXCEEDED", "full QA shared budget exhausted before dispatch"
                    )
                cap = min(cap, remaining - reserved)
                if any(
                    self.options.get(k, v) != v
                    for k, v in {
                        "n": 1,
                        "best_of": 1,
                        "top_p": 1,
                        "frequency_penalty": 0,
                        "presence_penalty": 0,
                        "logprobs": 0,
                    }.items()
                ):
                    raise DriverError(
                        "GENERATOR_OPTIONS_UNSUPPORTED",
                        "native full QA received unsupported sampling options",
                    )
                fitted = binding.fit_prompt(
                    original_prompt=prompt.rstrip(),
                    model_length_limit=8000,
                    estimated_generation_length=cap,
                    demonstration_delimiter="\n\n\n",
                    shuffle=False,
                    remove_method=self.options.get("remove_method", "first"),
                    tokenizer_model_name="gpt2",
                    last_is_test_example=True,
                )
                call = GenerationRequest(
                    prompt=fitted,
                    temperature=self.options.get("temperature", 0),
                    max_tokens=cap,
                    seed=owner.seed + len(binding.generations),
                    stop=tuple(self.options.get("stop", ["\n"])),
                    metadata={
                        "release_id": "acquire.ircot.native_like_substituted",
                        "stage": stage,
                        "native_generator_options": self.options,
                        "native_default_output_cap": 300,
                    },
                )
                event = dict(
                    operation="native_generator",
                    stage=stage,
                    request_hash=call.request_hash,
                    prompt_sha256=content_hash(fitted),
                    max_tokens=cap,
                    seed=call.seed,
                    stop=list(call.stop),
                    response_observed=False,
                )
                binding.generations.append(event)
                owner.last_events.append(event)
                response = owner.backend.generate(call)
                event.update(
                    response_observed=True,
                    response_text=response.text,
                    finish_reason=response.finish_reason,
                    usage=to_jsonable(response.usage),
                )
                binding.input_tokens += response.usage.input_tokens
                binding.output_tokens += response.usage.output_tokens
                if response.usage.output_tokens > cap:
                    raise DriverError(
                        "OUTPUT_BUDGET_EXCEEDED",
                        "provider exceeded native full QA output cap",
                        pre_execution=False,
                    )
                return [(response.text, 0.0)]

        return Generator


__all__ = [
    "QA_BINDING_MODE",
    "QA_PROFILE",
    "QA_ROLES",
    "PinnedIRCoTNativeQARuntime",
    "patch_native_qa_config",
]
