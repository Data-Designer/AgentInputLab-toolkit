"""Explicit file entrypoint for ACE's existing transductive online protocol.

Validation never generates, trains or downloads. Execution preserves stream
order, original controls, feedback provenance and every unsuccessful primary.
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path

from agentinputlab.configuration import _known, _object
from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.fit_configuration import FitConfig, _path, _reference, _save, _sha


class OnlineConfig(FitConfig):
    """A deliberate online stream, never an implicit fit or held-out evaluation."""

    def _ace_feedback(self):
        feedback = _object(self.value.get("feedback"), "online feedback")
        if feedback.get("kind") == "bfcl":
            _known(feedback, {"kind", "source_input", "model_id"}, "BFCL online feedback")
            _reference(feedback.get("source_input"), "feedback.source_input")
            _path(feedback.get("model_id"), "feedback.model_id")
            return feedback
        return super()._ace_feedback()

    def _controls(self):
        from agentinputlab.recipes.ace_online import ACEOnlineControls

        value = _object(self.value.get("controls"), "online controls")
        if set(value) != {f.name for f in fields(ACEOnlineControls)}:
            raise ValidationError("online controls must explicitly specify every native control")
        return ACEOnlineControls(**value)

    def validate(self):
        from agentinputlab.recipes.ace_online import audit_online_stream
        from workers.ace.driver import LLMConfig, WorkerFailure

        v = self.value
        _known(
            v,
            {
                "release_id",
                "checkout",
                "benchmark",
                "controls",
                "deployment",
                "stream",
                "feedback",
                "allow_test_feedback",
                "source_and_asset_acknowledged",
            },
            "online configuration",
        )
        if v.get("release_id") != "recipe.ace.online_transductive":
            raise ValidationError("online supports recipe.ace.online_transductive")
        _path(v.get("checkout"), "checkout")
        _path(v.get("benchmark"), "benchmark")
        if v["benchmark"].lower().startswith("tooltalk"):
            raise ValidationError("ToolTalk is test-only; online feedback is not allowed")
        if (
            v.get("allow_test_feedback") is not True
            or v.get("source_and_asset_acknowledged") is not True
        ):
            raise ValidationError(
                "online requires explicit transductive feedback and source acknowledgement"
            )
        controls = self._controls()
        try:
            deployment = LLMConfig.from_json(json.dumps(to_jsonable(v.get("deployment"))))
        except WorkerFailure:
            raise ValidationError(
                "online deployment must match the native three-role configuration"
            ) from None
        if (
            deployment.base_url is None
            or deployment.api_provider != "openai"
            or deployment.max_tokens_per_call != controls.per_call_output_tokens
        ):
            raise ValidationError(
                "online requires an OpenAI-compatible endpoint and matching per-call cap"
            )
        if ("stream" in v) == ("feedback" in v):
            raise ValidationError("choose an explicit stream or built-in feedback source, not both")
        if "stream" in v:
            stream = v["stream"]
            if not isinstance(stream, (list, tuple)):
                raise ValidationError("online stream must be an ordered array")
            if any(not isinstance(r, Mapping) or r.get("split") != "test" for r in stream):
                raise ValidationError(
                    "online stream must explicitly declare native test roles; no implicit relabeling"
                )
            audit_online_stream(stream, controls)
        else:
            feedback = self._ace_feedback()
            expected = {
                "statebench": "state_bench.common_support",
                "toolsandbox": "toolsandbox.common_support",
                "bfcl": "bfcl.v3.simple_python",
            }
            if (feedback["kind"] in expected and v["benchmark"] != expected[feedback["kind"]]) or (
                feedback["kind"] == "tau"
                and v["benchmark"] not in ("tau3.retail", "tau3.banking_knowledge")
            ):
                raise ValidationError("online benchmark must match its native feedback kind")
            if (
                feedback["kind"] == "bfcl"
                and feedback["model_id"] != v["deployment"]["roles"]["target"]["model"]
            ):
                raise ValidationError("BFCL online scorer model_id must match target deployment")
        return dict(
            validated=True,
            executed=False,
            release_id=v["release_id"],
            benchmark=v["benchmark"],
            transductive_not_offline_or_heldout=True,
            automatic_retry=False,
            model_calls=0,
            runtime_ready=None,
            controls=to_jsonable(v["controls"]),
        )

    def execute(self, output_dir):
        from agentinputlab.artifacts import load_pinned_json_mapping
        from agentinputlab.recipes.ace_online import ACEOnlineRecipe
        from workers.ace.driver import LLMConfig

        self.validate()
        v = self.value
        output = Path(output_dir).absolute()
        if output.exists() or output.is_symlink():
            raise ValidationError(
                "online output must be a new directory; consumed streams are not retried"
            )
        deployment = to_jsonable(v["deployment"])
        parsed = LLMConfig.from_json(json.dumps(deployment))
        if not os.environ.get(parsed.api_key_env, ""):
            raise ExternalDependencyError("online deployment API-key environment variable is unset")
        feedback = self._ace_feedback() if "feedback" in v else None
        if feedback is not None:
            pin = feedback["source_input"]
            source = load_pinned_json_mapping(
                pin["path"], expected_sha256=pin["sha256"], max_bytes=16 * 1024 * 1024
            )
            if source.get("benchmark") != v["benchmark"]:
                raise ValidationError("online feedback source benchmark differs from configuration")
            for name in self._feedback_environment_names(feedback, source):
                if not os.environ.get(name):
                    raise ExternalDependencyError(f"missing online environment variable: {name}")
        output.mkdir(parents=True)
        identity = dict(
            release_id=v["release_id"],
            benchmark=v["benchmark"],
            configuration_sha256=content_hash(v),
            automatic_retry=False,
            transductive_not_offline_or_heldout=True,
        )
        _save(output / "started.json", dict(identity, status="started"))
        processor = None
        try:
            if feedback is None:
                stream = v["stream"]
            else:
                kind = feedback["kind"]
                if kind in ("bfcl", "retrieval"):
                    from agentinputlab.recipes.ace_online_files import create_online_feedback

                    processor = create_online_feedback(
                        kind=kind,
                        source_input=feedback["source_input"],
                        benchmark=v["benchmark"],
                        model_id=feedback.get("model_id"),
                    )
                else:
                    module = importlib.import_module(f"agentinputlab.recipes.ace_{kind}")
                    config = {
                        k: to_jsonable(value)
                        for k, value in feedback.items()
                        if k not in ("kind", "environment_names")
                    }
                    processor = getattr(module, f"create_{kind}_feedback")(
                        **config, mode="online", output_root=str(output / "feedback")
                    )
                stream = processor.stream_examples()
                processor.validate_online_examples(stream)
            result = ACEOnlineRecipe(self._controls()).run_stream(
                stream,
                checkout=v["checkout"],
                llm_config=deployment,
                output_root=output / "native",
                allow_test_feedback=True,
                source_and_asset_acknowledged=True,
                data_processor=processor,
            )
            terminal = result.terminal
            summary = dict(
                identity,
                status=terminal["status"],
                primary_outcomes=to_jsonable(terminal["primary_outcomes"]),
                per_case_usage=to_jsonable(terminal["per_case_usage"]),
                usage=to_jsonable(terminal["usage"]),
                native_terminal=dict(
                    path=str(output / "native/terminal.json"),
                    sha256=_sha(output / "native/terminal.json"),
                ),
                native_manifest=dict(
                    path=str(output / "native/manifest.json"),
                    sha256=_sha(output / "native/manifest.json"),
                ),
                primary_native_feedback=[],
            )
            if processor is not None:
                if hasattr(processor, "retrieval_attempts"):
                    summary["retrieval_attempts"] = to_jsonable(processor.retrieval_attempts)
                    summary["retrieval_limit_per_case"] = processor.retrieval_limit_per_case
                events = to_jsonable(processor.events)
                _save(output / "feedback-events.json", events)
                summary["feedback_events"] = dict(
                    path=str(output / "feedback-events.json"),
                    sha256=_sha(output / "feedback-events.json"),
                )
                # Link only the original pre-update primary, never a later
                # adapted answer for the same case or a success-shaped wrapper.
                for primary in terminal["primary_outcomes"]:
                    prediction = primary.get("native_prediction")
                    if not isinstance(prediction, Mapping):
                        continue
                    answer = prediction.get("final_answer")
                    matches = [
                        (i, e)
                        for i, e in enumerate(events)
                        if e.get("target") == primary["case_id"]
                        and e.get("prediction_sha256") == content_hash(answer)
                    ]
                    if matches:
                        index, event = matches[0]
                        summary["primary_native_feedback"].append(
                            dict(
                                case_id=primary["case_id"],
                                prediction_sha256=content_hash(answer),
                                feedback_event_index=index,
                                result=event.get("result"),
                                metrics=event.get("metrics"),
                                score=event.get("score"),
                                status=event.get("status"),
                                native_task_score_NOT_claimed=event.get(
                                    "native_task_score_NOT_claimed", False
                                ),
                            )
                        )
            _save(output / "terminal.json", summary)
            return summary
        except Exception as exc:
            _save(
                output / "terminal.json",
                dict(
                    identity,
                    status="online_execution_failed",
                    error_type=type(exc).__name__,
                    usage_may_have_been_consumed=True,
                    native_error_sha256=content_hash(str(exc)),
                ),
            )
            raise ExternalDependencyError(
                f"online execution failed; inspect saved evidence in {output}; no automatic retry"
            ) from None
