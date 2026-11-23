from __future__ import annotations

from dataclasses import dataclass
from unittest import SkipTest

try:
    import pytest
except ModuleNotFoundError as error:
    if error.name != "pytest":
        raise
    raise SkipTest("pytest is required for this test module") from error

from agentinputlab.algorithms.acquire import HyDECleanRoom, HyDEConfig, HyDEControlledBindings
from agentinputlab.algorithms.acquire.ircot import (
    IRCOT_UPSTREAM_COMMIT,
    IRCoTControlledWorker,
)
from agentinputlab.algorithms.clarify import (
    ActiveDisambiguationControls,
    ActiveTaskDisambiguation,
    FutureTurnClarifier,
    FutureTurnControls,
)
from agentinputlab.algorithms.decompose.plan_and_solve import PlanAndSolvePSPrompt
from agentinputlab.algorithms.reformulate.query2doc import (
    QUERYGYM_REVISION,
    Query2DocLiveBindings,
    QueryGymRawResult,
)
from agentinputlab.backends.llm import ReplayBackend
from agentinputlab.backends.retrieval import Document, InMemoryBM25Retriever, RetrievalResult
from agentinputlab.contracts import FrozenAgentState, InterventionContext
from agentinputlab.contracts.enums import ProfileKind, SemanticFamily
from agentinputlab.contracts.errors import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.experiments.builders import RegistryInterventionBuilder
from agentinputlab.experiments.drivers.static_production import (
    StaticArmBuildRequest,
    StaticRuntimeComponents,
)
from agentinputlab.integrations.external_workers.ace import (
    ACE_UPSTREAM_COMMIT,
    ACEPlaybook,
    PlaybookBullet,
)
from agentinputlab.recipes import (
    SKILLOPT_PROTOCOL,
    SKILLOPT_UPSTREAM_COMMIT,
    ACEPlaybookCommonSupport,
    MCETextArtifactCommonSupport,
    RQRAGCommonSupportQueries,
    SkillOptSkillCommonSupport,
)
from agentinputlab.runner import InterventionRunner
from agentinputlab.selectors import NoOpIntervention


@dataclass
class _Runtime:
    identity: str

    def fingerprint(self):
        return {"identity": self.identity}


class _Embedder:
    embedder_id = "fixture-embedder"

    def embed(self, texts):
        return tuple((float(index + 1),) for index, _ in enumerate(texts))

    def fingerprint(self):
        return {"embedder_id": self.embedder_id, "revision": "fixture-v1"}


class _Searcher:
    searcher_id = "fixture-index"

    def search(self, vector, *, top_k):
        del vector
        return (
            RetrievalResult(
                document=Document(
                    doc_id="doc-1",
                    text="Paris is the capital of France.",
                    corpus_revision="fixture-v1",
                ),
                score=1.0,
                rank=1,
            ),
        )[:top_k]

    def fingerprint(self):
        return {"searcher_id": self.searcher_id, "revision": "fixture-v1"}


class _IRCoTWorker:
    worker_id = "registry-builder-ircot-fixture"

    def call(self, payload):
        return {
            "status": "ok",
            "request_id": payload["request_id"],
            "protocol_version": "agentinputlab.ircot.v1",
            "upstream_commit": IRCOT_UPSTREAM_COMMIT,
            "initial_query": payload["question"],
            "initial_document_ids": ["d1"],
            "steps": [],
            "documents": [
                {
                    "document_id": "d1",
                    "text": "Locked fixture evidence.",
                    "title": "Fixture",
                    "corpus_revision": "corpus-v1",
                    "score": 1.0,
                }
            ],
            "active_document_ids": ["d1"],
            "exit_reason": "controller",
            "answer": "fixture",
            "usage": {
                "retrieval_calls": 1,
                "model_calls": 1,
                "input_tokens": 4,
                "output_tokens": 2,
            },
        }

    def fingerprint(self):
        return {
            "worker_id": self.worker_id,
            "image_digest": "sha256:registry-builder-fixture",
        }


def _request(
    *,
    release_id: str,
    family: SemanticFamily,
    parameters,
    profile_kind: ProfileKind = ProfileKind.CONTROLLED_ADAPTER,
    arm_parameters=None,
    runtime=None,
) -> StaticArmBuildRequest:
    return StaticArmBuildRequest(
        arm_id="candidate",
        release_id=release_id,
        family=family,
        profile_kind=profile_kind,
        model_slot="anchor",
        seed=17,
        builder_parameters=parameters,
        arm_parameters=arm_parameters or {},
        runtime=runtime or StaticRuntimeComponents(backend=_Runtime("backend")),
    )


def test_builds_no_op_and_simple_registered_release():
    builder = RegistryInterventionBuilder()
    assert builder.fingerprint()["no_op_profile_policy"] == "inherit_requested_profile.v1"
    no_op = builder.build(
        _request(
            release_id="baseline.no_op",
            family=SemanticFamily.NO_OP,
            parameters={"schema_version": 1, "release_id": "baseline.no_op"},
        )
    )
    assert isinstance(no_op, NoOpIntervention)
    assert no_op.profile_kind == ProfileKind.CONTROLLED_ADAPTER.value

    official_no_op = builder.build(
        _request(
            release_id="baseline.no_op",
            family=SemanticFamily.NO_OP,
            parameters={"schema_version": 1, "release_id": "baseline.no_op"},
            profile_kind=ProfileKind.OFFICIAL_NATIVE,
        )
    )
    assert official_no_op.profile_kind == ProfileKind.OFFICIAL_NATIVE.value

    plan = builder.build(
        _request(
            release_id="plan.plan_and_solve.ps_prompt",
            family=SemanticFamily.DECOMPOSE,
            parameters={
                "schema_version": 1,
                "release_id": "plan.plan_and_solve.ps_prompt",
            },
        )
    )
    assert isinstance(plan, PlanAndSolvePSPrompt)
    assert plan.release_id == "plan.plan_and_solve.ps_prompt"


@pytest.mark.parametrize(
    ("release_id", "entrypoint", "kwargs", "expected_type", "controls_type"),
    (
        (
            "clarify.future_turns.future_outcome_match",
            "agentinputlab.algorithms.clarify.future_turns:FutureTurnControls",
            {
                "max_interpretations": 4,
                "max_candidates": 2,
                "max_tokens": 256,
                "temperature": 0.0,
            },
            FutureTurnClarifier,
            FutureTurnControls,
        ),
        (
            "clarify.active_task_disambiguation.eig_uniform",
            (
                "agentinputlab.algorithms.clarify.active_task_disambiguation:"
                "ActiveDisambiguationControls"
            ),
            {
                "total_hypotheses": 4,
                "total_questions": 2,
                "max_tokens": 256,
                "temperature": 0.5,
            },
            ActiveTaskDisambiguation,
            ActiveDisambiguationControls,
        ),
    ),
)
def test_builds_new_clarification_releases_with_typed_controls(
    release_id,
    entrypoint,
    kwargs,
    expected_type,
    controls_type,
):
    parameters = {
        "schema_version": 1,
        "release_id": release_id,
        "constructor_kwargs": {
            "controls": {
                "$type": "dataclass",
                "entrypoint": entrypoint,
                "kwargs": kwargs,
            }
        },
    }
    builder = RegistryInterventionBuilder()
    assert builder.preflight(parameters) == ()
    intervention = builder.build(
        _request(
            release_id=release_id,
            family=SemanticFamily.CLARIFY,
            parameters=parameters,
        )
    )
    assert isinstance(intervention, expected_type)
    assert isinstance(intervention.controls, controls_type)


def test_builds_ace_eval_only_from_a_frozen_typed_artifact():
    bullet = {
        "$type": "dataclass",
        "entrypoint": "agentinputlab.integrations.external_workers.ace:PlaybookBullet",
        "kwargs": {
            "bullet_id": "ctx-00001",
            "content": "Prefer explicit visible evidence.",
        },
    }
    parameters = {
        "schema_version": 1,
        "release_id": "recipe.ace.eval_only",
        "constructor_kwargs": {
            "artifact": {
                "$type": "dataclass",
                "entrypoint": "agentinputlab.recipes.ace:ACEContextArtifact",
                "kwargs": {
                    "artifact_id": "a" * 64,
                    "release_id": "recipe.ace.offline",
                    "playbook": {
                        "$type": "dataclass",
                        "entrypoint": (
                            "agentinputlab.integrations.external_workers.ace:ACEPlaybook"
                        ),
                        "kwargs": {
                            "bullets": {"$type": "tuple", "items": [bullet]},
                        },
                    },
                    "selected_iteration": 0,
                    "split_audit_hash": "b" * 64,
                    "upstream_commit": ACE_UPSTREAM_COMMIT,
                    "worker_fingerprint": {"worker_id": "fixture"},
                    "model_fingerprints": {
                        role: {"model": "fixture"} for role in ("target", "reflector", "curator")
                    },
                    "controls": {"seed": 17},
                    "usage": {"model_calls": 5, "tokens": 20},
                    "iteration_hashes": {"$type": "tuple", "items": ["c" * 64]},
                },
            }
        },
    }
    artifact_values = parameters["constructor_kwargs"]["artifact"]["kwargs"]
    identity = {
        k: v for k, v in artifact_values.items() if k not in {"artifact_id", "playbook", "usage"}
    }
    identity["playbook_hash"] = ACEPlaybook((PlaybookBullet(**bullet["kwargs"]),)).playbook_hash
    identity["iteration_hashes"] = ("c" * 64,)
    artifact_values["artifact_id"] = content_hash(identity)
    builder = RegistryInterventionBuilder()
    assert builder.preflight(parameters) == ()
    intervention = builder.build(
        _request(
            release_id="recipe.ace.eval_only",
            family=SemanticFamily.CURATE,
            parameters=parameters,
        )
    )
    assert isinstance(intervention, ACEPlaybookCommonSupport)


def test_builds_rqrag_common_support_with_frozen_controls():
    parameters = {
        "schema_version": 1,
        "release_id": "recipe.rqrag.common_support_queries",
        "constructor_kwargs": {
            "controls": {
                "$type": "dataclass",
                "entrypoint": "agentinputlab.recipes.rqrag:RQRAGMethodControls",
                "kwargs": {
                    "max_depth": 1,
                    "max_nodes": 8,
                    "max_documents": 15,
                    "retrieval_top_k": 5,
                    "max_output_tokens_per_call": 64,
                    "query_temperature": 1.0,
                    "answer_temperature": 0.0,
                    "selection_mode": "confidence",
                    "allow_partial_tree": False,
                },
            }
        },
    }
    builder = RegistryInterventionBuilder()
    assert builder.preflight(parameters) == ()
    intervention = builder.build(
        _request(
            release_id="recipe.rqrag.common_support_queries",
            family=SemanticFamily.REFORMULATE,
            parameters=parameters,
        )
    )
    assert isinstance(intervention, RQRAGCommonSupportQueries)
    assert intervention.controls.max_nodes == 8
    assert intervention.controls.retrieval_top_k == 5


def test_builds_skillopt_eval_only_from_frozen_typed_artifact():
    skill = "Use visible evidence."
    import hashlib

    skill_sha256 = hashlib.sha256(skill.encode()).hexdigest()
    parameters = {
        "schema_version": 1,
        "release_id": "recipe.skillopt.eval_only",
        "constructor_kwargs": {
            "artifact": {
                "$type": "dataclass",
                "entrypoint": "agentinputlab.recipes.skillopt:SkillOptArtifact",
                "kwargs": {
                    "artifact_id": "skillopt-fixture",
                    "release_id": "recipe.skillopt.offline",
                    "best_skill": skill,
                    "best_skill_sha256": skill_sha256,
                    "upstream_commit": SKILLOPT_UPSTREAM_COMMIT,
                    "protocol": SKILLOPT_PROTOCOL,
                    "config_sha256": "a" * 64,
                    "source_hashes": {"source.py": "b" * 64},
                    "output_hashes": {"best_skill.md": skill_sha256},
                    "controls": {"seed": 17},
                    "model_fingerprints": {
                        "target": {"model": "fixture"},
                        "optimizer": {"model": "fixture"},
                    },
                    "environment_keys": {"$type": "tuple", "items": []},
                    "command_sha256": "c" * 64,
                },
            }
        },
    }
    artifact_values = parameters["constructor_kwargs"]["artifact"]["kwargs"]
    identity = {k: v for k, v in artifact_values.items() if k not in {"artifact_id", "best_skill"}}
    identity["environment_keys"] = ()
    artifact_values["artifact_id"] = content_hash(identity)
    builder = RegistryInterventionBuilder()
    assert builder.preflight(parameters) == ()
    intervention = builder.build(
        _request(
            release_id="recipe.skillopt.eval_only",
            family=SemanticFamily.CURATE,
            parameters=parameters,
        )
    )
    assert isinstance(intervention, SkillOptSkillCommonSupport)


def test_builds_mce_text_artifact_from_frozen_typed_artifact():
    from agentinputlab.recipes import MCETextArtifact

    artifact = MCETextArtifact.build(
        selected_iteration=1,
        skill_markdown="## Skill Overview\nUse visible evidence.",
        context_files={"context/guidance.md": "Be precise."},
        source_hashes={"mce/main.py": "a" * 64},
        iteration_scores=(0.1, 0.2),
        discovery_split_sha256="b" * 64,
        model_fingerprint={"model": "fixture"},
        training_usage={"model_calls": 5},
    )
    kwargs = artifact.as_dict()
    kwargs["iteration_scores"] = {"$type": "tuple", "items": [0.1, 0.2]}
    parameters = {
        "schema_version": 1,
        "release_id": "recipe.mce.text_artifact_eval",
        "constructor_kwargs": {
            "artifact": {
                "$type": "dataclass",
                "entrypoint": "agentinputlab.recipes.mce_text_only:MCETextArtifact",
                "kwargs": kwargs,
            }
        },
    }
    builder = RegistryInterventionBuilder()
    assert builder.preflight(parameters) == ()
    intervention = builder.build(
        _request(
            release_id="recipe.mce.text_artifact_eval",
            family=SemanticFamily.CURATE,
            parameters=parameters,
        )
    )
    assert isinstance(intervention, MCETextArtifactCommonSupport)


def test_uses_context_runtime_and_materializes_typed_config_end_to_end():
    backend = ReplayBackend(["hypothesis one", "hypothesis two"])
    embedder = _Embedder()
    searcher = _Searcher()
    runtime = StaticRuntimeComponents(
        backend=backend,
        dense_embedder=embedder,
        vector_searcher=searcher,
    )
    parameters = {
        "schema_version": 1,
        "release_id": "acquire.hyde.paper_reimplementation",
        "constructor_kwargs": {
            "config": {
                "$type": "dataclass",
                "entrypoint": "agentinputlab.algorithms.acquire:HyDEConfig",
                "kwargs": {
                    "num_hypotheses": 2,
                    "max_tokens": 64,
                    "temperature": 0.0,
                    "top_k": 3,
                    "prompt_template": "Question: {query}",
                },
            }
        },
    }
    assert RegistryInterventionBuilder().preflight(parameters) == ()
    result = RegistryInterventionBuilder().build(
        _request(
            release_id="acquire.hyde.paper_reimplementation",
            family=SemanticFamily.ACQUIRE,
            parameters=parameters,
            runtime=runtime,
        )
    )
    assert isinstance(result, HyDECleanRoom)
    assert result.generator is None
    assert result.embedder is None
    assert result.searcher is None
    assert result.config.num_hypotheses == 2
    record = InterventionRunner().run(
        result,
        FrozenAgentState(query="What is the capital of France?"),
        InterventionContext(
            backend=backend,
            dense_embedder=embedder,
            vector_searcher=searcher,
            seed=17,
        ),
    )
    assert record.output_state_hash != record.input_state_hash


def test_hyde_controlled_builder_accepts_data_only_binding_recipe():
    state = FrozenAgentState(query="What is the capital of France?")
    backend = ReplayBackend(["A passage about Paris."])
    context = InterventionContext(
        backend=backend, dense_embedder=_Embedder(), vector_searcher=_Searcher(), seed=17
    )
    bindings = HyDEControlledBindings.from_context(
        context, budget=state.budget, config=HyDEConfig(num_hypotheses=1, top_k=1)
    )
    parameters = {
        "schema_version": 1,
        "release_id": "acquire.hyde.controlled",
        "constructor_kwargs": {"bindings": to_jsonable(bindings)},
    }
    builder = RegistryInterventionBuilder()
    assert builder.preflight(parameters) == ()
    method = builder.build(
        _request(
            release_id="acquire.hyde.controlled",
            family=SemanticFamily.ACQUIRE,
            parameters=parameters,
            runtime=StaticRuntimeComponents(
                backend=backend,
                dense_embedder=context.dense_embedder,
                vector_searcher=context.vector_searcher,
            ),
        )
    )
    result = InterventionRunner().run(method, state, context)
    assert result.patch.release_id == "acquire.hyde.controlled"
    assert result.usage.model_calls == 1


def test_query2doc_controlled_builder_uses_frozen_data_and_context_runtime():
    state = FrozenAgentState(query="Fixed query")
    backend = ReplayBackend(["A synthetic passage."])

    class Runtime:
        def fingerprint(self):
            return {"revision": QUERYGYM_REVISION, "source_closure_sha256": "a"*64,
                    "backend": backend.fingerprint(), "fixture": True}

        def reformulate(self, request):
            from agentinputlab.backends.llm import GenerationRequest
            response = backend.generate(GenerationRequest(prompt=request.query))
            return QueryGymRawResult(reformulated=request.query+" "+response.text,
                generated_text=response.text, usage={"model_calls": 1, "retrieval_calls": 0,
                "input_tokens": 0, "output_tokens": 0})

    retriever = InMemoryBM25Retriever([Document("doc", "Corpus fixture.", corpus_revision="fixture")])
    context = InterventionContext(backend=backend, retriever=retriever, external_runtime=Runtime())
    bindings = Query2DocLiveBindings.from_context(context, budget=state.budget)
    parameters = {"schema_version": 1, "release_id": "rewrite.query2doc.controlled_live",
                  "constructor_kwargs": {"bindings": to_jsonable(bindings)}}
    builder = RegistryInterventionBuilder()
    assert builder.preflight(parameters) == ()
    method = builder.build(_request(release_id="rewrite.query2doc.controlled_live",
        family=SemanticFamily.REFORMULATE, parameters=parameters,
        runtime=StaticRuntimeComponents(backend=backend, retriever=retriever,
                                        external_runtime=context.external_runtime)))
    result = InterventionRunner().run(method, state, context)
    assert result.patch.release_id == "rewrite.query2doc.controlled_live"
    assert result.usage.model_calls == 1


def test_ircot_builder_keeps_worker_in_context_and_runs_end_to_end():
    parameters = {
        "schema_version": 1,
        "release_id": "acquire.ircot.controlled",
        "constructor_kwargs": {
            "corpus_revision": "corpus-v1",
            "index_revision": "index-v1",
            "prompt_hash": "prompt-v1",
        },
    }
    builder = RegistryInterventionBuilder()
    assert builder.preflight(parameters) == ()
    intervention = builder.build(
        _request(
            release_id="acquire.ircot.controlled",
            family=SemanticFamily.ACQUIRE,
            parameters=parameters,
            runtime=StaticRuntimeComponents(
                backend=_Runtime("backend"),
                retriever=_Runtime("retriever"),
                external_runtime=_IRCoTWorker(),
            ),
        )
    )
    assert isinstance(intervention, IRCoTControlledWorker)
    assert intervention.client is None
    record = InterventionRunner().run(
        intervention,
        FrozenAgentState(query="Find the supporting fact.", task_id="fixture"),
        InterventionContext(
            backend=_Runtime("backend"),
            retriever=_Runtime("retriever"),
            external_runtime=_IRCoTWorker(),
            seed=17,
        ),
    )
    assert (
        record.patch.apply(FrozenAgentState(query="Find the supporting fact.", task_id="fixture"))
        .context[0]
        .block_id
        == "d1"
    )


def test_arm_parameter_binding_is_explicit_and_required():
    parameters = {
        "schema_version": 1,
        "release_id": "clarify.udcs.repo_native_prompt_route",
        "arm_parameter_bindings": {"clarification_threshold": "clarification_threshold"},
    }
    request = _request(
        release_id="clarify.udcs.repo_native_prompt_route",
        family=SemanticFamily.CLARIFY,
        parameters=parameters,
        arm_parameters={"clarification_threshold": 0.75},
    )
    result = RegistryInterventionBuilder().build(request)
    assert result.clarification_threshold == 0.75

    with pytest.raises(ArtifactError, match="arm parameter"):
        RegistryInterventionBuilder().build(
            _request(
                release_id="clarify.udcs.repo_native_prompt_route",
                family=SemanticFamily.CLARIFY,
                parameters=parameters,
            )
        )


def test_composite_release_is_routed_out_of_static_builder():
    parameters = {
        "schema_version": 1,
        "release_id": "plan.decomp.full",
        "constructor_kwargs": {"config_artifact_id": "locked"},
    }
    blockers = RegistryInterventionBuilder().preflight(parameters)
    assert blockers
    assert "composite/offline" in blockers[0]


def test_rejects_runtime_bindings_because_the_runner_requires_context_injection():
    blockers = RegistryInterventionBuilder().preflight(
        {
            "schema_version": 1,
            "release_id": "acquire.hyde.paper_reimplementation",
            "runtime_bindings": {"generator": "backend"},
        }
    )
    assert len(blockers) == 1
    assert "InterventionContext" in blockers[0]


def test_typed_dataclass_preflight_reports_missing_required_arguments():
    blockers = RegistryInterventionBuilder().preflight(
        {
            "schema_version": 1,
            "release_id": "plan.self_ask.full_loop",
            "constructor_kwargs": {
                "config": {
                    "$type": "dataclass",
                    "entrypoint": (
                        "agentinputlab.algorithms.decompose.self_ask.loop:SelfAskConfig"
                    ),
                    "kwargs": {},
                }
            },
        }
    )
    assert len(blockers) == 1
    assert "missing required arguments" in blockers[0]


def test_release_and_identity_mismatches_fail_closed():
    with pytest.raises(ValidationError, match="release differs"):
        RegistryInterventionBuilder().build(
            _request(
                release_id="plan.plan_and_solve.ps_prompt",
                family=SemanticFamily.DECOMPOSE,
                parameters={
                    "schema_version": 1,
                    "release_id": "acquire.hyde.paper_reimplementation",
                },
            )
        )

    blockers = RegistryInterventionBuilder().preflight(
        {
            "schema_version": 1,
            "release_id": "plan.plan_and_solve.ps_prompt",
            "api_key": "must-not-be-accepted",
        }
    )
    assert len(blockers) == 1
    assert "unknown fields" in blockers[0]


def test_required_environment_is_preflight_only(monkeypatch):
    parameters = {
        "schema_version": 1,
        "release_id": "plan.plan_and_solve.ps_prompt",
        "required_environment": ["AILAB_LOCKED_TEST_ASSET"],
    }
    monkeypatch.delenv("AILAB_LOCKED_TEST_ASSET", raising=False)
    assert RegistryInterventionBuilder().preflight(parameters) == (
        "missing required environment AILAB_LOCKED_TEST_ASSET",
    )
    monkeypatch.setenv("AILAB_LOCKED_TEST_ASSET", "/tmp/asset")
    assert RegistryInterventionBuilder().preflight(parameters) == ()
