"""Real four-reader inference using source-bound trained-router captures."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from prepare_adaptive_trained_acceptance import sha
from run_adaptive_controlled_acceptance import (
    PROMPTS,
    RecordedDriver,
    backend_for,
    state_for,
)
from run_clariq_acceptance import verify_services
from run_static_acceptance import digest, save

from agentinputlab import InterventionContext
from agentinputlab.algorithms.acquire.adaptive_rag import (
    AdaptiveRAGArtifactIdentity,
    AdaptiveRAGControlledWorker,
    AdaptiveRAGControls,
)
from agentinputlab.backends import Document, InMemoryBM25Retriever
from agentinputlab.benchmarks import load_production_benchmark_adapter
from agentinputlab.benchmarks.qa import QAPrediction
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from workers.adaptive_rag.driver import (
    AdaptiveRAGWorkerDriver,
    RouterResult,
    RuntimeBundle,
    RuntimeIdentity,
    _path_closure_sha256,
)
from workers.adaptive_rag.runtime import OpenAICompatibleGenerator, ToolkitRetriever

PIPELINE = "acquire.adaptive_rag.rebuilt_offline"
INFERENCE = "acquire.adaptive_rag.controlled_reimplementation"


def capture_identity(row):
    value = dict(row)
    identity = value.pop("capture_id", None)
    if identity != content_hash(value):
        raise ValueError("trained router capture identity differs")
    return identity


class StrictCapturedRouter:
    """Replay only the physically observed input/seed/artifact-specific route.

    This is declared reuse of an independent helper, not a synthetic model.
    Every treatment still charges one logical helper call/output token; actual
    eight physical helper attempts are counted once in the preparation receipt.
    """

    def __init__(self, capture, identity):
        capture_identity(capture)
        if any(
            getattr(identity, key) != value for key, value in capture["router_bindings"].items()
        ):
            raise ValueError("router capture snapshot/weights/tokenizer identity differs")
        self.capture, self.identity, self.logical_calls = capture, identity, []

    def component_manifest(self):
        return dict(
            component_type="router",
            fixture_runtime=False,
            implementation_class=f"{type(self).__module__}.{type(self).__qualname__}",
            revision=self.identity.router_revision,
            snapshot_sha256=self.identity.router_snapshot_sha256,
            weights_sha256=self.identity.router_weights_sha256,
            tokenizer_sha256=self.identity.router_tokenizer_sha256,
        )

    def _check(self, question):
        if (
            question != self.capture["case"]["question"]
            or content_hash(question) != self.capture["case"]["question_sha256"]
        ):
            raise ValueError("router capture query differs; no permissive fallback")

    def estimate_input_tokens(self, question):
        self._check(question)
        return self.capture.get("result", {}).get("input_tokens", 0)

    def first_decoder_step(self, question, *, seed):
        self._check(question)
        if seed != self.capture["seed"] or seed != 17 or self.logical_calls:
            raise ValueError("router capture seed or once-only use differs")
        self.logical_calls.append(
            dict(capture_id=self.capture["capture_id"], seed=seed, new_physical_router_calls=0)
        )
        if self.capture["status"] != "native_router_returned":
            raise RuntimeError("preserved failed native router capture; no retry/substitution")
        return RouterResult(**self.capture["result"])


def run(config, output):
    import agentinputlab.backends.retrieval.memory as bm25_module
    import workers.adaptive_rag.driver as driver_module
    import workers.adaptive_rag.runtime as runtime_module

    if output.exists() or len(config["models"]) != 1:
        raise ValueError("one model/create-only output required; no consumed-case retries")
    prep = Path(config["preparation_root"])
    complete_path = prep / "complete.json"
    if sha(complete_path) != config["preparation_complete_sha256"]:
        raise ValueError("trained router preparation closure differs")
    complete = json.loads(complete_path.read_text())
    manifest = json.loads((prep / "manifest.json").read_text())
    if (
        sha(prep / "manifest.json") != complete["manifest_sha256"]
        or complete["actual_physical_router_attempts"] != 8
        or len(manifest["cases"]) != 8
        or complete["target_calls"] != 0
    ):
        raise ValueError("requires complete source-bound eight-case router preparation")
    captures = {}
    for spec in complete["captures"]:
        path = Path(spec["path"])
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(prep.resolve())
            or sha(path) != spec["sha256"]
        ):
            raise ValueError("router capture path/hash differs")
        row = json.loads(path.read_text())
        if (
            capture_identity(row) != spec["capture_id"]
            or row["trained_artifact_id"] != manifest["trained_artifact_id"]
            or row["router_bindings"] != manifest["router_bindings"]
            or row["router_configuration"] != manifest["router_configuration"]
        ):
            raise ValueError("router capture trained artifact differs")
        captures[row["case"]["benchmark"], row["case"]["case_id"]] = row
    if len(captures) != 8:
        raise ValueError("requires unique frozen helper cases")
    model = config["models"][0]
    binding = verify_services([model])[model["id"]]
    adapters, examples = {}, {}
    for case in manifest["cases"]:
        cell = case["benchmark"]
        if cell not in adapters:
            adapters[cell] = load_production_benchmark_adapter(cell, config["prepared_root"])
        adapter = adapters[cell]
        if dict(adapter.prepared.file_sha256) != manifest["datasets"][cell]:
            raise ValueError("prepared QA source files differ")
        ex = sorted(adapter.examples(case["split"]), key=lambda x: x.example_id)[
            case["source_index"]
        ]
        if (
            ex.example_id != case["case_id"]
            or ex.visible_state().state_hash != case["visible_state_hash"]
            or ex.visible_state().query != case["question"]
        ):
            raise ValueError("frozen source case/visible state differs")
        examples[cell, ex.example_id] = ex
    output.mkdir()
    for name in ("claims", "terminals", "deployment", "prompts"):
        (output / name).mkdir()
    prompt_paths = {}
    for name, text in PROMPTS.items():
        p = output / "prompts" / (name + ".txt")
        with p.open("x") as stream:
            stream.write(text)
        prompt_paths[name] = p
    router_root = Path(config["trained_artifact"]) / "checkpoint"
    common = dict(
        manifest["router_bindings"],
        retriever_revision="toolkit.in_memory_bm25.v1",
        retriever_implementation_sha256=_path_closure_sha256(Path(bm25_module.__file__)),
    )
    save(
        output / "manifest.json",
        dict(
            pipeline_release=PIPELINE,
            inference_release=INFERENCE,
            model=model,
            model_binding=binding,
            preparation_complete_sha256=sha(complete_path),
            trained_artifact_id=manifest["trained_artifact_id"],
            cases=manifest["cases"],
            router_configuration=manifest["router_configuration"],
            expected_positions=8,
            per_case_logical_calls=8,
            per_case_output_tokens=4096,
            per_generator_cap=512,
            seed=17,
            helper_physical_calls_already_in_preparation=8,
            new_router_calls_in_this_lane=0,
            exact_input_seed_artifact_reuse=True,
            native_training_not_repeated=True,
            controlled_arms_not_upstream_arm_parity=True,
            operator_sha256=sha(Path(__file__)),
            runtime_sha256=sha(Path(runtime_module.__file__)),
            publication_ready=False,
            automatic_retry=False,
        ),
    )
    for case in manifest["cases"]:
        cell, case_id = case["benchmark"], case["case_id"]
        example = examples[cell, case_id]
        visible = example.visible_state()
        key = digest(
            dict(
                model=model["id"],
                cell=cell,
                case_id=case_id,
                trained_artifact=manifest["trained_artifact_id"],
                release=PIPELINE,
            )
        )
        save(
            output / "claims" / (key + ".json"),
            dict(
                case=case,
                model=model,
                trained_artifact_id=manifest["trained_artifact_id"],
                job_id=key,
                scientific_attempt_once=True,
            ),
        )
        capture = captures[cell, case_id]
        calls, row = (
            [],
            dict(
                job_id=key,
                pipeline_release=PIPELINE,
                inference_release=INFERENCE,
                model=model,
                benchmark=cell,
                case_id=case_id,
                case=case,
                capture_id=capture["capture_id"],
                capture_status=capture["status"],
                trained_artifact_id=manifest["trained_artifact_id"],
            ),
        )
        try:
            documents = tuple(
                Document(
                    block.block_id or f"p{i}",
                    block.text,
                    title=str(block.metadata.get("title", "")),
                    corpus_revision=content_hash(visible.context),
                )
                for i, block in enumerate(visible.context)
            )
            if not documents or len({d.doc_id for d in documents}) != len(documents):
                raise ValueError("requires unambiguous original visible paragraphs")
            retriever = InMemoryBM25Retriever(documents)
            prefix = output / "deployment" / key
            corpus, index, model_file, gen_file, ret_file = (
                prefix.with_suffix(suffix)
                for suffix in (
                    ".corpus.json",
                    ".index.json",
                    ".model.json",
                    ".generator.json",
                    ".retriever.json",
                )
            )
            save(
                corpus,
                dict(
                    documents=documents,
                    visible_state_hash=visible.state_hash,
                    selection="all_original_visible_paragraphs_no_gold_enrichment",
                ),
            )
            save(index, retriever.fingerprint())
            save(model_file, binding)
            save(
                gen_file,
                dict(
                    model=model,
                    max_output_tokens=512,
                    temperature=0,
                    seed=17,
                    router_configuration=manifest["router_configuration"],
                    input_token_upper_bound=65536,
                    final_text_only=True,
                    implicit_retry=False,
                    runtime_sha256=sha(Path(runtime_module.__file__)),
                ),
            )
            save(ret_file, dict(retriever=retriever.fingerprint(), top_k_controlled_by_driver=True))
            identity = RuntimeIdentity(
                **common,
                base_model_revision=digest(binding["model_json_hashes"]),
                generator_artifact_sha256=sha(model_file),
                generator_config_sha256=sha(gen_file),
                retriever_config_sha256=sha(ret_file),
                corpus_revision=content_hash(visible.context),
                corpus_manifest_sha256=sha(corpus),
                index_revision=content_hash(retriever.fingerprint()),
                index_manifest_sha256=sha(index),
            )
            router = StrictCapturedRouter(capture, identity)
            generator = OpenAICompatibleGenerator(
                backend=backend_for(model, identity.base_model_revision, calls),
                identity=identity,
                input_token_upper_bound=65536,
            )
            driver = AdaptiveRAGWorkerDriver(
                source_root=Path(config["upstream"]),
                router_weights_path=router_root / "model.safetensors",
                prompt_root=output / "prompts",
                runtime=RuntimeBundle(
                    identity,
                    router,
                    generator,
                    ToolkitRetriever(retriever=retriever, identity=identity),
                ),
                router_snapshot_path=router_root,
                router_tokenizer_path=router_root / "tokenizer.json",
                generator_artifact_manifest_path=model_file,
                generator_config_path=gen_file,
                retriever_implementation_path=Path(bm25_module.__file__),
                retriever_config_path=ret_file,
                corpus_manifest_path=corpus,
                index_manifest_path=index,
            )
            worker_identity = dict(
                worker_id="adaptive-trained-captured-live",
                driver_sha256=sha(Path(driver_module.__file__)),
                runtime_sha256=sha(Path(runtime_module.__file__)),
                identity_sha256=content_hash(asdict(identity)),
                capture_id=capture["capture_id"],
                preparation_complete_sha256=sha(complete_path),
                router_configuration=manifest["router_configuration"],
                fixture_mode=False,
            )
            client = RecordedDriver(driver, worker_identity)
            row["worker_requests"], row["worker_responses"] = client.requests, client.responses
            artifacts = AdaptiveRAGArtifactIdentity(
                **{k: v for k, v in asdict(identity).items() if k != "fixture_runtime"},
                router_checkpoint="native-trained-" + manifest["trained_artifact_id"],
                worker_fingerprint_sha256=content_hash(worker_identity),
                **{k + "_prompt_sha256": sha(p) for k, p in prompt_paths.items()},
            )
            result = AdaptiveRAGControlledWorker(
                artifacts=artifacts,
                client=client,
                controls=AdaptiveRAGControls(max_output_tokens=512),
            ).run(state_for(example), InterventionContext(seed=17))
            row.update(
                composite_result=to_jsonable(result), logical_router_calls=router.logical_calls
            )
            answer = result.final_answer
            if not answer.strip():
                raise ValueError("empty native final answer; no reasoning promotion")
            row.update(
                status="scored_terminal",
                prediction=dict(answer=answer),
                metrics=dict(adapters[cell].score(example, QAPrediction(answer=answer))),
            )
        except Exception as exc:
            row.update(status="failed_terminal", error_type=type(exc).__name__, error=str(exc))
        row.update(
            calls=calls,
            physical_new_generator_attempts=len(calls),
            physical_new_router_attempts=0,
            known_generator_responses=sum("response" in c for c in calls),
            helper_logical_charge=1,
            failure_denominator_preserved=True,
            automatic_retry=False,
        )
        save(output / "terminals" / (key + ".json"), row)
        print(
            json.dumps(
                dict(model=model["id"], case=case_id, status=row["status"], requests=len(calls))
            ),
            flush=True,
        )
    rows = [json.loads(p.read_text()) for p in (output / "terminals").glob("*.json")]
    if len(rows) != 8:
        raise ValueError("missing fixed-denominator terminals")
    save(
        output / "complete.json",
        dict(
            terminal_positions=8,
            expected_positions=8,
            statuses=dict(Counter(r["status"] for r in rows)),
            new_generator_attempts=sum(r["physical_new_generator_attempts"] for r in rows),
            new_router_attempts=0,
            separate_preparation_physical_router_attempts=8,
            no_extra_training_or_hidden_inference=True,
            publication_ready=False,
            automatic_retry=False,
        ),
    )
    save(output / "post-service-identities.json", verify_services([model]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)
