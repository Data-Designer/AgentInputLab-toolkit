"""Full selective QA runtime: exact existing compressor captures, fresh readers.

Compression is strictly reused only for byte-identical worker requests. Logical
per-case helper usage stays in the shared budget; physical old helper compute is
referenced, not executed or charged as new. No inference fallback on mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_clariq_acceptance import verify_services
from run_recomp_acceptance import CapturedContextSource
from run_static_acceptance import QA_CELLS, RecordingBackend, digest, local_transport, save

from agentinputlab import ExecutionBudget, InterventionContext
from agentinputlab.algorithms.curate.recomp import RECOMPAbstractiveExternal
from agentinputlab.api import create_intervention, run_composite_intervention, run_intervention
from agentinputlab.backends import FinalContentStopBackend
from agentinputlab.benchmarks import load_production_benchmark_adapter
from agentinputlab.benchmarks.qa import QAPrediction
from agentinputlab.contracts.serialization import content_hash, to_jsonable
from agentinputlab.observability import RunMonitor

RELEASE = "curate.recomp.selective"
OLD_MANIFEST_SHA = "6a986c990e78c59842afc6600a7b1b351e14c41a62c7c785a155aababf829021"
OLD_COMPLETE_SHA = "60dbc36c58f8845bf5e8103d27a35fcf2d8ef723ff80fdff7e56450a5b6251ee"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExactCapturedCompressor:
    """Read-only cache of a complete original two-phase native worker execution."""

    worker_id = "strict-reused-native-recomp-not-new-compression"

    def __init__(self, directory):
        self.directory = directory
        self.terminal = json.loads((directory / "terminal.json").read_text())
        assert self.terminal["status"] == "native_compressed"
        self.record = self.terminal["record"]
        self.identity = self.record["patch"]["metadata"]["worker_fingerprint"]
        self.requests = [
            json.loads((directory / f"worker-{n:02d}-request.json").read_text()) for n in (1, 2)
        ]
        self.responses = [
            json.loads((directory / f"worker-{n:02d}-response.json").read_text()) for n in (1, 2)
        ]
        self.used = []
        assert self.record["patch"]["metadata"]["execution_evidence"] == "pinned_upstream_source"
        assert self.terminal["physical_helper_model_calls"] == 1
        assert self.responses == self.terminal["worker_responses"]

    def fingerprint(self):
        # This is the original EXECUTION identity, not a claim of a live process.
        # Explicit cache provenance and zero new helper executions are in proof().
        return self.identity

    def call(self, payload):
        index = len(self.used)
        assert index < 2 and to_jsonable(payload) == self.requests[index], (
            "exact native cache request mismatch; no fallback inference"
        )
        request_path = self.directory / f"worker-{index + 1:02d}-request.json"
        response_path = self.directory / f"worker-{index + 1:02d}-response.json"
        self.used.append(dict(request_sha256=sha(request_path), response_sha256=sha(response_path)))
        return self.responses[index]

    def proof(self):
        return dict(
            directory=str(self.directory),
            terminal_sha256=sha(self.directory / "terminal.json"),
            original_execution_identity=self.identity,
            matched_requests=self.used,
            reused_original_physical_helper_calls=1,
            new_physical_helper_calls=0,
            logical_helper_usage_charged_in_current_budget=self.record["usage"],
            not_a_fresh_helper_inference=True,
        )


def prepared(config):
    old = Path(config["old_compressor_cohort"])
    assert (
        sha(old / "manifest.json") == OLD_MANIFEST_SHA
        and sha(old / "complete.json") == OLD_COMPLETE_SHA
    )
    originals = json.loads((old / "manifest.json").read_text())
    captures = {}
    for path in (old / "preparation").glob("*/terminal.json"):
        worker = ExactCapturedCompressor(path.parent)
        captures[worker.record["input_state_hash"]] = path.parent
    assert len(captures) == 8
    adapters = {
        cell: load_production_benchmark_adapter(cell, config["prepared_root"])
        for cell, _ in QA_CELLS
    }
    selected = {}
    for cell, split in QA_CELLS:
        for example in sorted(adapters[cell].examples(split), key=lambda item: item.example_id)[
            64:66
        ]:
            state = example.visible_state(
                budget=ExecutionBudget(max_model_calls=8, max_output_tokens=4096)
            )
            capture = captures[state.state_hash]
            recorded_jobs = [
                j
                for j in originals["jobs"]
                if j["benchmark"] == cell and j["case_id"] == example.example_id
            ]
            assert len(recorded_jobs) == 8
            selected[cell, example.example_id] = (example, state, capture)
    assert len(selected) == 8
    return old, adapters, selected


def parameters(worker):
    payload = worker.requests[0]["payload"]
    return dict(
        checkpoint_revision=payload["checkpoint_revision"],
        tokenizer_revision=payload["tokenizer_revision"],
        asset_manifest_hash=payload["asset_manifest_hash"],
        asset_license_acknowledged=True,
    )


def verify_cache(config, output):
    old, adapters, selected = prepared(config)
    rows = []
    for (cell, case), (_, state, directory) in selected.items():
        worker = ExactCapturedCompressor(directory)
        compressor = RECOMPAbstractiveExternal(
            **parameters(worker),
            input_template_hash=worker.requests[0]["payload"]["input_template_hash"],
        )
        result = run_intervention(
            compressor,
            state,
            InterventionContext(
                seed=17,
                external_runtime=worker,
                retriever=CapturedContextSource(cell, case, content_hash(state.context)),
            ),
        )
        assert len(worker.used) == 2
        assert result["record"]["patch"] == worker.record["patch"]
        rows.append(
            dict(benchmark=cell, case=case, input_state_hash=state.state_hash, **worker.proof())
        )
    save(
        output,
        dict(
            exact_native_captures_verified=8,
            physical_new_model_calls=0,
            old_manifest_sha256=sha(old / "manifest.json"),
            old_complete_sha256=sha(old / "complete.json"),
            package=__import__("agentinputlab").__file__,
            evidence=rows,
            publication_ready=False,
        ),
    )


def run(config, output):
    assert len(config["models"]) == 1 and not output.exists()
    bindings = verify_services(config["models"])
    old, adapters, selected = prepared(config)
    model = config["models"][0]
    jobs = []
    for (cell, case), (_, state, directory) in selected.items():
        job = dict(
            release=RELEASE,
            model=model,
            benchmark=cell,
            case_id=case,
            seed=17,
            source_state_hash=state.state_hash,
            reuse_terminal_sha256=sha(directory / "terminal.json"),
            per_call_max_tokens=512,
            budget={"max_model_calls": 8, "max_output_tokens": 4096},
            full_runtime_not_old_component_or_old_reader=True,
        )
        job["id"] = digest(job)
        jobs.append(job)
    output.mkdir()
    for name in ("claims", "terminals", "model-outcomes"):
        (output / name).mkdir()
    save(
        output / "manifest.json",
        dict(
            jobs=jobs,
            model_binding=bindings[model["id"]],
            dataset_files={
                cell: dict(adapter.prepared.file_sha256) for cell, adapter in adapters.items()
            },
            old_manifest_sha256=sha(old / "manifest.json"),
            old_complete_sha256=sha(old / "complete.json"),
            prerequisite_sha256=sha(Path(config["prerequisite"])),
            automatic_retry=False,
            physical_new_compressor_inference=False,
            paper_results_changed=False,
        ),
    )

    def execute(job):
        example, state, directory = selected[job["benchmark"], job["case_id"]]
        worker = ExactCapturedCompressor(directory)
        calls, monitor = [], RunMonitor()
        backend = RecordingBackend(
            calls=calls,
            model_id=model["id"],
            base_url=model["base_url"],
            extra_body=model["extra_body"],
            transport=local_transport,
            timeout_seconds=240,
            deployment_identity={
                "revision": digest(bindings[model["id"]]["model_json_hashes"]),
                "server": "vllm",
            },
        )
        if model["id"] == "gpt-oss-20b":
            backend = FinalContentStopBackend(backend)
        save(output / "claims" / (job["id"] + ".json"), {"job": job})
        row = dict(
            job=job,
            additional_reader_attempts=1,
            additional_compressor_attempts=0,
            automatic_retry=False,
        )
        record = None
        stage = "full_selective_runtime"
        try:
            record = run_composite_intervention(
                create_intervention(RELEASE, **parameters(worker)),
                state,
                InterventionContext(
                    backend=backend,
                    external_runtime=worker,
                    monitor=monitor,
                    seed=17,
                    retriever=CapturedContextSource(
                        job["benchmark"], job["case_id"], content_hash(state.context)
                    ),
                ),
            )
            row.update(
                status="answer_returned",
                execution=record,
                prediction={"answer": record["final_answer"]},
            )
        except Exception as exc:
            row.update(
                status="failed_terminal", stage=stage, error_type=type(exc).__name__, error=str(exc)
            )
        row.update(
            calls=calls,
            logical_compressor_reuse=worker.proof(),
            monitor_snapshots=to_jsonable(monitor.list_snapshots()),
        )
        save(
            output / "model-outcomes" / (job["id"] + ".json"),
            {**row, "native_scoring_started": False},
        )
        if record is not None:
            try:
                row.update(
                    status="scored_terminal",
                    metrics=dict(
                        adapters[job["benchmark"]].score(example, QAPrediction(**row["prediction"]))
                    ),
                )
            except Exception as exc:
                row.update(
                    status="failed_terminal",
                    stage="native_QA_scoring",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
        assert len(worker.used) == 2 and len(calls) <= 1
        save(output / "terminals" / (job["id"] + ".json"), row)

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(execute, jobs))
    rows = [json.loads(p.read_text()) for p in (output / "terminals").glob("*.json")]
    assert len(rows) == 8
    save(
        output / "complete.json",
        dict(
            expected_positions=8,
            terminal_positions=8,
            scored=sum(r["status"] == "scored_terminal" for r in rows),
            failed=sum(r["status"] == "failed_terminal" for r in rows),
            physical_new_reader_calls=sum(len(r["calls"]) for r in rows),
            physical_new_compressor_calls=0,
            automatic_retry=False,
            publication_ready=False,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--verify-cache", action="store_true")
    a = p.parse_args()
    (verify_cache if a.verify_cache else run)(json.loads(a.config.read_text()), a.output)
