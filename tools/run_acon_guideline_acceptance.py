"""Controlled native ACON guideline fit from actual disjoint QASPER train histories.

Native smolagents directory/F1 layout is an explicit input-format bridge, NOT a
claim to execute the original smolagents benchmark. Fixed train papers are chosen
before outcomes. Genuine failure/no-regression branches remain terminal. Generated
templates are opaque candidates, not deployed or selected on heldout performance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_clariq_acceptance import verify_services
from run_static_acceptance import RecordingBackend, digest, local_transport, save

from agentinputlab.backends import GenerationRequest, GenerationResponse, Usage
from agentinputlab.benchmarks import BenchmarkSmokeAssetRegistry
from agentinputlab.benchmarks.preprocess import _qasper_answer, _qasper_context_blocks
from agentinputlab.benchmarks.qa import QAPrediction, _score_qasper
from agentinputlab.contracts.serialization import to_jsonable
from agentinputlab.recipes import ACONGuidelineControls, ACONGuidelinePipeline
from workers.acon.driver import _construct_optimizer, _load_manifest, _upstream_import_scope
from workers.acon.guideline import validate_guideline_checkout


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_train(source):
    asset = BenchmarkSmokeAssetRegistry.default().get("qasper.dev.official")
    if source.stat().st_size != asset.size_bytes or sha(source) != asset.sha256:
        raise ValueError("cached official QASPER archive identity differs")
    with tarfile.open(source, "r:gz") as archive:
        data = {}
        for split in ("train", "dev"):
            member = archive.getmember(f"qasper-{split}-v0.3.json")
            if not member.isfile() or member.size > 64 * 1024**2:
                raise ValueError("official QASPER member identity/size differs")
            data[split] = json.load(archive.extractfile(member))
    rows = []
    for paper_id, paper in sorted(data["train"].items()):
        if paper_id in data["dev"] or len(paper["qas"]) < 2:
            continue
        first, second = sorted(paper["qas"], key=lambda q: q["question_id"])[:2]
        groups = []
        for annotation in second["answers"]:
            answer = annotation["answer"]
            value, kind = _qasper_answer(answer)
            groups.append(
                {
                    "answers": (value,),
                    "answer_type": kind,
                    "evidence_texts": tuple(answer.get("evidence", [])),
                }
            )
        rows.append(
            dict(
                paper_id=paper_id,
                source_split="train",
                history_question=first["question"],
                history_id=first["question_id"],
                question=second["question"],
                case_id=second["question_id"],
                context="\n\n".join(
                    b["text"] for b in _qasper_context_blocks(paper, paper_id=paper_id)
                )[:12000],
                evaluator_only_reference_groups=groups,
            )
        )
        if len(rows) == 2:
            break
    if len(rows) != 2:
        raise ValueError("two disjoint official training papers required")
    excluded = [q["question_id"] for paper in data["dev"].values() for q in paper["qas"]]
    assert not set(excluded) & {r["case_id"] for r in rows}
    return rows, sorted(excluded), asset.sha256


class DurableBackend(RecordingBackend):
    def __init__(self, *, journal, **kwargs):
        super().__init__(**kwargs)
        self.journal = journal
        self.journal.mkdir()

    def generate(self, request):
        number = len(self.calls) + 1
        save(self.journal / f"{number:04d}-request.json", request)
        try:
            return super().generate(request)
        finally:
            save(self.journal / f"{number:04d}-terminal.json", self.calls[-1])


class StrictReusingBackend(DurableBackend):
    """Reuse exact previously successful requests; journal references, not calls."""

    def __init__(self, *, reuses, fit_reuses=(), **kwargs):
        super().__init__(**kwargs)
        self.reuses = reuses
        self.fit_reuses = tuple(fit_reuses)

    def generate(self, request):
        number = len(self.calls) + 1
        reference = self.reuses.get(number)
        if reference is None:
            # A changed analysis can still produce an identical native update
            # request. Cache by the ENTIRE request, never by score or task name.
            for candidate in self.fit_reuses:
                path = Path(candidate["path"])
                if sha(path) != candidate["sha256"]:
                    raise ValueError("prior fit-response checksum differs")
                old = json.loads(path.read_text())
                response = old.get("response", {})
                if old["request"] == to_jsonable(request):
                    if (
                        response.get("model") != self.model_id
                        or response.get("usage", {}).get("input_tokens", 0) <= 0
                    ):
                        raise ValueError(
                            "identical prior request lacks verified usage/model; do not retry it"
                        )
                    reference = {**candidate, "known_outcome_cache": True}
                    break
        if reference is None:
            return super().generate(request)
        path = Path(reference["path"])
        if sha(path) != reference["sha256"]:
            raise ValueError("strictly reused training response identity differs")
        original = json.loads(path.read_text())
        if to_jsonable(request) != original["request"]:
            raise ValueError("strictly reused training request differs before any new model call")
        response = original["response"]
        if response["usage"]["output_tokens"] > request.max_tokens or (
            not reference.get("known_outcome_cache") and not response["text"].strip()
        ):
            raise ValueError("failed responses cannot be reused as successful training history")
        self.calls.append({**original, "strict_reuse": reference})
        save(self.journal / f"{number:04d}-reuse.json", self.calls[-1])
        return GenerationResponse(**{**response, "usage": Usage(**response["usage"])})


def system_message_reuses(prior, train, model, expected_job_sha256, expected_terminal_sha256):
    """Validate a concrete source-system omission, not an outcome-selected retry."""
    prior = Path(prior)
    if (
        sha(prior / "job.json") != expected_job_sha256
        or sha(prior / "terminal.json") != expected_terminal_sha256
    ):
        raise ValueError("prior implementation-retest binding differs")
    job = json.loads((prior / "job.json").read_text())
    if job["model"]["id"] != model["id"] or job["train"] != train:
        raise ValueError("system-message repair cannot change model or training cases")
    if {k: v for k, v in job["model"].items() if k != "service_directory"} != {
        k: v for k, v in model.items() if k != "service_directory"
    }:
        raise ValueError("system-message repair backend configuration differs")
    # Compression happened at logical calls3/7. A two-call Q27 failure is not
    # affected by this omission and must NOT be sent to this repair cohort.
    affected = [prior / "trajectory-attempts" / f"{i:04d}-terminal.json" for i in (3, 7)]
    if not any(p.exists() for p in affected):
        raise ValueError("no consumed native compression affected by system-message omission")
    for p in affected:
        if p.exists() and json.loads(p.read_text())["request"]["system"]:
            raise ValueError("prior native compression already received a system message")
    reuses = {}
    for number in (1, 2, 5, 6):
        p = prior / "trajectory-attempts" / f"{number:04d}-terminal.json"
        if not p.exists():
            raise ValueError("complete warmup/baseline references required; do not rerun them")
        row = json.loads(p.read_text())
        if row["request"]["system"] or row["request"]["seed"] != 16 + number:
            raise ValueError("prior unchanged warmup request differs")
        response = row["response"]
        if (
            not response["text"].strip()
            or response["usage"]["input_tokens"] <= 0
            or response["usage"]["output_tokens"] > 512
        ):
            raise ValueError("prior warmup/baseline reference is not a usable actual result")
        reuses[number] = {"path": str(p), "sha256": sha(p), "additional_model_calls": 0}
    return reuses


class NativeHistorySeam:
    def __init__(self, backend, calls):
        self.backend, self.calls = backend, calls
        self.system_message = ""

    def generate(self, prompt, temperature=0.0):
        return ask(
            self.backend, self.calls, prompt, temperature=temperature, system=self.system_message
        )


def ask(backend, calls, prompt, temperature=0.0, *, system="", allow_empty=False):
    # The train-trajectory phase has 2 fixed papers x 4 calls, including the
    # shared first response once. Fit-generation costs are separate below.
    output = sum(c.get("response", {}).get("usage", {}).get("output_tokens", 0) for c in calls)
    if len(calls) >= 8 or output + 512 > 4096:
        raise ValueError("training-trajectory budget exhausted before transport")
    response = backend.generate(
        GenerationRequest(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=512,
            seed=17 + len(calls),
        )
    )
    if response.usage.input_tokens == response.usage.output_tokens == 0:
        raise ValueError("training trajectory has unknown token usage")
    if response.usage.output_tokens > 512 or (not response.text.strip() and not allow_empty):
        raise ValueError("training trajectory exceeds cap or has empty final content")
    return response.text


def history_class(checkout):
    return importlib.import_module("productive_agents.ctxopt.history_optimizer").HistoryOptimizer


def model_worker(job):
    out, checkout = Path(job["output"]), Path(job["checkout"])
    model, binding = job["model"], job["binding"]
    calls = []
    reuse_args = {}
    backend_class = DurableBackend
    if job.get("implementation_retest_reason") == "verified_native_history_system_message_omission":
        reuse_args["reuses"] = system_message_reuses(
            job["prior_output"],
            job["train"],
            model,
            job["prior_job_sha256"],
            job["prior_terminal_sha256"],
        )
        backend_class = StrictReusingBackend
        prior = Path(job["prior_output"])
        prior_job = json.loads((prior / "job.json").read_text())
        if prior_job["binding"]["model_json_hashes"] != binding["model_json_hashes"]:
            raise ValueError("strict reuse cannot change backbone/model-file bindings")
        reuse_args["fit_reuses"] = [
            {"path": str(p), "sha256": sha(p), "additional_model_calls": 0}
            for p in sorted((prior / "trajectory-attempts").glob("*-terminal.json"))
        ]
    backend = backend_class(
        journal=out / "trajectory-attempts",
        calls=calls,
        model_id=model["id"],
        base_url=model["base_url"],
        api_key="",
        deployment_identity={"revision": digest(binding["model_json_hashes"]), "server": "vllm"},
        extra_body=model.get("extra_body", {}),
        timeout_seconds=180,
        transport=local_transport,
        **reuse_args,
    )
    stage, fit = "real_training_trajectories", None
    try:
        source, _ = validate_guideline_checkout(checkout)
        core, _ = _load_manifest(
            Path(__import__("workers.acon.driver", fromlist=["x"]).__file__).with_name(
                "compatibility_manifest.json"
            )
        )
        profile = core["prompt_profiles"]["smolagents.history.v1"]
        train = out / "native-train-bundle"
        train.mkdir()
        evaluation = {"baseline": {}, "optimized": {}}
        with _upstream_import_scope(source):
            native = history_class(source)
            for row in job["train"]:
                case = row["case_id"]
                system = "Answer the research-paper question concisely. Return the answer only."
                initial = (
                    f"{system}\n\nPaper:\n{row['context']}\n\nQuestion: {row['history_question']}"
                )
                first = ask(backend, calls, initial, allow_empty=True)
                history = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": initial},
                    {"role": "assistant", "content": first},
                ]
                flat = "\n\n".join(m["role"].upper() + ":\n" + m["content"] for m in history)
                baseline_prompt = flat + "\n\nUSER:\n" + row["question"]
                baseline = ask(backend, calls, baseline_prompt, allow_empty=True)
                optimizer = _construct_optimizer(
                    native,
                    {
                        "history_prompt_dir": str(source / profile["prompt_dir"]),
                        "prompts": {
                            "prompt_system": profile["system_template"],
                            "prompt_history_user": profile["user_template"],
                        },
                        "history_summarization_threshold": -1,
                        "model": model["id"],
                        "temperature": 0.0,
                        "use_llmlingua": False,
                    },
                    NativeHistorySeam(backend, calls),
                    "HistoryOptimizer",
                )
                if optimizer.tokenizer is None:
                    raise ValueError("exact native tokenizer unavailable; no approximate fallback")
                summary = optimizer.process(
                    task=row["question"],
                    history=flat,
                    prev_history_summary=None,
                    raw_history=history,
                    opt_args=None,
                )
                if not isinstance(summary, str) or not summary.strip():
                    raise ValueError("native ACON produced an empty history summary")
                optimized_history = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": "Previous session summary:\n" + summary},
                ]
                optimized_prompt = system + "\n\n" + summary + "\n\nQuestion: " + row["question"]
                optimized = ask(backend, calls, optimized_prompt, allow_empty=True)
                for role, text, hist in (
                    ("baseline", baseline, history),
                    ("optimized", optimized, optimized_history),
                ):
                    complete = [
                        *hist,
                        {"role": "user", "content": row["question"]},
                        {"role": "assistant", "content": text},
                    ]
                    directory = train / role / "samples" / case
                    directory.mkdir(parents=True)
                    save(directory / "llm_history.json", complete)
                    metrics = dict(
                        _score_qasper(
                            QAPrediction(answer=text), tuple(row["evaluator_only_reference_groups"])
                        )
                    )
                    evaluation[role][case] = {
                        "f1": metrics["answer_f1"],
                        "success": metrics["answer_f1"] == 1.0,
                        "metrics": metrics,
                        "failures": "measured QASPER answer F1 below 1"
                        if metrics["answer_f1"] < 1
                        else None,
                    }
                save(
                    out / (case + "-native-compression.json"),
                    {
                        "source": str(source),
                        "original_class": native.__module__ + "." + native.__name__,
                        "raw_interactions": optimizer.history,
                        "summary": summary,
                        "source_dataset": "qasper.v0_3_train",
                        "layout_adapter": "smolagents F1 format only",
                    },
                )
        for role in evaluation:
            save(train / (role + "-evaluations.json"), {"individual": evaluation[role]})
        files = {p.relative_to(train).as_posix(): sha(p) for p in train.rglob("*") if p.is_file()}
        save(
            train / "manifest.json",
            {
                "schema": "agentinputlab.acon.training-input.v1",
                "split": "train",
                "benchmark": "smolagents",
                "source_dataset": "qasper.v0_3_train",
                "layout_adapter_not_smolagents_benchmark_parity": True,
                "train_ids": [r["case_id"] for r in job["train"]],
                "excluded_evaluation_ids": job["excluded_ids"],
                "files": files,
                "source_archive_sha256": job["source_archive_sha256"],
            },
        )
        stage = "source_native_analysis_and_update"
        fit = ACONGuidelinePipeline(ACONGuidelineControls()).fit(
            checkout=source,
            training_root=train,
            training_manifest_sha256=sha(train / "manifest.json"),
            output_root=out / "native-fit",
            backend=backend,
        )
        for relative in fit["candidates"]:
            if not (out / "native-fit" / relative).read_text().strip():
                raise ValueError(
                    "native updater produced an empty candidate; not a usable artifact"
                )
        status = "candidate_fit_terminal" if fit["candidates"] else "native_no_regression_terminal"
    except Exception as exc:
        import traceback

        status = "failed_terminal"
        save(
            out / "failure.json",
            {
                "stage": stage,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
    physical = [row for row in calls if "strict_reuse" not in row]
    save(
        out / "terminal.json",
        {
            "status": status,
            "stage": stage,
            "fit": fit,
            "physical_model_attempts": len(physical),
            "logical_requests_including_strict_reuses": len(calls),
            "strict_reused_responses": len(calls) - len(physical),
            "implementation_retest_reason": job.get("implementation_retest_reason"),
            "prior_terminal_sha256": job.get("prior_terminal_sha256"),
            "known_input_tokens": sum(
                r.get("response", {}).get("usage", {}).get("input_tokens", 0) for r in physical
            ),
            "known_output_tokens": sum(
                r.get("response", {}).get("usage", {}).get("output_tokens", 0) for r in physical
            ),
            "unknown_usage_attempts": sum("response" not in r for r in physical),
            "automatic_retry": False,
            "paper_results_changed": False,
            "generated_templates_executed": False,
            "deployment_acceptance_complete": False,
            "package_location": __import__("agentinputlab").__file__,
        },
    )


def run(args):
    config = json.loads(args.config.read_text())
    bindings = verify_services(config["models"])
    rows, excluded, archive_hash = select_train(args.archive)
    args.output.mkdir(exist_ok=False)
    save(
        args.output / "manifest.json",
        {
            "models": config["models"],
            "train": rows,
            "bindings": bindings,
            "source_archive_sha256": archive_hash,
            "source_split": "official_train",
            "heldout_dev_ids_excluded": excluded,
            "trajectory_budget_per_model": {"calls": 8, "output_tokens": 4096, "per_call": 512},
            "fit_controls": to_jsonable(ACONGuidelineControls()),
            "automatic_retry": False,
            "scope": "controlled native candidate producer, NOT full original benchmark/deployment acceptance",
            "script_sha256": sha(__file__),
            "paper_results_changed": False,
        },
    )

    def one(model):
        out = args.output / model["id"]
        out.mkdir()
        save(
            out / "job.json",
            {
                "model": model,
                "binding": bindings[model["id"]],
                "train": rows,
                "excluded_ids": excluded,
                "source_archive_sha256": archive_hash,
                "checkout": str(args.checkout),
                "output": str(out),
            },
        )
        with (out / "process.log").open("x") as log:
            try:
                done = subprocess.run(
                    [sys.executable, __file__, "--worker-job", str(out / "job.json")],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=1500,
                )
                result = {"returncode": done.returncode}
            except subprocess.TimeoutExpired:
                result = {"timed_out": True, "automatic_retry": False}
            save(out / "process-exit.json", result)
        (args.output.parent / "keep-services-alive").touch()

    with ThreadPoolExecutor(len(config["models"])) as pool:
        list(pool.map(one, config["models"]))
    terminals = {
        m["id"]: json.loads((args.output / m["id"] / "terminal.json").read_text())
        if (args.output / m["id"] / "terminal.json").exists()
        else {"status": "missing_terminal_inspect_process_no_retry"}
        for m in config["models"]
    }
    save(
        args.output / "complete.json",
        {
            "terminals": terminals,
            "full_toolkit_acceptance_complete": False,
            "deployment_acceptance_complete": False,
            "automatic_retry": False,
        },
    )
    print(json.dumps({k: v["status"] for k, v in terminals.items()}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    p.add_argument("--archive", type=Path)
    p.add_argument("--checkout", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--worker-job", type=Path)
    args = p.parse_args()
    if args.worker_job:
        model_worker(json.loads(args.worker_job.read_text()))
    else:
        run(args)


if __name__ == "__main__":
    main()
