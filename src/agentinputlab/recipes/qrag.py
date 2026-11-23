"""Pinned Q-RAG value-policy training with native rollout, rewards and updates.

Run fit in a dedicated process. Hydra's dataset transport reads the caller's
closed native records; QADatasetAdapter, QAEnv, GroundTruthFeedback,
ParallelTextEnv and PQN own all label projection, rollout and gradient math.
Training checkpoints are new artifacts, never aliases of the released policy.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import runpy
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path

from agentinputlab.algorithms.acquire.qrag.bridge import QRAG_UPSTREAM_COMMIT
from agentinputlab.algorithms.acquire.qrag.common_support import (
    _source_import,
    _unused_vllm_import_compatibility,
)
from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash

from .recomp_extractive import _LOCK, _save, _scope

RELEASE = "training.qrag.value_policy"
PROTOCOL = "agentinputlab.qrag.native_value_fit.v1"
SOURCE_FILES = {
    "train_q_rag.py": "e412a85fe6c2fe06eb6c0f406c8b824fe6063d975dbfb54b8c3c1615558fa4c2",
    "envs/parallel_env.py": "d77f6f79ba3a5aecf09795075274378ccb806ec0514ed8b2d625e1e9e5d93853",
    "envs/qa_env.py": "267c27c248f332478bf810a50f097a898a818cf6f8f4bd7b70a2893312c0eae1",
    "envs/qa_dataset_adapter.py": "b2cf231177a1f079f7fdb213838df2ab967caf6c9977df5447bc31cf8abba5f3",
    "rl/agents/pqn.py": "5e69ee5e20f098c101ceebb0c3713a3e63ffcd0a1b614d7180c1b474465cce39",
    "rl/q_module.py": "8f5e50f1033aa04a326bba2ba14eff2ff8631d494f01913179652644e7832148",
    "rl/bert_predictor.py": "7d82d0749afd0b57d1416dd6c39477f164fd710421f23faaadbd8a29e2f142c8",
    "rl/optim.py": "1b54bee66a2496a6cfa5c07365e573ecd597b218e3055b4bd8c8997275a02459",
    "rl/feedback/feedback.py": "d28a12b111c81ce30d1fee701282e36f5b4c67b6bad30de4d62ac547b99bc7d3",
    "configs/training.yaml": "eb0bae23a1c9a6e18cb83e2596b9197b7ed36cee55a546be66f1bfdf83220f4e",
    "configs/algo/pqn_e5_hotpotqa.yaml": "f333f83f4b0b0b7337468970a9b95a9672556c85cdb6248c3f7f25755e20c2f8",
    "configs/envs/hotpotqa.yaml": "1820a78023a04b133ec3da97f86927b5910bdc3389807825dc32a213d4a5156a",
    "configs/feedback/defaults.yaml": "c14bfa71848eb6381438592a85bf7ed1edf2d8ef109dfed0d061ded601aa53ad",
    "configs/logger/logging.yaml": "38dea1930fd7c69e5782db3d64dbf23c73811711d8f0f2637ad20973c7e0f501",
}
SOURCE_RECEIPT_SHA256 = "f722670a11ed666d95130a61f54d0556764e0fab6b780503226646079af2a57a"
TASKS = {"hotpotqa", "2WikiMultihopQA", "musique", "babilong"}


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _file(value, label):
    path = Path(value).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValidationError(f"Q-RAG {label} requires a regular non-symlink file")
    return path.resolve()


def _text(value):
    return isinstance(value, str) and bool(value.strip()) and "\0" not in value


@dataclass(frozen=True, slots=True)
class QRAGFitControls:
    steps_count: int = 50000
    batch_size: int = 64
    accumulate_grads: int = 2
    environments: int = 4
    learning_start: int = 200
    eval_interval: int = 500
    eval_episodes: int = 300
    warmup_steps: int = 1000
    seed: int = 17
    device: str = "cuda:0"
    task: str = "hotpotqa"
    max_optimizer_steps: int = 50000
    max_data_bytes: int = 64 * 1024 * 1024
    min_free_output_bytes: int = 128 * 1024**3

    def __post_init__(self):
        for name in (
            "steps_count",
            "batch_size",
            "accumulate_grads",
            "environments",
            "learning_start",
            "eval_interval",
            "eval_episodes",
            "warmup_steps",
            "seed",
            "max_optimizer_steps",
            "max_data_bytes",
            "min_free_output_bytes",
        ):
            value = getattr(self, name)
            lower = 0 if name in {"learning_start", "warmup_steps", "seed"} else 1
            if type(value) is not int or value < lower:
                raise ValidationError(f"Q-RAG fit {name} must be an integer in range")
        if self.steps_count > self.max_optimizer_steps:
            raise ValidationError("Q-RAG optimizer plan exceeds the declared step cap")
        if self.task not in TASKS or self.device not in {"cpu", "cuda:0"}:
            raise ValidationError(
                "Q-RAG fit needs a supported native labeled task and one logical device"
            )


def audit_qrag_training_data(train_file, dev_file, controls):
    """Reject missing labels and cross-split leakage; do not invent support labels."""
    queries, identities, audit = {}, {}, {}
    paths = {"train": _file(train_file, "train data"), "dev": _file(dev_file, "dev data")}
    for split, path in paths.items():
        if path.stat().st_size > controls.max_data_bytes:
            raise ValidationError("Q-RAG data exceeds the declared byte cap")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise ValidationError("Q-RAG native data must be a non-empty record array")
        seen, qs = set(), set()
        for row in rows:
            if (
                not isinstance(row, dict)
                or not _text(row.get("question"))
                or not _text(row.get("answer"))
            ):
                raise ValidationError("Q-RAG training requires native question and answer text")
            task = controls.task
            identifier = (
                row.get("_id") if task in {"hotpotqa", "2WikiMultihopQA"} else row.get("id")
            )
            if task == "babilong":
                identifier = row.get("id", content_hash(row["question"]))
            if not _text(identifier) or identifier in seen:
                raise ValidationError("Q-RAG native training IDs must be non-empty and unique")
            seen.add(identifier)
            query = " ".join(row["question"].casefold().split())
            if query in qs:
                raise ValidationError("Q-RAG duplicated training query")
            qs.add(query)
            if task in {"hotpotqa", "2WikiMultihopQA"}:
                context, support = row.get("context"), row.get("supporting_facts")
                if (
                    not isinstance(context, list)
                    or not context
                    or not isinstance(support, list)
                    or not support
                ):
                    raise ValidationError(
                        "Q-RAG native training requires context and real support labels"
                    )
                titles = {}
                for item in context:
                    if (
                        not isinstance(item, list)
                        or len(item) != 2
                        or not _text(item[0])
                        or not isinstance(item[1], list)
                        or not item[1]
                        or not all(_text(x) for x in item[1])
                        or item[0] in titles
                    ):
                        raise ValidationError("Q-RAG native context titles/sentences are invalid")
                    titles[item[0]] = item[1]
                if any(
                    not isinstance(s, list)
                    or len(s) != 2
                    or not _text(s[0])
                    or s[0] not in titles
                    or type(s[1]) is not int
                    or not 0 <= s[1] < len(titles[s[0]])
                    for s in support
                ):
                    raise ValidationError(
                        "Q-RAG support label references unavailable source sentences"
                    )
            elif task == "musique":
                paragraphs, decomposition = row.get("paragraphs"), row.get("question_decomposition")
                if (
                    not isinstance(paragraphs, list)
                    or not paragraphs
                    or not isinstance(decomposition, list)
                    or not decomposition
                ):
                    raise ValidationError(
                        "Q-RAG MuSiQue requires native paragraphs and decomposition labels"
                    )
                if any(
                    not isinstance(p, dict)
                    or not isinstance(p.get("title"), str)
                    or "\0" in p["title"]
                    or not _text(p.get("paragraph_text"))
                    for p in paragraphs
                ):
                    raise ValidationError("Q-RAG native MuSiQue paragraph text is invalid")
                if any(
                    not isinstance(d, dict)
                    or type(d.get("paragraph_support_idx")) is not int
                    or not 0 <= d["paragraph_support_idx"] < len(paragraphs)
                    for d in decomposition
                ):
                    raise ValidationError("Q-RAG MuSiQue support label is invalid")
            else:
                chunks, labels = row.get("chunks"), row.get("references_idx")
                if not isinstance(chunks, list) or not chunks or not all(_text(x) for x in chunks):
                    raise ValidationError("Q-RAG Babilong requires real native chunks")
                if (
                    not isinstance(labels, list)
                    or not labels
                    or any(type(i) is not int or not 0 <= i < len(chunks) for i in labels)
                ):
                    raise ValidationError("Q-RAG Babilong requires valid native reference indices")
        queries[split], identities[split] = qs, seen
        audit[split] = dict(
            path=str(path),
            sha256=_sha(path),
            rows=len(rows),
            query_set_sha256=content_hash(sorted(qs)),
        )
    if (
        paths["train"] == paths["dev"]
        or queries["train"] & queries["dev"]
        or identities["train"] & identities["dev"]
    ):
        raise ValidationError("Q-RAG train/dev IDs and queries must be disjoint")
    return dict(
        files=audit,
        query_overlap=False,
        id_overlap=False,
        task=controls.task,
        native_updates=controls.steps_count * controls.accumulate_grads,
        optimizer_steps=controls.steps_count,
        transitions_per_rollout=math.ceil(controls.batch_size / controls.environments)
        * controls.environments,
        native_support_labels_required=True,
        no_dummy_or_fabricated_reward_labels=True,
    )


class _NativeRecords:
    def __init__(self, path, task, expected_sha256):
        if _sha(_file(path, "native dataset")) != expected_sha256:
            raise ValidationError("Q-RAG native dataset differs from the fitted input plan")
        self.rows = json.loads(Path(path).read_text(encoding="utf-8"))
        if _sha(path) != expected_sha256:
            raise ValidationError("Q-RAG native dataset changed while being loaded")
        self.task = task

    def name(self):
        return self.task

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class NativeQRAGDataset:
    """Hydra data-format seam; the upstream adapter still projects every label."""

    def __new__(cls, path, task, expected_sha256):
        from envs.qa_dataset_adapter import QADatasetAdapter

        return QADatasetAdapter(_NativeRecords(path, task, expected_sha256))


def validate_qrag_training_source(checkout):
    supplied = Path(checkout).expanduser()
    if supplied.is_symlink() or not supplied.is_dir():
        raise ValidationError("Q-RAG training needs a regular mounted source directory")
    root = supplied.resolve()
    if (root / ".git").exists():

        def git(*args):
            return subprocess.check_output(
                ["git", "-C", str(root), *args], text=True, timeout=30
            ).strip()

        if git("rev-parse", "HEAD") != QRAG_UPSTREAM_COMMIT or git(
            "status", "--porcelain=v1", "--untracked-files=all"
        ):
            raise ExternalDependencyError("Q-RAG source checkout must be pinned and clean")
    else:
        receipt = _file(root / "agentinputlab-source-receipt.json", "source receipt")
        if _sha(receipt) != SOURCE_RECEIPT_SHA256:
            raise ExternalDependencyError("Q-RAG full native source receipt differs")
        data = json.loads(receipt.read_text())
        expected = {r["path"]: r for r in data["files"]}
        actual = set()
        for p in root.rglob("*"):
            if "__pycache__" in p.parts or p.suffix == ".pyc":
                continue
            if p.is_symlink():
                raise ExternalDependencyError("Q-RAG mounted source contains a symlink")
            if p.is_file() and p.name != receipt.name:
                relative = p.relative_to(root).as_posix()
                actual.add(relative)
                row = expected.get(relative)
                if row is None or p.stat().st_size != row["size"] or _sha(p) != row["sha256"]:
                    raise ExternalDependencyError("Q-RAG full mounted source file differs")
        if actual != set(expected):
            raise ExternalDependencyError("Q-RAG source closure has missing files")
    for name, expected in SOURCE_FILES.items():
        p = _file(root / name, "native source")
        if not p.is_relative_to(root) or _sha(p) != expected:
            raise ExternalDependencyError(f"Q-RAG pinned native training source differs: {name}")
    return root


def validate_qrag_initial_model(folder, manifest, expected_sha256):
    from .llmlingua2_training import _closed_files

    root, data = _closed_files(
        folder, manifest, expected_sha256, "agentinputlab.qrag.initial_model.v1"
    )
    if not {"model.safetensors", "config.json", "tokenizer_config.json"} <= data["files"].keys():
        raise ValidationError("Q-RAG initialization needs safe model and tokenizer files")
    if any(n.endswith((".bin", ".pkl", ".pickle", ".py")) for n in data["files"]):
        raise ValidationError("Q-RAG initial model contains an unsafe payload")
    model = json.loads((root / "config.json").read_text())
    tokenizer = json.loads((root / "tokenizer_config.json").read_text())
    if (
        model.get("model_type") != "xlm-roberta"
        or model.get("hidden_size") != 1024
        or model.get("num_hidden_layers") != 24
        or model.get("auto_map")
        or tokenizer.get("auto_map")
    ):
        raise ValidationError(
            "native E5 PQN initialization needs the full standard 24-layer 1024-dim architecture"
        )
    return root, data


def _parameter_hashes(module):
    return {
        name: hashlib.sha256(
            value.detach()
            .cpu()
            .contiguous()
            .view(-1)
            .view(dtype=__import__("torch").uint8)
            .numpy()
            .tobytes()
        ).hexdigest()
        for name, value in module.named_parameters()
    }


def _counted_embedding_forward(original, observed):
    """Observe direct ``.forward`` and ``__call__`` without replacing outputs."""

    def forward(module, *args, **kwargs):
        observed["embedding_forward_attempts"] += 1
        response = original(module, *args, **kwargs)
        observed["embedding_forward_responses"] += 1
        return response

    return forward


class _NativeCheckpointStore:
    """Serialize each unchanged native update once; preserve best/last names."""

    def __init__(self, root, native_save, events):
        self.root, self.native_save, self.events = root.resolve(), native_save, events
        self.snapshots = {}

    def __call__(self, agent, checkpoint_path, verbose=False):
        destination = Path(checkpoint_path).absolute()
        if not destination.resolve().is_relative_to(self.root) or destination.is_symlink():
            raise ValidationError("Q-RAG checkpoint output escapes its unique fit directory")
        step = agent._update_step
        created = step not in self.snapshots
        if created:
            cached = self.root / "native-checkpoints" / f"update-{step:08d}.pt"
            if cached.exists():
                raise ValidationError("Q-RAG native checkpoint snapshot already exists")
            self.native_save(agent, str(cached), verbose=verbose)
            self.snapshots[step] = cached
        cached = self.snapshots[step]
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".native-link")
        os.link(cached, temporary)
        os.replace(temporary, destination)
        self.events.append(
            dict(
                native_update_step=step,
                logical_path=str(destination),
                immutable_snapshot_path=str(cached),
                native_serialization_performed=created,
                storage_only_hardlink_alias=True,
            )
        )


@dataclass(frozen=True, slots=True)
class QRAGTrainedPolicyArtifact:
    root: Path
    artifact_id: str

    @property
    def checkpoint(self):
        return self.root / "native-final.pt"

    def runtime(self, **kwargs):
        from .qrag_inference import QRAGTrainedPolicyRuntime

        return QRAGTrainedPolicyRuntime(self, **kwargs)

    @classmethod
    def load(cls, root):
        p = Path(root).expanduser()
        if p.is_symlink() or not p.is_dir():
            raise ValidationError("Q-RAG trained artifact must be a regular directory")
        metadata = json.loads(_file(p / "artifact.json", "artifact metadata").read_text())
        identity = metadata.pop("artifact_id")
        if identity != content_hash(metadata) or metadata.get("protocol") != PROTOCOL:
            raise ValidationError("Q-RAG trained artifact identity differs")
        for name in ("fit-plan.json", "fit-terminal.json", "native-final.pt"):
            if _sha(_file(p / name, "artifact file")) != metadata["files"][name]:
                raise ValidationError("Q-RAG trained artifact file differs")
        terminal = json.loads((p / "fit-terminal.json").read_text())
        if terminal.get("status") != "trained_candidate" or terminal.get("optimizer_steps", 0) < 1:
            raise ValidationError("Q-RAG artifact is not a closed trained candidate")
        return cls(p.resolve(), identity)


@contextmanager
def _observe_native(root, source, initial, audit, controls, observed):
    import hydra
    from omegaconf import OmegaConf
    from rl.agents.pqn import PQN
    from rl.bert_predictor import BertPredictor

    compose, initialize, update, native_save = hydra.compose, PQN.__init__, PQN.update, PQN.save
    embedding_forward = BertPredictor.forward
    store = _NativeCheckpointStore(root, native_save, observed["checkpoint_storage_events"])

    def composed(*args, **kwargs):
        cfg = compose(*args, **kwargs)
        cfg.algo.model.model_name = str(initial)
        cfg.algo.model.revision = None
        for section, split in (("env", "train"), ("test_env", "dev")):
            cfg.envs[section].dataset = OmegaConf.create(
                dict(
                    _target_="agentinputlab.recipes.qrag.NativeQRAGDataset",
                    path=audit["files"][split]["path"],
                    task=controls.task,
                    expected_sha256=audit["files"][split]["sha256"],
                )
            )
        cfg.envs.task = controls.task
        OmegaConf.resolve(cfg)
        _save(root / "native-composed-config.json", OmegaConf.to_container(cfg, resolve=True))
        return cfg

    def initialized(agent, *args, **kwargs):
        observed["native_policy_initialization_started"] = True
        initialize(agent, *args, **kwargs)
        observed["agent"] = agent
        observed["native_policy_initialized"] = True
        observed["initial_critic_parameter_sha256"] = _parameter_hashes(agent.critic)

        def stepped(optimizer, args, kwargs):
            observed["optimizer_steps"] += 1

        observed["step_hook"] = agent.critic_optim.register_step_post_hook(stepped)

    def updated(agent, state, action, next_state, values, reward, not_done):
        value = update(agent, state, action, next_state, values, reward, not_done)
        observed["updates"].append(
            dict(
                loss=float(value),
                native_rewards=reward.detach().cpu().tolist(),
                native_not_done=not_done.detach().cpu().tolist(),
                native_update_step=agent._update_step,
                current_learning_rates=[float(g["lr"]) for g in agent.critic_optim.param_groups],
                alpha=float(agent.alpha),
            )
        )
        return value

    def saved(agent, checkpoint_path, verbose=False):
        return store(agent, checkpoint_path, verbose=verbose)

    hydra.compose, PQN.__init__, PQN.update, PQN.save = composed, initialized, updated, saved
    BertPredictor.forward = _counted_embedding_forward(embedding_forward, observed)
    try:
        yield
    finally:
        hydra.compose, PQN.__init__, PQN.update, PQN.save = compose, initialize, update, native_save
        BertPredictor.forward = embedding_forward
        if "step_hook" in observed:
            observed.pop("step_hook").remove()


class QRAGValuePolicyTraining:
    release_id = RELEASE

    def __init__(self, controls=None):
        self.controls = (
            QRAGFitControls(**controls)
            if isinstance(controls, Mapping)
            else controls or QRAGFitControls()
        )
        if not isinstance(self.controls, QRAGFitControls):
            raise ValidationError("Q-RAG fit requires typed controls")

    def fit(
        self,
        *,
        checkout,
        initial_checkpoint,
        initial_manifest,
        initial_manifest_sha256,
        train_file,
        dev_file,
        output_root,
        source_and_asset_acknowledged=False,
    ):
        if source_and_asset_acknowledged is not True:
            raise ValidationError(
                "Q-RAG fit requires explicit mounted-source/model/data acknowledgement"
            )
        c = self.controls
        source = validate_qrag_training_source(checkout)
        initial, _ = validate_qrag_initial_model(
            initial_checkpoint, initial_manifest, initial_manifest_sha256
        )
        audit = audit_qrag_training_data(train_file, dev_file, c)
        root = Path(output_root).expanduser().absolute()
        if root.exists() or root.is_symlink():
            raise ValidationError("Q-RAG fit output must be create-only; no consumed-attempt retry")
        if root.resolve().is_relative_to(source) or root.resolve().is_relative_to(initial):
            raise ValidationError("Q-RAG fit output must not mutate mounted source or assets")
        parent = root.parent
        while not parent.exists():
            parent = parent.parent
        if shutil.disk_usage(parent).free < c.min_free_output_bytes:
            raise ValidationError(
                "Q-RAG native checkpoints exceed the declared free-output-space guard"
            )
        versions = {
            n: importlib.metadata.version(n)
            for n in (
                "torch",
                "transformers",
                "numpy",
                "hydra-core",
                "omegaconf",
                "safetensors",
                "tensorboard",
            )
        }
        root.mkdir(parents=True)
        argv = [
            str(source / "train_q_rag.py"),
            "envs=hotpotqa",
            "algo=pqn_e5_hotpotqa",
            f"seed={c.seed}",
            f"device={c.device}",
            f"steps_count={c.steps_count}",
            f"batch_size={c.batch_size}",
            f"accumulate_grads={c.accumulate_grads}",
            f"envs_parallel={c.environments}",
            f"learning_start={c.learning_start}",
            f"eval_interval={c.eval_interval}",
            f"eval_episodes={c.eval_episodes}",
            f"algo.pqn.scheduler.warmup={c.warmup_steps}",
            f"logger.log_dir={json.dumps(str(root / 'native'))}",
        ]
        plan = dict(
            protocol=PROTOCOL,
            release_id=RELEASE,
            controls=asdict(c),
            split_audit=audit,
            source_commit=QRAG_UPSTREAM_COMMIT,
            source_files=SOURCE_FILES,
            initial_manifest_sha256=initial_manifest_sha256,
            dependency_versions=versions,
            native_argv=argv,
            native_reward="GroundTruthFeedback",
            dataset_transport_only=True,
            hyperparameters_are_explicit_configuration_not_paper_parity=True,
            source_load_unsafe_checkpoint_method_not_used=True,
            checkpoint_storage="native serialization once per unchanged update; hardlinked best/last/final paths",
            automatic_retry=False,
        )
        _save(root / "fit-plan.json", plan)
        observed = dict(
            optimizer_steps=0,
            updates=[],
            native_policy_initialization_started=False,
            native_policy_initialized=False,
            embedding_forward_attempts=0,
            embedding_forward_responses=0,
            checkpoint_storage_events=[],
        )
        start, cpu = time.monotonic(), time.process_time()
        try:
            with (
                _LOCK,
                _scope(root, c.seed),
                _source_import(source),
                _unused_vllm_import_compatibility(),
            ):
                import torch

                with _observe_native(root, source, initial, audit, c, observed):
                    sys.argv[:] = argv
                    with (
                        (root / "native-training.log").open("x") as log,
                        redirect_stdout(log),
                        redirect_stderr(log),
                    ):
                        native = runpy.run_path(str(source / "train_q_rag.py"), run_name="__main__")
                agent = native["agent"]
                if (
                    observed["optimizer_steps"] != c.steps_count
                    or len(observed["updates"]) != audit["native_updates"]
                ):
                    raise ValidationError(
                        "Q-RAG actual native updates differ from the bounded plan"
                    )
                if not all(
                    math.isfinite(x["loss"]) and math.isfinite(x["alpha"])
                    for x in observed["updates"]
                ):
                    raise ValidationError("Q-RAG native loss/temperature is non-finite")
                final = _parameter_hashes(agent.critic)
                changed = sum(
                    v != observed["initial_critic_parameter_sha256"][k] for k, v in final.items()
                )
                if not changed:
                    raise ValidationError("Q-RAG native fit produced no changed critic parameter")
                # The context restored PQN.save; reuse the already immutable
                # final native snapshot when evaluation saved this update.
                store = _NativeCheckpointStore(
                    root, type(agent).save, observed["checkpoint_storage_events"]
                )
                for event in observed["checkpoint_storage_events"]:
                    store.snapshots[event["native_update_step"]] = Path(
                        event["immutable_snapshot_path"]
                    )
                store(agent, str(root / "native-final.pt"))
                _save(
                    root / "parameter-proof.json",
                    dict(
                        initial=observed["initial_critic_parameter_sha256"],
                        final=final,
                        changed_critic_parameter_tensors=changed,
                        new_neural_forwards=0,
                    ),
                )
                peak = torch.cuda.max_memory_allocated() if c.device == "cuda:0" else 0
            terminal = dict(
                status="trained_candidate",
                optimizer_steps=observed["optimizer_steps"],
                embedding_forward_attempts=observed["embedding_forward_attempts"],
                embedding_forward_responses=observed["embedding_forward_responses"],
                checkpoint_storage_events=observed["checkpoint_storage_events"],
                updates=observed["updates"],
                changed_critic_parameter_tensors=changed,
                wall_seconds=time.monotonic() - start,
                cpu_seconds=time.process_time() - cpu,
                peak_cuda_allocated_bytes=peak,
                new_generative_model_calls=0,
                training_compute_is_separate_not_free=True,
                official_native_ground_truth_feedback=True,
                automatic_retry=False,
                published_score_parity=False,
            )
            _save(root / "fit-terminal.json", terminal)
            metadata = dict(
                protocol=PROTOCOL,
                release_id=RELEASE,
                files={
                    n: _sha(root / n)
                    for n in ("fit-plan.json", "fit-terminal.json", "native-final.pt")
                },
                native_final_candidate_not_released_checkpoint=True,
                paper_results_changed=False,
            )
            artifact_id = content_hash(metadata)
            _save(root / "artifact.json", dict(metadata, artifact_id=artifact_id))
            return QRAGTrainedPolicyArtifact(root.resolve(), artifact_id)
        except Exception as exc:
            if not (root / "fit-terminal.json").exists():
                _save(
                    root / "fit-terminal.json",
                    dict(
                        status="failed_terminal",
                        optimizer_steps=observed["optimizer_steps"],
                        embedding_forward_attempts=observed["embedding_forward_attempts"],
                        embedding_forward_responses=observed["embedding_forward_responses"],
                        native_policy_initialization_started=observed[
                            "native_policy_initialization_started"
                        ],
                        native_policy_initialized=observed["native_policy_initialized"],
                        updates=observed["updates"],
                        error_type=type(exc).__name__,
                        error=str(exc),
                        wall_seconds=time.monotonic() - start,
                        cpu_seconds=time.process_time() - cpu,
                        automatic_retry=False,
                    ),
                )
            raise


__all__ = [
    "QRAGFitControls",
    "QRAGTrainedPolicyArtifact",
    "QRAGValuePolicyTraining",
    "audit_qrag_training_data",
]
