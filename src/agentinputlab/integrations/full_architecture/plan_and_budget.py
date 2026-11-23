"""Source-pinned native P&B prompt construction, with no provider side effects.

Only the released allocation functions and reasoning prompt are evaluated. The
native dotenv/client setup, gold-aware evaluator and retry loop are not imported.
The composite adapter owns the single, observed solver call after this boundary.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import textwrap
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from agentinputlab.algorithms.decompose.plan_and_budget.cached_soft_budget import (
    P_AND_B_UPSTREAM_REVISION,
    CachedPlanBudgetRecord,
    DecaySchedule,
)
from agentinputlab.contracts import ArtifactError, ExternalDependencyError, ValidationError

PAB_NATIVE_PROMPT_MODE = "source_pinned_weighted_prompt.v1"
PAB_SOURCE_HASHES = {
    "inference/_5_planned_local_weighted_model.py":
        "c4463d2a4e14882d4ff2b060b7b9cc9a93ecc7732ace5814dc72f1432c62c5ed",
    "utils/utils.py":
        "9c448f585251e94302e47c0c87ca61861dc5069d6255d7e106d28c4ca6610fc7",
}
_FUNCTIONS = {
    "allocate_tokens", "constant_schedule", "linear_decay", "polynomial_decay",
    "exponential_decay", "cosine_anneal",
}
_SCHEDULES = {
    "constant": "constant_schedule", "linear": "linear_decay",
    "polynomial": "polynomial_decay", "exponential": "exponential_decay",
    "cosine": "cosine_anneal",
}


@dataclass(frozen=True, slots=True)
class PlanBudgetPromptRequest:
    record: CachedPlanBudgetRecord
    instruction: str
    reference: str
    output_format: str
    schedule: str

    def __post_init__(self):
        if not isinstance(self.record, CachedPlanBudgetRecord):
            raise ValidationError("P&B prompt requires a solver-only cached record")
        if not all(isinstance(x, str) for x in (self.instruction, self.reference)):
            raise ValidationError("P&B instruction/reference must be strings")
        if not isinstance(self.output_format, str) or not self.output_format.strip():
            raise ValidationError("P&B requires an explicit output format")
        object.__setattr__(self, "schedule", DecaySchedule.parse(self.schedule).value)


class _PromptBoundary(Exception):
    def __init__(self, messages):
        self.messages = messages


class PlanAndBudgetNativePromptRuntime:
    """Execute the hash-pinned native reasoning prefix up to its provider call.

    This runtime does not call a model, return a fake response, execute a source
    evaluator, or load the source module's environment/credentials. Source assets
    and NumPy are explicit optional dependencies; nothing is downloaded here.
    """

    worker_id = "agentinputlab.plan-and-budget.native-prompt.v1"

    def __init__(self, *, source_root: str | Path):
        self.source_root = Path(source_root).expanduser().resolve()

    def _sources(self):
        sources = {}
        for name, expected in PAB_SOURCE_HASHES.items():
            path = self.source_root / name
            try:
                if path.stat().st_size > 128 * 1024:
                    raise ArtifactError("P&B native source exceeds the size limit")
                raw = path.read_bytes()
            except OSError as exc:
                raise ArtifactError(f"P&B native source unavailable: {name}") from exc
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ArtifactError(f"P&B native source SHA-256 differs: {name}")
            sources[name] = raw.decode("utf-8")
        return sources

    def fingerprint(self):
        self._sources()
        try:
            numpy = importlib.import_module("numpy")
        except ImportError as exc:
            raise ExternalDependencyError(
                "P&B native solver requires agentinputlab[plan-and-budget]"
            ) from exc
        return {
            "runtime_id": self.worker_id,
            "upstream_revision": P_AND_B_UPSTREAM_REVISION,
            "source_sha256": dict(PAB_SOURCE_HASHES),
            "numpy_version": numpy.__version__,
            "binding_mode": PAB_NATIVE_PROMPT_MODE,
            "model_calls": 0,
            "native_retry_decorator": "removed_at_explicit_provider_boundary",
            "provider_binding": "caller_observed_chat_backend",
        }

    def run(self, request: PlanBudgetPromptRequest):
        if not isinstance(request, PlanBudgetPromptRequest):
            raise ValidationError("P&B native prompt requires PlanBudgetPromptRequest")
        identity = self.fingerprint()
        sources = self._sources()
        source = ast.parse(sources["inference/_5_planned_local_weighted_model.py"])
        helpers = [n for n in source.body if isinstance(n, ast.FunctionDef)
                   and n.name in _FUNCTIONS]
        native_class = next(n for n in source.body if isinstance(n, ast.ClassDef)
                            and n.name == "PlannedLocalWeighted_Model")
        prompt = next(n for n in native_class.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "_prompt"
                              for t in n.targets))
        reasoning = copy.deepcopy(next(
            n for n in native_class.body if isinstance(n, ast.AsyncFunctionDef)
            and n.name == "reasoning"
        ))
        # The source body is unchanged. Do not import or invoke its ten-attempt
        # retry decorator around a boundary that may already have spent tokens.
        reasoning.decorator_list = []
        utility = ast.parse(sources["utils/utils.py"])
        level_map = ast.literal_eval(next(
            n.value for n in utility.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "level_map" for t in n.targets)
        ))
        logger = SimpleNamespace(debug=lambda *_args: None)

        async def capture(messages, *, logger, return_raw):
            del logger
            if return_raw is not True:
                raise ArtifactError("P&B native provider boundary differs")
            raise _PromptBoundary(messages)

        namespace = {
            "np": importlib.import_module("numpy"), "textwrap": textwrap,
            "Query": SimpleNamespace, "level_map": level_map, "logger": logger,
            "MODEL_NAME": "agentinputlab-controlled-chat",
            "COMPLETION_LLM": set(), "REASONING_LLM": set(),
            "CHAT_LLM": {"agentinputlab-controlled-chat"}, "generate_response": capture,
        }
        nodes = [*helpers, prompt, reasoning]
        exec(compile(ast.Module(body=nodes, type_ignores=[]),
                     "pinned-p-and-b-reasoning-prefix", "exec"), namespace)
        instance = SimpleNamespace(
            _prompt=namespace["_prompt"], instruction=request.instruction,
            output_format=request.output_format, decay=request.schedule, logger=logger,
        )
        record = request.record
        query = SimpleNamespace(description=record.query, reference=request.reference,
                                steps=list(record.steps), credits=list(record.credits),
                                level=record.level)
        coroutine = namespace["reasoning"](instance, query)
        try:
            # capture() has no await/yield: this enters the genuine native prefix
            # and stops at its first provider call, without an asyncio event loop.
            coroutine.send(None)
        except _PromptBoundary as boundary:
            messages = boundary.messages
        else:  # pragma: no cover - fixed source invariant
            raise ArtifactError("P&B native reasoning did not reach its provider boundary")
        finally:
            coroutine.close()
        total_words = level_map[record.level][1]
        allocations = tuple(int(x) for x in namespace["allocate_tokens"](
            record.credits, total_words, namespace[_SCHEDULES[request.schedule]]
        ))
        return {
            "messages": messages, "allocations": allocations,
            "soft_word_budget": total_words, "identity": identity,
            "usage": {"model_calls": 0, "retrieval_calls": 0,
                      "input_tokens": 0, "output_tokens": 0},
        }


__all__ = ["PlanAndBudgetNativePromptRuntime", "PlanBudgetPromptRequest"]
