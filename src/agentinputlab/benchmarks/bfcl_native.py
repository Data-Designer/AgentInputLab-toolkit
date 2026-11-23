"""Source-verified BFCL v1.3 Python AST scoring for local FC transports.

The original checker, tool-schema converter and OpenAI FC decoder are loaded
from the checksum-pinned archive, not reimplemented or vendored. Only their
required Python definitions execute. Commercial API handlers, model inference,
Java/JS parsers and the unsafe prompting decoder are not loaded. This is a
case-level Python AST evaluation, not the aggregate BFCL leaderboard workflow.
"""

from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import json
import re
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

from agentinputlab.contracts import ArtifactError, ValidationError
from agentinputlab.contracts.serialization import to_jsonable

from .bfcl_v3 import (
    BFCL_V3_AST_CHECKER_REVISION,
    BFCL_V3_OFFICIAL_SOURCE_REVISION,
    BFCL_V3_SOURCE_ARCHIVE_SHA256,
    BFCLV3EvaluationResult,
)

BFCL_NATIVE_SCOPE = "source_verified_python_ast_openai_fc"
BFCL_NATIVE_SCORER_ID = "bfcl_eval.eval_checker.ast_eval.ast_checker:ast_checker"
_PREFIX = "gorilla-1.3/berkeley-function-call-leaderboard/"
_SOURCES = {
    "bfcl_eval.eval_checker.ast_eval.ast_checker": BFCL_V3_AST_CHECKER_REVISION,
    "bfcl_eval.constants.type_mappings": "1702fb67afbe2c492608e58e2b7d02e46381f50166b47f3c952f76e34c7cd3bd",
    "bfcl_eval.constants.model_config": "26bff7e39e1ea018c5fd5a55882a0020a98f63f76068ece29e68f4ff6e8c4cb1",
    "bfcl_eval.utils": "255e6dad9a6533c8e52b4c8b2bfc78c0744b915605e09c04808df6c023f4a43f",
    "bfcl_eval.model_handler.model_style": "0244f6108e85b40379d7ddd80d4f7b699a270972c4d8cf37966a4aa677f8ba57",
    "bfcl_eval.model_handler.utils": "153bb33a9515ea2d6ee6a1387faaa464c7be4fe9c6a9f88cc17ad23a1cc306d2",
    "bfcl_eval.model_handler.api_inference.openai_completion": "b84c67a86190e1b8dbc38e72cb5f175ad106b1017cf14f19655318f3379ba00a",
    "bfcl_eval.eval_checker.ast_eval.type_convertor.java_type_converter": "2fd4f4b0443b3dd974a1723bb4e45c086d7b352631062da7807ad1ad40706604",
    "bfcl_eval.eval_checker.ast_eval.type_convertor.js_type_converter": "a114e9ff75c025cb52787ac33d6c2fbaa390905c6125a2b3c6afebab232bb5e4",
}


@dataclass(frozen=True, slots=True)
class SourceVerifiedBFCLV3Result(BFCLV3EvaluationResult):
    provenance_status: str = field(init=False, default="source_verified")
    evaluation_scope: str = field(init=False, default=BFCL_NATIVE_SCOPE)
    # The full official CLI/aggregate workflow is intentionally not asserted.
    official_runtime_verified: bool = field(init=False, default=False)
    diagnostic: Mapping[str, Any] = field(default_factory=dict)


class _PinnedPythonDefinitions:
    """Isolated source namespace; no upstream package is installed globally."""

    def __init__(self, archive, model_id):
        self.archive = archive
        self.model_id = model_id
        self.modules = {}
        self.hashes = {}

    def source(self, name):
        path = _PREFIX + name.replace(".", "/") + ".py"
        member = self.archive.getmember(path)
        if not member.isfile() or member.size > 2 * 1024 * 1024:
            raise ArtifactError("BFCL source member is not a bounded regular file")
        raw = self.archive.extractfile(member).read()
        observed = hashlib.sha256(raw).hexdigest()
        expected = _SOURCES.get(name)
        if expected is not None and observed != expected:
            raise ArtifactError("BFCL source definition checksum differs")
        self.hashes[name] = observed
        return raw.decode()

    def imported(self, name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("bfcl_eval."):
            return self.module(name)
        return builtins.__import__(name, globals, locals, fromlist, level)

    def module(self, name):
        if name in self.modules:
            return self.modules[name]
        if name not in _SOURCES:
            raise ArtifactError("BFCL scorer requested an unapproved source module")
        module = ModuleType("_agentinputlab_native_" + name)
        self.modules[name] = module
        namespace = module.__dict__
        namespace["__builtins__"] = {**vars(builtins), "__import__": self.imported}
        source = self.source(name)
        if name == "bfcl_eval.constants.model_config":
            # The native checker only needs the transport's name-normalization
            # flag. Register the ACTUAL operator model ID for the standard FC
            # protocol; do not label it as an upstream leaderboard model.
            namespace["MODEL_CONFIG_MAPPING"] = {
                self.model_id.replace("_", "/"): SimpleNamespace(underscore_to_dot=True)
            }
        elif name == "bfcl_eval.model_handler.utils":
            tree = ast.parse(source)
            selected = [
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef)
                and n.name in {"_cast_to_openai_type", "convert_to_tool"}
            ]
            if len(selected) != 2:
                raise ArtifactError("BFCL native tool converter definitions differ")
            namespace.update(
                copy=copy,
                re=re,
                json=json,
                ModelStyle=self.module("bfcl_eval.model_handler.model_style").ModelStyle,
                GORILLA_TO_OPENAPI=self.module(
                    "bfcl_eval.constants.type_mappings"
                ).GORILLA_TO_OPENAPI,
            )
            exec(compile(ast.Module(body=selected, type_ignores=[]), name, "exec"), namespace)
        elif name == "bfcl_eval.model_handler.api_inference.openai_completion":
            tree = ast.parse(source)
            parent = next(
                n
                for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == "OpenAICompletionsHandler"
            )
            decode = next(
                n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == "decode_ast"
            )
            # Keep the exact upstream FC branch; no API constructor or unsafe
            # eval-based prompting decoder is reachable on this bound object.
            namespace["json"] = json
            exec(compile(ast.Module(body=[decode], type_ignores=[]), name, "exec"), namespace)
        elif name == "bfcl_eval.utils":
            tree = ast.parse(source)
            validate = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "is_function_calling_format_output"
            )
            exec(compile(ast.Module(body=[validate], type_ignores=[]), name, "exec"), namespace)
        else:
            exec(compile(source, name, "exec"), namespace)
        return module


class SourceVerifiedBFCLV3ASTBridge:
    bfcl_bridge_contract = "bfcl_v3_scorer_bridge_v1"
    benchmark_id = "bfcl.v3.simple_python"
    source_revision = BFCL_V3_OFFICIAL_SOURCE_REVISION
    scorer_id = BFCL_NATIVE_SCORER_ID
    scorer_revision = BFCL_V3_AST_CHECKER_REVISION
    ast_checker_revision = BFCL_V3_AST_CHECKER_REVISION
    model_handler_id = "bfcl_eval.model_handler.api_inference.openai_completion:OpenAICompletionsHandler.decode_ast.FC"
    model_handler_revision = _SOURCES["bfcl_eval.model_handler.api_inference.openai_completion"]
    bridge_id = "agentinputlab.bfcl.source_verified_python_fc.v1"
    bridge_revision = hashlib.sha256(
        b"BFCL-v1.3-pinned-Python-AST+native-FC-decoder+native-schema-converter-v1"
    ).hexdigest()
    test_category = "simple"
    language = "Python"
    provenance_status = "source_verified"
    evaluation_scope = BFCL_NATIVE_SCOPE
    official_runtime_verified = False

    def __init__(self, *, source_archive: str | Path, model_id: str):
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValidationError("BFCL scorer requires the actual model ID")
        archive = Path(source_archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != BFCL_V3_SOURCE_ARCHIVE_SHA256:
            raise ArtifactError("BFCL official source archive checksum differs")
        with tarfile.open(archive, "r:gz") as tar:
            loader = _PinnedPythonDefinitions(tar, model_id)
            self._checker = loader.module("bfcl_eval.eval_checker.ast_eval.ast_checker").ast_checker
            self._decoder = loader.module(
                "bfcl_eval.model_handler.api_inference.openai_completion"
            ).decode_ast
            self._format_valid = loader.module("bfcl_eval.utils").is_function_calling_format_output
            self._utilities = loader.module("bfcl_eval.model_handler.utils")
            self.source_hashes = dict(loader.hashes)
        self.model_id = model_id
        self._handler = SimpleNamespace(model_name=model_id, is_fc_model=True)
        self._verified_source_sha256 = BFCL_V3_SOURCE_ARCHIVE_SHA256

    def tools(self, function: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        tools = self._utilities.convert_to_tool(
            [to_jsonable(function)],
            self._utilities.GORILLA_TO_OPENAPI,
            self._utilities.ModelStyle.OpenAI_Completions,
        )
        return tuple(tools)

    def evaluate(
        self, evaluator_payload: Mapping[str, Any], prediction: Any
    ) -> SourceVerifiedBFCLV3Result:
        if (
            evaluator_payload.get("language") != "Python"
            or evaluator_payload.get("test_category") != "simple"
        ):
            raise ValidationError("native BFCL bridge supports Python simple only")
        try:
            if not isinstance(prediction, (list, tuple)):
                raise ValueError("FC result must be a function-call sequence")
            decoded = self._decoder(self._handler, to_jsonable(prediction), language="Python")
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            return SourceVerifiedBFCLV3Result(
                0.0,
                diagnostic={
                    "error_type": "ast_decoder:decoder_failed",
                    "decoder_error": type(exc).__name__,
                },
            )
        if not self._format_valid(decoded):
            return SourceVerifiedBFCLV3Result(
                0.0, diagnostic={"error_type": "ast_decoder:decoder_wrong_output_format"}
            )
        result = self._checker(
            [to_jsonable(evaluator_payload["function"])],
            decoded,
            to_jsonable(evaluator_payload["ground_truth"]),
            "Python",
            "simple",
            self.model_id,
        )
        return SourceVerifiedBFCLV3Result(
            float(bool(result["valid"])), diagnostic=to_jsonable(result)
        )

    def fingerprint(self):
        return {
            "source_archive_sha256": self._verified_source_sha256,
            "source_revision": self.source_revision,
            "model_id": self.model_id,
            "transport": "openai_compatible_function_calling",
            "name_normalization": "native_dots_to_underscores",
            "source_hashes": dict(self.source_hashes),
            "source_code_verified": True,
            "official_runtime_verified": False,
            "scope": self.evaluation_scope,
            "aggregate_leaderboard_claimed": False,
        }
