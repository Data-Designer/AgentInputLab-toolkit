"""Explicit caller-owned EnvAdapter extension for the pinned SkillOpt trainer.

The factory is trusted Python code, not a sandbox. Source and configuration are
checked before import. Native training, reflection, selection and CLI controls
remain in the caller-mounted upstream checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentinputlab.artifacts import load_pinned_json_mapping
from agentinputlab.contracts import ExternalDependencyError, ValidationError
from agentinputlab.contracts.serialization import content_hash, to_jsonable

ENVIRONMENT_RECEIPT = "agentinputlab_environment_binding.json"
_SHA = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class SkillOptEnvironmentBinding:
    factory_path: str | Path
    factory_sha256: str
    configuration_path: str | Path
    configuration_sha256: str
    factory_name: str = "create"

    def __post_init__(self) -> None:
        for name in ("factory_sha256", "configuration_sha256"):
            if not isinstance(getattr(self, name), str) or not _SHA.fullmatch(getattr(self, name)):
                raise ValidationError(f"SkillOpt environment {name} must be a SHA-256")
        for name in ("factory_path", "configuration_path"):
            if not isinstance(getattr(self, name), (str, Path)) or not str(getattr(self, name)).strip():
                raise ValidationError(f"SkillOpt environment {name} must be a path")
        if (not isinstance(self.factory_name, str) or not self.factory_name.isidentifier()
                or self.factory_name.startswith("_")):
            raise ValidationError("SkillOpt environment factory_name must be a public identifier")

    def verify(self) -> dict[str, Any]:
        factory = Path(self.factory_path).expanduser().absolute()
        if (factory.is_symlink() or not factory.is_file() or factory.suffix != ".py"
                or factory.stat().st_size > 1024 * 1024):
            raise ExternalDependencyError("SkillOpt environment factory must be a bounded regular Python file")
        if hashlib.sha256(factory.read_bytes()).hexdigest() != self.factory_sha256:
            raise ExternalDependencyError("SkillOpt environment factory checksum changed")
        config = Path(self.configuration_path).expanduser().absolute()
        load_pinned_json_mapping(config, expected_sha256=self.configuration_sha256,
                                 max_bytes=8 * 1024 * 1024)
        return dict(factory_path=str(factory.resolve()), factory_sha256=self.factory_sha256,
                    configuration_path=str(config.resolve()),
                    configuration_sha256=self.configuration_sha256, factory_name=self.factory_name)

    def command_options(self) -> tuple[str, ...]:
        return ("--factory", str(Path(self.factory_path).expanduser().absolute()),
                "--factory-sha256", self.factory_sha256,
                "--factory-name", self.factory_name,
                "--factory-config", str(Path(self.configuration_path).expanduser().absolute()),
                "--factory-config-sha256", self.configuration_sha256)


def run_with_environment(*, checkout, binding: SkillOptEnvironmentBinding,
                         receipt_path, native_args: list[str]) -> None:
    from .skillopt import validate_skillopt_checkout

    root = validate_skillopt_checkout(checkout)
    identity = binding.verify()
    configuration = load_pinned_json_mapping(binding.configuration_path,
        expected_sha256=binding.configuration_sha256, max_bytes=8 * 1024 * 1024)
    receipt = Path(receipt_path)
    if receipt.exists() or receipt.is_symlink() or not receipt.parent.is_dir():
        raise ValidationError("SkillOpt environment receipt must be a new file in the fit output")
    prior_path, prior_argv = list(sys.path), list(sys.argv)
    native_name = "_agentinputlab_skillopt_cli_" + content_hash(str(root))[:16]
    factory_name = "_agentinputlab_skillopt_factory_" + binding.factory_sha256[:16]
    previous = {name: sys.modules.get(name) for name in (native_name, factory_name)}
    try:
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location(native_name, root / "scripts/train.py")
        if spec is None or spec.loader is None:
            raise ExternalDependencyError("SkillOpt native CLI cannot be imported")
        native = importlib.util.module_from_spec(spec)
        sys.modules[native_name] = native
        spec.loader.exec_module(native)
        spec = importlib.util.spec_from_file_location(factory_name, identity["factory_path"])
        if spec is None or spec.loader is None:
            raise ExternalDependencyError("SkillOpt environment factory cannot be imported")
        factory_module = importlib.util.module_from_spec(spec)
        sys.modules[factory_name] = factory_module
        spec.loader.exec_module(factory_module)
        factory = getattr(factory_module, binding.factory_name, None)
        if not callable(factory):
            raise ValidationError("SkillOpt environment factory is not callable")
        from skillopt.envs.base import EnvAdapter

        bound = False

        def get_adapter(cfg):
            nonlocal bound
            if bound:
                raise ValidationError("SkillOpt native CLI requested a second environment binding")
            binding.verify()
            adapter = factory(cfg=dict(cfg), configuration=to_jsonable(configuration))
            if not isinstance(adapter, EnvAdapter) or not callable(getattr(adapter, "fingerprint", None)):
                raise ValidationError("SkillOpt factory must return a native EnvAdapter with a fingerprint")
            fingerprint = to_jsonable(adapter.fingerprint())
            if not isinstance(fingerprint, dict) or not fingerprint:
                raise ValidationError("SkillOpt environment fingerprint must be a nonempty mapping")
            content_hash(fingerprint)
            binding.verify()
            value = dict(schema="agentinputlab.skillopt.environment_binding.v1",
                         binding=identity, environment_fingerprint=fingerprint,
                         native_training_algorithm_unchanged=True,
                         profile_kind="caller_environment_adapter")
            with receipt.open("x", encoding="utf-8") as stream:
                json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            bound = True
            return adapter

        native.get_adapter = get_adapter
        sys.argv = [str(root / "scripts/train.py"), *native_args]
        native.main()
        if not bound:
            raise ValidationError("SkillOpt native CLI did not instantiate the configured environment")
        binding.verify()
        validate_skillopt_checkout(root)
    finally:
        sys.path[:] = prior_path
        sys.argv[:] = prior_argv
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--" not in args:
        raise ValidationError("SkillOpt environment wrapper requires explicit native arguments after --")
    boundary = args.index("--")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--factory", required=True)
    parser.add_argument("--factory-sha256", required=True)
    parser.add_argument("--factory-name", default="create")
    parser.add_argument("--factory-config", required=True)
    parser.add_argument("--factory-config-sha256", required=True)
    parser.add_argument("--receipt", required=True)
    a = parser.parse_args(args[:boundary])
    binding = SkillOptEnvironmentBinding(a.factory, a.factory_sha256, a.factory_config,
                                        a.factory_config_sha256, a.factory_name)
    run_with_environment(checkout=a.checkout, binding=binding,
                         receipt_path=a.receipt, native_args=args[boundary + 1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
