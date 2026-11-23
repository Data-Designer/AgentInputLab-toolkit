"""Execution contracts for the pinned native GRPO driver, not reward math.

The native manager omits ``reward_extra_info`` while its PPO driver indexes
that key and catches failures by invoking the manager again. Complete the
dictionary without another scoring pass, and latch failures across the whole
batch so a fallback cannot repeat already consumed helper calls.
"""
from __future__ import annotations

from collections.abc import Mapping

from agentinputlab.contracts import ExternalDependencyError, ValidationError


class NativeGRPORewardManagerContract:
    """Keep native rewards unchanged and prohibit automatic batch retries.

    This adapter does not itself load a model or certify a GRPO fit. The
    execution worker must construct its manager from the verified native
    source, bind the metered scorer, and retain original per-item costs.
    """

    def __init__(self, native_manager):
        if not callable(native_manager):
            raise ValidationError("SmartSearch GRPO requires a callable native reward manager")
        self.native_manager = native_manager
        self.native_calls = 0
        self._failed = False

    def __call__(self, data, return_dict=False, curr_save_path=None):
        if self._failed:
            raise ExternalDependencyError("SmartSearch native reward batch failed; no automatic scoring retry")
        if type(return_dict) is not bool:
            raise ValidationError("SmartSearch reward dictionary flag must be a boolean")
        try:
            self.native_calls += 1
            result = self.native_manager(data, return_dict=return_dict, curr_save_path=curr_save_path)
            if not return_dict:
                return result
            if not isinstance(result, Mapping):
                # The native manager's explicit cached-rm branch returns the
                # exact cached tensor, even when return_dict=True.
                batch = getattr(data, "batch", None)
                cached = batch.get("rm_scores") if hasattr(batch, "get") else None
                if cached is None or result is not cached:
                    raise ExternalDependencyError("SmartSearch native reward dictionary contract differs")
                result = {"reward_tensor": result}
            if "reward_tensor" not in result or result["reward_tensor"] is None:
                raise ExternalDependencyError("SmartSearch native reward tensor is missing")
            extras = result.get("reward_extra_info", {})
            if not isinstance(extras, Mapping):
                raise ExternalDependencyError("SmartSearch native reward extra-info mapping differs")
            # Preserve the original tensor object and existing extra fields.
            # No tensor arithmetic, rescaling, default positive or re-score.
            return {**result, "reward_extra_info": extras}
        except BaseException:
            self._failed = True
            raise
