from __future__ import annotations

import dataclasses
import unittest
from types import MappingProxyType

from agentinputlab.contracts import (
    ContextBlock,
    Evidence,
    EvidenceKind,
    ExecutionTrace,
    FrozenAgentState,
    InputPatch,
    InterventionContext,
    Message,
    PatchOp,
    PatchOperation,
    StateConflictError,
    TraceEvent,
    TracePhase,
)
from agentinputlab.contracts.serialization import canonical_json, content_hash


class ContractTests(unittest.TestCase):
    def test_add_message_accepts_frozen_mapping_and_typed_message(self) -> None:
        original = {"role": "assistant", "content": "saved answer", "name": "reader"}
        for value in (original, MappingProxyType(original), Message(**original)):
            with self.subTest(type=type(value).__name__):
                state = FrozenAgentState(query="q", messages=(Message("user", "q"),))
                patch = dataclasses.replace(
                    InputPatch.no_op(state),
                    operations=(PatchOperation(PatchOp.ADD_MESSAGE, value),),
                )
                before = patch.patch_hash
                output = patch.apply(state)
                self.assertEqual(output.messages, (*state.messages, Message(**original)))
                self.assertEqual(patch.patch_hash, before)
                self.assertEqual(len(state.messages), 1)

    def test_add_message_detaches_caller_mapping_and_rejects_non_message(self) -> None:
        source = {"role": "assistant", "content": "original"}
        state = FrozenAgentState(query="q")
        patch = dataclasses.replace(InputPatch.no_op(state), operations=(
            PatchOperation(PatchOp.ADD_MESSAGE, source),
        ))
        source['content'] = 'mutated'
        self.assertEqual(patch.apply(state).messages[0].content, 'original')
        invalid = dataclasses.replace(patch, operations=(PatchOperation(PatchOp.ADD_MESSAGE, 'raw'),))
        with self.assertRaisesRegex(ValueError, 'Message or mapping'):
            invalid.apply(state)

    def test_state_hash_is_order_independent_for_mappings(self) -> None:
        left = FrozenAgentState(query="q", metadata={"b": 2, "a": 1})
        right = FrozenAgentState(query="q", metadata={"a": 1, "b": 2})
        self.assertEqual(left.state_hash, right.state_hash)

    def test_state_is_frozen(self) -> None:
        state = FrozenAgentState(query="q")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            state.query = "changed"  # type: ignore[misc]

    def test_nested_state_metadata_is_detached_and_deeply_frozen(self) -> None:
        source = {"nested": {"values": [1, 2]}}
        state = FrozenAgentState(query="q", metadata=source)
        original_hash = state.state_hash
        source["nested"]["values"].append(3)
        self.assertEqual(state.state_hash, original_hash)
        self.assertEqual(state.metadata["nested"]["values"], (1, 2))
        with self.assertRaises(TypeError):
            state.metadata["nested"]["values"][0] = 9

    def test_state_rejects_caller_owned_untyped_context_and_messages(self) -> None:
        with self.assertRaisesRegex(ValueError, "ContextBlock"):
            FrozenAgentState(query="q", context=({"text": "mutable"},))
        with self.assertRaisesRegex(ValueError, "Message"):
            FrozenAgentState(query="q", messages=({"role": "user", "content": "mutable"},))

    def test_trace_and_patch_reject_untyped_nested_records(self) -> None:
        with self.assertRaisesRegex(TypeError, "TraceEvent"):
            ExecutionTrace(method_id="m", release_id="r", events=({"name": "mutable"},))
        state = FrozenAgentState(query="q")
        with self.assertRaisesRegex(ValueError, "PatchOperation"):
            InputPatch(
                method_id="m",
                release_id="r",
                source_state_hash=state.state_hash,
                operations=({"op": "no_op"},),
                trace=ExecutionTrace(method_id="m", release_id="r"),
                implementation_identity="fixture",
                profile_kind="fixture_unverified",
            )

    def test_intervention_context_run_config_is_deeply_frozen(self) -> None:
        source = {"decode": {"stop": ["END"]}}
        context = InterventionContext(run_config=source)
        source["decode"]["stop"].append("LATE")
        self.assertEqual(context.run_config["decode"]["stop"], ("END",))

    def test_intervention_context_rejects_invalid_seed_values(self) -> None:
        for seed in (True, -1, 1.5, "7"):
            with (
                self.subTest(seed=seed),
                self.assertRaisesRegex(ValueError, "non-negative integer"),
            ):
                InterventionContext(seed=seed)

    def test_canonical_serialization_rejects_sets(self) -> None:
        with self.assertRaises(TypeError):
            canonical_json({"unstable": {"a", "b"}})

    def test_patch_applies_in_order(self) -> None:
        state = FrozenAgentState(query="original")
        trace = ExecutionTrace(method_id="m", release_id="r")
        patch = InputPatch(
            method_id="m",
            release_id="r",
            source_state_hash=state.state_hash,
            operations=(
                PatchOperation(PatchOp.REPLACE_QUERY, "rewritten"),
                PatchOperation(
                    PatchOp.ADD_CONTEXT,
                    ContextBlock(text="evidence", block_id="d1"),
                ),
            ),
            trace=trace,
            implementation_identity="test",
            profile_kind="controlled_adapter",
        )
        output = patch.apply(state)
        self.assertEqual(output.query, "rewritten")
        self.assertEqual(output.context[0].block_id, "d1")
        self.assertEqual(state.query, "original")

    def test_patch_and_trace_detach_nested_metadata(self) -> None:
        mutable = {"nested": {"values": [1]}}
        state = FrozenAgentState(query="q")
        trace = ExecutionTrace(
            method_id="m",
            release_id="r",
            events=(
                TraceEvent(
                    phase=TracePhase.PROPOSE,
                    name="fixture",
                    payload=mutable,
                ),
            ),
        )
        patch = InputPatch(
            method_id="m",
            release_id="r",
            source_state_hash=state.state_hash,
            operations=(PatchOperation(PatchOp.SET_METADATA, mutable),),
            trace=trace,
            implementation_identity="fixture",
            profile_kind="controlled_adapter",
            metadata=mutable,
        )
        patch_hash = patch.patch_hash
        trace_hash = trace.trace_hash
        mutable["nested"]["values"].append(2)
        self.assertEqual(patch.patch_hash, patch_hash)
        self.assertEqual(trace.trace_hash, trace_hash)
        self.assertEqual(patch.apply(state).metadata["nested"]["values"], (1,))

    def test_stale_patch_is_rejected(self) -> None:
        source = FrozenAgentState(query="q1")
        other = FrozenAgentState(query="q2")
        patch = InputPatch.no_op(source)
        with self.assertRaises(StateConflictError):
            patch.apply(other)

    def test_explicit_no_op_does_not_change_hash(self) -> None:
        state = FrozenAgentState(query="q")
        self.assertEqual(InputPatch.no_op(state).apply(state).state_hash, state.state_hash)

    def test_synthetic_surrogate_is_not_citable(self) -> None:
        surrogate = Evidence(
            kind=EvidenceKind.SYNTHETIC_QUERY_SURROGATE,
            text="hypothetical answer",
            evidence_id="hyde-1",
        )
        self.assertFalse(surrogate.citable)

    def test_evidence_metadata_is_detached_and_deeply_frozen(self) -> None:
        metadata = {"source": {"pages": [1, 2]}}
        evidence = Evidence(
            kind=EvidenceKind.TOOL_OBSERVATION,
            text="observation",
            evidence_id="tool-1",
            metadata=metadata,
        )
        original_hash = content_hash(evidence)
        metadata["source"]["pages"].append(3)
        self.assertEqual(content_hash(evidence), original_hash)
        self.assertEqual(evidence.metadata["source"]["pages"], (1, 2))


if __name__ == "__main__":
    unittest.main()
