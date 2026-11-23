from __future__ import annotations

import unittest

from agentinputlab.algorithms.decompose.plan_and_budget import allocate_tokens
from agentinputlab.algorithms.decompose.plan_and_solve import (
    PLAN_AND_SOLVE_TRIGGER,
    render_plan_and_solve_prompt,
)
from agentinputlab.algorithms.decompose.self_ask import (
    SelfAskConfig,
    SelfAskLoop,
)
from agentinputlab.algorithms.reformulate.query2doc import compose_sparse_expansion
from agentinputlab.backends.llm.callable import ReplayBackend
from agentinputlab.backends.retrieval.base import Document
from agentinputlab.backends.retrieval.memory import InMemoryBM25Retriever
from agentinputlab.contracts import FrozenAgentState, InterventionContext


class AlgorithmGoldenTests(unittest.TestCase):
    def test_query2doc_five_repeat_golden(self) -> None:
        self.assertEqual(
            compose_sparse_expansion("when", "a hypothetical document"),
            "when when when when when a hypothetical document",
        )

    def test_plan_and_solve_prompt_201_golden(self) -> None:
        self.assertEqual(
            render_plan_and_solve_prompt("problem"),
            f"Q:problem\nA: {PLAN_AND_SOLVE_TRIGGER}",
        )

    def test_plan_and_budget_allocation_golden(self) -> None:
        self.assertEqual(
            allocate_tokens((15, 15, 20, 20, 10), 100, "constant"),
            (18, 18, 26, 26, 12),
        )

    def test_self_ask_trajectory_golden(self) -> None:
        demonstrations = (
            "Question: Demo?\nAre follow up questions needed here: No.\n"
            "So the final answer is: Demo."
        )
        result = SelfAskLoop(SelfAskConfig(demonstrations=demonstrations)).run(
            FrozenAgentState(query="Composite question?"),
            InterventionContext(
                backend=ReplayBackend(
                    (
                        "Yes.\nFollow up: Atomic question?",
                        "So the final answer is: final",
                    )
                ),
                retriever=InMemoryBM25Retriever((Document(doc_id="doc-1", text="atomic"),)),
            ),
        )
        self.assertEqual(
            result.transcript,
            f"{demonstrations}\n\n"
            "Question: Composite question?\n"
            "Are follow up questions needed here: Yes.\n"
            "Follow up: Atomic question?\n"
            "Intermediate answer: atomic\n"
            "So the final answer is: final",
        )
        self.assertEqual(result.final_answer, "final")


if __name__ == "__main__":
    unittest.main()
