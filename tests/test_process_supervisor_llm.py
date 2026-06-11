"""Integration test for ProcessSupervisor with a real LLM call.

Uses the project's configured LLM (Anthropic proxy at localhost:6655 per
.env) — no mocks. Builds a minimal in-memory process model, feeds it a
hardcoded order_agent message, and asserts the supervisor lands on either
an Execution/Termination line (happy path) or a Violation line (off-model).
"""
import unittest

from langchain_core.messages import AIMessage

from src.control_plane.process_supervisor import ProcessSupervisor
from src.llm import create_chat_llm


_DESCRIPTION = """\
Tiny coffee-shop process for testing.

Lanes:
- order_agent: handles customer requests.

Activities:
- A01 Identify Customer Request (order_agent, message): the agent reads the
  customer's incoming message and acknowledges what they want to order.
- A02 Create Order (order_agent, tool_call=process_order): the agent calls
  the process_order tool to record the order.

There is no other activity. Any other behaviour is a violation.
"""


def _write_minimal_model(tmp_path):
    yaml_path = tmp_path / "model.yaml"
    desc_path = tmp_path / "description.md"
    desc_path.write_text(_DESCRIPTION, encoding="utf-8")
    yaml_path.write_text(
        f"""\
name: test_minimal_v1
description_source: {desc_path.name}
activities:
  - id: A01
    name: identify_customer_request
    display_name: Identify Customer Request
    agent: order_agent
    trigger: message
    terminal: false
  - id: A02
    name: create_order
    display_name: Create Order
    agent: order_agent
    trigger: tool_call
    tool: process_order
    terminal: false
""",
        encoding="utf-8",
    )
    return yaml_path


class TestProcessSupervisorLLM(unittest.TestCase):
    """Real-LLM integration: happy path + violation path."""

    @classmethod
    def setUpClass(cls):
        cls.llm = create_chat_llm()

    def _make_supervisor(self, tmp_dir):
        model_path = _write_minimal_model(tmp_dir)
        log_path = tmp_dir / "process.log"
        return ProcessSupervisor(
            process_model_path=model_path,
            log_path=log_path,
            llm=self.llm,
        )

    def test_happy_path_identifies_a01(self):
        """A clear 'Identify Customer Request'-style message → Execution:A01."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            sup = self._make_supervisor(Path(td))
            msg = AIMessage(
                content=(
                    "Hi! I see you'd like a cappuccino and a croissant. "
                    "Let me get that order started for you."
                )
            )
            line = sup.observe(msg, agent_name="order_agent")

            self.assertIsNotNone(line, "supervisor returned None for an AI message")
            head = line.split(" | ", 1)[0]
            self.assertTrue(
                head.startswith("Execution:A01:") or head.startswith("Termination:A01:"),
                f"expected Execution/Termination on A01, got: {head!r}",
            )

    def test_violation_path_off_model_message(self):
        """A message with nothing to do with the catalog → Violation."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            sup = self._make_supervisor(Path(td))
            msg = AIMessage(
                content=(
                    "I just finished writing a haiku about autumn leaves "
                    "drifting through the park at sunset."
                )
            )
            line = sup.observe(msg, agent_name="order_agent")

            self.assertIsNotNone(line, "supervisor returned None for an AI message")
            head = line.split(" | ", 1)[0]
            self.assertTrue(
                head.startswith("Violation:"),
                f"expected Violation, got: {head!r}",
            )

    def test_critique_returns_grounded_prose(self):
        """supervisor.critique(off-model AIMessage, ...) returns natural-language
        guidance grounded in the allowed activities and is NOT a verdict line."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            sup = self._make_supervisor(Path(td))
            off_msg = AIMessage(
                content=(
                    "I just finished writing a haiku about autumn leaves "
                    "drifting through the park at sunset."
                )
            )
            text = sup.critique(
                off_msg,
                agent_name="order_agent",
                violation_reason="llm_unknown_activity_A99",
            )

            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 20, "critique should be more than a stub")
            for prefix in ("Execution:", "Termination:", "Violation:"):
                self.assertFalse(
                    text.lstrip().startswith(prefix),
                    f"critique must not look like a passive-log line, got: {text!r}",
                )
            # Either the slug or display name of one of the allowed activities
            # for order_agent should appear, demonstrating BPMN grounding.
            allowed = sup.allowed_next_activities_for("order_agent")
            self.assertTrue(allowed, "model should expose order_agent activities")
            tokens = []
            for a in allowed:
                tokens.append(a.name)
                if a.display_name:
                    tokens.append(a.display_name)
                    tokens.append(a.display_name.lower())
            self.assertTrue(
                any(t in text or t.lower() in text.lower() for t in tokens),
                f"critique should reference at least one allowed activity; "
                f"got: {text!r}; tokens checked: {tokens}",
            )


if __name__ == "__main__":
    unittest.main()
