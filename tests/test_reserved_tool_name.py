"""Tests for reserved-name enforcement in `resolve_tools`.

The synthetic `assistant_message` pseudo tool call is emitted by the response
guardrail's gateway node and must never collide with a real registered tool.
`resolve_tools` refuses to build an agent whose YAML declares the reserved
name.
"""
import unittest

from src.control_plane.subgraph import RESPONSE_GUARDRAIL_TOOL_NAME
from src.control_plane.tool_registry import resolve_tools


class TestReservedToolName(unittest.TestCase):
    def test_reserved_tool_name_raises(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_tools([RESPONSE_GUARDRAIL_TOOL_NAME])
        self.assertIn(RESPONSE_GUARDRAIL_TOOL_NAME, str(ctx.exception))
        self.assertIn("reserved", str(ctx.exception).lower())

    def test_reserved_name_alongside_real_tool_raises(self):
        with self.assertRaises(ValueError):
            resolve_tools(["process_order", RESPONSE_GUARDRAIL_TOOL_NAME])

    def test_normal_tools_resolve(self):
        tools = resolve_tools(["process_order", "calculate_total"])
        self.assertEqual(len(tools), 2)


if __name__ == "__main__":
    unittest.main()
