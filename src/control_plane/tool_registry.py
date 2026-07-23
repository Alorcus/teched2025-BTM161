"""Single Python registry that resolves tool names (from YAML AgentDefinitions)
into the concrete `@tool`-decorated callables that already live in `src/agents/`.

Only tool *bindings* live here; tool *implementations* stay in their respective
agent files so the existing imports keep working.
"""
from langchain_core.tools import BaseTool

from src.agents.shared_components import transfer_to_agent
from src.agents.order_agent import process_order, calculate_total
from src.agents.inventory_agent import (
    check_inventory, update_stock, get_alternatives,
)
from src.agents.barista_agent import (
    start_preparation, end_preparation, estimate_prep_time, clean_machine,
)
from src.agents.customer_service_agent import (
    offer_refund, offer_partial_refund,
)
from src.agents.order_store import get_order
from src.agents.tray_tools import place_on_tray, check_tray


TOOL_REGISTRY: dict[str, BaseTool] = {
    t.name: t
    for t in [
        process_order, calculate_total,
        check_inventory, update_stock, get_alternatives,
        start_preparation, end_preparation, estimate_prep_time, clean_machine,
        offer_refund, offer_partial_refund,
        get_order,
        place_on_tray, check_tray,
        transfer_to_agent,
    ]
}


def resolve_tools(names: list[str]) -> list[BaseTool]:
    from .subgraph import RESPONSE_GUARDRAIL_TOOL_NAME

    reserved = [n for n in names if n == RESPONSE_GUARDRAIL_TOOL_NAME]
    if reserved:
        raise ValueError(
            f"Tool name {RESPONSE_GUARDRAIL_TOOL_NAME!r} is reserved for the "
            f"response guardrail's synthetic pseudo-call and cannot be used as "
            f"a real tool. Rename the tool in AgentDefinition."
        )
    missing = [n for n in names if n not in TOOL_REGISTRY]
    if missing:
        raise KeyError(f"Unknown tool names in AgentDefinition: {missing}")
    return [TOOL_REGISTRY[n] for n in names]
