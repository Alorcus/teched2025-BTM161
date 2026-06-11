"""Single Python registry that resolves tool names (from YAML AgentDefinitions)
into the concrete `@tool`-decorated callables that already live in `src/agents/`.

Only tool *bindings* live here; tool *implementations* stay in their respective
agent files so the existing imports keep working.
"""

from langchain_core.tools import BaseTool

from src.agents.barista_agent import (
    clean_machine,
    end_preparation,
    estimate_prep_time,
    start_preparation,
)
from src.agents.customer_service_agent import (
    offer_partial_refund,
    offer_refund,
)
from src.agents.inventory_agent import (
    check_inventory,
    get_alternatives,
    update_stock,
)
from src.agents.order_agent import calculate_total, process_order
from src.agents.order_store import get_order
from src.agents.shared_components import transfer_to_agent
from src.agents.tray_tools import check_tray, place_on_tray

TOOL_REGISTRY: dict[str, BaseTool] = {
    t.name: t
    for t in [
        process_order,
        calculate_total,
        check_inventory,
        update_stock,
        get_alternatives,
        start_preparation,
        end_preparation,
        estimate_prep_time,
        clean_machine,
        offer_refund,
        offer_partial_refund,
        get_order,
        place_on_tray,
        check_tray,
        transfer_to_agent,
    ]
}


def resolve_tools(names: list[str]) -> list[BaseTool]:
    missing = [n for n in names if n not in TOOL_REGISTRY]
    if missing:
        raise KeyError(f"Unknown tool names in AgentDefinition: {missing}")
    return [TOOL_REGISTRY[n] for n in names]
