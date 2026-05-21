from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import logging
import json

logger = logging.getLogger("coffee_shop.inventory_agent")

from .shared_components import (
    OrderIdSchema, OrderStatus,
    transfer_to_barista, transfer_to_customer_service,
)
from ..llm import bind_tools_sequential
from .order_store import (
    load_order, save_order, get_order,
    check_inventory_availability, check_and_update_stock,
    get_inventory_item, get_alternatives_from_db,
)
from .context_isolation import create_context_isolation_hook


# INVENTORY AGENT TOOLS
@tool(args_schema=OrderIdSchema)
def check_inventory(order_id: str) -> str:
    """Check if all items in the order are available in inventory."""
    order = load_order(order_id)
    if order is None:
        return f"Error: Order '{order_id}' not found."

    report = check_inventory_availability(order)
    if "error" in report:
        return report["error"]

    new_status = OrderStatus.INVENTORY_CONFIRMED if report["all_available"] else OrderStatus.INVENTORY_ISSUES
    order.status = new_status
    save_order(order)
    if report["all_available"]:
        logger.debug("Inventory check passed for %s", order_id)
    else:
        logger.debug("Inventory issues for %s: %s", order_id, ", ".join(report["unavailable_items"]))

    summary = f"Order {order_id}: {new_status}."
    if not report["all_available"]:
        summary += f" Unavailable: {', '.join(report['unavailable_items'])}."
    for d in report["details"]:
        summary += f"\n  {d['name']}: {d['status']} (requested {d['requested']}, available {d['available']})"

    return json.dumps({
        "order_id": order_id,
        "status": new_status.value,
        "all_available": report["all_available"],
        "summary": summary,
    })


@tool(args_schema=OrderIdSchema)
def update_stock(order_id: str) -> str:
    """Update inventory after order confirmation."""
    order = load_order(order_id)
    if order is None:
        return f"Error: Order '{order_id}' not found."
    if order.status != OrderStatus.INVENTORY_CONFIRMED:
        return (
            f"Error: Cannot update stock - order {order_id} status is "
            f"'{order.status.value}', not 'inventory_confirmed'."
        )

    try:
        items_report = check_and_update_stock(order)
    except (KeyError, ValueError) as e:
        order.status = OrderStatus.INVENTORY_ISSUES
        save_order(order)
        return f"Error updating stock: {e}"

    summary = f"Stock updated for order {order_id}. {len(items_report)} item(s) deducted."
    for item in items_report:
        summary += f"\n  {item['name']}: {item['previous_stock']} -> {item['new_stock']}"

    return json.dumps({
        "order_id": order_id,
        "status": "success",
        "items_updated": len(items_report),
        "summary": summary,
    })


@tool
def get_alternatives(item_name: str) -> str:
    """Get alternative items for out-of-stock products."""
    item = get_inventory_item(item_name.lower())
    if item is None:
        return f"Error: Item '{item_name}' not found in menu."

    alts = get_alternatives_from_db(item_name.lower())
    alt_strs = [f"{a['name'].title()} (${a['price']:.2f}) - {a['stock']} available" for a in alts]

    return json.dumps({
        "alternatives": alt_strs,
        "original_item": item_name,
        "category": item["category"],
    })


DEFAULT_PROMPT = """\
You are the inventory management agent for a coffee shop.

Your job:
- Check item availability for an order using check_inventory.
- If all items are available: update stock levels with update_stock, then MUST transfer to the barista agent.
- If items are unavailable: suggest alternatives using get_alternatives, then transfer to customer service.

After checking inventory and updating stock, you MUST transfer immediately.
Do NOT tell the customer the order is ready — you only handle stock.

You can transfer to:
- Barista agent: when all items are confirmed available and stock is updated
- Customer service agent: when items are unavailable and need resolution"""

DEFAULT_TOOLS = [check_inventory, update_stock, get_alternatives, get_order,
                 transfer_to_barista, transfer_to_customer_service]
DEFAULT_TOOL_NAMES = [t.name for t in DEFAULT_TOOLS]


def create_inventory_agent(chat_llm, prompt=None):
    """Create and return the inventory agent."""
    if not prompt:
        prompt = DEFAULT_PROMPT

    tools = list(DEFAULT_TOOLS)

    llm_with_tools = bind_tools_sequential(chat_llm, tools)

    return create_react_agent(
        model=llm_with_tools,
        name="inventory_agent",
        tools=tools,
        prompt=prompt,
        pre_model_hook=create_context_isolation_hook("inventory_agent"),
    )
