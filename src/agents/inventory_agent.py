from langchain_core.tools import tool
import logging
import json

logger = logging.getLogger("coffee_shop.inventory_agent")

from .shared_components import (
    OrderIdSchema, OrderStatus,
)
from .order_store import (
    load_order,
    check_inventory_availability, check_and_update_stock,
    get_inventory_item, get_alternatives_from_db,
)
from .order_state_machine import state_machine, InvalidTransitionError


@tool(args_schema=OrderIdSchema)
def check_inventory(order_id: str) -> str:
    """Check if all items in the order are available in inventory."""
    logger.debug("check_inventory called for %s", order_id)
    order = load_order(order_id)
    if order is None:
        return f"Error: Order '{order_id}' not found."

    report = check_inventory_availability(order)
    if "error" in report:
        return report["error"]

    new_status = OrderStatus.INVENTORY_CONFIRMED if report["all_available"] else OrderStatus.INVENTORY_ISSUES
    try:
        order = state_machine.transition(order, new_status, context="check_inventory")
    except InvalidTransitionError as e:
        return json.dumps({
            "order_id": order_id,
            "error": f"Cannot record inventory check result: {e}",
        })
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
    logger.debug("update_stock called for %s", order_id)
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
        try:
            order = state_machine.transition(order, OrderStatus.INVENTORY_ISSUES, context=f"update_stock: {e}")
        except InvalidTransitionError:
            pass
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
    logger.debug("get_alternatives called for %s", item_name)
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


