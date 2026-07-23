import json
import logging

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .shared_components import MENU, OrderStatus
from .order_store import load_order, is_item_in_order
from . import tray as tray_store

logger = logging.getLogger("coffee_shop.tray_tools")


class PlaceOnTraySchema(BaseModel):
    order_id: str = Field(description="The order ID (e.g. 'ORD0001')")
    item_name: str = Field(
        description="Name of the item to place on the tray (e.g. 'latte', 'croissant')"
    )
    quantity: int = Field(description="Quantity of the item to place", default=1)


class CheckTraySchema(BaseModel):
    order_id: str = Field(description="The order ID (e.g. 'ORD0001')")


@tool(args_schema=PlaceOnTraySchema)
def place_on_tray(order_id: str, item_name: str, quantity: int = 1) -> str:
    """Place an item on the customer's tray. Use after stock is deducted (food/pastry) or after brewing completes (coffee)."""
    logger.debug(
        f"tray_tools.py place_on_tray called: order={order_id}, item={item_name}, qty={quantity}"
    )
    logger.debug(
        f"place_on_tray called: order={order_id}, item={item_name}, qty={quantity}"
    )

    # check if item is in MENU
    item_key = item_name.lower()
    menu_item = MENU.get(item_key)
    if not menu_item:
        logger.debug(f"Attempted to place unknown item on tray: {item_name}")
        return json.dumps({"status": "error", "message": f"Unknown item: {item_name}"})

    category = menu_item.category
    contaminated = False

    if category == "coffee":
        from .barista_agent import ORDER_STATUS_CACHE

        cache = ORDER_STATUS_CACHE.get(order_id, {})
        last_contaminated = cache.get("last_brew_contaminated", False)
        if last_contaminated:
            contaminated = True
    else:
        order = load_order(order_id)
        if not order:
            return json.dumps(
                {"status": "error", "message": f"Order {order_id} not found"}
            )
        if order.status not in (
            OrderStatus.INVENTORY_CONFIRMED,
            OrderStatus.IN_PREPARATION,
        ):
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Cannot place items — order status is {order.status.value}",
                }
            )

    tray_contents = tray_store.place_on_tray(
        order_id, item_key, quantity, category, contaminated=contaminated
    )

    logger.debug(
        f"Placed {quantity}x {item_name} on tray for order {order_id}. Tray contents: {tray_contents}"
    )
    return json.dumps(
        {
            "status": "success",
            "message": f"Placed {quantity}x {item_name} on the tray.",
            "tray": tray_contents,
            "order_id": order_id,
        }
    )


@tool(args_schema=CheckTraySchema)
def check_tray(order_id: str) -> str:
    """Check what items are currently on the customer's tray."""
    logger.debug(f"check_tray called for {order_id}")
    tray_contents = tray_store.tray_as_list(order_id)
    return json.dumps(
        {
            "order_id": order_id,
            "tray": tray_contents,
            "item_count": len(tray_contents),
        }
    )
