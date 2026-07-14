from langchain_core.tools import tool
import logging

logger = logging.getLogger("coffee_shop.customer_service_agent")

from .shared_components import (
    OrderIdSchema,
    OrderStatus,
)
from pydantic import BaseModel, Field
import json

from .order_store import load_order, save_order, set_order_status


class PartialRefundSchema(BaseModel):
    order_id: str = Field(description="The order ID string")
    refund_percent: int = Field(default=50, description="Refund percentage to apply")


@tool(args_schema=OrderIdSchema)
def offer_refund(order_id: str) -> str:
    """Process a full refund for an order."""
    logger.debug("offer_refund called for %s", order_id)
    order = load_order(order_id)
    if order is None:
        return f"Error: Order '{order_id}' not found."

    refund_amount = order.total
    set_order_status(order, OrderStatus.REFUNDED, context="offer_refund: full refund")
    order.total = 0.0
    save_order(order)

    return json.dumps(
        {
            "order_id": order_id,
            "refund_amount": refund_amount,
            "summary": f"Full refund of ${refund_amount:.2f} processed for order {order_id}.",
        }
    )


@tool(args_schema=PartialRefundSchema)
def offer_partial_refund(order_id: str, refund_percent: int = 50) -> str:
    """Process a partial refund for an order."""
    logger.debug(
        "offer_partial_refund called for %s, refund=%d%%", order_id, refund_percent
    )
    order = load_order(order_id)
    if order is None:
        return f"Error: Order '{order_id}' not found."

    # clamping makes business sense but exploration of how agentic processes
    # can go wrong is also interesting...
    original_total = order.total
    discount_amount = original_total * (refund_percent / 100)
    final_total = original_total - discount_amount
    order.total = final_total
    save_order(order)
    logger.debug(
        "Partial refund %d%% ($%.2f) for %s, new total $%.2f",
        refund_percent,
        discount_amount,
        order_id,
        final_total,
    )

    return json.dumps(
        {
            "order_id": order_id,
            "refund_amount": discount_amount,
            "original_total": original_total,
            "new_total": final_total,
            "summary": f"Partial refund ({refund_percent}%) of ${discount_amount:.2f} for order {order_id}. New total: ${final_total:.2f}",
        }
    )
