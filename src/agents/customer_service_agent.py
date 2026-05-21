from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import logging

logger = logging.getLogger("coffee_shop.customer_service_agent")

from .shared_components import (
    transfer_to_order_agent, transfer_to_barista, transfer_to_inventory,
    OrderIdSchema, OrderStatus,
)
from ..llm import bind_tools_sequential
from .context_isolation import create_context_isolation_hook
from pydantic import BaseModel, Field
import json

from .order_store import load_order, save_order, get_order


class PartialRefundSchema(BaseModel):
    order_id: str = Field(description="The order ID string")
    refund_percent: int = Field(default=50, description="Refund percentage to apply")


# CUSTOMER SERVICE AGENT TOOLS
@tool(args_schema=OrderIdSchema)
def offer_refund(order_id: str) -> str:
    """Process a full refund for an order."""
    order = load_order(order_id)
    if order is None:
        return f"Error: Order '{order_id}' not found."

    refund_amount = order.total
    order.status = OrderStatus.REFUNDED
    order.total = 0.0
    save_order(order)
    logger.debug("Full refund $%.2f for %s", refund_amount, order_id)

    return json.dumps({
        "order_id": order_id,
        "refund_amount": refund_amount,
        "summary": f"Full refund of ${refund_amount:.2f} processed for order {order_id}.",
    })


@tool(args_schema=PartialRefundSchema)
def offer_partial_refund(order_id: str, refund_percent: int = 50) -> str:
    """Process a partial refund for an order."""
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
    logger.debug("Partial refund %d%% ($%.2f) for %s, new total $%.2f", refund_percent, discount_amount, order_id, final_total)

    return json.dumps({
        "order_id": order_id,
        "refund_amount": discount_amount,
        "original_total": original_total,
        "new_total": final_total,
        "summary": f"Partial refund ({refund_percent}%) of ${discount_amount:.2f} for order {order_id}. New total: ${final_total:.2f}",
    })


DEFAULT_PROMPT = """\
You are a customer service agent focused on customer satisfaction.

Your job:
- Handle complaints, failed preparations, and unavailable items with empathy.
- Offer full or partial refunds when appropriate.
- Help the customer decide on next steps (new order, alternative items, or refund).

You can transfer to:
- Order agent: when the customer wants to place a new or modified order
- Inventory agent: to check availability of alternative items
- Barista agent: to retry preparation of an item

Always prioritize customer satisfaction and be generous with compensation when needed."""

DEFAULT_TOOLS = [offer_refund, offer_partial_refund, get_order, transfer_to_order_agent, transfer_to_barista, transfer_to_inventory]
DEFAULT_TOOL_NAMES = [t.name for t in DEFAULT_TOOLS]


def create_customer_service_agent(chat_llm, prompt=None):
    """Create and return the customer service agent."""
    if not prompt:
        prompt = DEFAULT_PROMPT

    tools = list(DEFAULT_TOOLS)

    llm_with_tools = bind_tools_sequential(chat_llm, tools)

    return create_react_agent(
        model=llm_with_tools,
        name="customer_service_agent",
        tools=tools,
        prompt=prompt,
        pre_model_hook=create_context_isolation_hook("customer_service_agent"),
    )
