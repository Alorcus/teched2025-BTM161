from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import logging
import random
import json
from pydantic import BaseModel, Field

logger = logging.getLogger("coffee_shop.barista_agent")

from src.llm import bind_tools_sequential

from .shared_components import (
    OrderIdSchema, OrderStatus,
    transfer_to_agent,
)
from .order_store import load_order, save_order, get_order
from .context_isolation import create_context_isolation_hook


class RemakeItemSchema(BaseModel):
    order_id: str = Field(description="The order ID string")
    item_name: str = Field(description="Name of the item to remake")


# BARISTA AGENT TOOLS
@tool(args_schema=OrderIdSchema)
def prepare_order(order_id: str) -> str:
    """Simulate drink and food preparation with potential for errors.
    Returns a preparation report."""
    order = load_order(order_id)
    if order is None:
        return f"Error: Order '{order_id}' not found."
    if order.status != OrderStatus.INVENTORY_CONFIRMED:
        return f"Error: Inventory not confirmed for order {order_id}. Cannot prepare."

    order.status = OrderStatus.IN_PREPARATION
    save_order(order)

    # Simulate preparation with 20% chance of error
    preparation_success = random.random() > 0.2

    prep_report = f"Preparing Order {order_id}...\n"
    for item in order.items:
        prep_report += f"- Making {item.quantity}x {item.name.title()}\n"

    if preparation_success:
        order.status = OrderStatus.COMPLETED
        prep_report += "\nAll items prepared successfully!"
        logger.debug("Order %s prepared successfully", order_id)
    else:
        order.status = OrderStatus.PREPARATION_ERROR
        failed_item = random.choice(order.items)
        prep_report += f"\nError preparing {failed_item.name.title()}"
        logger.debug("Order %s preparation failed on %s", order_id, failed_item.name)

    save_order(order)
    return json.dumps({"order_id": order_id, "status": order.status.value, "summary": prep_report})


@tool(args_schema=RemakeItemSchema)
def remake_order_item(order_id: str, item_name: str) -> str:
    """Remake a specific drink or food item."""
    order = load_order(order_id)
    if order is None:
        return f"Error: Order '{order_id}' not found."

    for item in order.items:
        if item.name == item_name.lower():
            # Simulate remake with 90% success rate
            remake_success = random.random() > 0.1

            if remake_success:
                order.status = OrderStatus.COMPLETED
                save_order(order)
                logger.debug("Remade %s for order %s successfully", item_name, order_id)
                return json.dumps({
                    "order_id": order_id,
                    "status": OrderStatus.COMPLETED.value,
                    "summary": f"Successfully remade {item_name.title()} for order {order_id}.",
                })
            else:
                logger.debug("Remake of %s for order %s failed", item_name, order_id)
                return json.dumps({
                    "order_id": order_id,
                    "status": order.status.value,
                    "summary": f"Failed to remake {item_name.title()}. Please try again.",
                })

    return f"Error: Item '{item_name}' not found in order {order_id}."


@tool(args_schema=OrderIdSchema)
def estimate_prep_time(order_id: str) -> str:
    """Estimate preparation time for an order."""
    order = load_order(order_id)
    if order is None:
        return f"Error: Order '{order_id}' not found."

    total_items = sum(item.quantity for item in order.items)
    base_time = 2  # 2 minutes base time
    time_per_item = 1.5  # 1.5 minutes per additional item
    estimated_time = base_time + (total_items - 1) * time_per_item

    return f"Estimated preparation time for Order {order_id}: {estimated_time:.1f} minutes ({total_items} item(s))"


DEFAULT_PROMPT = """\
You are a skilled barista agent responsible for drink and food preparation.

Your job:
- Prepare the order using prepare_order.
- If preparation succeeds: inform the customer their order is ready. Your job is done.
- If preparation fails: attempt a remake with remake_order_item. If that also fails, transfer to customer service.

You can transfer to:
- Customer service agent: when preparation fails and cannot be resolved by remaking

Take pride in your craft. If something goes wrong, be honest about it."""

DEFAULT_TOOLS = [prepare_order, remake_order_item, estimate_prep_time, get_order, transfer_to_agent]
DEFAULT_TOOL_NAMES = [t.name for t in DEFAULT_TOOLS]


def create_barista_agent(chat_llm, prompt=None):
    """Create and return the barista agent."""
    if not prompt:
        prompt = DEFAULT_PROMPT

    tools = list(DEFAULT_TOOLS)

    llm_with_tools = bind_tools_sequential(chat_llm, tools)

    return create_react_agent(
        model=llm_with_tools,
        name="barista_agent",
        tools=tools,
        prompt=prompt,
        pre_model_hook=create_context_isolation_hook("barista_agent"),
    )
