from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import logging
import json
from pydantic import BaseModel, Field
from typing import Optional

logger = logging.getLogger("coffee_shop.order_agent")

from .shared_components import (
    MENU, Order, OrderItem, Size, ALLOWED_EXTRAS,
    transfer_to_inventory, transfer_to_customer_service,
)
from ..llm import bind_tools_sequential
from .order_store import save_order, load_order, get_order
from .context_isolation import create_context_isolation_hook


class CustomerOrderItemSchema(BaseModel):
    name: str = Field(description="Name of the item")
    quantity: int = Field(default=1, description="Quantity of the item")
    size: Optional[Size] = Field(default=None, description="Size of the item (small, medium, large)")
    extras: list[str] = Field(default_factory=list, description="List of extra options (e.g., soy milk, extra shot)")

class ProcessOrderInputSchema(BaseModel):
    order: list[CustomerOrderItemSchema] = Field(description="List of items in the order")
    customer: str = Field(description="Customer's name")

class CalculateTotalInputSchema(BaseModel):
    order_id: str = Field(description="The order ID string")
    discount_percent: int = Field(default=0, description="Discount percentage to apply")


# ORDER AGENT TOOLS
@tool(args_schema=ProcessOrderInputSchema)
def process_order(order: list[CustomerOrderItemSchema], customer) -> str:
    """Process a customer order.
    Returns the created order details."""

    try:
        ordered_items = []
        unknown_items = []
        total = 0.0

        for item in order:
            item.name = item.name.lower().strip()

            if item.name not in MENU:
                unknown_items.append(item.name)
                continue

            # Validate extras
            invalid_extras = [e for e in item.extras if e.lower() not in ALLOWED_EXTRAS]
            if invalid_extras:
                return (
                    f"Error: Unknown extras: {', '.join(invalid_extras)}. "
                    f"Allowed: {', '.join(sorted(ALLOWED_EXTRAS))}"
                )

            price = MENU[item.name].price * item.quantity

            # Charge $0.50 for each paid extra
            num_paid_extras = len([extra for extra in item.extras if extra.lower() not in {"hot", "cold", "iced"}])
            price += num_paid_extras * 0.50 * item.quantity

            # Charge $0.50 less for small size, $0.75 more for large
            if item.size:
                if item.size == Size.SMALL:
                    price -= 0.50 * item.quantity
                elif item.size == Size.LARGE:
                    price += 0.75 * item.quantity

            ordered_items.append(OrderItem(
                name=item.name, quantity=item.quantity, price=price,
                size=item.size, extras=item.extras,
            ))
            total += price

        if unknown_items:
            logger.debug("Unknown items in order: %s", ", ".join(unknown_items))
            return f"Error processing order.\nItems not on menu: {', '.join(unknown_items)}\nAvailable items: {', '.join(MENU.keys())}"

        current_order = Order(total=total, customer=customer, items=ordered_items)
        save_order(current_order)
        order_id = current_order.order_id_str

        order_summary = f"Order {order_id} created for {customer}:\n"
        for item in ordered_items:
            extras_str = f" with {', '.join(item.extras)}" if item.extras else ""
            size_str = f" ({item.size.value})" if item.size else ""
            order_summary += f"  - {item.quantity}x {item.name.title()}{size_str}{extras_str}: ${item.price:.2f}\n"
        order_summary += f"Total: ${total:.2f}\nStatus: pending"

        return json.dumps({"order_id": order_id, "summary": order_summary})
    except Exception as e:
        return f"Error processing order: {str(e)}. Please use specified format."


@tool(args_schema=CalculateTotalInputSchema)
def calculate_total(order_id: str, discount_percent: int = 0) -> str:
    """Updates the order's total with optional discount."""
    order = load_order(order_id)
    if order is None:
        return f"Error: Order '{order_id}' not found."

    original_total = order.total
    discount_amount = original_total * (discount_percent / 100)
    final_total = original_total - discount_amount
    order.total = final_total
    save_order(order)
    if discount_percent > 0:
        logger.debug("Discount %d%% applied to %s: $%.2f -> $%.2f", discount_percent, order_id, original_total, final_total)

    result = f"Order {order.order_id_str} total: ${original_total:.2f}"
    if discount_percent > 0:
        result += f", discount ({discount_percent}%): -${discount_amount:.2f}, final: ${final_total:.2f}"

    return json.dumps({"order_id": order_id, "total": final_total, "discount": discount_amount, "summary": result})


DEFAULT_PROMPT = """\
You are a friendly, chatty order-taking agent at a coffee shop.

Your conversation flow:
1. Greet the customer and take their drink order.
2. If they don't specify a size, ask: "Would you like that as a large or normal?"
   Do NOT offer "small" as an option.
3. Once drinks are settled, ask if they'd like something to eat as well.
4. Confirm the full order and tell them the total price.
5. Process the order using process_order, then IMMEDIATELY transfer to the inventory agent.

After you process an order, you MUST transfer to the inventory agent.
Do NOT tell the customer the order is complete or ready — you only take and price orders.
Do NOT ask the customer whether they want you to check availability — just hand off.

You can transfer to:
- Inventory agent: to check item availability (mandatory after processing an order)
- Customer service agent: if the customer has a complaint or wants a modification

Be warm, conversational, and guide the customer through their order naturally."""

DEFAULT_TOOLS = [process_order, calculate_total, get_order, transfer_to_inventory, transfer_to_customer_service]
DEFAULT_TOOL_NAMES = [t.name for t in DEFAULT_TOOLS]


def create_order_agent(chat_llm, prompt=None):
    """Create and return the order agent."""
    if not prompt:
        prompt = DEFAULT_PROMPT

    tools = list(DEFAULT_TOOLS)

    llm_with_tools = bind_tools_sequential(chat_llm, tools)

    return create_react_agent(
        model=llm_with_tools,
        name="order_agent",
        tools=tools,
        prompt=prompt,
        pre_model_hook=create_context_isolation_hook("order_agent"),
    )
