from enum import Enum
from typing import List, Optional, Annotated, Any, TypedDict
from datetime import datetime, timezone
import logging

from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langgraph_swarm import SwarmState
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field
from sqlalchemy import Enum as SAEnum
from sqlmodel import SQLModel, Field as SQLField, Relationship, Column, JSON

logger = logging.getLogger("coffee_shop.handoff")


def _resolve_from_agent(state: dict) -> str:
    """Determine which agent is calling this tool.

    Checks active_agent first (available when ToolNode gets parent state),
    then falls back to the .name attribute on the last AIMessage.
    """
    active = state.get("active_agent")
    if active and active != "unknown":
        return active
    for msg in reversed(state.get("messages", [])):
        name = getattr(msg, "name", None)
        if name and name.endswith("_agent"):
            return name
    return "unknown"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OrderStatus(str, Enum):
    PENDING = "pending"
    INVENTORY_CONFIRMED = "inventory_confirmed"
    INVENTORY_ISSUES = "inventory_issues"
    IN_PREPARATION = "in_preparation"
    COMPLETED = "completed"
    PREPARATION_ERROR = "preparation_error"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"


class Size(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


# ---------------------------------------------------------------------------
# Global allowed extras
# ---------------------------------------------------------------------------

ALLOWED_EXTRAS: set[str] = {
    "soy milk", "oat milk", "almond milk",
    "extra shot", "decaf",
    "whipped cream", "vanilla syrup", "caramel syrup",
    "hot", "cold", "iced",
}


# ---------------------------------------------------------------------------
# SQLModel table classes
# ---------------------------------------------------------------------------

class MenuItem(SQLModel, table=True):
    __tablename__ = "inventory"

    name: str = SQLField(primary_key=True)
    price: float
    stock: int
    category: str


class OrderItem(SQLModel, table=True):
    __tablename__ = "order_items"

    id: int | None = SQLField(default=None, primary_key=True)
    order_id: int = SQLField(foreign_key="orders.id")
    name: str = SQLField(foreign_key="inventory.name")
    quantity: int
    price: float
    size: Size | None = SQLField(
        default=None,
        sa_column=Column(SAEnum(Size, values_callable=lambda e: [m.value for m in e]),
                         nullable=True),
    )
    extras: list[str] = SQLField(default_factory=list, sa_column=Column(JSON))

    order: Optional["Order"] = Relationship(back_populates="items")


class Order(SQLModel, table=True):
    __tablename__ = "orders"

    id: int | None = SQLField(default=None, primary_key=True)
    customer: str
    status: OrderStatus = SQLField(
        default=OrderStatus.PENDING,
        sa_column=Column(SAEnum(OrderStatus, values_callable=lambda e: [m.value for m in e]),
                         nullable=False),
    )
    total: float = 0.0
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(timezone.utc))
    last_modified: datetime = SQLField(default_factory=lambda: datetime.now(timezone.utc))

    items: List[OrderItem] = Relationship(
        back_populates="order",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "lazy": "selectin"},
    )

    @property
    def order_id_str(self) -> str:
        return f"ORD{self.id:04d}" if self.id else "ORD????"


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------

MENU = {
    "espresso": MenuItem(name="espresso", price=2.50, stock=20, category="coffee"),
    "latte": MenuItem(name="latte", price=4.00, stock=15, category="coffee"),
    "cappuccino": MenuItem(name="cappuccino", price=3.75, stock=18, category="coffee"),
    "americano": MenuItem(name="americano", price=3.00, stock=22, category="coffee"),
    "croissant": MenuItem(name="croissant", price=2.75, stock=8, category="pastry"),
    "muffin": MenuItem(name="muffin", price=3.25, stock=12, category="pastry"),
    "sandwich": MenuItem(name="sandwich", price=6.50, stock=5, category="food"),
}


# ---------------------------------------------------------------------------
# Extended Swarm State with handoff context
# ---------------------------------------------------------------------------

class HandoffContext(TypedDict, total=False):
    from_agent: str
    context_summary: str
    expectation: str


class CoffeeShopState(SwarmState):
    handoff_context: Optional[HandoffContext]


# ---------------------------------------------------------------------------
# Pydantic schema for tools that operate on an existing order by ID
# ---------------------------------------------------------------------------

class OrderIdSchema(BaseModel):
    order_id: str = Field(description="The order ID (e.g. 'ORD0001')")


# ---------------------------------------------------------------------------
# Handoff Tools — each requires explicit context summary and expectation
# ---------------------------------------------------------------------------

@tool
def transfer_to_inventory(
    context_summary: Annotated[str, "Summary of what you know so far that is relevant for the next agent"],
    expectation: Annotated[str, "What you expect the next agent to accomplish"],
    state: Annotated[Any, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Transfer to inventory agent to check item availability."""
    from_agent = _resolve_from_agent(state)
    logger.debug("handoff %s -> inventory_agent | summary=%s", from_agent, str(context_summary)[:80])
    tool_message = ToolMessage(
        content=f"Successfully transferred to inventory_agent. Context: {context_summary}",
        name="transfer_to_inventory",
        tool_call_id=tool_call_id,
    )
    return Command(
        goto="inventory_agent",
        graph=Command.PARENT,
        update={
            "messages": [tool_message],
            "active_agent": "inventory_agent",
            "handoff_context": {
                "from_agent": from_agent,
                "context_summary": context_summary,
                "expectation": expectation,
            },
        },
    )


@tool
def transfer_to_barista(
    context_summary: Annotated[str, "Summary of what you know so far that is relevant for the next agent"],
    expectation: Annotated[str, "What you expect the next agent to accomplish"],
    state: Annotated[Any, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Transfer to barista agent to prepare the order."""
    from_agent = _resolve_from_agent(state)
    logger.debug("handoff %s -> barista_agent | summary=%s", from_agent, str(context_summary)[:80])
    tool_message = ToolMessage(
        content=f"Successfully transferred to barista_agent. Context: {context_summary}",
        name="transfer_to_barista",
        tool_call_id=tool_call_id,
    )
    return Command(
        goto="barista_agent",
        graph=Command.PARENT,
        update={
            "messages": [tool_message],
            "active_agent": "barista_agent",
            "handoff_context": {
                "from_agent": from_agent,
                "context_summary": context_summary,
                "expectation": expectation,
            },
        },
    )


@tool
def transfer_to_customer_service(
    context_summary: Annotated[str, "Summary of what you know so far that is relevant for the next agent"],
    expectation: Annotated[str, "What you expect the next agent to accomplish"],
    state: Annotated[Any, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Transfer to customer service agent for issues, complaints, or order modifications."""
    from_agent = _resolve_from_agent(state)
    logger.debug("handoff %s -> customer_service_agent | summary=%s", from_agent, str(context_summary)[:80])
    tool_message = ToolMessage(
        content=f"Successfully transferred to customer_service_agent. Context: {context_summary}",
        name="transfer_to_customer_service",
        tool_call_id=tool_call_id,
    )
    return Command(
        goto="customer_service_agent",
        graph=Command.PARENT,
        update={
            "messages": [tool_message],
            "active_agent": "customer_service_agent",
            "handoff_context": {
                "from_agent": from_agent,
                "context_summary": context_summary,
                "expectation": expectation,
            },
        },
    )


@tool
def transfer_to_order_agent(
    context_summary: Annotated[str, "Summary of what you know so far that is relevant for the next agent"],
    expectation: Annotated[str, "What you expect the next agent to accomplish"],
    state: Annotated[Any, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Transfer back to order agent for new or modified orders."""
    from_agent = _resolve_from_agent(state)
    logger.debug("handoff %s -> order_agent | summary=%s", from_agent, str(context_summary)[:80])
    tool_message = ToolMessage(
        content=f"Successfully transferred to order_agent. Context: {context_summary}",
        name="transfer_to_order_agent",
        tool_call_id=tool_call_id,
    )
    return Command(
        goto="order_agent",
        graph=Command.PARENT,
        update={
            "messages": [tool_message],
            "active_agent": "order_agent",
            "handoff_context": {
                "from_agent": from_agent,
                "context_summary": context_summary,
                "expectation": expectation,
            },
        },
    )
