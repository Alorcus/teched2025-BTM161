import logging

from langgraph.graph import StateGraph
from langgraph_swarm import add_active_agent_router
from langgraph.checkpoint.memory import InMemorySaver

from src.agents.shared_components import CoffeeShopState
from src.agents import (
    create_order_agent, create_inventory_agent,
    create_barista_agent, create_customer_service_agent,
)

logger = logging.getLogger("coffee_shop.graph")


def build_coffee_shop_graph(llm, agent_definitions=None):
    """Construct and compile the multi-agent coffee shop graph.

    Returns a compiled LangGraph StateGraph.
    """
    if agent_definitions is None:
        agent_definitions = {}

    order_agent = create_order_agent(llm, agent_definitions.get('order_agent', None))
    inventory_agent = create_inventory_agent(llm, agent_definitions.get('inventory_agent', None))
    barista_agent = create_barista_agent(llm, agent_definitions.get('barista_agent', None))
    customer_service_agent = create_customer_service_agent(llm, agent_definitions.get('customer_service_agent', None))

    checkpointer = InMemorySaver()

    agent_names = ["order_agent", "inventory_agent", "barista_agent", "customer_service_agent"]
    builder = StateGraph(CoffeeShopState)
    add_active_agent_router(builder, route_to=agent_names, default_active_agent="order_agent")
    builder.add_node("order_agent", order_agent, destinations=("inventory_agent", "customer_service_agent"))
    builder.add_node("inventory_agent", inventory_agent, destinations=("barista_agent", "customer_service_agent"))
    builder.add_node("barista_agent", barista_agent, destinations=("customer_service_agent",))
    builder.add_node("customer_service_agent", customer_service_agent, destinations=("order_agent", "inventory_agent", "barista_agent"))

    return builder.compile(checkpointer=checkpointer)
