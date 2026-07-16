import logging
from pathlib import Path

from langgraph.graph import StateGraph
from langgraph_swarm import add_active_agent_router
from langgraph.checkpoint.memory import InMemorySaver

from src.agents.shared_components import CoffeeShopState
from src.control_plane import AgentRepo, Catalog, JsonlLogSink, NullLogSink, build
from src.control_plane.gateway import Gateway

logger = logging.getLogger("coffee_shop.graph")

AGENT_IDS = ("order_agent", "inventory_agent", "barista_agent", "customer_service_agent")


def build_coffee_shop_graph(
    llm,
    repo: AgentRepo,
    catalog: Catalog,
    log_sink: JsonlLogSink | NullLogSink,
):
    """Construct and compile the multi-agent coffee shop swarm graph.

    Each agent is built via the Gateway Factory (guarded subgraph). Allowed
    handovers come from the AgentDefinitions in the repo. Returns
    `(app, gateways)` — `gateways` maps agent_id to the per-agent Gateway so
    runners can evaluate response-scoped guardrails on streamed AIMessages
    before publishing them.
    """
    subgraphs: dict[str, tuple] = {}
    gateways: dict[str, Gateway] = {}
    for agent_id in AGENT_IDS:
        sg, defn, snapshot, gateway = build(agent_id, llm, repo, catalog, log_sink)
        subgraphs[agent_id] = (sg, defn, snapshot)
        gateways[agent_id] = gateway
        logger.info("built %s | snapshot=%s | allowed_handovers=%s",
                    agent_id, snapshot, list(defn.allowed_handovers))

    checkpointer = InMemorySaver()
    builder = StateGraph(CoffeeShopState)
    add_active_agent_router(
        builder,
        route_to=list(AGENT_IDS),
        default_active_agent="order_agent",
    )
    for agent_id, (sg, defn, _) in subgraphs.items():
        builder.add_node(
            agent_id, sg, destinations=tuple(defn.allowed_handovers) or None,
        )

    return builder.compile(checkpointer=checkpointer), gateways
