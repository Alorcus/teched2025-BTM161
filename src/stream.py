from dataclasses import dataclass
from typing import Iterator, Callable

from langchain_core.messages import BaseMessage

from src.llm import normalize_content


@dataclass
class StreamMessage:
    agent_name: str
    content: str
    message: BaseMessage
    is_agent_reply: bool


SWARM_AGENTS = frozenset(
    ("order_agent", "inventory_agent", "barista_agent", "customer_service_agent")
)


def extract_messages(stream) -> Iterator[StreamMessage]:
    """Deduplicate and yield messages from a LangGraph subgraph stream."""
    seen = set()
    for ns, update in stream:
        for node, node_updates in update.items():
            if node_updates is None:
                continue

            if isinstance(node_updates, (dict, tuple)):
                node_updates_list = [node_updates]
            elif isinstance(node_updates, list):
                node_updates_list = node_updates
            else:
                continue

            for nu in node_updates_list:
                if isinstance(nu, tuple):
                    continue
                messages_key = next(
                    (k for k in nu.keys() if k == "messages"), None
                )
                if messages_key is None:
                    continue

                msgs = nu[messages_key]
                if not msgs:
                    continue
                message = msgs[-1]
                content = getattr(message, "content", "")
                name = getattr(message, "name", "unknown")
                msg_id = f"{content}_{name}"

                if msg_id in seen:
                    continue
                seen.add(msg_id)

                agent_name = name
                is_reply = (
                    agent_name in SWARM_AGENTS
                    and bool(content)
                    and not getattr(message, "tool_calls", None)
                )

                yield StreamMessage(
                    agent_name=agent_name,
                    content=normalize_content(content) if content else "",
                    message=message,
                    is_agent_reply=is_reply,
                )
