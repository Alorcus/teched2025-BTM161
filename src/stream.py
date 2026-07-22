from dataclasses import dataclass
from typing import Iterator

from langchain_core.messages import BaseMessage, HumanMessage

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
    """Deduplicate and yield messages from a LangGraph subgraph stream.

    Assistant reply messages that a later stream event marks as guardrail-
    rejected are downgraded to `is_agent_reply=False`. Detection: the
    response_gateway node emits a corrective HumanMessage carrying the rejected
    text in `additional_kwargs[REJECTED_CONTENT_KWARG]` immediately after the
    offending AIMessage. We buffer only the *most recent* candidate agent
    reply — if the very next stream event is such a correction we rewrite the
    buffered reply before yielding, otherwise we flush it unchanged. This keeps
    the yield semantics streaming (bounded delay of one stream event) rather
    than batching to end-of-stream.
    """
    from src.control_plane.subgraph import CORRECTION_KWARG, REJECTED_CONTENT_KWARG

    seen: set[str] = set()
    pending: StreamMessage | None = None

    def _make_stream_message(message: BaseMessage) -> StreamMessage | None:
        content = getattr(message, "content", "")
        name = getattr(message, "name", "unknown")
        msg_id = f"{content}_{name}"
        if msg_id in seen:
            return None
        seen.add(msg_id)
        agent_name = name
        is_reply = (
            agent_name in SWARM_AGENTS
            and bool(content)
            and not getattr(message, "tool_calls", None)
        )
        return StreamMessage(
            agent_name=agent_name,
            content=normalize_content(content) if content else "",
            message=message,
            is_agent_reply=is_reply,
        )

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

                if isinstance(message, HumanMessage):
                    kwargs = message.additional_kwargs or {}
                    if kwargs.get(CORRECTION_KWARG) is True:
                        rejected = kwargs.get(REJECTED_CONTENT_KWARG, "")
                        if pending is not None and pending.content == rejected:
                            pending = StreamMessage(
                                agent_name=pending.agent_name,
                                content=pending.content,
                                message=pending.message,
                                is_agent_reply=False,
                            )
                        if pending is not None:
                            yield pending
                            pending = None
                        continue

                new_msg = _make_stream_message(message)
                if new_msg is None:
                    continue

                if pending is not None:
                    yield pending
                pending = new_msg

    if pending is not None:
        yield pending
