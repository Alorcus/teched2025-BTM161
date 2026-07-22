from dataclasses import dataclass
from typing import Iterator

from langchain_core.messages import AIMessage, BaseMessage

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


def _matches_pending(pending: StreamMessage, rejected_id: str, rejected_content: str) -> bool:
    """Pair a rejection marker with the pending StreamMessage.

    Prefers message-id match when the marker carries one (unique even when two
    agents produce identical short text like 'Sure!'), and falls back to
    content match otherwise.
    """
    pending_id = getattr(pending.message, "id", "") or ""
    if rejected_id:
        return bool(pending_id) and pending_id == rejected_id
    return pending.content == rejected_content


def extract_messages(stream) -> Iterator[StreamMessage]:
    """Deduplicate and yield messages from a LangGraph subgraph stream.

    Assistant reply messages that a later stream event marks as guardrail-
    rejected are downgraded to `is_agent_reply=False`. Detection: the
    response_gateway node emits a corrective message stamped with
    `additional_kwargs[CORRECTION_KWARG]=True` immediately after the offending
    AIMessage — either a HumanMessage (on retry, correcting the LLM) or a
    canned-fallback AIMessage (on retry-cap exhaustion, replacing the reply
    the customer would otherwise have seen). Both carry the rejected
    message's id in `REJECTED_MESSAGE_ID_KWARG` and its text in
    `REJECTED_CONTENT_KWARG`. Pairing prefers id-match (unique per message
    even when two turns happen to produce identical text) and falls back to
    content-match. We buffer only the *most recent* candidate agent reply —
    if the very next stream event is such a correction we rewrite the
    buffered reply before yielding, otherwise we flush it unchanged. This
    keeps the yield semantics streaming (bounded delay of one stream event)
    rather than batching to end-of-stream. The fallback AIMessage itself is
    yielded normally as the new pending reply.
    """
    from src.control_plane.subgraph import (
        CORRECTION_KWARG,
        REJECTED_CONTENT_KWARG,
        REJECTED_MESSAGE_ID_KWARG,
    )

    seen: set[str] = set()
    pending: StreamMessage | None = None

    def _make_stream_message(message: BaseMessage) -> StreamMessage | None:
        content = getattr(message, "content", "")
        name = getattr(message, "name", "unknown")
        mid = getattr(message, "id", "") or ""
        # Include the message id in the dedup key when available so distinct
        # AIMessages that happen to carry identical content (e.g. two "Sure!"
        # replies, one rejected and one valid) both surface. Fall back to
        # content-only for messages that arrive without an id.
        msg_id = f"{mid}|{content}_{name}" if mid else f"{content}_{name}"
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

                kwargs = getattr(message, "additional_kwargs", None) or {}
                if kwargs.get(CORRECTION_KWARG) is True:
                    rejected_id = kwargs.get(REJECTED_MESSAGE_ID_KWARG, "")
                    rejected_content = kwargs.get(REJECTED_CONTENT_KWARG, "")
                    if pending is not None and _matches_pending(
                        pending, rejected_id, rejected_content
                    ):
                        pending = StreamMessage(
                            agent_name=pending.agent_name,
                            content=pending.content,
                            message=pending.message,
                            is_agent_reply=False,
                        )
                    if pending is not None:
                        yield pending
                        pending = None
                    # The correction marker may itself be an AIMessage (the
                    # cap-exhausted fallback) — treat it as the new pending
                    # reply so the customer sees the fallback text. A
                    # corrective HumanMessage is internal-only and should
                    # never surface as a StreamMessage.
                    if isinstance(message, AIMessage):
                        new_msg = _make_stream_message(message)
                        if new_msg is not None:
                            pending = new_msg
                    continue

                new_msg = _make_stream_message(message)
                if new_msg is None:
                    continue

                if pending is not None:
                    yield pending
                pending = new_msg

    if pending is not None:
        yield pending
