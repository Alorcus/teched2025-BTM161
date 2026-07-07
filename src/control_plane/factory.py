"""Gateway Factory.

Resolves an AgentDefinition from the Agent Repo, builds the Gateway with the
applicable guardrails and Log Sink, composes the base prompt with applicable
guideline prompts, and returns a compiled per-agent subgraph.
"""

from .agent_repo import AgentDefinition, AgentRepo
from .catalog import Catalog
from .gateway import Gateway
from .log_sink import JsonlLogSink, NullLogSink
from .nemo_guardrail import bind_llm_to_nemo
from .snapshot import snapshot_id as compute_snapshot_id
from .subgraph import create_agent_subgraph
from .tool_registry import resolve_tools


def build(
    agent_id: str,
    llm,
    repo: AgentRepo,
    catalog: Catalog,
    log_sink: JsonlLogSink | NullLogSink,
):
    """Build (compile) an agent subgraph for the given agent_id."""
    definition: AgentDefinition = repo.get(agent_id)
    tools = resolve_tools(list(definition.tools))
    guardrails = catalog.guardrails(list(definition.guardrail_ids))
    guidelines = catalog.guidelines(list(definition.guideline_ids))

    # NeMo guardrails need the agents' LLM to back their LLM-based rails. Bind it
    # now (also warms the shared LLMRails cache so config errors surface early).
    bind_llm_to_nemo(guardrails, llm)

    composed_prompt = definition.base_prompt
    if guidelines:
        appendix = "\n\n## Guidelines\n\n" + "\n\n".join(
            f"- {g.prompt.strip()}" for g in guidelines
        )
        composed_prompt = composed_prompt.rstrip() + appendix

    snapshot = compute_snapshot_id(
        agent_id=definition.id,
        agent_version=definition.version,
        guardrails=guardrails,
        guidelines=[(g.id, g.version) for g in guidelines],
    )

    gateway = Gateway(
        agent_id=definition.id,
        guardrails=guardrails,
        allowed_handovers=list(definition.allowed_handovers),
        snapshot_id=snapshot,
        log_sink=log_sink,
    )

    subgraph = create_agent_subgraph(
        agent_id=definition.id,
        llm=llm,
        tools=tools,
        prompt=composed_prompt,
        gateway=gateway,
    )
    return subgraph, definition, snapshot
