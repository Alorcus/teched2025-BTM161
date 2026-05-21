import html as html_mod
import logging
import time

import panel as pn

from src.coffee_shop import CoffeeShop
from src.agents import CUSTOMER_SCENARIOS
from src.agents.order_agent import DEFAULT_PROMPT as ORDER_PROMPT, DEFAULT_TOOL_NAMES as ORDER_TOOLS
from src.agents.inventory_agent import DEFAULT_PROMPT as INVENTORY_PROMPT, DEFAULT_TOOL_NAMES as INVENTORY_TOOLS
from src.agents.barista_agent import DEFAULT_PROMPT as BARISTA_PROMPT, DEFAULT_TOOL_NAMES as BARISTA_TOOLS
from src.agents.customer_service_agent import DEFAULT_PROMPT as CS_PROMPT, DEFAULT_TOOL_NAMES as CS_TOOLS
from .event_bus import EventBus, EventType
from .agent_panel import AgentPanel
from .conversation_runner import ConversationRunner

logger = logging.getLogger("coffee_shop.dashboard")

AGENT_REGISTRY = {
    "order_agent": {"prompt": ORDER_PROMPT, "tools": ORDER_TOOLS},
    "inventory_agent": {"prompt": INVENTORY_PROMPT, "tools": INVENTORY_TOOLS},
    "barista_agent": {"prompt": BARISTA_PROMPT, "tools": BARISTA_TOOLS},
    "customer_service_agent": {"prompt": CS_PROMPT, "tools": CS_TOOLS},
}


def create_dashboard():
    pn.extension(sizing_mode="stretch_both")

    shop = CoffeeShop()
    shop.open_shop()
    event_bus = EventBus()
    runner = ConversationRunner(shop, event_bus)

    agent_panels: dict[str, AgentPanel] = {}
    for agent_name, config in shop.agent_config.items():
        if agent_name == "user":
            continue
        reg = AGENT_REGISTRY.get(agent_name, {})
        agent_panels[agent_name] = AgentPanel(
            agent_name=agent_name,
            config=config,
            system_prompt=reg.get("prompt", ""),
            tools=reg.get("tools", []),
        )

    grid = pn.GridSpec(ncols=2, nrows=2, sizing_mode="stretch_both")
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (agent_name, panel_obj), (r, c) in zip(agent_panels.items(), positions):
        grid[r, c] = panel_obj.panel()

    scenario_options = {
        f"{i}: {s[:50]}": i for i, s in enumerate(CUSTOMER_SCENARIOS)
    }
    scenario_select = pn.widgets.Select(
        name="", options=scenario_options, sizing_mode="stretch_width",
        margin=(0, 0, 5, 0),
    )
    run_button = pn.widgets.Button(
        name="Run Conversation", button_type="primary", sizing_mode="stretch_width"
    )
    status_indicator = pn.indicators.LoadingSpinner(value=False, size=25)
    conversation_log = pn.pane.HTML(
        '<div style="font-size:12px;color:#999;">No conversation yet.</div>',
        sizing_mode="stretch_both",
        styles={"overflow-y": "auto", "flex": "1"},
    )
    log_entries: list[str] = []

    def on_run(event):
        if runner.is_running:
            return
        for p in agent_panels.values():
            p.reset()
        log_entries.clear()
        conversation_log.object = ""
        status_indicator.value = True
        runner.start(scenario_index=scenario_select.value)

    run_button.on_click(on_run)

    def poll_events():
        events = event_bus.drain()
        for ev in events:
            _dispatch_event(ev, agent_panels, log_entries, conversation_log)
        if not runner.is_running and not events:
            status_indicator.value = False

    sidebar = pn.Column(
        pn.pane.HTML(
            '<h2 style="margin:0 0 24px 0;padding:0;">Agent Observatory</h2>',
            sizing_mode="stretch_width",
        ),
        pn.pane.HTML('<label style="font-size:13px;font-weight:500;">Scenario</label>',
                     sizing_mode="stretch_width", margin=(0, 0, 4, 0)),
        scenario_select,
        run_button,
        pn.Row(status_indicator, pn.pane.Markdown("", width=10)),
        pn.layout.Divider(),
        pn.pane.HTML('<label style="font-size:14px;font-weight:600;margin-bottom:8px;display:block;">Conversation Log</label>',
                     sizing_mode="stretch_width"),
        conversation_log,
        width=340,
        sizing_mode="stretch_height",
        styles={"display": "flex", "flex-direction": "column"},
    )

    template = pn.template.FastListTemplate(
        title="Coffee Shop Agent Observatory",
        sidebar=[sidebar],
        main=[grid],
        accent_base_color="#795548",
        header_background="#4E342E",
        theme="default",
    )

    # Register periodic callback after template is built — Panel will attach it
    # to the document when served.
    pn.state.add_periodic_callback(poll_events, period=100)

    return template


def _dispatch_event(
    event, agent_panels: dict[str, AgentPanel],
    log_entries: list[str], conversation_log
):
    panel = agent_panels.get(event.agent_name)

    if event.event_type == EventType.AGENT_THINKING:
        if panel:
            if event.content == "thinking":
                panel.set_status("thinking")
            else:
                panel.set_status("idle")

    elif event.event_type == EventType.AGENT_MESSAGE:
        if panel:
            panel.set_status("idle")
            panel.add_message("ai", event.content)
        _log(log_entries, conversation_log,
             f'<span style="color:{panel.color if panel else "#333"}">'
             f'<b>{event.agent_name}</b></span>: {_truncate(event.content)}')

    elif event.event_type == EventType.TOOL_CALL:
        if panel:
            panel.set_status("executing_tool")
            panel.add_tool_call(event.tool_name or "?", event.tool_args)

    elif event.event_type == EventType.TOOL_RESULT:
        if panel:
            panel.set_status("idle")
            panel.set_tool_result(event.tool_name or "?", event.tool_result or "")
            panel.add_message("tool", f"{event.tool_name}: {_truncate(event.tool_result or '', 100)}")

    elif event.event_type == EventType.HANDOFF:
        if panel:
            panel.set_status("handed_off")
        target = agent_panels.get(event.target_agent or "")
        if target and event.handoff_context:
            target.set_handoff(event.handoff_context)
            target.add_message(
                "handoff",
                f"[From {event.handoff_context.get('from_agent', '?')}] "
                f"{event.handoff_context.get('context_summary', '')}",
            )
        _log(log_entries, conversation_log,
             f'<span style="color:#9C27B0"><b>HANDOFF</b></span> '
             f'{event.agent_name} → {event.target_agent}')

    elif event.event_type == EventType.CUSTOMER_MESSAGE:
        _log(log_entries, conversation_log,
             f'<span style="color:#424242"><b>Customer</b></span>: '
             f'{_truncate(event.content)}')

    elif event.event_type == EventType.CONVERSATION_START:
        _log(log_entries, conversation_log,
             f'<span style="color:#4CAF50"><b>START</b></span> {_truncate(event.content)}')

    elif event.event_type == EventType.CONVERSATION_END:
        _log(log_entries, conversation_log,
             '<span style="color:#F44336"><b>END</b></span> Conversation complete')


def _log(entries: list[str], pane, html_line: str):
    ts = time.strftime("%H:%M:%S")
    entries.append(
        f'<div style="padding:2px 0;border-bottom:1px solid #f0f0f0;font-size:12px;">'
        f'<span style="color:#999;margin-right:6px;">{ts}</span>{html_line}</div>'
    )
    pane.object = "\n".join(entries[-50:])


def _truncate(text: str, max_len: int = 150) -> str:
    text = text.replace("\n", " ").strip()
    full_escaped = html_mod.escape(text)
    if len(text) > max_len:
        short = html_mod.escape(text[:max_len]) + "..."
        return f'<span title="{full_escaped}">{short}</span>'
    return full_escaped


def main():
    logging.getLogger("bokeh.server.views.static_handler").setLevel(logging.WARNING)
    logging.getLogger("tornado.access").setLevel(logging.WARNING)
    pn.serve(
        create_dashboard,
        port=5006,
        show=False,
        title="Coffee Shop Agent Observatory",
    )
    print("Dashboard running at http://localhost:5006")


if __name__ == "__main__":
    main()
