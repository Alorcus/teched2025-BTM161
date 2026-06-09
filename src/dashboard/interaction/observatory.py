import atexit
import html as html_mod
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import panel as pn

from src.coffee_shop import CoffeeShop
from src.agents import CUSTOMER_SCENARIOS, build_default_prompt
from src.agents.order_agent import DEFAULT_PROMPT as ORDER_PROMPT, DEFAULT_TOOL_NAMES as ORDER_TOOLS
from src.agents.inventory_agent import DEFAULT_PROMPT as INVENTORY_PROMPT, DEFAULT_TOOL_NAMES as INVENTORY_TOOLS
from src.agents.barista_agent import DEFAULT_PROMPT as BARISTA_PROMPT, DEFAULT_TOOL_NAMES as BARISTA_TOOLS, start_coffee_machine, stop_coffee_machine
from src.agents.customer_service_agent import DEFAULT_PROMPT as CS_PROMPT, DEFAULT_TOOL_NAMES as CS_TOOLS
from .event_bus import EventBus, EventType, DashboardEvent
from .agent_panel import AgentPanel
from .conversation_runner import ConversationRunner
from .stock_panel import StockPanel
from .coffee_machine_panel import CoffeeMachinePanel
from .tray_panel import TrayPanel
from .log_saver import DashboardLogSaver

logger = logging.getLogger("coffee_shop.dashboard")


class _EventBusLogHandler(logging.Handler):
    """Forwards Python log records to the dashboard event bus."""

    def __init__(self, event_bus: EventBus):
        super().__init__(level=logging.DEBUG)
        self._event_bus = event_bus

    def emit(self, record: logging.LogRecord):
        agent = record.name.replace("coffee_shop.", "").split(".")[0]
        self._event_bus.publish(DashboardEvent(
            event_type=EventType.LOG_MESSAGE,
            agent_name=agent,
            content=record.getMessage(),
            log_level=record.levelno,
        ))

AGENT_REGISTRY = {
    "order_agent": {"prompt": ORDER_PROMPT, "tools": ORDER_TOOLS},
    "inventory_agent": {"prompt": INVENTORY_PROMPT, "tools": INVENTORY_TOOLS},
    "barista_agent": {"prompt": BARISTA_PROMPT, "tools": BARISTA_TOOLS},
    "customer_service_agent": {"prompt": CS_PROMPT, "tools": CS_TOOLS},
}


def create_observatory_dashboard():
    """Create the Agent Observatory dashboard page."""
    pn.extension(sizing_mode="stretch_both")

    shop = CoffeeShop()
    shop.open_shop()
    event_bus = EventBus()
    runner = ConversationRunner(shop, event_bus)
    log_saver = DashboardLogSaver(event_bus)

    coffee_shop_logger = logging.getLogger("coffee_shop")
    coffee_shop_logger.setLevel(logging.DEBUG)
    coffee_shop_logger.addHandler(_EventBusLogHandler(event_bus))

    start_coffee_machine()
    atexit.register(stop_coffee_machine)

    stock_panel = StockPanel()
    coffee_machine_panel = CoffeeMachinePanel()
    tray_panel = TrayPanel()

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

    grid = pn.GridSpec(ncols=2, nrows=2, sizing_mode="stretch_both",
                       styles={"gap": "5px"})
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (agent_name, panel_obj), (r, c) in zip(agent_panels.items(), positions):
        grid[r, c] = panel_obj.panel()

    scenario_labels = [
        "Latte & croissant (friendly)",
        "2 espressos (in a hurry)",
        "Complaint (cold cappuccino)",
        "Ask for a recommendation",
    ]
    scenario_options = {
        f"{i}: {scenario_labels[i]}": i for i in range(len(CUSTOMER_SCENARIOS))
    }
    scenario_select = pn.widgets.Select(
        name="", options=scenario_options, sizing_mode="stretch_width",
        margin=(0, 0, 5, 0),
    )

    log_level_options = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
    log_level_select = pn.widgets.Select(
        name="", options=log_level_options, value=20,
        sizing_mode="stretch_width", margin=(0, 0, 5, 0),
    )
    prompt_textarea = pn.widgets.TextAreaInput(
        name="Customer Prompt",
        value=build_default_prompt(0),
        height=200,
        sizing_mode="stretch_width",
        margin=(0, 0, 10, 0),
    )

    def on_scenario_change(event):
        prompt_textarea.value = build_default_prompt(event.new)

    scenario_select.param.watch(on_scenario_change, "value")

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
        log_saver.reset()  # Reset log saver for new conversation
        runner.start(scenario_index=scenario_select.value, custom_prompt=prompt_textarea.value)

    run_button.on_click(on_run)

    def on_save_log(event):
        if not log_saver.events:
            pn.state.notifications.error("No conversation data to save", duration=3000)
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")[:-3]
            filename = f"{timestamp}.eventlog.csv"
            filepath = Path("generated_event_log") / filename

            log_saver.save_to_csv(str(filepath))
            pn.state.notifications.success(f"✓ Saved {filename}", duration=4000)
        except Exception as e:
            pn.state.notifications.error(f"Failed to save: {e}", duration=4000)

    save_button.on_click(on_save_log)

    def update_save_button_state():
        """Enable save button when conversation is complete and has events."""
        has_events = len(log_saver.events) > 0
        is_idle = not runner.is_running
        save_button.disabled = not (has_events and is_idle)

    def poll_events():
        events = log_saver.capture_events()  # Capture events via log saver
        for ev in events:
            _dispatch_event(ev, agent_panels, log_entries, conversation_log,
                            coffee_machine_panel, tray_panel, log_level_select.value)
        if not runner.is_running and not events:
            status_indicator.value = False
        stock_panel.refresh()
        coffee_machine_panel.update_progress()
        update_save_button_state()  # Update button state after processing events

    sidebar = pn.Column(
        pn.Row(
            pn.Column(
                pn.pane.HTML('<label style="font-size:13px;font-weight:500;">Scenario</label>',
                             sizing_mode="stretch_width", margin=(0, 0, 2, 0)),
                scenario_select,
                sizing_mode="stretch_width", styles={"flex": "2"},
            ),
            pn.Column(
                pn.pane.HTML('<label style="font-size:13px;font-weight:500;">Log Level</label>',
                             sizing_mode="stretch_width", margin=(0, 0, 2, 0)),
                log_level_select,
                sizing_mode="stretch_width", styles={"flex": "1"},
            ),
            sizing_mode="stretch_width", margin=(0, 0, 5, 0),
            styles={"gap": "5px"},
        ),
        prompt_textarea,
        run_button,
        save_button,
        pn.Row(status_indicator, pn.pane.Markdown("", width=10)),
        pn.layout.Divider(),
        pn.pane.HTML('<label style="font-size:14px;font-weight:600;margin-bottom:8px;display:block;">Conversation Log</label>',
                     sizing_mode="stretch_width"),
        conversation_log,
        width=340,
        sizing_mode="stretch_height",
        styles={"display": "flex", "flex-direction": "column"},
    )

    # Navigation tabs for header
    nav_tabs = pn.Row(
        pn.pane.HTML(
            '<div style="display:inline-block;padding:6px 14px;background:#6D4C41;color:white;'
            'border-radius:4px;font-weight:600;margin-right:8px;font-size:13px;">Interaction Observatory</div>',
            sizing_mode="fixed"
        ),
        pn.pane.HTML(
            '<a href="/metrics" style="display:inline-block;padding:6px 14px;background:rgba(255,255,255,0.15);color:white;'
            'border-radius:4px;text-decoration:none;font-weight:500;font-size:13px;">'
            'Metrics Observatory</a>',
            sizing_mode="fixed"
        ),
        margin=(0, 0, 0, 0),
    )

    template = pn.template.FastListTemplate(
        title="Coffee Shop Agent Observatory",
        sidebar=[sidebar],
        header=[nav_tabs],
        main=[pn.Column(
            pn.Row(
                pn.Column(tray_panel.panel(), width=160, height=160),
                pn.Column(stock_panel.panel(), sizing_mode="stretch_both", styles={"flex": "2"}),
                pn.Column(coffee_machine_panel.panel(), sizing_mode="stretch_width", styles={"flex": "1"}),
                sizing_mode="stretch_width",
            ),
            grid,
            sizing_mode="stretch_both",
            styles={"gap": "5px"},
        )],
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
    log_entries: list[str], conversation_log,
    coffee_machine_panel: CoffeeMachinePanel,
    tray_panel: TrayPanel,
    min_log_level: int = 20,
):
    panel = agent_panels.get(event.agent_name)

    if event.event_type == EventType.LOG_MESSAGE:
        if event.log_level >= min_log_level:
            level_name = logging.getLevelName(event.log_level)
            color = {
                "DEBUG": "#9E9E9E",
                "INFO": "#2196F3",
                "WARNING": "#FF9800",
                "ERROR": "#F44336",
            }.get(level_name, "#666")
            _log(log_entries, conversation_log,
                 f'<span style="font-family:monospace;font-size:11px;color:{color};'
                 f'border-left:3px solid {color};padding-left:6px;">'
                 f'[{level_name}] {event.agent_name}: '
                 f'{_truncate(event.content, 120)}</span>')
        return

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
        if event.tool_name == "start_preparation":
            coffee_machine_panel.start_brewing("coffee")

    elif event.event_type == EventType.TOOL_RESULT:
        if panel:
            panel.set_status("idle")
            panel.set_tool_result(event.tool_name or "?", event.tool_result or "")
            panel.add_message("tool", f"{event.tool_name}: {_truncate(event.tool_result or '', 100)}")
        if event.tool_name == "end_preparation" and event.tool_result:
            try:
                result_data = json.loads(event.tool_result)
                status = result_data.get("status", "")
                if status == "ready":
                    coffee_machine_panel.complete(True)
                elif status == "contaminated":
                    coffee_machine_panel.complete(True)
                elif status in ("failed", "error"):
                    coffee_machine_panel.complete(False)
                    coffee_machine_panel.mark_dirty()
            except (json.JSONDecodeError, TypeError):
                pass
        elif event.tool_name == "clean_machine" and event.tool_result:
            try:
                result_data = json.loads(event.tool_result)
                if result_data.get("status") in ("cleaned", "already_clean"):
                    coffee_machine_panel.reset()
            except (json.JSONDecodeError, TypeError):
                pass
        elif event.tool_name == "place_on_tray" and event.tool_result:
            try:
                result_data = json.loads(event.tool_result)
                order_id = result_data.get("order_id")
                if order_id:
                    tray_panel.refresh(order_id)
            except (json.JSONDecodeError, TypeError):
                pass

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

    elif event.event_type == EventType.USER_VISIBLE:
        if panel:
            panel.add_message("user", event.content)

    elif event.event_type == EventType.CONVERSATION_START:
        _log(log_entries, conversation_log,
             f'<span style="color:#4CAF50"><b>START</b></span> {_truncate(event.content)}')

    elif event.event_type == EventType.CONVERSATION_END:
        _log(log_entries, conversation_log,
             '<span style="color:#F44336"><b>END</b></span> Conversation complete')
        tray_panel.clear()


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
