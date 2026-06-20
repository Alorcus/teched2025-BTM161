import atexit
import html as html_mod
import json
import logging
import time
import threading
from typing import Optional
import panel as pn

from src.coffee_shop import CoffeeShop
from src.config import CoffeeShopConfig
from src.agents import CUSTOMER_SCENARIOS, build_default_prompt
from src.agents.barista_agent import start_coffee_machine, stop_coffee_machine
from ..nav import header_nav
from .event_bus import EventBus, EventType, DashboardEvent
from .agent_panel import AgentPanel
from .conversation_runner import ConversationRunner
from .stock_panel import StockPanel
from .coffee_machine_panel import CoffeeMachinePanel
from .tray_panel import TrayPanel
from src.trace_processing.trace_processor import TraceProcessor

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


def _agent_registry_from_repo(shop: CoffeeShop) -> dict[str, dict]:
    """Read agent prompts/tool-names from the Agent Repo set up during open_shop()."""
    repo = shop.agent_repo
    if repo is None:
        return {}
    return {
        agent_id: {
            "prompt": d.base_prompt,
            "tools": list(d.tools),
        }
        for agent_id, d in repo.all().items()
    }


def create_observatory_dashboard(setup_name: str):
    """Create the Agent Observatory dashboard page."""
    pn.extension(sizing_mode="stretch_both")

    shop = CoffeeShop(CoffeeShopConfig(setup_name=setup_name))
    shop.open_shop()
    event_bus = EventBus()
    runner = ConversationRunner(shop, event_bus)
    agent_registry = _agent_registry_from_repo(shop)

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
        reg = agent_registry.get(agent_name, {})
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
    export_button = pn.widgets.Button(
        name="Export to Event Log",
        button_type="default",
        sizing_mode="stretch_width",
        disabled=True,
    )
    export_status = pn.pane.HTML(
        "",
        sizing_mode="stretch_width",
        styles={"font-size": "11px", "min-height": "16px"},
    )
    _export_done_flag: list[str] = []   # thread-safe message queue: ["ok"] or ["err: …"]
    _conversation_has_run: list[bool] = [False]  # mutable container so closure can write to it

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
        runner.start(scenario_index=scenario_select.value, custom_prompt=prompt_textarea.value)

    run_button.on_click(on_run)

    def on_export(event):
        if runner.is_running:
            return
        export_button.disabled = True
        export_status.object = '<span style="color:#FF9800;">⏳ Exporting…</span>'
        _export_done_flag.clear()

        def _run_export():
            try:
                TraceProcessor().process_all_traces(export_as_json=False)
                _export_done_flag.append("ok")
            except Exception as e:
                _export_done_flag.append(f"err:{e}")

        threading.Thread(target=_run_export, daemon=True).start()

    export_button.on_click(on_export)

    def poll_events():
        events = event_bus.drain()
        for ev in events:
            _dispatch_event(ev, agent_panels, log_entries, conversation_log,
                            coffee_machine_panel, tray_panel, log_level_select.value,
                            export_button, _conversation_has_run)
        if not runner.is_running and not events:
            status_indicator.value = False
        stock_panel.refresh()
        coffee_machine_panel.update_progress()

        if _export_done_flag:
            msg = _export_done_flag.pop(0)
            if msg == "ok":
                export_status.object = '<span style="color:#4CAF50;">✅ Export complete — saved to ./generated_event_log/</span>'
            else:
                export_status.object = f'<span style="color:#F44336;">❌ {msg[4:]}</span>'
            if not runner.is_running and _conversation_has_run[0]:
                export_button.disabled = False

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
        export_button,
        export_status,
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
    nav_tabs = header_nav(active="/")

    template = pn.template.FastListTemplate(
        title=f"Coffee Shop Agent Observatory — {setup_name}",
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
    export_button = None,
    conversation_has_run: Optional[list[bool]] = None
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

    elif event.event_type == EventType.AGENT_MESSAGE_REJECTED:
        if panel:
            panel.set_status("idle")
            panel.add_message("ai_rejected", event.content, reason=event.rejection_reason)
        reason_short = ""
        if event.rejection_reason:
            reason_short = (
                f' <span style="color:#8a3a34;font-style:italic;font-size:11px;">'
                f'⚠ {_truncate(event.rejection_reason, 160)}</span>'
            )
        _log(log_entries, conversation_log,
             f'<span style="color:#b3261e"><b>{event.agent_name} [REJECTED]</b></span>: '
             f'<span style="color:#b3261e;">'
             f'{_truncate(event.content)}</span>{reason_short}')

    elif event.event_type == EventType.TOOL_CALL:
        # Render-only TOOL_CALL events for rejected attempts: show in agent
        # panel but skip coffee-machine side effects.
        rejected_render_only = (event.supervisor_line or "").startswith("REJECTED")
        if panel:
            if not rejected_render_only:
                panel.set_status("executing_tool")
            panel.add_tool_call(event.tool_name or "?", event.tool_args)
        if not rejected_render_only and event.tool_name == "start_preparation":
            coffee_machine_panel.start_brewing("coffee")

    elif event.event_type == EventType.TOOL_RESULT:
        if panel:
            panel.set_status("idle")
            panel.set_tool_result(event.tool_name or "?", event.tool_result or "")
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
        if conversation_has_run is not None:
            conversation_has_run[0] = True
        if export_button is not None:
            export_button.disabled = False


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
