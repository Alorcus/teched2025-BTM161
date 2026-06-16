"""Trace Table dashboard page.

Mounted as the ``/trace`` route of the multi-page ``dashboard`` server in
``src.dashboard.app``. Focused on the global message trace: one row per
message, columns per agent + a Process Supervisor column.

The page is composed of four regions:

  - Top status strip — Tray, Inventory/Stock, Coffee Machine (same widgets the
    original dashboard uses).
  - Sidebar — Scenario, Log Level, Customer Prompt, Run button, status.
  - Below the sidebar — the full Conversation Log (chat-style HTML pane).
  - Main — the Trace Table (one row per emitted message, ordered globally).

The trace pane has smart auto-scroll: it sticks to the bottom only when the
user is already at the bottom; otherwise it preserves their scroll position.
"""
from __future__ import annotations

import atexit
import html as html_mod
import json
import logging
import time

import panel as pn

from src.coffee_shop import CoffeeShop
from src.agents import CUSTOMER_SCENARIOS, build_default_prompt
from src.agents.barista_agent import start_coffee_machine, stop_coffee_machine
from .interaction.agent_panel import AgentPanel  # noqa: F401  (used indirectly by status colors)
from .interaction.coffee_machine_panel import CoffeeMachinePanel
from .interaction.conversation_runner import ConversationRunner
from .interaction.event_bus import EventBus, EventType, DashboardEvent
from .interaction.stock_panel import StockPanel
from .interaction.tray_panel import TrayPanel
from .nav import header_nav
from .trace_table_panel import TraceTablePanel

logger = logging.getLogger("coffee_shop.dashboard.trace")


_PAGE_CSS = """
<style>
:root {
  --tt-bg: #f5f1ec;
  --tt-card: #ffffff;
  --tt-ink: #2b211d;
  --tt-muted: #8d7b6f;
  --tt-accent: #4E342E;
}
body, .bk-root, .pn-Template {
  background: var(--tt-bg) !important;
  color: var(--tt-ink) !important;
}
.tt-sidebar {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  padding: 18px 18px 22px 18px;
  background: var(--tt-card);
  border-radius: 12px;
  box-shadow: 0 1px 3px rgba(78, 52, 46, 0.06), 0 8px 24px rgba(78, 52, 46, 0.04);
  border: 1px solid #ece4dc;
}
.tt-sidebar h3 {
  margin: 0 0 4px 0; font-size: 15px; color: var(--tt-accent); font-weight: 600;
}
.tt-sidebar p.lead {
  margin: 0 0 14px 0; font-size: 12px; color: var(--tt-muted);
}
.tt-label {
  display: block; font-size: 11px; font-weight: 600; letter-spacing: 0.4px;
  text-transform: uppercase; color: var(--tt-accent);
  margin: 14px 0 4px 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.tt-status {
  display: flex; align-items: center; gap: 10px; margin-top: 14px;
  font-size: 12px; color: var(--tt-muted);
}
.tt-conversation-log {
  margin-top: 14px;
  background: var(--tt-card);
  border-radius: 12px;
  border: 1px solid #ece4dc;
  box-shadow: 0 1px 3px rgba(78, 52, 46, 0.06);
  padding: 14px 16px 12px 16px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  display: flex;
  flex-direction: column;
  flex: 1 1 auto;
  min-height: 0;
}
.tt-conversation-log h4 {
  margin: 0 0 8px 0; font-size: 12px; font-weight: 600;
  letter-spacing: 0.4px; text-transform: uppercase; color: var(--tt-accent);
  flex: 0 0 auto;
}
.tt-conv-body {
  flex: 1 1 auto;
  min-height: 0;
  overflow-y: auto;
}
</style>
"""


# Inline conversation-log scroll-preservation script. Emitted at the end of
# every rendered log HTML (after .tt-conv-body exists in the DOM).
_CONV_SCROLL_SCRIPT = """
<script>
(function () {
  const NEAR = 32;
  const KEY = '__convLogScrollState';

  function findScroller() {
    let s = document.currentScript;
    let root = s ? s.parentNode : null;
    while (root && root.querySelector && !root.querySelector('.tt-conv-body')) {
      root = root.parentNode;
    }
    return root && root.querySelector ? root.querySelector('.tt-conv-body') : null;
  }

  function apply() {
    const el = findScroller();
    if (!el) return;
    const prev = window[KEY];
    if (prev && typeof prev.atBottom === 'boolean') {
      if (prev.atBottom) el.scrollTop = el.scrollHeight;
      else if (typeof prev.top === 'number') el.scrollTop = prev.top;
    } else {
      el.scrollTop = el.scrollHeight;
      window[KEY] = { atBottom: true, top: el.scrollTop };
    }
    if (!el.__convPersistInstalled) {
      el.__convPersistInstalled = true;
      el.addEventListener('scroll', () => {
        const slack = el.scrollHeight - el.scrollTop - el.clientHeight;
        window[KEY] = { atBottom: slack <= NEAR, top: el.scrollTop };
      }, { passive: true });
    }
  }

  apply();
  if (typeof requestAnimationFrame === 'function') requestAnimationFrame(apply);
  else setTimeout(apply, 0);
})();
</script>
"""


def _scenario_options() -> dict[str, int]:
    labels = [
        "Latte & croissant (friendly)",
        "2 espressos (in a hurry)",
        "Complaint (cold cappuccino)",
        "Ask for a recommendation",
    ]
    return {f"{i}: {labels[i]}": i for i in range(min(len(labels), len(CUSTOMER_SCENARIOS)))}


def _truncate(text: str, max_len: int = 150) -> str:
    text = text.replace("\n", " ").strip()
    full_escaped = html_mod.escape(text)
    if len(text) > max_len:
        short = html_mod.escape(text[:max_len]) + "..."
        return f'<span title="{full_escaped}">{short}</span>'
    return full_escaped


_AGENT_COLORS = {
    "order_agent": "#2196F3",
    "inventory_agent": "#FF9800",
    "barista_agent": "#8BC34A",
    "customer_service_agent": "#E91E63",
    "customer": "#4E342E",
}


def _log(entries: list[str], pane, html_line: str) -> None:
    ts = time.strftime("%H:%M:%S")
    entries.append(
        f'<div style="padding:3px 0;border-bottom:1px solid #f4ede4;font-size:12px;">'
        f'<span style="color:#9b897c;margin-right:6px;font-variant-numeric:tabular-nums;">'
        f'{ts}</span>{html_line}</div>'
    )
    body = "\n".join(entries[-200:])
    pane.object = (
        '<div class="tt-conversation-log">'
        '<h4>Conversation Log</h4>'
        f'<div class="tt-conv-body">{body}</div>'
        f'{_CONV_SCROLL_SCRIPT}'
        '</div>'
    )


def _empty_log_html() -> str:
    return (
        '<div class="tt-conversation-log">'
        '<h4>Conversation Log</h4>'
        '<div class="tt-conv-body" style="color:#a8978a;font-size:12px;'
        'font-style:italic;">No conversation yet.</div>'
        f'{_CONV_SCROLL_SCRIPT}'
        '</div>'
    )


def _dispatch_to_log(
    event: DashboardEvent,
    log_entries: list[str],
    conversation_log: pn.pane.HTML,
    coffee_machine_panel: CoffeeMachinePanel,
    tray_panel: TrayPanel,
    min_log_level: int = 20,
) -> None:
    """Dispatch dashboard events to the conversation log + tray + coffee
    machine state. Mirrors src.dashboard.app._dispatch_event but only the
    subset relevant to this view (no agent panels)."""
    if event.event_type == EventType.LOG_MESSAGE:
        if event.log_level >= min_log_level:
            level_name = logging.getLevelName(event.log_level)
            color = {
                "DEBUG": "#9E9E9E", "INFO": "#2196F3",
                "WARNING": "#FF9800", "ERROR": "#F44336",
            }.get(level_name, "#666")
            _log(log_entries, conversation_log,
                 f'<span style="font-family:ui-monospace,Menlo,monospace;'
                 f'font-size:11px;color:{color};border-left:3px solid {color};'
                 f'padding-left:6px;">[{level_name}] {event.agent_name}: '
                 f'{_truncate(event.content, 120)}</span>')
        return

    if event.event_type == EventType.AGENT_MESSAGE:
        color = _AGENT_COLORS.get(event.agent_name, "#3a2f2a")
        _log(log_entries, conversation_log,
             f'<span style="color:{color}"><b>{event.agent_name}</b></span>: '
             f'{_truncate(event.content)}')

    elif event.event_type == EventType.AGENT_MESSAGE_REJECTED:
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
        # Render-only TOOL_CALL events for rejected attempts have a
        # supervisor_line tagged "REJECTED"; do NOT trigger physical side
        # effects (coffee machine, etc.) for those.
        if (event.supervisor_line or "").startswith("REJECTED"):
            pass
        elif event.tool_name == "start_preparation":
            coffee_machine_panel.start_brewing("coffee")

    elif event.event_type == EventType.TOOL_RESULT:
        if event.tool_name == "end_preparation" and event.tool_result:
            try:
                result_data = json.loads(event.tool_result)
                status = result_data.get("status", "")
                if status in ("ready", "contaminated"):
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
        _log(log_entries, conversation_log,
             f'<span style="color:#9C27B0"><b>HANDOFF</b></span> '
             f'{event.agent_name} → {event.target_agent}')

    elif event.event_type == EventType.CUSTOMER_MESSAGE:
        _log(log_entries, conversation_log,
             f'<span style="color:#4E342E"><b>Customer</b></span>: '
             f'{_truncate(event.content)}')

    elif event.event_type == EventType.CONVERSATION_START:
        _log(log_entries, conversation_log,
             f'<span style="color:#4CAF50"><b>START</b></span> '
             f'{_truncate(event.content)}')

    elif event.event_type == EventType.CONVERSATION_END:
        _log(log_entries, conversation_log,
             '<span style="color:#F44336"><b>END</b></span> Conversation complete')
        tray_panel.clear()


def create_trace_dashboard():
    """Per-session factory. Each browser session gets its own runner + panel."""
    pn.extension(sizing_mode="stretch_both")

    import os
    from src.config import CoffeeShopConfig
    # Start from the dataclass defaults; only OVERRIDE if env vars are set.
    # This keeps src/config.py as the single source of truth for the default.
    cfg_kwargs = {"setup_name": os.getenv("SETUP_NAME", "baseline")}
    env_enabled = os.getenv("PROCESS_SUPERVISOR_ENABLED")
    if env_enabled is not None:
        cfg_kwargs["process_supervisor_enabled"] = env_enabled.lower() in ("1", "true", "yes")
    env_active = os.getenv("PROCESS_SUPERVISOR_ACTIVE")
    if env_active is not None:
        cfg_kwargs["process_supervisor_active"] = env_active.lower() in ("1", "true", "yes")
    env_retries = os.getenv("PROCESS_SUPERVISOR_MAX_RETRIES")
    if env_retries is not None:
        cfg_kwargs["process_supervisor_max_retries"] = int(env_retries)
    cfg = CoffeeShopConfig(**cfg_kwargs)
    if not cfg.process_supervisor_enabled:
        logger.info("process supervisor: DISABLED (no observation, no critique)")
    else:
        logger.info(
            "active supervisor: %s (max_retries=%d)",
            "ON" if cfg.process_supervisor_active else "OFF",
            cfg.process_supervisor_max_retries,
        )
    shop = CoffeeShop(config=cfg)
    shop.open_shop()
    event_bus = EventBus()
    runner = ConversationRunner(shop, event_bus)

    trace_table = TraceTablePanel()
    stock_panel = StockPanel()
    coffee_machine_panel = CoffeeMachinePanel()
    tray_panel = TrayPanel()

    start_coffee_machine()
    atexit.register(stop_coffee_machine)

    # ------------------------------------------------------------------ sidebar
    scenario_select = pn.widgets.Select(
        name="", options=_scenario_options(), sizing_mode="stretch_width",
        margin=(0, 0, 8, 0),
    )
    log_level_options = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
    log_level_select = pn.widgets.Select(
        name="", options=log_level_options, value=20,
        sizing_mode="stretch_width", margin=(0, 0, 8, 0),
    )
    prompt_textarea = pn.widgets.TextAreaInput(
        name="",
        value=build_default_prompt(0),
        height=180,
        sizing_mode="stretch_width",
        margin=(0, 0, 12, 0),
    )

    def on_scenario_change(event):
        prompt_textarea.value = build_default_prompt(event.new)

    scenario_select.param.watch(on_scenario_change, "value")

    run_button = pn.widgets.Button(
        name="Run Conversation",
        button_type="primary",
        sizing_mode="stretch_width",
        margin=(8, 0, 0, 0),
    )
    status_indicator = pn.indicators.LoadingSpinner(value=False, size=22)
    status_text = pn.pane.HTML(
        '<span style="font-size:12px;color:#8d7b6f;">Idle</span>',
        sizing_mode="stretch_width",
    )

    def _set_status(running: bool):
        status_indicator.value = running
        if running:
            status_text.object = (
                '<span style="font-size:12px;color:#4E342E;font-weight:600;">'
                'Running…</span>'
            )
        else:
            status_text.object = (
                '<span style="font-size:12px;color:#8d7b6f;">Idle</span>'
            )

    # ---------------------------------------------------------- conversation log
    conversation_log = pn.pane.HTML(
        _empty_log_html(),
        sizing_mode="stretch_both",
        styles={"flex": "1 1 auto", "min-height": "0"},
    )
    log_entries: list[str] = []

    def _reset_log():
        log_entries.clear()
        conversation_log.object = _empty_log_html()

    def on_run(_event):
        if runner.is_running:
            return
        trace_table.reset()
        _reset_log()
        _set_status(True)
        runner.start(
            scenario_index=scenario_select.value,
            custom_prompt=prompt_textarea.value,
        )

    run_button.on_click(on_run)

    def poll_events():
        events: list[DashboardEvent] = event_bus.drain()
        for ev in events:
            try:
                trace_table.handle_event(ev)
            except Exception:
                logger.exception("trace table handle_event failed")
            try:
                _dispatch_to_log(
                    ev, log_entries, conversation_log,
                    coffee_machine_panel, tray_panel,
                    min_log_level=log_level_select.value,
                )
            except Exception:
                logger.exception("conversation log dispatch failed")
            if ev.event_type == EventType.CONVERSATION_END:
                _set_status(False)
        if events:
            trace_table.flush()
        if not runner.is_running and not events and status_indicator.value:
            _set_status(False)
        stock_panel.refresh()
        coffee_machine_panel.update_progress()

    sidebar_top = pn.pane.HTML(
        _PAGE_CSS
        + '<div class="tt-sidebar">'
          '<h3>Coffee Shop · Trace Table</h3>'
          '<p class="lead">Configure a scenario, then run a conversation. '
          'Every emitted message will appear as a row in the table on the right, '
          'in global order.</p>'
          '<span class="tt-label">Scenario</span>'
          '</div>',
        sizing_mode="stretch_width",
    )

    sidebar = pn.Column(
        sidebar_top,
        scenario_select,
        pn.pane.HTML('<span class="tt-label">Log Level</span>'),
        log_level_select,
        pn.pane.HTML('<span class="tt-label">Customer Prompt</span>'),
        prompt_textarea,
        run_button,
        pn.Row(status_indicator, status_text,
               margin=(12, 0, 0, 0),
               styles={"align-items": "center", "gap": "10px",
                       "flex": "0 0 auto"}),
        conversation_log,
        width=360,
        sizing_mode="stretch_height",
        styles={
            "padding": "12px",
            "display": "flex",
            "flex-direction": "column",
            "min-height": "0",
        },
    )

    # -------------------------------------------------------------- main column
    top_status_row = pn.Row(
        pn.Column(tray_panel.panel(), width=170, height=170),
        pn.Column(stock_panel.panel(),
                  sizing_mode="stretch_both", styles={"flex": "2"}),
        pn.Column(coffee_machine_panel.panel(),
                  sizing_mode="stretch_width", styles={"flex": "1"}),
        sizing_mode="stretch_width",
        margin=(0, 0, 12, 0),
    )

    main_column = pn.Column(
        top_status_row,
        trace_table.panel(),
        sizing_mode="stretch_both",
    )

    template = pn.template.FastListTemplate(
        title="Coffee Shop Agent Observatory",
        sidebar=[sidebar],
        header=[header_nav(active="/trace")],
        main=[main_column],
        accent_base_color="#795548",
        header_background="#4E342E",
        theme="default",
    )

    pn.state.add_periodic_callback(poll_events, period=100)
    return template
