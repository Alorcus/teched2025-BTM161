"""Trace Table panel.

Renders one row per emitted message in global order. Each agent gets its own
column; one extra column (Customer) for inbound user messages and one (Process
Supervisor) for the supervisor's per-message verdict. Exactly one of the agent
columns is non-blank per row — there is never any vertical overlap.

The panel listens on the existing dashboard ``EventBus``. Events that don't
correspond to a message (status pings, log lines, conversation lifecycle) are
filtered out via ``ROW_CREATORS``.
"""
from __future__ import annotations

import html

import panel as pn

from .event_bus import DashboardEvent, EventType


# Column key MUST match the agent_name string the runner publishes.
COLUMNS: list[tuple[str, str, str]] = [
    ("order_agent", "Order Agent", "\U0001F4DD"),
    ("inventory_agent", "Inventory Agent", "\U0001F4E6"),
    ("barista_agent", "Barista Agent", "☕"),
    ("customer_service_agent", "Customer Service", "\U0001F4AC"),
    ("customer", "Customer", "\U0001F464"),
]
COLUMN_KEYS = [k for k, _, _ in COLUMNS]

AGENT_ACCENT: dict[str, str] = {
    "order_agent": "#2196F3",
    "inventory_agent": "#FF9800",
    "barista_agent": "#8BC34A",
    "customer_service_agent": "#E91E63",
    "customer": "#4E342E",
}

# Event types that produce a row. HANDOFF is intentionally skipped: the
# transfer_to_* TOOL_CALL row already represents the transition and carries
# the supervisor's Termination verdict.
ROW_CREATORS: set[EventType] = {
    EventType.CUSTOMER_MESSAGE,
    EventType.AGENT_MESSAGE,
    EventType.AGENT_MESSAGE_REJECTED,
    EventType.TOOL_CALL,
    EventType.TOOL_RESULT,
}


_TABLE_CSS = """
<style>
.trace-wrap {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  background: linear-gradient(180deg, #fbfaf8 0%, #f5f1ec 100%);
  border-radius: 12px;
  padding: 18px 20px 22px 20px;
  box-shadow: 0 1px 3px rgba(78, 52, 46, 0.06), 0 8px 24px rgba(78, 52, 46, 0.04);
  border: 1px solid #ece4dc;
}
.trace-wrap h2 {
  margin: 0 0 4px 0;
  font-size: 15px;
  font-weight: 600;
  color: #4E342E;
  letter-spacing: 0.2px;
}
.trace-wrap .subtitle {
  margin: 0 0 14px 0;
  font-size: 12px;
  color: #8d7b6f;
}
.trace-scroll {
  max-height: calc(100vh - 220px);
  overflow: auto;
  border-radius: 8px;
  border: 1px solid #ece4dc;
  background: #ffffff;
}
table.trace {
  border-collapse: separate;
  border-spacing: 0;
  width: 100%;
  font-size: 12px;
  color: #2b211d;
  table-layout: fixed;
}
table.trace thead th {
  position: sticky;
  top: 0;
  z-index: 2;
  background: rgba(255, 255, 255, 0.92);
  backdrop-filter: saturate(140%) blur(6px);
  -webkit-backdrop-filter: saturate(140%) blur(6px);
  text-align: left;
  font-weight: 600;
  font-size: 11px;
  letter-spacing: 0.4px;
  text-transform: uppercase;
  color: #6b574c;
  padding: 8px 12px 7px 12px;
  border-bottom: 1px solid #e6dcd2;
}
table.trace thead th .accent {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-right: 8px;
  vertical-align: middle;
  box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.85);
}
table.trace thead th .icon { margin-right: 6px; opacity: 0.85; }
table.trace tbody tr { transition: background-color 120ms ease; }
table.trace tbody tr:hover { background: #fbf6ef; }
table.trace tbody tr + tr td { border-top: 1px solid #f1ebe4; }
table.trace td {
  padding: 4px 10px;
  vertical-align: top;
  white-space: normal;
  word-break: break-word;
  line-height: 1.3;
  color: #3a2f2a;
}
table.trace td.empty {
  background:
    repeating-linear-gradient(135deg,
      rgba(120, 100, 90, 0.025) 0px,
      rgba(120, 100, 90, 0.025) 6px,
      transparent 6px, transparent 12px);
}
table.trace td.owned {
  position: relative;
  background: #ffffff;
  cursor: pointer;
}
table.trace td.owned::before {
  content: "";
  position: absolute;
  left: 0; top: 4px; bottom: 4px;
  width: 3px;
  border-radius: 0 2px 2px 0;
  background: var(--accent, #cccccc);
}
table.trace td.owned .meta {
  display: inline;
  font-size: 10px;
  color: #9b897c;
  margin-right: 6px;
  letter-spacing: 0.3px;
  white-space: nowrap;
}
table.trace td.owned .meta .ts { font-variant-numeric: tabular-nums; }
table.trace td.owned .meta .kind {
  display: inline-block;
  margin-left: 4px;
  padding: 0 5px;
  border-radius: 7px;
  background: var(--accent-soft, #f3eee9);
  color: var(--accent, #6b574c);
  font-size: 9px;
  letter-spacing: 0.3px;
  text-transform: uppercase;
  font-weight: 600;
  vertical-align: 1px;
}
table.trace td.owned .body {
  color: #2b211d;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-overflow: ellipsis;
}
table.trace td.owned.expanded .body {
  display: block;
  -webkit-line-clamp: unset;
  overflow: visible;
  white-space: pre-wrap;
}
table.trace td.owned .body code.tool {
  font-family: ui-monospace, "SFMono-Regular", "JetBrains Mono", Menlo, Consolas, monospace;
  font-size: 11px;
  color: #4E342E;
  background: #f5efe8;
  padding: 0 4px;
  border-radius: 3px;
}
table.trace td.owned .body .args {
  font-family: ui-monospace, "SFMono-Regular", "JetBrains Mono", Menlo, Consolas, monospace;
  font-size: 10.5px;
  color: #6b574c;
  word-break: break-all;
}
table.trace td.owned.expanded .body .args { white-space: pre-wrap; }
table.trace td.owned .body .arrow { color: #9b897c; margin-right: 3px; }

table.trace td.supervisor {
  font-family: ui-monospace, "SFMono-Regular", "JetBrains Mono", Menlo, Consolas, monospace;
  font-size: 10.5px;
  color: #8d7b6f;
  background: #fcfaf7;
  border-left: 1px solid #efe7de;
  white-space: normal;
  word-break: break-word;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  cursor: pointer;
}
table.trace td.supervisor.expanded {
  display: table-cell;
  -webkit-line-clamp: unset;
  overflow: visible;
  white-space: pre-wrap;
}
table.trace td.supervisor .badge {
  display: inline-block;
  padding: 1px 6px;
  border-radius: 999px;
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  margin-right: 5px;
  vertical-align: 1px;
}
table.trace td.supervisor.execution   { color: #2e7d32; }
table.trace td.supervisor.execution   .badge { background: #e8f3ea; color: #2e7d32; }
table.trace td.supervisor.termination { color: #c25a00; }
table.trace td.supervisor.termination .badge { background: #fdecdc; color: #c25a00; }
table.trace td.supervisor.violation   { color: #b3261e; font-weight: 600; }
table.trace td.supervisor.violation   .badge { background: #fde8e6; color: #b3261e; }
table.trace td.supervisor.dash        { color: #b8a99c; }

table.trace td.owned.rejected {
  color: #b3261e;
  background: #fff7f6;
}
table.trace td.owned.rejected .body { color: #b3261e; opacity: 0.95; }
table.trace td.owned.rejected .body .reason {
  display: block;
  margin-top: 4px;
  font-style: italic;
  color: #8a3a34;
  opacity: 1;
  font-size: 11.5px;
}
table.trace td.owned.rejected .meta .kind {
  background: #fde8e6 !important;
  color: #b3261e !important;
}
table.trace td.owned.rejected::before { background: #b3261e !important; }

table.trace tbody:empty + .trace-empty,
.trace-empty {
  padding: 28px 14px;
  text-align: center;
  color: #a8978a;
  font-size: 13px;
  font-style: italic;
}
</style>
"""


# Inline scroll-preservation script. Re-emitted at the END of every
# _render_html() output. Each Panel re-render destroys and recreates
# .trace-scroll via container.innerHTML, so we restore scrollTop from a
# window-level cache (.__traceScrollState) on every render.
#
# Panel renders into a Bokeh shadow DOM, so document.querySelector(...) from
# the outer document never finds .trace-scroll. We have to locate the
# enclosing root (Document or ShadowRoot) the script is attached to and
# query within that. document.currentScript is unreliable for scripts that
# Panel re-injects via replaceChild, so we fall back to walking every
# shadow root in the page.
_SCROLL_SCRIPT = """
<script>
(function () {
  const NEAR = 32;
  const KEY = '__traceScrollState';

  function findScrollerRoot() {
    const cs = document.currentScript;
    if (cs && cs.getRootNode) {
      const r = cs.getRootNode();
      if (r && r.querySelector && r.querySelector('.trace-scroll')) return r;
    }
    const stack = [document];
    while (stack.length) {
      const node = stack.pop();
      if (node.querySelector && node.querySelector('.trace-scroll')) return node;
      const all = node.querySelectorAll ? node.querySelectorAll('*') : [];
      for (const el of all) {
        if (el.shadowRoot) stack.push(el.shadowRoot);
      }
    }
    return null;
  }

  function snapshot(el) {
    if (!el || !el.isConnected) return;
    const slack = el.scrollHeight - el.scrollTop - el.clientHeight;
    window[KEY] = {
      atBottom: slack <= NEAR,
      top: el.scrollTop,
    };
  }

  function apply() {
    const root = findScrollerRoot();
    if (!root) return;
    const el = root.querySelector('.trace-scroll');
    if (!el) return;

    const prev = window[KEY];
    if (prev && typeof prev.atBottom === 'boolean') {
      if (prev.atBottom) {
        el.scrollTop = el.scrollHeight;
      } else if (typeof prev.top === 'number') {
        const max = Math.max(0, el.scrollHeight - el.clientHeight);
        el.scrollTop = Math.min(prev.top, max);
      }
    } else {
      // First render: park at bottom so newest messages are visible.
      el.scrollTop = el.scrollHeight;
      window[KEY] = { atBottom: true, top: el.scrollTop };
    }

    if (!el.__tracePersistInstalled) {
      el.__tracePersistInstalled = true;
      el.addEventListener('scroll', () => snapshot(el), { passive: true });
    }

    if (!el.__traceExpandInstalled) {
      el.__traceExpandInstalled = true;
      window.__traceExpanded = window.__traceExpanded || new Set();
      el.addEventListener('click', (ev) => {
        const cell = ev.target.closest('td.owned, td.supervisor');
        if (!cell || !el.contains(cell)) return;
        if (cell.classList.contains('dash')) return;
        const key = cell.getAttribute('data-cell');
        if (!key) return;
        if (window.__traceExpanded.has(key)) {
          window.__traceExpanded.delete(key);
          cell.classList.remove('expanded');
        } else {
          window.__traceExpanded.add(key);
          cell.classList.add('expanded');
        }
      });
    }

    // Re-apply expand state after every re-render.
    if (window.__traceExpanded) {
      for (const key of window.__traceExpanded) {
        const cell = el.querySelector('td[data-cell="' + key + '"]');
        if (cell) cell.classList.add('expanded');
      }
    }
  }

  // Run synchronously, then again on rAF so layout has settled and
  // scrollHeight is accurate.
  apply();
  if (typeof requestAnimationFrame === 'function') {
    requestAnimationFrame(apply);
  } else {
    setTimeout(apply, 0);
  }
})();
</script>
"""


def _truncate(text: str, max_len: int = 600) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "…"


class TraceTablePanel:
    """Append-only trace of (agent, content, supervisor_line) rows."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self._pane = pn.pane.HTML(self._render_html(), sizing_mode="stretch_both")
        self._dirty = False

    def panel(self) -> pn.pane.HTML:
        return self._pane

    def reset(self) -> None:
        self.rows.clear()
        self._dirty = False
        self._pane.object = self._render_html()

    def handle_event(self, ev: DashboardEvent) -> None:
        """Append a row if the event is a row-creator. Caller must invoke flush()
        once per poll tick to actually re-render the pane."""
        if ev.event_type not in ROW_CREATORS:
            return
        agent, kind, content_html = self._classify(ev)
        if agent not in COLUMN_KEYS:
            return
        ts = ""
        try:
            import time as _time
            ts = _time.strftime("%H:%M:%S", _time.localtime(ev.timestamp))
        except Exception:
            ts = ""
        self.rows.append({
            "agent": agent,
            "kind": kind,
            "event_type": ev.event_type.name,
            "content_html": content_html,
            "supervisor_line": ev.supervisor_line,
            "ts": ts,
        })
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        self._pane.object = self._render_html()
        self._dirty = False

    # ----------------------------------------------------------------- helpers

    def _classify(self, ev: DashboardEvent) -> tuple[str, str, str]:
        agent = ev.agent_name or ""
        et = ev.event_type
        if et == EventType.CUSTOMER_MESSAGE:
            return (agent or "customer"), "say", html.escape(_truncate(ev.content or ""))
        if et == EventType.AGENT_MESSAGE:
            return agent, "say", html.escape(_truncate(ev.content or ""))
        if et == EventType.AGENT_MESSAGE_REJECTED:
            body = html.escape(_truncate(ev.content or ""))
            reason = ev.rejection_reason or ""
            if reason:
                body += (
                    '<span class="reason">'
                    f'⚠ supervisor: {html.escape(_truncate(reason, 600))}'
                    '</span>'
                )
            return agent, "rejected", body
        if et == EventType.TOOL_CALL:
            name = html.escape(ev.tool_name or "")
            args_text = ""
            if ev.tool_args:
                try:
                    import json as _json
                    args_text = _json.dumps(ev.tool_args, ensure_ascii=False)
                except (TypeError, ValueError):
                    args_text = str(ev.tool_args)
            args_text = html.escape(_truncate(args_text, 240))
            # Tool calls produced by a rejected AIMessage are emitted with a
            # synthetic supervisor_line starting with "REJECTED" so the
            # operator sees the structured args even though no execution
            # happened. Render them in the rejected lane.
            sup = ev.supervisor_line or ""
            if sup.startswith("REJECTED"):
                body = (
                    f'<code class="tool">{name}</code>'
                    f'<span class="args"> {args_text}</span>'
                    f'<span class="reason">⚠ not executed (supervisor rejected)</span>'
                )
                return agent, "rejected", body
            return agent, "tool", (
                f'<code class="tool">{name}</code>'
                f'<span class="args"> {args_text}</span>'
            )
        if et == EventType.TOOL_RESULT:
            name = html.escape(ev.tool_name or "")
            result = html.escape(_truncate(str(ev.tool_result or ""), 320))
            return agent, "result", (
                f'<span class="arrow">↩</span>'
                f'<code class="tool">{name}</code>'
                f'<span class="args"> {result}</span>'
            )
        return agent, "say", html.escape(_truncate(ev.content or ""))

    def _supervisor_cell(self, line: str | None, idx: int) -> str:
        if not line:
            return '<td class="supervisor dash">&mdash;</td>'
        # Strip the " | <serialized message>" suffix that the supervisor log
        # appends — the trace table already shows the message in its own column.
        verdict = line.split(" | ", 1)[0].strip()
        dc = f'data-cell="r{idx}-sup"'
        if verdict.startswith("Violation:"):
            return (
                f'<td class="supervisor violation" {dc}>'
                '<span class="badge">VIOL</span>'
                f'{html.escape(verdict[len("Violation:"):])}'
                '</td>'
            )
        if verdict.startswith("Execution:"):
            return (
                f'<td class="supervisor execution" {dc}>'
                '<span class="badge">EXEC</span>'
                f'{html.escape(verdict[len("Execution:"):])}'
                '</td>'
            )
        if verdict.startswith("Termination:"):
            return (
                f'<td class="supervisor termination" {dc}>'
                '<span class="badge">TERM</span>'
                f'{html.escape(verdict[len("Termination:"):])}'
                '</td>'
            )
        if verdict.startswith("NonAction:"):
            return f'<td class="supervisor dash">{html.escape(verdict)}</td>'
        return f'<td class="supervisor" {dc}>{html.escape(verdict)}</td>'

    def _render_html(self) -> str:
        parts: list[str] = [_TABLE_CSS]
        parts.append('<div class="trace-wrap">')
        parts.append('<h2>Trace Table</h2>')
        parts.append(
            '<p class="subtitle">One row per message, in global order. '
            'Each agent owns its own column. Click a cell to expand truncated content.</p>'
        )
        parts.append('<div class="trace-scroll">')
        # Column widths (CSS table-layout: fixed) — equal-ish for agents,
        # supervisor narrower.
        col_count = len(COLUMNS) + 1
        agent_pct = 88 / len(COLUMNS)
        parts.append('<table class="trace"><colgroup>')
        for _ in COLUMNS:
            parts.append(f'<col style="width:{agent_pct:.2f}%">')
        parts.append('<col style="width:12%">')
        parts.append('</colgroup>')

        parts.append('<thead><tr>')
        for key, label, icon in COLUMNS:
            color = AGENT_ACCENT.get(key, "#cccccc")
            parts.append(
                f'<th><span class="accent" style="background:{color}"></span>'
                f'<span class="icon">{icon}</span>{html.escape(label)}</th>'
            )
        parts.append('<th>Process Supervisor</th></tr></thead>')

        parts.append('<tbody>')
        for idx, row in enumerate(self.rows):
            owner = row["agent"]
            accent = AGENT_ACCENT.get(owner, "#cccccc")
            parts.append('<tr>')
            for key, _label, _icon in COLUMNS:
                if key == owner:
                    style = (
                        f'--accent:{accent};'
                        f'--accent-soft:{accent}1f;'
                    )
                    title = html.escape(row.get("ts", ""))
                    kind_label = {
                        "say": "msg",
                        "tool": "tool call",
                        "result": "tool result",
                        "rejected": "rejected",
                    }.get(row["kind"], row["kind"])
                    cell_classes = "owned"
                    if row["kind"] == "rejected":
                        cell_classes += " rejected"
                    parts.append(
                        f'<td class="{cell_classes}" data-cell="r{idx}-body" '
                        f'style="{style}" title="{title}">'
                        f'<span class="meta">'
                        f'<span class="ts">{title}</span>'
                        f'<span class="kind">{kind_label}</span>'
                        f'</span>'
                        f'<span class="body">{row["content_html"]}</span>'
                        f'</td>'
                    )
                else:
                    parts.append('<td class="empty"></td>')
            parts.append(self._supervisor_cell(row.get("supervisor_line"), idx))
            parts.append('</tr>')
        parts.append('</tbody>')
        parts.append('</table>')

        if not self.rows:
            parts.append(
                f'<div class="trace-empty" style="grid-column: 1 / span {col_count};">'
                'No messages yet — click "Run Conversation" to start a trace.'
                '</div>'
            )
        parts.append('</div>')  # /trace-scroll
        parts.append(_SCROLL_SCRIPT)
        parts.append('</div>')  # /trace-wrap
        return "".join(parts)
