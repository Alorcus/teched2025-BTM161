import json
import html as html_mod
import time

import param
import panel as pn


# Per-agent scroll-preservation script. Mirrors trace_app._CONV_SCROLL_SCRIPT:
# caches {atBottom, top} on `window` so re-renders restore position, with a
# 32px sticky-bottom threshold so new messages glue to the bottom unless the
# user has scrolled up. Selector and state key are parameterized by
# agent_name so the four AgentPanel instances do not share window state.
_AP_SCROLL_SCRIPT_TEMPLATE = """
<script>
(function () {
  const NEAR = 32;
  const KEY = '__agentPanelScrollState_%(agent)s';
  const SEL = '.ap-scroll-body-%(agent)s';

  // Panel injects scripts via replaceChild, so document.currentScript is
  // unreliable. Walk every shadow root from document to find the scroller.
  // Pattern mirrors trace_table_panel._SCROLL_SCRIPT.
  function findScrollerRoot() {
    const cs = document.currentScript;
    if (cs && cs.getRootNode) {
      const r = cs.getRootNode();
      if (r && r.querySelector && r.querySelector(SEL)) return r;
    }
    const stack = [document];
    while (stack.length) {
      const node = stack.pop();
      if (node.querySelector && node.querySelector(SEL)) return node;
      const all = node.querySelectorAll ? node.querySelectorAll('*') : [];
      for (const el of all) {
        if (el.shadowRoot) stack.push(el.shadowRoot);
      }
    }
    return null;
  }

  function apply() {
    const root = findScrollerRoot();
    if (!root) return;
    const el = root.querySelector(SEL);
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
      el.scrollTop = el.scrollHeight;
      window[KEY] = { atBottom: true, top: el.scrollTop };
    }
    if (!el.__apPersistInstalled) {
      el.__apPersistInstalled = true;
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


class AgentPanel(param.Parameterized):
    agent_name = param.String()
    display_name = param.String()
    icon = param.String()
    color = param.String()
    bg_color = param.String()
    system_prompt = param.String(default="")
    tools_list = param.List(default=[])

    status = param.String(default="idle")
    messages = param.List(default=[])
    tool_calls = param.List(default=[])

    def __init__(self, agent_name, config, system_prompt="", tools=None, **kwargs):
        # agent_name is interpolated raw into a JS selector and window
        # state key; reject anything that wouldn't be a safe identifier.
        assert agent_name.isidentifier(), (
            f"agent_name must be a valid identifier, got: {agent_name!r}"
        )
        super().__init__(
            agent_name=agent_name,
            display_name=config["name"],
            icon=config["icon"],
            color=config["color"],
            bg_color=config["bg_color"],
            system_prompt=system_prompt,
            tools_list=tools or [],
            **kwargs,
        )
        self._header_pane = pn.pane.HTML("", sizing_mode="stretch_width")
        # The messages pane is a flex child of the bordered Column (capped
        # by the Column's height). We make it position:relative so the
        # inner .ap-scroll-body can be absolutely positioned inside it —
        # required because Panel injects an unstyled wrapper <div> between
        # bk-panel-models-markup-HTML and our content; that wrapper has
        # min-height:auto and grows to its child, bursting out of the
        # bounded Bokeh wrapper. position:absolute on the scroll body
        # tethers it to the Bokeh-sized markup wrapper instead.
        #
        # CRITICAL: pn.pane.HTML must be constructed WITH the rendered
        # HTML (script included) — Panel only executes <script> tags
        # that are present on the FIRST render. Subsequent `.object`
        # updates do re-inject the script but do not execute it.
        # See trace_table_panel.TraceTablePanel.__init__ for the same
        # pattern.
        self._messages_pane = pn.pane.HTML(
            self._render_messages_html(),
            sizing_mode="stretch_width",
            styles={
                "flex": "1 1 auto",
                "min-height": "0",
                "position": "relative",
                "overflow": "hidden",
            },
        )

        self._render_header()
        # _messages_pane was constructed with rendered HTML above; skip
        # the redundant _render_messages() call — calling it would
        # reassign .object, and Panel only executes <script> tags from
        # the FIRST render. Subsequent .object updates leave the script
        # in the DOM but do not re-execute it.

    def panel(self):
        tools_html = " ".join(
            f'<span style="background:#E0E0E0;padding:2px 6px;border-radius:4px;'
            f'font-size:11px;margin:2px;display:inline-block;">{html_mod.escape(t)}</span>'
            for t in self.tools_list
        )
        tools_section = pn.pane.HTML(
            f'<div style="margin-bottom:8px;"><strong style="font-size:12px;">Tools:</strong>'
            f'<div style="margin-top:4px;">{tools_html}</div></div>',
            sizing_mode="stretch_width",
        )

        prompt_card = pn.Card(
            pn.pane.HTML(
                f'<pre style="font-size:11px;white-space:pre-wrap;margin:0;">'
                f'{html_mod.escape(self.system_prompt)}</pre>',
                sizing_mode="stretch_width",
            ),
            title="System Prompt",
            collapsed=True,
            sizing_mode="stretch_width",
            styles={"margin-bottom": "8px"},
        )

        # Pin header + tools so the message scroller's flex:1 growth
        # cannot squeeze them out.
        self._header_pane.styles = {**(self._header_pane.styles or {}), "flex": "0 0 auto"}
        tools_section.styles = {**(tools_section.styles or {}), "flex": "0 0 auto"}
        return pn.Column(
            self._header_pane,
            tools_section,
            self._messages_pane,
            sizing_mode="stretch_both",
            styles={
                "border": f"2px solid {self.color}",
                "border-radius": "8px",
                "padding": "12px",
                "background": f"{self.bg_color}66",
                # Outer Column never scrolls; the inner .ap-scroll-body
                # owns the single scrollbar. height:100% closes the
                # GridSpec→flex chain so flex min-height:0 has a bounded
                # ancestor to measure against.
                "display": "flex",
                "flex-direction": "column",
                "min-height": "0",
                "height": "100%",
            },
        )

    def add_message(self, role: str, content: str, reason: str | None = None):
        ts = time.strftime("%H:%M:%S")
        msgs = list(self.messages)
        msgs.append({"role": role, "content": content, "ts": ts, "reason": reason or ""})
        self.messages = msgs
        self._render_messages()

    def add_tool_call(self, name: str, args: dict | None):
        ts = time.strftime("%H:%M:%S")
        args_str = ""
        if args:
            # Force display order for transfer_to_agent: receiver first.
            # The LLM may emit args in any order; reorder so the receiver
            # (target_agent) appears before the rationale fields.
            if name == "transfer_to_agent" and "target_agent" in args:
                ordered = {"target_agent": args["target_agent"]}
                for k, v in args.items():
                    if k != "target_agent":
                        ordered[k] = v
                args = ordered
            try:
                args_str = json.dumps(args, ensure_ascii=False)
            except (TypeError, ValueError):
                args_str = "..."
        msgs = list(self.messages)
        msgs.append({"role": "tool_call", "content": f"{name}({args_str})", "ts": ts, "tool_name": name})
        self.messages = msgs
        self._render_messages()

    def set_tool_result(self, name: str, result: str):
        ts = time.strftime("%H:%M:%S")
        msgs = list(self.messages)
        msgs.append({"role": "tool_result", "content": f"{name} → {result}", "ts": ts, "tool_name": name})
        self.messages = msgs
        self._render_messages()

    def set_status(self, status: str):
        self.status = status
        self._render_header()

    def reset(self):
        self.status = "idle"
        self.messages = []
        self.tool_calls = []
        self._render_header()
        self._render_messages()

    def _render_header(self):
        status_colors = {
            "idle": "#9E9E9E",
            "thinking": "#FFC107",
            "executing_tool": "#2196F3",
            "handed_off": "#9C27B0",
        }
        badge_color = status_colors.get(self.status, "#9E9E9E")
        self._header_pane.object = (
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
            f'<span style="font-size:24px;">{self.icon}</span>'
            f'<strong style="color:{self.color};font-size:16px;">{html_mod.escape(self.display_name)}</strong>'
            f'<span style="background:{badge_color};color:white;padding:2px 8px;'
            f'border-radius:12px;font-size:11px;margin-left:auto;">'
            f'{self.status.replace("_", " ")}</span></div>'
        )

    def _render_messages(self):
        self._messages_pane.object = self._render_messages_html()

    def _render_messages_html(self) -> str:
        script = _AP_SCROLL_SCRIPT_TEMPLATE % {"agent": self.agent_name}
        if not self.messages:
            return (
                f'<div class="ap-scroll-body ap-scroll-body-{self.agent_name}" '
                f'style="position:absolute;inset:0;overflow-y:auto;">'
                f'<div style="color:#999;font-size:12px;padding:8px;">No messages yet</div>'
                f'</div>{script}'
            )

        html_parts = [
            '<style>'
            '.agent-msg{padding:4px 0;border-bottom:1px solid #eee;}'
            '.agent-msg-body{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'
            'display:inline-block;max-width:100%;vertical-align:bottom;}'
            '.agent-msg:hover{background:#fffbe6;}'
            '.agent-msg:hover .agent-msg-body{white-space:pre-wrap;word-break:break-word;'
            'overflow:visible;text-overflow:clip;display:block;}'
            '.agent-msg:hover .agent-msg-truncated{display:none;}'
            '.agent-msg-reason{margin-top:4px;font-size:11px;color:#8a3a34;'
            'font-style:italic;border-left:2px solid #b3261e;padding-left:6px;}'
            '.agent-msg-reason-body{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'
            'display:inline-block;max-width:100%;vertical-align:bottom;}'
            '.agent-msg-reason:hover .agent-msg-reason-body{white-space:pre-wrap;'
            'word-break:break-word;overflow:visible;text-overflow:clip;display:block;}'
            '</style>'
        ]
        html_parts.append(
            f'<div class="ap-scroll-body ap-scroll-body-{self.agent_name}" '
            f'style="font-size:12px;position:absolute;inset:0;overflow-y:auto;">'
        )
        for msg in self.messages[-20:]:
            role = msg["role"]
            full_content = str(msg["content"])
            full_escaped = html_mod.escape(full_content)
            ts = msg.get("ts", "")
            if role == "ai":
                prefix = f'<span style="color:{self.color};font-weight:bold;">AI:</span>'
            elif role == "ai_rejected":
                prefix = '<span style="color:#b3261e;font-weight:bold;">AI&nbsp;[REJECTED]:</span>'
            elif role == "user":
                prefix = '<span style="color:#2E7D32;font-weight:bold;">User:</span>'
            elif role == "tool":
                prefix = '<span style="color:#666;font-weight:bold;">Tool:</span>'
            elif role == "tool_call":
                prefix = '<span style="color:#2196F3;font-weight:bold;">⚙️</span>'
            elif role == "tool_result":
                prefix = '<span style="color:#666;">→</span>'
            else:
                prefix = f'<span style="font-weight:bold;">{html_mod.escape(role)}:</span>'
            reason = msg.get("reason") or ""
            reason_html = ""
            if reason:
                reason_full = html_mod.escape(reason)
                reason_html = (
                    f'<div class="agent-msg-reason">⚠ supervisor: '
                    f'<span class="agent-msg-reason-body">{reason_full}</span></div>'
                )
            body_style = ""
            if role == "ai_rejected":
                body_style = "color:#b3261e;"
            html_parts.append(
                f'<div class="agent-msg">'
                f'<span style="color:#999;font-size:10px;margin-right:4px;">{ts}</span>'
                f'{prefix} <span class="agent-msg-body" style="{body_style}">{full_escaped}</span>'
                f'{reason_html}</div>'
            )
        html_parts.append("</div>")
        html_parts.append(script)
        return "\n".join(html_parts)
