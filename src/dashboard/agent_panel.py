import json
import html as html_mod
import time

import param
import panel as pn


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
    handoff_context = param.Dict(default=None, allow_None=True)

    def __init__(self, agent_name, config, system_prompt="", tools=None, **kwargs):
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
        self._handoff_pane = pn.pane.HTML("", sizing_mode="stretch_width")
        self._messages_pane = pn.pane.HTML("", sizing_mode="stretch_width")
        self._tool_calls_pane = pn.pane.HTML("", sizing_mode="stretch_width")

        self._render_header()
        self._render_handoff()
        self._render_messages()
        self._render_tool_calls()

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

        return pn.Column(
            self._header_pane,
            prompt_card,
            tools_section,
            self._handoff_pane,
            self._messages_pane,
            self._tool_calls_pane,
            sizing_mode="stretch_both",
            styles={
                "border": f"2px solid {self.color}",
                "border-radius": "8px",
                "padding": "12px",
                "background": f"{self.bg_color}66",
                "overflow-y": "auto",
            },
        )

    def add_message(self, role: str, content: str):
        ts = time.strftime("%H:%M:%S")
        msgs = list(self.messages)
        msgs.append({"role": role, "content": content, "ts": ts})
        self.messages = msgs
        self._render_messages()

    def add_tool_call(self, name: str, args: dict | None):
        ts = time.strftime("%H:%M:%S")
        tcs = list(self.tool_calls)
        tcs.append({"name": name, "args": args, "result": None, "ts": ts})
        self.tool_calls = tcs
        self._render_tool_calls()

    def set_tool_result(self, name: str, result: str):
        if self.tool_calls:
            updated = list(self.tool_calls)
            for i in range(len(updated) - 1, -1, -1):
                if updated[i]["name"] == name and updated[i]["result"] is None:
                    updated[i] = {**updated[i], "result": result}
                    break
            self.tool_calls = updated
            self._render_tool_calls()

    def set_status(self, status: str):
        self.status = status
        self._render_header()

    def set_handoff(self, context: dict | None):
        self.handoff_context = context
        self._render_handoff()

    def reset(self):
        self.status = "idle"
        self.messages = []
        self.tool_calls = []
        self.handoff_context = None
        self._render_header()
        self._render_handoff()
        self._render_messages()
        self._render_tool_calls()

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

    def _render_handoff(self):
        if not self.handoff_context:
            self._handoff_pane.object = ""
            return
        hc = self.handoff_context
        self._handoff_pane.object = (
            f'<div style="background:#F3E5F5;border-left:3px solid #9C27B0;'
            f'padding:8px;margin-bottom:8px;border-radius:4px;font-size:12px;">'
            f'<strong>Handoff from:</strong> {html_mod.escape(str(hc.get("from_agent", "?")))}<br>'
            f'<strong>Context:</strong> {html_mod.escape(str(hc.get("context_summary", "")))}<br>'
            f'<strong>Expectation:</strong> {html_mod.escape(str(hc.get("expectation", "")))}'
            f'</div>'
        )

    def _render_messages(self):
        if not self.messages:
            self._messages_pane.object = (
                '<div style="color:#999;font-size:12px;padding:8px;">No messages yet</div>'
            )
            return

        html_parts = ['<div style="font-size:12px;max-height:300px;overflow-y:auto;">']
        for msg in self.messages[-20:]:
            role = msg["role"]
            full_content = str(msg["content"])
            full_escaped = html_mod.escape(full_content)
            display_content = html_mod.escape(full_content[:500])
            ts = msg.get("ts", "")
            if role == "ai":
                prefix = f'<span style="color:{self.color};font-weight:bold;">AI:</span>'
            elif role == "tool":
                prefix = '<span style="color:#666;font-weight:bold;">Tool:</span>'
            elif role == "handoff":
                prefix = '<span style="color:#9C27B0;font-weight:bold;">Handoff:</span>'
            else:
                prefix = f'<span style="font-weight:bold;">{html_mod.escape(role)}:</span>'
            html_parts.append(
                f'<div style="padding:4px 0;border-bottom:1px solid #eee;" title="{full_escaped}">'
                f'<span style="color:#999;font-size:10px;margin-right:4px;">{ts}</span>'
                f'{prefix} {display_content}</div>'
            )
        html_parts.append("</div>")
        self._messages_pane.object = "\n".join(html_parts)

    def _render_tool_calls(self):
        if not self.tool_calls:
            self._tool_calls_pane.object = ""
            return

        html_parts = [
            '<div style="font-size:11px;margin-top:8px;">'
            '<strong style="font-size:12px;">Tool Calls:</strong>'
        ]
        for i, tc in enumerate(self.tool_calls[-10:], 1):
            name = html_mod.escape(tc["name"])
            ts = tc.get("ts", "")
            args_full = ""
            args_display = ""
            if tc["args"]:
                try:
                    args_full = json.dumps(tc["args"], ensure_ascii=False)
                    if len(args_full) > 100:
                        args_display = html_mod.escape(args_full[:100]) + "..."
                    else:
                        args_display = html_mod.escape(args_full)
                except (TypeError, ValueError):
                    args_full = "..."
                    args_display = "..."

            args_title = html_mod.escape(args_full)

            result_badge = ""
            if tc["result"] is not None:
                result_full = str(tc["result"])
                result_short = html_mod.escape(result_full[:80])
                result_title = html_mod.escape(result_full)
                if len(result_full) > 80:
                    result_short += "..."
                result_badge = (
                    f'<div style="color:#666;margin-left:16px;" title="{result_title}">'
                    f'→ {result_short}</div>'
                )
            else:
                result_badge = (
                    '<div style="color:#FFC107;margin-left:16px;">⏳ pending</div>'
                )

            html_parts.append(
                f'<div style="padding:3px 0;" title="{args_title}">'
                f'<span style="color:#999;font-size:10px;margin-right:4px;">{ts}</span>'
                f'<code>{name}</code>({args_display})'
                f'{result_badge}</div>'
            )
        html_parts.append("</div>")
        self._tool_calls_pane.object = "\n".join(html_parts)
