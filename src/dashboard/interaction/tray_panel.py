import panel as pn

from src.agents.tray import get_tray, tray_as_list

CATEGORY_ICONS = {
    "coffee": "☕",
    "pastry": "🥐",
    "food": "🥪",
}


class TrayPanel:
    def __init__(self):
        self._pane = pn.pane.HTML("", width=160, height=160)
        self._last_snapshot: list[dict] = []
        self._render_empty()

    def panel(self):
        return self._pane

    def refresh(self, order_id: str | None):
        if not order_id:
            return
        contents = tray_as_list(order_id)
        if contents == self._last_snapshot:
            return
        self._last_snapshot = contents
        if not contents:
            self._render_empty()
        else:
            self._render(contents)

    def clear(self):
        self._last_snapshot = []
        self._render_empty()

    def _render_empty(self):
        self._pane.object = (
            '<div style="border:2px solid #e0e0e0;border-radius:12px;padding:12px;'
            'background:#fafafa;width:140px;height:140px;display:flex;flex-direction:column;'
            'justify-content:center;align-items:center;">'
            '<div style="font-weight:600;font-size:14px;margin-bottom:8px;">🍽️ Tray</div>'
            '<div style="font-size:12px;color:#999;text-align:center;">Empty</div>'
            '</div>'
        )

    def _render(self, contents: list[dict]):
        items_html = []
        for entry in contents:
            icon = CATEGORY_ICONS.get(entry.get("category", ""), "📦")
            name = entry.get("item_name", "?").capitalize()
            qty = entry.get("quantity", 1)
            contaminated = entry.get("contaminated", False)

            warning = ""
            if contaminated:
                warning = ' <span style="color:#FF9800;" title="Contaminated">⚠️</span>'

            items_html.append(
                f'<div style="display:inline-block;margin:3px 6px 3px 0;padding:4px 8px;'
                f'border:1px solid #e0e0e0;border-radius:6px;background:#fff;font-size:12px;">'
                f'{icon} {qty}x {name}{warning}</div>'
            )

        self._pane.object = (
            '<div style="border:2px solid #4CAF50;border-radius:12px;padding:12px;'
            'background:#fafafa;width:140px;height:140px;overflow-y:auto;">'
            '<div style="font-weight:600;font-size:14px;margin-bottom:8px;">🍽️ Tray</div>'
            f'<div>{"".join(items_html)}</div>'
            '</div>'
        )
