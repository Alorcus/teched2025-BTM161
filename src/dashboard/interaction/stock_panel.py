import panel as pn
from sqlalchemy.exc import OperationalError

from src.agents import get_all_inventory

CATEGORY_ICONS = {
    "coffee": "☕",
    "pastry": "🥐",
    "food": "🥪",
}

MAX_ICONS = 20


class StockPanel:
    def __init__(self):
        self._pane = pn.pane.HTML("", sizing_mode="stretch_width")
        self._last_snapshot: dict[str, int] = {}

    def panel(self):
        return self._pane

    def refresh(self):
        try:
            inventory = get_all_inventory()
        except OperationalError:
            # DB not initialised yet (open_shop() has not run). Nothing to
            # render on this pass; the next refresh after initialize_runtime
            # will succeed.
            return
        snapshot = {name: item.stock for name, item in inventory.items()}

        if snapshot == self._last_snapshot:
            return
        self._last_snapshot = snapshot

        cards = []
        for name, item in inventory.items():
            icon = CATEGORY_ICONS.get(item.category, "📦")
            count = item.stock
            icons_display = icon * min(count, MAX_ICONS)
            if count > MAX_ICONS:
                icons_display += "…"

            cards.append(
                f'<div style="display:inline-block;vertical-align:top;margin:4px 8px 4px 0;'
                f'padding:8px 12px;border:1px solid #e0e0e0;border-radius:8px;'
                f'background:#fafafa;min-width:100px;">'
                f'<div style="font-weight:600;font-size:13px;margin-bottom:2px;">'
                f'{name.capitalize()}</div>'
                f'<div style="font-size:11px;color:#666;">Stock: {count}</div>'
                f'<div style="font-size:14px;line-height:1.4;word-break:break-all;">'
                f'{icons_display}</div>'
                f'</div>'
            )

        self._pane.object = (
            '<div style="padding:8px 0;display:flex;flex-wrap:wrap;align-items:flex-start;">'
            + "".join(cards)
            + '</div>'
        )
