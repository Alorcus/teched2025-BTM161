import time

import panel as pn


class CoffeeMachinePanel:
    def __init__(self):
        self._pane = pn.pane.HTML("", sizing_mode="stretch_width", min_width=250)
        self._state = "idle"
        self._drink = ""
        self._brew_start: float | None = None
        self._brew_eta: float = 3.0
        self._render()

    def panel(self):
        return self._pane

    def start_brewing(self, drink: str = "coffee"):
        self._state = "brewing"
        self._drink = drink
        self._brew_start = time.time()
        self._render()

    def update_progress(self):
        if self._state != "brewing" or not self._brew_start:
            return
        self._render()

    def complete(self, success: bool):
        self._state = "ready" if success else "failed"
        self._brew_start = None
        self._render()

    def mark_dirty(self):
        self._state = "dirty"
        self._render()

    def reset(self):
        self._state = "idle"
        self._drink = ""
        self._brew_start = None
        self._render()

    @property
    def state(self):
        return self._state

    def _progress_fraction(self) -> float:
        if not self._brew_start:
            return 0.0
        elapsed = time.time() - self._brew_start
        return min(elapsed / self._brew_eta, 0.99)

    def _render(self):
        machine_svg = self._build_svg()
        progress_html = self._build_progress()
        status_html = self._build_status()

        border_color = {
            "idle": "#e0e0e0",
            "brewing": "#2196F3",
            "ready": "#4CAF50",
            "failed": "#F44336",
            "dirty": "#FF9800",
        }.get(self._state, "#e0e0e0")

        self._pane.object = (
            f'<div style="border:2px solid {border_color};border-radius:12px;'
            f'padding:12px;background:#fafafa;min-width:220px;">'
            f'<div style="font-weight:600;font-size:14px;margin-bottom:8px;">'
            f'Coffee Machine</div>'
            f'{machine_svg}'
            f'{progress_html}'
            f'{status_html}'
            f'</div>'
        )

    def _build_svg(self) -> str:
        fill = {
            "idle": "#9E9E9E",
            "brewing": "#2196F3",
            "ready": "#4CAF50",
            "failed": "#F44336",
            "dirty": "#FF9800",
        }.get(self._state, "#9E9E9E")

        steam = ""
        if self._state == "brewing":
            steam = (
                '<g opacity="0.6">'
                '<path d="M30 5 Q32 0 34 5" stroke="#999" fill="none" stroke-width="1.5"/>'
                '<path d="M38 7 Q40 2 42 7" stroke="#999" fill="none" stroke-width="1.5"/>'
                '<path d="M46 5 Q48 0 50 5" stroke="#999" fill="none" stroke-width="1.5"/>'
                '</g>'
            )

        return (
            f'<svg width="80" height="70" viewBox="0 0 80 70" style="display:block;margin:0 auto 8px;">'
            f'{steam}'
            f'<rect x="15" y="15" width="50" height="45" rx="5" fill="{fill}" opacity="0.85"/>'
            f'<rect x="25" y="25" width="30" height="20" rx="3" fill="#fff" opacity="0.9"/>'
            f'<circle cx="40" cy="53" r="4" fill="#fff" opacity="0.7"/>'
            f'</svg>'
        )

    def _build_progress(self) -> str:
        if self._state != "brewing":
            return ""
        fraction = self._progress_fraction()
        pct = int(fraction * 100)
        return (
            f'<div style="background:#e0e0e0;border-radius:4px;height:8px;margin:6px 0;overflow:hidden;">'
            f'<div style="background:#2196F3;height:100%;width:{pct}%;'
            f'border-radius:4px;transition:width 0.3s;"></div>'
            f'</div>'
        )

    def _build_status(self) -> str:
        if self._state == "idle":
            return '<div style="font-size:12px;color:#999;">Idle — waiting for orders</div>'
        elif self._state == "brewing":
            return (
                f'<div style="font-size:12px;color:#2196F3;">'
                f'⏳ Brewing {self._drink}...</div>'
            )
        elif self._state == "ready":
            return (
                f'<div style="font-size:12px;color:#4CAF50;">'
                f'✅ {self._drink.capitalize()} ready!</div>'
            )
        elif self._state == "failed":
            return '<div style="font-size:12px;color:#F44336;">❌ Brew failed!</div>'
        elif self._state == "dirty":
            return (
                '<div style="font-size:12px;color:#FF9800;">'
                '🔧 Machine dirty — needs cleaning!</div>'
            )
        return ""
