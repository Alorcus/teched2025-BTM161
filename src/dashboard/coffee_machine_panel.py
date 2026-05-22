import os
import random
import time

import panel as pn
import requests

from services.coffee_machine.state import (
    reseed as backend_reseed,
    get_queue as backend_get_queue,
)

FAILURE_RATE = 0.2
SEED = int(os.environ.get("COFFEE_MACHINE_SEED", "100"))
COFFEE_MACHINE_URL = "http://127.0.0.1:8001"


class CoffeeMachinePanel:
    def __init__(self):
        self._title_pane = pn.pane.HTML(
            '<div style="font-weight:600;font-size:14px;">Coffee Machine</div>',
            sizing_mode="stretch_width",
        )
        self._queue_pane = pn.pane.HTML("", sizing_mode="stretch_width")
        self._regen_button = pn.widgets.Button(
            name="♻️", button_type="light", width=36, height=28,
            margin=(0, 0, 0, 0),
        )
        self._regen_button.on_click(lambda e: self.regenerate_queue())
        self._pane = pn.pane.HTML("", sizing_mode="stretch_width", min_width=250)
        self._state = "idle"
        self._drink = ""
        self._brew_start: float | None = None
        self._brew_eta: float = 3.0
        self._last_result: str = "INIT"
        self._queue: list[str] = backend_get_queue()
        self._render()

    def panel(self):
        self._frame = pn.Column(
            self._title_pane,
            pn.Row(
                self._queue_pane,
                self._regen_button,
                sizing_mode="stretch_width",
                styles={"align-items": "center"},
            ),
            self._pane,
            sizing_mode="stretch_width",
            styles={
                "border": "2px solid #e0e0e0",
                "border-radius": "12px",
                "padding": "12px",
                "background": "#fafafa",
            },
        )
        return self._frame

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
        self._shift_queue(success)
        self._render()

    def mark_dirty(self):
        self._state = "dirty"
        self._render()

    def reset(self):
        self._state = "idle"
        self._drink = ""
        self._brew_start = None
        self._render()

    def regenerate_queue(self):
        new_seed = random.randint(0, 2**31)
        backend_reseed(new_seed)
        self._queue = backend_get_queue()
        self._render()

    @property
    def state(self):
        return self._state

    def peek_next_result(self) -> str:
        return self._queue[0] if self._queue else "SUCC"

    def _fetch_queue(self) -> list[str]:
        return backend_get_queue()

    def _shift_queue(self, success: bool):
        self._last_result = "SUCC" if success else "FAIL"
        self._queue = backend_get_queue()

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

        if hasattr(self, "_frame"):
            self._frame.styles = {
                "border": f"2px solid {border_color}",
                "border-radius": "12px",
                "padding": "12px",
                "background": "#fafafa",
            }

        self._pane.object = (
            f'<div style="min-width:220px;">'
            f'{machine_svg}'
            f'{progress_html}'
            f'{status_html}'
            f'</div>'
        )

        self._queue_pane.object = self._build_queue()

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

    def _build_queue(self) -> str:
        def badge(label: str, is_last: bool = False) -> str:
            if label == "INIT":
                bg, color = "#9E9E9E", "#fff"
            elif label == "SUCC":
                bg, color = "#4CAF50", "#fff"
            else:
                bg, color = "#F44336", "#fff"
            opacity = "1" if is_last else "0.7"
            return (
                f'<span style="display:inline-block;padding:2px 5px;border-radius:3px;'
                f'font-size:10px;font-weight:600;font-family:monospace;'
                f'background:{bg};color:{color};opacity:{opacity};margin:0 2px;">'
                f'{label}</span>'
            )

        parts = [badge(self._last_result, is_last=True)]
        parts.append('<span style="font-size:11px;margin:0 3px;color:#666;">▶</span>')
        for r in self._queue:
            parts.append(badge(r))

        return (
            f'<div style="padding:4px 0;font-size:11px;display:flex;align-items:center;flex-wrap:wrap;">'
            f'{"".join(parts)}'
            f'</div>'
        )
