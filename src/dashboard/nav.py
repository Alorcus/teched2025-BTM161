"""Shared header navigation for the multi-page dashboard.

Each page (Interaction / Metrics) renders the same two-tab strip in its
template ``header`` slot. The currently active tab is shown solid; the
other is a subtle pill link.
"""

from __future__ import annotations

import panel as pn

_TABS: tuple[tuple[str, str], ...] = (
    ("/", "Interaction Observatory"),
    ("/metrics", "Metrics Dashboard"),
)

_ACTIVE_STYLE = (
    "display:inline-block;padding:6px 14px;background:#6D4C41;color:white;"
    "border-radius:4px;font-weight:600;font-size:13px;"
)
_LINK_STYLE = (
    "display:inline-block;padding:6px 14px;background:rgba(255,255,255,0.15);"
    "color:white;border-radius:4px;text-decoration:none;font-weight:500;"
    "font-size:13px;"
)


def header_nav(active: str) -> pn.Row:
    """Return the two-tab header nav with ``active`` highlighted.

    ``active`` must be one of ``"/"``, ``"/metrics"``.
    """
    items: list[pn.pane.HTML] = []
    for i, (route, label) in enumerate(_TABS):
        margin_right = "margin-right:8px;" if i < len(_TABS) - 1 else ""
        if route == active:
            html = f'<div style="{_ACTIVE_STYLE}{margin_right}">{label}</div>'
        else:
            html = f'<a href="{route}" style="{_LINK_STYLE}{margin_right}">{label}</a>'
        items.append(pn.pane.HTML(html, sizing_mode="fixed"))
    return pn.Row(*items, margin=(0, 0, 0, 0))
