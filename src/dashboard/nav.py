"""Shared header navigation for the multi-page dashboard.

Each page (Interaction / Metrics / Trace) renders the same three-tab strip
in its template ``header`` slot. The currently active tab is shown solid;
the others are subtle pill links.
"""

from __future__ import annotations

import panel as pn

_TABS: tuple[tuple[str, str], ...] = (
    ("/", "Interaction Observatory"),
    ("/metrics", "Metrics Dashboard"),
    ("/trace", "Trace Table"),
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


def header_nav(active: str) -> pn.pane.HTML:
    """Return the three-tab header nav with ``active`` highlighted.

    ``active`` must be one of ``"/"``, ``"/metrics"``, ``"/trace"``.
    """
    parts: list[str] = []
    for route, label in _TABS:
        if route == active:
            parts.append(f'<div style="{_ACTIVE_STYLE}">{label}</div>')
        else:
            parts.append(f'<a href="{route}" style="{_LINK_STYLE}">{label}</a>')
    html = (
        '<div style="display:flex;gap:8px;align-items:center;">'
        f'{"".join(parts)}</div>'
    )
    return pn.pane.HTML(html, sizing_mode="stretch_width")
