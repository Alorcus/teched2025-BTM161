import panel as pn
import polars as pl


COLOR_SCHEME = {
    "off-white": "#F8F4E8",
    "beige": "#EBDBCB",
    "yellow": "#FDCA40",
    "orange": "#D87F12",
    "red": "#8D0209",
    "dark_red": "#721A0D",
    "brown": "#563210",
}

AGENT_COLORS = {
    "order_agent": COLOR_SCHEME["yellow"],
    "barista_agent": COLOR_SCHEME["orange"],
    "inventory_agent": COLOR_SCHEME["red"],
    "customer_service_agent": COLOR_SCHEME["brown"],
}


def fmt_seconds(value: float | None) -> str:
    """Format a duration in seconds for KPI display (e.g. ``1m 29s`` / ``39.4s``)."""
    if value is None:
        return "—"
    if value >= 60:
        m, s = divmod(int(round(value)), 60)
        return f"{m}m {s:02d}s"
    return f"{value:.1f}s"


def section_header(title: str) -> pn.Column:
    """Bold section title + thin divider, matching the Time Metrics / System Metrics style."""
    return pn.Column(
        pn.pane.HTML(
            f'<div style="font-size:13px;font-weight:600;margin:6px 0 2px;">{title}</div>',
            height=22, sizing_mode="stretch_width",
        ),
        pn.layout.Divider(margin=(0, 0, 2, 0), height=1),
        sizing_mode="stretch_width",
    )


def subsection_header(title: str, top_margin: int = 8) -> pn.pane.HTML:
    """Smaller header used for sub-charts within a section (e.g. ``Agent Workload``)."""
    return pn.pane.HTML(
        f'<div style="font-size:11px;font-weight:500;margin:{top_margin}px 0 4px;">{title}</div>',
        height=24, sizing_mode="stretch_width",
    )


# ---- Shared KPI card styling ------------------------------------------------
# Every KPI card in the metrics dashboard is a flex-column with the same
# outer shell (padding, border, radius, background, height alignment) and
# the same title / subtitle typography. Only the "body" underneath the
# subtitle differs — a single value for `subtitled_kpi_card`, or the
# avg/min·max/n block for `per_order_kpi_card`. Keeping these tokens in one
# place is what makes every section on the page look like one system.

_CARD_SHELL_OPEN = (
    '<div style="padding:8px 10px;border:1px solid #e0e0e0;border-radius:6px;'
    'background:#fafafa;display:flex;flex-direction:column;height:100%;'
    'box-sizing:border-box;">'
)
_CARD_SHELL_CLOSE = "</div>"
_CARD_TITLE_STYLE = "font-size:12px;font-weight:600;color:#4E342E;margin-bottom:2px;"
_CARD_SUBTITLE_STYLE = "font-size:10px;color:#777;margin-bottom:6px;line-height:1.35;"
_CARD_VALUE_STYLE = (
    "font-weight:600;font-size:18px;color:#333;line-height:1.2;"
    "margin-top:auto;"
)


def kpi_card(title: str, value: str) -> str:
    """KPI card with title and value only — no subtitle line.

    Shares the outer shell, title typography, and value typography with
    ``subtitled_kpi_card`` so a row of these lines up beside the subtitled
    variant used in other sections.
    """
    return (
        f'{_CARD_SHELL_OPEN}'
        f'<div style="{_CARD_TITLE_STYLE}">{title}</div>'
        f'<div style="{_CARD_VALUE_STYLE}">{value}</div>'
        f'{_CARD_SHELL_CLOSE}'
    )


def subtitled_kpi_card(title: str, subtitle: str, value: str) -> str:
    """KPI card with title, one-sentence subtitle, and a single value.

    Shares its outer shell and title/subtitle typography with
    ``per_order_kpi_card`` so a mixed grid of both lines up perfectly.
    """
    return (
        f'{_CARD_SHELL_OPEN}'
        f'<div style="{_CARD_TITLE_STYLE}">{title}</div>'
        f'<div style="{_CARD_SUBTITLE_STYLE}">{subtitle}</div>'
        f'<div style="{_CARD_VALUE_STYLE}">{value}</div>'
        f'{_CARD_SHELL_CLOSE}'
    )


def per_order_kpi_card(title: str, subtitle: str, durations: pl.Series,
                       unit: str = "orders") -> str:
    """Per-order KPI card with title, human-readable subtitle, avg/med/n.

    Designed to live inside a CSS grid so the four cards distribute evenly
    across the available width and match heights.
    """
    clean = durations.drop_nulls()
    n = clean.len()
    if n == 0:
        avg_str = min_str = max_str = "—"
    else:
        avg_str = fmt_seconds(float(clean.mean()))
        min_str = fmt_seconds(float(clean.min()))
        max_str = fmt_seconds(float(clean.max()))
    avg_style = "font-weight:600;font-size:13px;color:#333;line-height:1.25;"
    range_style = "font-weight:500;font-size:11px;color:#555;line-height:1.25;"
    footer_style = "font-size:10px;color:#999;margin-top:auto;padding-top:4px;"
    return (
        f'{_CARD_SHELL_OPEN}'
        f'<div style="{_CARD_TITLE_STYLE}">{title}</div>'
        f'<div style="{_CARD_SUBTITLE_STYLE}">{subtitle}</div>'
        f'<div style="{avg_style}">avg {avg_str}</div>'
        f'<div style="{range_style}">min {min_str} &nbsp;·&nbsp; max {max_str}</div>'
        f'<div style="{footer_style}">n={n} {unit}</div>'
        f'{_CARD_SHELL_CLOSE}'
    )


def kpi_row(cards_html: str, columns: int, top_padding: int = 2) -> pn.pane.HTML:
    """Standard KPI-card row wrapper — one grid style used everywhere."""
    return pn.pane.HTML(
        f'<div style="padding:{top_padding}px 0 2px;display:grid;'
        f'grid-template-columns:repeat({columns}, 1fr);gap:8px;width:100%;">'
        f'{cards_html}</div>',
        sizing_mode="stretch_width",
    )
