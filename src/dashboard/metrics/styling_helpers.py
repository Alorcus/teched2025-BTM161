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


def small_kpi_card(label: str, value: str) -> str:
    """Compact label-over-value card used in the Overview row."""
    return (
        '<div style="display:inline-block;margin:0 5px 5px 0;padding:4px 9px;'
        'border:1px solid #e0e0e0;border-radius:5px;background:#fafafa;min-width:78px;">'
        f'<div style="font-size:10px;color:#666;margin-bottom:1px;">{label}</div>'
        f'<div style="font-weight:600;font-size:13px;color:#333;line-height:1.2;">{value}</div>'
        '</div>'
    )


def subtitled_kpi_card(title: str, subtitle: str, value: str) -> str:
    """KPI card with title, one-sentence subtitle, and a single value.

    Visually matches ``per_order_kpi_card`` (title + subtitle + value column),
    so a grid of these lines up with the Time Metrics per-order cards.
    """
    title_style = "font-size:12px;font-weight:600;color:#4E342E;margin-bottom:2px;"
    subtitle_style = "font-size:10px;color:#777;margin-bottom:6px;line-height:1.35;"
    value_style = (
        "font-weight:600;font-size:18px;color:#333;line-height:1.2;"
        "margin-top:auto;"
    )
    return (
        '<div style="padding:8px 10px;border:1px solid #e0e0e0;border-radius:6px;'
        'background:#fafafa;display:flex;flex-direction:column;height:100%;'
        'box-sizing:border-box;">'
        f'<div style="{title_style}">{title}</div>'
        f'<div style="{subtitle_style}">{subtitle}</div>'
        f'<div style="{value_style}">{value}</div>'
        '</div>'
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
    title_style = "font-size:12px;font-weight:600;color:#4E342E;margin-bottom:2px;"
    subtitle_style = "font-size:10px;color:#777;margin-bottom:6px;line-height:1.35;"
    avg_style = "font-weight:600;font-size:13px;color:#333;line-height:1.25;"
    range_style = "font-weight:500;font-size:11px;color:#555;line-height:1.25;"
    footer_style = "font-size:10px;color:#999;margin-top:auto;padding-top:4px;"
    return (
        '<div style="padding:8px 10px;border:1px solid #e0e0e0;border-radius:6px;'
        'background:#fafafa;display:flex;flex-direction:column;height:100%;'
        'box-sizing:border-box;">'
        f'<div style="{title_style}">{title}</div>'
        f'<div style="{subtitle_style}">{subtitle}</div>'
        f'<div style="{avg_style}">avg {avg_str}</div>'
        f'<div style="{range_style}">min {min_str} &nbsp;·&nbsp; max {max_str}</div>'
        f'<div style="{footer_style}">n={n} {unit}</div>'
        '</div>'
    )
