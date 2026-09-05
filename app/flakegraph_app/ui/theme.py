"""Shared visual treatment for the operational Streamlit interface.

The app runs both standalone and inside Snowsight, which imposes its own light
or dark theme on the embedded frame. A fixed palette therefore cannot work: dark
ink on Snowsight's dark canvas renders headings invisible. Colours are resolved
per render from the theme Streamlit reports, with a media-query fallback for
hosts that do not report one.
"""

from __future__ import annotations

import streamlit as st

_LIGHT_PALETTE = {
    "fg-ink": "#17212b",
    "fg-muted": "#66727e",
    "fg-line": "#dfe5e8",
    "fg-line-strong": "#9aa6af",
    "fg-surface": "#f7f9fa",
    "fg-card": "#ffffff",
    "fg-track": "#e7ecef",
    "fg-green": "#16806a",
    "fg-coral": "#d5654f",
    "fg-ready-halo": "#e2f1ed",
    "fg-offline-halo": "#f9e8e4",
    "fg-tooltip-bg": "#17212b",
    "fg-tooltip-fg": "#ffffff",
    "fg-shadow": "rgba(23, 33, 43, .18)",
}

# Contrast, not inversion: greens and corals are lightened so they stay legible
# against a dark canvas, and surfaces sit slightly above the page rather than
# below it as they do in light mode.
_DARK_PALETTE = {
    "fg-ink": "#e6ecf2",
    "fg-muted": "#9aa7b4",
    "fg-line": "#33404d",
    "fg-line-strong": "#5a6b7a",
    "fg-surface": "#1b242e",
    "fg-card": "#161e27",
    "fg-track": "#2b3641",
    "fg-green": "#3fc9a4",
    "fg-coral": "#f0876f",
    "fg-ready-halo": "rgba(63, 201, 164, .22)",
    "fg-offline-halo": "rgba(240, 135, 111, .22)",
    "fg-tooltip-bg": "#e6ecf2",
    "fg-tooltip-fg": "#111820",
    "fg-shadow": "rgba(0, 0, 0, .45)",
}


def _reported_theme() -> str | None:
    """Return the host's theme when Streamlit exposes one, otherwise None.

    Older Streamlit versions and non-standard hosts do not provide the context,
    so absence is a supported answer rather than an error.
    """

    try:
        theme_type = st.context.theme.type
    except Exception:
        return None
    return theme_type if theme_type in {"light", "dark"} else None


def is_dark_theme() -> bool:
    """Return whether the host is rendering the app on a dark canvas.

    Charts are drawn by Plotly rather than CSS, so they cannot inherit the
    variables above and need the resolved answer in Python.
    """

    return _reported_theme() == "dark"


def _variables(palette: dict[str, str]) -> str:
    return "".join(f"--{name}: {value};" for name, value in palette.items())


def apply_theme() -> None:
    """Apply restrained styling without changing Streamlit interaction semantics."""

    reported = _reported_theme()
    active = _DARK_PALETTE if reported == "dark" else _LIGHT_PALETTE
    # Only guess from the operating system when the host reported nothing.
    # Applying it unconditionally would override a host that deliberately
    # renders light while the workstation prefers dark.
    fallback = (
        ""
        if reported is not None
        else f"@media (prefers-color-scheme: dark) {{ :root {{ {_variables(_DARK_PALETTE)} }} }}"
    )

    palette_css = (
        f":root {{ {_variables(_LIGHT_PALETTE)} }}"
        f":root {{ {_variables(active)} }}"
        f"{fallback}"
        # Neutrals are re-derived from the host's own text colour, so surfaces and
        # rules stay legible even where the theme cannot be detected at all. Each
        # declaration follows a literal above it: a browser without color-mix
        # discards these and keeps the detected palette.
        ":root {"
        "--fg-muted: color-mix(in srgb, currentColor 62%, transparent);"
        "--fg-line: color-mix(in srgb, currentColor 18%, transparent);"
        "--fg-line-strong: color-mix(in srgb, currentColor 42%, transparent);"
        "--fg-surface: color-mix(in srgb, currentColor 5%, transparent);"
        "--fg-card: color-mix(in srgb, currentColor 3%, transparent);"
        "--fg-track: color-mix(in srgb, currentColor 12%, transparent);"
        # The accents need the same treatment. Snowsight sets the frame's theme
        # from its own appearance setting, which a Streamlit that reports no
        # theme leaves the workstation's preference to guess at — and the guess
        # is wrong whenever the two disagree. Observed in Snowsight's light theme
        # on a dark workstation: the page eyebrow rendered the dark palette's
        # mint on white and was effectively invisible. Blending the brand hue
        # toward the host's own text colour keeps it the same colour family and
        # legible on either canvas, without having to know which one it is.
        "--fg-green: color-mix(in srgb, #1aa183 72%, currentColor);"
        "--fg-coral: color-mix(in srgb, #e0705a 72%, currentColor);"
        "}"
    )

    st.html(
        "<style>"
        + palette_css
        + _STATIC_CSS
        + "</style>"
    )


_STATIC_CSS = """
          [data-testid="stSidebar"] { border-right: 1px solid var(--fg-line); }
          /* Streamlit normally reserves a full header row above sidebar content.
             Overlay its controls so navigation starts at the top of the panel. */
          [data-testid="stSidebarContent"] { position: relative; }
          [data-testid="stSidebarHeader"] {
            position: absolute;
            top: .65rem;
            right: .65rem;
            z-index: 2;
            width: auto;
            height: auto;
          }
          [data-testid="stSidebarUserContent"] {
            padding-top: .65rem !important;
          }
          /* The main content deliberately starts under Streamlit's header row so
             the page begins at the top of the frame. A header that paints its own
             background therefore paints over the page's first line: in Snowsight's
             light theme the near-opaque white band washed out both the page eyebrow
             and the rename control, while dark mode — already transparent — looked
             correct. The header stays transparent in every theme. */
          [data-testid="stHeader"] { background: transparent; }
          [data-testid="stMainBlockContainer"], .block-container {
            padding-top: .65rem !important;
          }
          /* Live fragments refresh independently. Streamlit's global status
             widget makes those small polls look like full-page reloads, so the
             app presents its own timestamp inside the refreshed region. */
          [data-testid="stStatusWidget"] { display: none !important; }
          h1, h2, h3 { letter-spacing: 0 !important; }
          h1 { font-size: 2rem !important; }
          h2 { font-size: 1.35rem !important; margin-top: 1.2rem !important; }
          h3 { font-size: 1rem !important; }
          div[data-testid="stMetric"] {
            background: var(--fg-surface);
            border: 1px solid var(--fg-line);
            border-radius: 6px;
            padding: .8rem 1rem;
          }
          div[data-testid="stMetricValue"] { font-size: 1.55rem; }
          .fg-kicker {
            color: var(--fg-green);
            font-weight: 700;
            font-size: .75rem;
            text-transform: uppercase;
            letter-spacing: .08em;
            margin-bottom: .2rem;
          }
          .fg-subtle { color: var(--fg-muted); font-size: .9rem; }
          .fg-stage {
            border-bottom: 1px solid var(--fg-line);
            padding: .65rem 0 .75rem;
          }
          .fg-stage:last-child { border-bottom: 0; }
          .fg-stage-header {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 1rem;
            margin-bottom: .42rem;
          }
          .fg-stage-status {
            color: var(--fg-muted);
            max-width: 52%;
            overflow-wrap: anywhere;
            text-align: right;
            text-transform: capitalize;
          }
          .fg-help-icon {
            display: inline-flex;
            position: relative;
            align-items: center;
            justify-content: center;
            width: .95rem;
            height: .95rem;
            margin: 0 .32rem;
            border: 1px solid var(--fg-line-strong);
            border-radius: 50%;
            background: transparent;
            color: var(--fg-muted);
            font-size: .66rem;
            font-weight: 700;
            line-height: 1;
            padding: 0;
            cursor: help;
            vertical-align: .08rem;
          }
          .fg-help-icon:focus,
          .fg-help-icon:hover {
            border-color: var(--fg-green);
            color: var(--fg-green);
            outline: none;
          }
          .fg-help-icon:focus::after,
          .fg-help-icon:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            left: 1.35rem;
            top: 50%;
            z-index: 1000;
            width: min(25rem, 62vw);
            transform: translateY(-50%);
            border-radius: 5px;
            background: var(--fg-tooltip-bg);
            box-shadow: 0 4px 14px var(--fg-shadow);
            color: var(--fg-tooltip-fg);
            font-size: .78rem;
            font-weight: 400;
            line-height: 1.4;
            padding: .55rem .7rem;
            pointer-events: none;
            text-align: left;
            white-space: normal;
          }
          .fg-stage-track {
            width: 100%;
            height: .38rem;
            overflow: hidden;
            border-radius: 3px;
            background: var(--fg-track);
          }
          .fg-stage-track > span {
            display: block;
            height: 100%;
            border-radius: inherit;
            background: var(--fg-green);
            transition: width .25s ease-out;
          }
          .fg-run-timestamp {
            color: var(--fg-muted);
            font-size: .8rem;
            line-height: 2.4rem;
            text-align: right;
            width: 100%;
          }
          .fg-detail {
            border-left: 3px solid var(--fg-green);
            background: var(--fg-surface);
            padding: .8rem 1rem;
          }
          .fg-node {
            display: grid;
            grid-template-columns: minmax(12rem, 1.8fr)
              repeat(5, minmax(5.5rem, .8fr)) minmax(12rem, 1.4fr);
            align-items: center;
            gap: 1rem;
            border: 1px solid var(--fg-line);
            border-radius: 7px;
            background: var(--fg-card);
            padding: .9rem 1rem;
            margin-bottom: .55rem;
          }
          .fg-node-identity { display: flex; align-items: center; gap: .7rem; min-width: 0; }
          .fg-node-identity div { display: flex; flex-direction: column; min-width: 0; }
          .fg-node-identity strong,
          .fg-node-stat strong { overflow-wrap: anywhere; }
          .fg-node-identity span:not(.fg-status-dot),
          .fg-node-stat span { color: var(--fg-muted); font-size: .74rem; }
          .fg-status-dot { width: .62rem; height: .62rem; border-radius: 50%; flex: 0 0 auto; }
          .fg-status-dot.ready {
            background: var(--fg-green);
            box-shadow: 0 0 0 3px var(--fg-ready-halo);
          }
          .fg-status-dot.offline {
            background: var(--fg-coral);
            box-shadow: 0 0 0 3px var(--fg-offline-halo);
          }
          .fg-node-stat { display: flex; flex-direction: column; min-width: 0; }
          .fg-node-stat strong { font-size: .86rem; font-weight: 650; }
          @media (max-width: 900px) {
            .fg-node { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .fg-node-identity, .fg-node-stat-wide { grid-column: 1 / -1; }
          }
          div[data-testid="stButtonGroup"] [role="radiogroup"] {
            column-gap: .35rem !important;
          }
          button, [data-baseweb="select"] > div, input, textarea { border-radius: 5px !important; }
          [data-testid="stFileUploaderDropzone"] { border-radius: 6px; }
          /* Graph history is a navigation list, so its labels scan from one
             stable left edge. Scope these rules to run buttons so compact row
             actions keep a centered, accessible hit area of their own. */
          [data-testid="stSidebar"] .st-key-graph_run_list {
            gap: .4rem !important;
          }
          [data-testid="stSidebar"] .st-key-graph_run_list
          [class*="st-key-run_"] button {
            justify-content: flex-start !important;
            text-align: left !important;
            height: 3.125rem;
            padding: .42rem .62rem !important;
          }
          [data-testid="stSidebar"] .st-key-graph_run_list
          [class*="st-key-run_"] button > div,
          [data-testid="stSidebar"] .st-key-graph_run_list
          [class*="st-key-run_"] button > div > span {
            justify-content: flex-start !important;
            width: 100%;
          }
          [data-testid="stSidebar"] .st-key-graph_run_list
          [class*="st-key-run_"] button p {
            flex: 1 1 auto;
            width: 100%;
            font-size: .82rem;
            line-height: 1.32;
            text-align: left !important;
            white-space: pre-line;
          }
          [data-testid="stSidebar"] .st-key-graph_run_list
          [class*="st-key-forget_"] button {
            width: 100% !important;
            min-width: 0 !important;
            height: 2.25rem;
            justify-content: center !important;
            padding: 0 !important;
          }
          @keyframes fg-status-spin {
            to { transform: rotate(360deg); }
          }
          [data-testid="stSidebar"] [class*="st-key-run_active_"]
          [data-testid="stIconMaterial"] {
            animation: fg-status-spin 1.1s linear infinite;
          }
          @media (prefers-reduced-motion: reduce) {
            [data-testid="stSidebar"] [class*="st-key-run_active_"]
            [data-testid="stIconMaterial"] { animation: none; }
          }
          [data-testid="stSidebar"] .st-key-graph_run_list
          [data-testid="stHorizontalBlock"] {
            align-items: center;
            gap: .4rem;
          }
"""


def page_heading(kicker: str, title: str, description: str) -> None:
    """Render a consistent compact heading for one operational view."""

    st.html(f'<div class="fg-kicker">{kicker}</div>')
    st.title(title)
    st.caption(description)
