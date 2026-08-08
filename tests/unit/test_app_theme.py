"""The app is embedded in Snowsight, which imposes its own light or dark canvas.

A fixed palette renders headings invisible on a dark host, which is how this was
first reported, so both the CSS palette and the Plotly canvas are covered here.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from flakegraph_app.explorer import build_graph_figure
from flakegraph_app.ui import theme
from flakegraph_app.ui.graph_explorer import _projection_signature


def _force_theme(monkeypatch: pytest.MonkeyPatch, value: object) -> list[str]:
    emitted: list[str] = []
    # Patched by dotted path: streamlit is imported into theme, not re-exported
    # by it, so reaching through the module attribute is not a typed access.
    monkeypatch.setattr(
        "flakegraph_app.ui.theme.st.html", lambda markup: emitted.append(str(markup))
    )
    context = (
        SimpleNamespace()
        if value is None
        else SimpleNamespace(theme=SimpleNamespace(type=value))
    )
    monkeypatch.setattr("flakegraph_app.ui.theme.st.context", context, raising=False)
    return emitted


def test_dark_host_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_theme(monkeypatch, "dark")

    assert theme.is_dark_theme() is True


def test_light_host_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_theme(monkeypatch, "light")

    assert theme.is_dark_theme() is False


def test_missing_context_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hosts that expose no theme must still render rather than raise."""

    _force_theme(monkeypatch, None)

    assert theme.is_dark_theme() is False


def test_dark_host_gets_readable_ink(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reported bug: dark ink on a dark canvas made titles unreadable."""

    emitted = _force_theme(monkeypatch, "dark")

    theme.apply_theme()
    css = "".join(emitted)

    assert f"--fg-ink: {theme._DARK_PALETTE['fg-ink']};" in css
    # The light value may still appear as the base layer, but the dark override
    # must come after it so it wins the cascade.
    assert css.rindex(theme._DARK_PALETTE["fg-ink"]) > css.rindex(
        theme._LIGHT_PALETTE["fg-ink"]
    )


def test_light_host_does_not_get_a_dark_media_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host rendering light must win over a workstation preferring dark."""

    emitted = _force_theme(monkeypatch, "light")

    theme.apply_theme()

    assert "prefers-color-scheme: dark" not in "".join(emitted)


def test_unknown_host_falls_back_to_the_operating_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted = _force_theme(monkeypatch, None)

    theme.apply_theme()

    assert "prefers-color-scheme: dark" in "".join(emitted)


def _figure_colors(dark: bool) -> dict[str, Any]:
    nodes = [
        {"id": "n1", "name": "Aikido", "primary_type": "CONCEPT", "description": ""},
        {"id": "n2", "name": "Judo", "primary_type": "CONCEPT", "description": ""},
    ]
    edges = [
        {
            "id": "e1",
            "source_node_id": "n1",
            "target_node_id": "n2",
            "relation_type": "RELATED_TO",
            "confidence": 0.9,
        }
    ]
    figure = build_graph_figure(nodes, edges, dark=dark)
    return {
        "paper": figure.layout.paper_bgcolor,
        "plot": figure.layout.plot_bgcolor,
        "font": figure.layout.font.color,
    }


def test_graph_canvas_never_paints_its_own_background() -> None:
    """A canvas that must be told its theme paints a lit panel where detection fails.

    Streamlit does not report a theme on every supported version — Snowflake pins
    one that does not — so the chart inherits the host background instead of
    choosing a colour it cannot reliably determine.
    """

    for dark in (True, False):
        colors = _figure_colors(dark=dark)
        assert colors["paper"] == "rgba(0,0,0,0)"
        assert colors["plot"] == "rgba(0,0,0,0)"


def test_graph_canvas_still_adapts_its_text() -> None:
    """Text is drawn by Plotly, so it cannot inherit and must follow the theme."""

    assert _figure_colors(dark=True)["font"] == "#e6ecf2"
    assert _figure_colors(dark=False)["font"] == "#17212b"


def test_neutrals_are_derived_from_the_host_text_colour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Surfaces must stay legible even where the host theme cannot be detected.

    Snowflake pins a Streamlit version that reports no theme, so a palette that
    depended entirely on detection would fall back to light surfaces on a dark
    canvas for anyone whose workstation preference disagrees with Snowsight.
    """

    emitted = _force_theme(monkeypatch, None)

    theme.apply_theme()
    css = "".join(emitted)

    for name in ("fg-muted", "fg-line", "fg-surface", "fg-card", "fg-track"):
        assert f"--{name}: color-mix(in srgb, currentColor" in css
    # A literal must precede each derived value so browsers without color-mix
    # keep a usable palette rather than dropping the variable entirely.
    assert css.index(f"--fg-line: {theme._LIGHT_PALETTE['fg-line']}") < css.index(
        "--fg-line: color-mix"
    )


def test_graph_canvas_avoids_webgl_traces() -> None:
    """Snowsight renders the app in a sandboxed frame that provides no WebGL.

    A WebGL trace there shows "WebGL is not supported by your browser" instead of
    the graph, while rendering normally in local development — so the constraint
    is invisible without an explicit check.
    """

    nodes = [
        {"id": "n1", "name": "Aikido", "primary_type": "CONCEPT", "description": ""},
        {"id": "n2", "name": "Judo", "primary_type": "CONCEPT", "description": ""},
    ]
    edges = [
        {
            "id": "e1",
            "source_node_id": "n1",
            "target_node_id": "n2",
            "relation_type": "RELATED_TO",
            "confidence": 0.9,
        }
    ]

    figure = build_graph_figure(nodes, edges)

    assert figure.data
    for trace in figure.data:
        assert "gl" not in trace.type, f"{trace.type} requires WebGL"


def test_accents_survive_a_host_theme_the_workstation_disagrees_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The brand hues must be legible without knowing which canvas they sit on.

    Snowsight sets the frame's theme from its own appearance setting, which the
    Streamlit it pins does not report, leaving the workstation preference to
    guess. Observed in Snowsight's light theme on a dark workstation: the page
    eyebrow drew the dark palette's mint on white and was invisible. Blending
    toward the host's own text colour removes the guess.
    """

    emitted = _force_theme(monkeypatch, None)

    theme.apply_theme()
    css = "".join(emitted)

    for name in ("fg-green", "fg-coral"):
        assert f"--{name}: color-mix(in srgb," in css
        assert "currentColor);" in css.split(f"--{name}: color-mix(in srgb,")[1][:80]
        # The literal still comes first for browsers without color-mix.
        assert css.index(f"--{name}: {theme._LIGHT_PALETTE[name]}") < css.index(
            f"--{name}: color-mix"
        )


def test_the_header_never_paints_over_the_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """Content starts under Streamlit's header, so the header must not be opaque.

    The main block's top padding is reduced so the page begins at the top of the
    frame. A header with its own background therefore covers the page's first
    line: in Snowsight's light theme a 94%-opaque white band washed out both the
    eyebrow and the rename control, while dark mode looked correct because its
    header was already transparent.
    """

    for reported in ("light", "dark", None):
        emitted = _force_theme(monkeypatch, reported)
        theme.apply_theme()
        css = "".join(emitted)

        assert '[data-testid="stHeader"] { background: transparent; }' in css
        assert "fg-header-bg" not in css


def test_the_graph_cache_key_tracks_the_projection_it_stands_for() -> None:
    """Identify a projection by its members, not by hashing every field of it.

    The figure cache is keyed on a signature so a lookup does not walk twelve
    hundred node dictionaries on every rerun. That only holds if the signature
    separates projections that must draw differently, and shares one entry for
    projections that are genuinely the same.
    """

    nodes = [{"id": f"n{index}"} for index in range(5)]
    edges = [{"id": "e1", "source_node_id": "n0", "target_node_id": "n1"}]

    assert _projection_signature(nodes, edges) == _projection_signature(list(nodes), list(edges))
    assert _projection_signature(nodes, edges) != _projection_signature(nodes[:4], edges)
    assert _projection_signature(nodes, edges) != _projection_signature(nodes, [])
    # Order decides layout, so a reordering is a different projection.
    assert _projection_signature(nodes, edges) != _projection_signature(nodes[::-1], edges)
