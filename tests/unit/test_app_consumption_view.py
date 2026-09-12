"""The consumption view's contract with an operator reading it."""

from __future__ import annotations

from flakegraph_app.ui.consumption import _usd


def test_sub_cent_costs_are_not_rounded_away() -> None:
    """A real cost shown as "$0.00" reads as free in a transparency view."""

    assert _usd(0.0025) == "$0.0025"
    assert _usd(12.5) == "$12.50"
    assert _usd(None) == "$0.00"
