"""What a graph consumed, and what it cost.

The view answers two different questions with the same data. "What did this cost"
is the billed total. "What did running it ourselves save" is the avoided total,
priced at a stated hosted alternative — a saving with no named comparison is not
a number anyone can check.

Where the rate card could not price a call, the view says so rather than showing a
confident total that quietly excludes it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, SupportsFloat, cast

import pandas as pd
import streamlit as st

_EVENT_COLUMNS = [
    "stage",
    "operation",
    "provider",
    "model",
    "locality",
    "calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "pages",
]

# Below one cent, two decimals would render a real cost as "$0.00", which reads
# as free in a view whose whole purpose is transparency.
_SUB_CENT = 0.01

_GROUPINGS = {
    "Stage": "by_stage",
    "Provider": "by_provider",
    "Model": "by_model",
    "Where it ran": "by_locality",
}


def render_consumption(metrics: Mapping[str, Any], graph_id: str) -> None:
    """Render the consumption view for one graph's metrics.

    ``graph_id`` namespaces the filter and sort controls. Streamlit keys a widget
    by name for the whole session, so shared keys would carry one graph's stage,
    provider, and model selections onto the next graph opened — where they match
    nothing, and the table reports no calls for a graph that made thousands.
    """

    consumption = metrics.get("consumption")
    if not isinstance(consumption, Mapping) or not consumption:
        # Age is one reason a graph carries no usage and not the only one: a
        # distributed run assembles its report from worker artifacts and records
        # none either. Naming a cause the app cannot check told an operator their
        # freshly built graph predated a feature it was built with.
        st.info(
            "This graph reports no recorded usage, so there are no tokens, pages "
            "or cost to show. Graphs built before usage was recorded, and runs "
            "whose runtime does not report it, both arrive here."
        )
        return

    totals = consumption.get("totals")
    totals = totals if isinstance(totals, Mapping) else {}
    _render_totals(totals)
    _render_incomplete_warning(totals)

    events = [event for event in consumption.get("events", []) if isinstance(event, Mapping)]
    if not events:
        st.caption("No provider calls were recorded for this graph.")
        return

    _render_breakdown(consumption, graph_id)
    _render_events(events, graph_id)


def _render_totals(totals: Mapping[str, Any]) -> None:
    """Show spend, saving and the volume behind both."""

    billed, avoided, tiles = st.columns([1, 1, 2])
    billed.metric(
        "Billed",
        _usd(totals.get("billed_usd")),
        help="Cost of work that ran on a hosted provider.",
    )
    avoided.metric(
        "Avoided",
        _usd(totals.get("avoided_usd")),
        help=(
            "What the locally executed work would have cost on the configured "
            "reference model. Nothing was billed for it."
        ),
    )
    with tiles:
        tokens, pages, calls = st.columns(3)
        tokens.metric("Tokens", f"{int(totals.get('total_tokens') or 0):,}")
        pages.metric("Pages parsed", f"{int(totals.get('pages') or 0):,}")
        calls.metric("Provider calls", f"{int(totals.get('calls') or 0):,}")


def _render_incomplete_warning(totals: Mapping[str, Any]) -> None:
    """Say plainly when a total excludes calls that could not be priced."""

    unpriced = int(totals.get("unpriced_calls") or 0)
    if not unpriced:
        return
    st.warning(
        f"{unpriced} of {int(totals.get('calls') or 0)} calls have no rate covering "
        "what they used, so the totals above exclude them. Add the missing rate "
        "under `consumption.rates` to price them — the recorded usage is already "
        "there, so no run has to be repeated."
    )


def _render_breakdown(consumption: Mapping[str, Any], graph_id: str) -> None:
    """Break spend down along whichever axis the operator picks."""

    st.subheader("Breakdown")
    label = st.radio(
        "Group by",
        list(_GROUPINGS),
        horizontal=True,
        key=f"consumption_grouping_{graph_id}",
        help="Attribute spend to the dimension you want to act on.",
    )
    grouped = consumption.get(_GROUPINGS[label])
    if not isinstance(grouped, Mapping) or not grouped:
        st.caption("Nothing recorded for this grouping.")
        return
    frame = pd.DataFrame(
        [
            {
                label: name,
                "Calls": int(value.get("calls") or 0),
                "Tokens": int(value.get("total_tokens") or 0),
                "Pages": int(value.get("pages") or 0),
                "Billed (USD)": float(value.get("billed_usd") or 0.0),
                "Avoided (USD)": float(value.get("avoided_usd") or 0.0),
                "Unpriced calls": int(value.get("unpriced_calls") or 0),
            }
            for name, value in grouped.items()
            if isinstance(value, Mapping)
        ]
    )
    st.dataframe(
        frame.sort_values("Billed (USD)", ascending=False),
        width="stretch",
        hide_index=True,
    )


def _render_events(events: Sequence[Mapping[str, Any]], graph_id: str) -> None:
    """Show every recorded call, filterable and sortable.

    The per-call table is kept rather than only totals: an unexpected bill is
    usually one stage or one model, and an aggregate cannot show which.
    """

    st.subheader("Every recorded call")
    st.caption(
        "One row per provider call. Cached results are absent because they called "
        "nobody and were never billed."
    )
    frame = pd.DataFrame(
        [
            {
                **{column: event.get(column) for column in _EVENT_COLUMNS if column != "pages"},
                "prompt_tokens": int(_usage(event).get("prompt_tokens") or 0),
                "completion_tokens": int(_usage(event).get("completion_tokens") or 0),
                "total_tokens": int(_usage(event).get("total_tokens") or 0),
                "pages": int(event.get("pages") or 0),
            }
            for event in events
        ]
    )

    filters = st.columns(4)
    for column, widget in zip(("stage", "provider", "model", "locality"), filters, strict=False):
        options = sorted({str(value) for value in frame[column].dropna().unique()})
        chosen = widget.multiselect(
            column.replace("_", " ").title(),
            options,
            key=f"cf_{column}_{graph_id}",
        )
        if chosen:
            frame = frame[frame[column].isin(chosen)]

    if frame.empty:
        st.caption("No calls match these filters.")
        return

    sort_column = st.selectbox(
        "Sort by",
        [column for column in _EVENT_COLUMNS if column in frame.columns],
        index=_EVENT_COLUMNS.index("total_tokens"),
        key=f"consumption_sort_{graph_id}",
    )
    descending = st.toggle(
        "Largest first",
        value=True,
        key=f"consumption_sort_desc_{graph_id}",
    )
    st.dataframe(
        frame.sort_values(sort_column, ascending=not descending),
        width="stretch",
        hide_index=True,
    )
    st.caption(f"{len(frame):,} of {len(events):,} calls shown.")


def _usage(event: Mapping[str, Any]) -> Mapping[str, Any]:
    usage = event.get("usage")
    return usage if isinstance(usage, Mapping) else {}


def _usd(value: object) -> str:
    """Format a cost without pretending to more precision than a rate card has.

    Sub-cent amounts keep four decimals; a run that cost a fraction of a cent
    should not display as ``$0.00`` when the point of the view is transparency.
    """

    # Costs reach the view as object: Snowflake returns NUMBER as Decimal and
    # VARIANT members untyped. Converting under except keeps Decimal working
    # while a value that is not a number formats as zero instead of raising in
    # the middle of a page render.
    try:
        amount = float(cast(SupportsFloat, value)) if value is not None else 0.0
    except (TypeError, ValueError):
        amount = 0.0
    if amount and abs(amount) < _SUB_CENT:
        return f"${amount:,.4f}"
    return f"${amount:,.2f}"
