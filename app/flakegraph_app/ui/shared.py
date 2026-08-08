"""Reusable Streamlit renderers for progress and provider controls."""

from __future__ import annotations

import html
from collections.abc import Sequence
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any, Protocol, cast

import pandas as pd
import streamlit as st
from flakegraph_app.models import RUN_STAGE_ORDER, ProviderSelection, RunSnapshot
from flakegraph_app.providers import (
    CORTEX_EMBEDDING_MODELS_BY_WIDTH,
    DEFAULTS,
    ProviderOption,
    cortex_embedding_model_for_width,
    option_by_name,
)

_BYTES_PER_UNIT = 1024
_JUST_NOW_SECONDS = 10
_SECONDS_PER_MINUTE = 60
_MINUTES_PER_HOUR = 60
_HOURS_PER_DAY = 24
_STAGE_HELP = {
    "prepare_document": (
        "Read the source, extract embedded text or run OCR when needed, normalize pages, "
        "and split the document into overlapping evidence chunks."
    ),
    "extract_document_context": (
        "Identify document-level subjects that help later windows keep important "
        "entities consistent across the whole source."
    ),
    "extract_entity_window": (
        "Ask the selected LLM to extract typed entity mentions from one bounded text "
        "window, retaining exact source evidence for every accepted mention."
    ),
    "compact_entity_inventory": (
        "Combine independent entity windows into a document-wide inventory for the relation pass."
    ),
    "extract_relation_window": (
        "Extract and verify evidence-backed relationships between locally grounded "
        "entities in one text window."
    ),
    "compact_document": (
        "Assemble the document's entity inventory and relation windows into one "
        "immutable document graph shard."
    ),
    "finalize_graph": (
        "Resolve duplicate identities, build canonical nodes and edges, detect "
        "communities, run quality checks, and publish the graph destination."
    ),
    # A Snowflake run reports the single-process pipeline's own stages. Left
    # unexplained they read as jargon to anyone who did not build the pipeline.
    "file_source": (
        "List the documents the run will process from the selected upload or "
        "Snowflake stage, and record where each one came from."
    ),
    "ocr": (
        "Read text from each document, using the selected OCR provider only for "
        "the pages that carry no embedded text."
    ),
    "chunking": (
        "Normalize pages and split every document into overlapping evidence "
        "chunks small enough for the model to quote exactly."
    ),
    "chunk_embeddings": (
        "Embed each evidence chunk so the graph can later be searched by meaning "
        "rather than by keyword."
    ),
    "graph_extraction": (
        "Ask the selected LLM for typed entities and evidence-backed relations, "
        "one bounded window at a time, keeping the source quote for each claim."
    ),
    "filtering": (
        "Drop candidates whose cited evidence cannot be found in the documents "
        "the run actually read."
    ),
    "merge": (
        "Resolve mentions that name the same real thing into one canonical node, "
        "combining their descriptions and evidence."
    ),
    "community_detection": (
        "Group densely connected nodes into communities and summarize each one so "
        "the graph can be read at a higher level."
    ),
    "quality": (
        "Score the finished graph — evidence support, isolates, duplicate "
        "identities — and record the result alongside the snapshot."
    ),
    "graph_embeddings": (
        "Embed canonical nodes and edges so the finished graph supports semantic "
        "search and neighbour suggestions."
    ),
    "write": (
        "Publish the finished snapshot to the configured destination, replacing "
        "the previous version of this graph atomically."
    ),
}


class _WidgetState(Protocol):
    """Structural subset shared by dictionaries and Streamlit session state."""

    def get(self, key: str, default: Any = None, /) -> Any:
        """Return a stored widget value or its default."""
        ...

    def __setitem__(self, key: str, value: Any) -> None:
        """Store a widget value."""
        ...

    def __delitem__(self, key: str) -> None:
        """Remove a widget value."""
        ...


def concise_error(error: Exception) -> str:
    """Return the final actionable line from a noisy CLI or API failure."""

    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    return lines[-1] if lines else type(error).__name__


def seeded_widget(state: _WidgetState, key: str, default: object) -> str:
    """Seed a keyed widget's value once and return its key.

    Streamlit reports a widget that carries both a default and a session-state
    entry as a conflict on the page itself, so the default is written to state
    before the widget is created rather than passed alongside it.
    """

    if state.get(key, None) is None:
        state[key] = default
    return key


def secret_text_input(
    label: str,
    *,
    key: str,
    help_text: str | None = None,
) -> str:
    """Render a masked text field with Streamlit's built-in reveal control.

    Password mode prevents shoulder-surfing and adds the familiar eye button for
    explicit show/hide control. The submitted value remains under the caller's
    normal widget key and is never copied into labels, help text, logs, or another
    session-state entry.
    """

    return str(
        st.text_input(
            label,
            type="password",
            key=key,
            help=help_text,
        )
    )


def provider_controls(
    title: str,
    key_prefix: str,
    options: tuple[ProviderOption, ...],
    default_name: str,
    *,
    required_dimension: int | None = None,
) -> ProviderSelection:
    """Render one adapter selector and only the fields required by that adapter.

    Streamlit retains widget values after a selectbox changes. The synchronization
    step resets provider-owned model, endpoint, credential, and dimension fields
    to the newly selected adapter's neutral defaults so a prior provider cannot
    leak stale connection data into the effective request.

    ``required_dimension`` is the width a destination already stores. Where one
    exists it is the only width that can be written, so it seeds the field and is
    stated next to it rather than left for the operator to discover.
    """

    labels = {option.name: option.label for option in options}
    names = list(labels)
    # Changing runtime clears this key rather than assigning the new default, so
    # the selector falls back to the index below. Assigning it instead makes
    # Streamlit report the widget's default and its stored value as disagreeing,
    # on the page, and a selector that carries no default becomes clearable —
    # offering an empty adapter this form has no meaning for.
    selected_name = st.selectbox(
        title,
        names,
        index=names.index(default_name) if default_name in names else 0,
        format_func=lambda name: labels[name],
        key=f"{key_prefix}_provider",
        help=f"Choose the {key_prefix} adapter used for this graph run.",
    )
    selected = option_by_name(options, selected_name)
    endpoint_default, model_default = DEFAULTS.get(key_prefix, {}).get(selected_name, ("", ""))
    # The destination's stored width decides which models can be written at all,
    # so it replaces the adapter's own default rather than only the dimension
    # beside it. A default model and a default width that disagree would send
    # vectors of one shape at a column of another.
    fitted_model = cortex_embedding_model_for_width(required_dimension)
    if fitted_model and model_default in CORTEX_EMBEDDING_MODELS_BY_WIDTH.values():
        model_default = fitted_model
    _synchronize_provider_fields(
        cast(_WidgetState, st.session_state),
        key_prefix,
        selected,
        endpoint_default=endpoint_default,
        model_default=model_default,
    )
    # A keyed widget takes its value from session state. Passing a default as
    # well makes Streamlit warn on the page that the two disagree, so each
    # default is seeded into state and the widget then reads only from there.
    widget_state = cast(_WidgetState, st.session_state)
    model = (
        st.text_input(
            "Model",
            key=seeded_widget(widget_state, f"{key_prefix}_model", model_default),
            help=f"Provider model identifier used by the selected {key_prefix} adapter.",
        )
        if selected.needs_model
        else None
    )
    endpoint = (
        st.text_input(
            "Endpoint",
            key=seeded_widget(widget_state, f"{key_prefix}_endpoint", endpoint_default),
            help=f"Base URL for the selected {key_prefix} service.",
        )
        if selected.needs_endpoint
        else None
    )
    credential = (
        secret_text_input(
            "API key environment variable",
            key=seeded_widget(
                widget_state,
                f"{key_prefix}_credential",
                {
                    "ocr": "KG_OCR_API_KEY",
                    "llm": "KG_LLM_API_KEY",
                    "embedding": "KG_EMBED_API_KEY",
                }[key_prefix],
            ),
            help_text="The app stores the variable name, never the credential value.",
        )
        if selected.needs_api_key
        else None
    )
    dimension = (
        int(
            st.number_input(
                "Dimensions",
                min_value=1,
                step=1,
                key=seeded_widget(
                    widget_state,
                    f"{key_prefix}_dimension",
                    required_dimension or selected.default_dimension or 384,
                ),
                help=(
                    f"The graph tables store {required_dimension}-wide vectors, so the "
                    "embedding model has to produce that width."
                    if required_dimension
                    else (
                        "Vector width produced by the selected embedding model and used "
                        "by graph storage schemas."
                    )
                ),
            )
        )
        if selected.default_dimension is not None
        else None
    )
    return ProviderSelection(
        provider=selected_name,
        model=model or None,
        endpoint=endpoint or None,
        api_key_environment_variable=credential or None,
        dimension=dimension,
    )


def _synchronize_provider_fields(
    state: _WidgetState,
    key_prefix: str,
    selected: ProviderOption,
    *,
    endpoint_default: str,
    model_default: str,
) -> None:
    """Clear stale widget state when an adapter selection changes.

    The first render records ownership without disturbing values restored by
    Streamlit. Later provider changes clear all provider-specific fields and seed
    only defaults valid for the new adapter.
    """

    owner_key = f"_{key_prefix}_fields_provider"
    previous = state.get(owner_key)
    state[owner_key] = selected.name
    if previous is None or previous == selected.name:
        return
    for suffix in ("model", "endpoint", "credential", "dimension"):
        with suppress(KeyError):
            del state[f"{key_prefix}_{suffix}"]
    if selected.needs_model:
        state[f"{key_prefix}_model"] = model_default
    if selected.needs_endpoint:
        state[f"{key_prefix}_endpoint"] = endpoint_default
    if selected.default_dimension is not None:
        state[f"{key_prefix}_dimension"] = selected.default_dimension


def render_run_snapshot(snapshot: RunSnapshot) -> None:
    """Present one run with durable counters, stages, warnings, and recent events."""

    columns = st.columns(4)
    columns[0].metric(
        "Status",
        snapshot.status.replace("_", " ").title(),
        help="Current state of the complete graph run, aggregated across every task.",
    )
    columns[1].metric(
        "Documents",
        snapshot.documents_total if snapshot.documents_total is not None else "—",
        help="Number of source documents registered for this run.",
    )
    columns[2].metric(
        "Completed",
        snapshot.documents_completed,
        help="Documents whose graph shards have completed successfully.",
    )
    columns[3].metric(
        "Failed",
        snapshot.documents_failed,
        help="Documents that stopped after exhausting their configured retries.",
    )
    st.caption(f"Last update: {format_relative_time(snapshot.updated_at)}")
    if snapshot.warnings:
        for warning in snapshot.warnings:
            st.warning(warning)
    if snapshot.error:
        st.error(snapshot.error)
    if snapshot.stages:
        st.subheader("Pipeline stages")
        stage_order = {stage: index for index, stage in enumerate(RUN_STAGE_ORDER)}
        ordered_stages = sorted(
            snapshot.stages,
            key=lambda item: stage_order.get(item.stage, len(stage_order)),
        )
        for stage in ordered_stages:
            total = (
                f"{stage.completed}/{stage.total}"
                if stage.total is not None
                else str(stage.completed or "")
            )
            duration = f" · {_format_duration(stage.elapsed_ms)}" if stage.elapsed_ms else ""
            progress = (
                min(100, round(stage.completed / stage.total * 100))
                if stage.total
                else (100 if stage.status.lower() in {"completed", "succeeded"} else 0)
            )
            help_text = _STAGE_HELP.get(
                stage.stage,
                f"Processing stage: {stage.stage.replace('_', ' ')}.",
            )
            status_label = stage.message or stage.status
            st.html(
                '<div class="fg-stage">'
                '<div class="fg-stage-header">'
                f"<span><strong>{html.escape(stage.stage.replace('_', ' ').title())}</strong>"
                f'<button class="fg-help-icon" type="button" '
                f'aria-label="{html.escape(help_text)}" '
                f'data-tooltip="{html.escape(help_text)}">?</button> '
                f'<span class="fg-subtle">{html.escape(total)}{duration}</span></span>'
                f'<span class="fg-stage-status">{html.escape(status_label)}</span>'
                "</div>"
                '<div class="fg-stage-track" role="progressbar" '
                f'aria-label="{html.escape(stage.stage)}" '
                f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{progress}">'
                f'<span style="width:{progress}%"></span>'
                "</div>"
                "</div>"
            )
    if snapshot.events:
        with st.expander("Recent events"):
            frame = pd.DataFrame(
                [
                    {
                        "Time": event.timestamp,
                        "Stage": event.stage,
                        "Status": event.status,
                        "File": event.file_id,
                        "Message": event.message,
                        "Duration (ms)": event.elapsed_ms,
                    }
                    for event in reversed(snapshot.events[-100:])
                ]
            )
            st.dataframe(frame, width="stretch", hide_index=True)


def format_bytes(value: int) -> str:
    """Format byte counts for compact source-object tables."""

    amount = float(value)
    units: Sequence[str] = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if amount < _BYTES_PER_UNIT or unit == units[-1]:
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= _BYTES_PER_UNIT
    return f"{value} B"


def format_relative_time(value: str | None, *, now: datetime | None = None) -> str:
    """Format an activity timestamp as a concise age for operator-facing pages.

    Provider timestamps may be absent, timezone-naive, or malformed. Returning a
    stable fallback keeps status rendering available while making healthy worker
    heartbeats and genuinely stale runs easy to distinguish at a glance.
    """

    if not value:
        return "Not reported"
    try:
        observed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return "Not reported"
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=UTC)
    elapsed_seconds = max(0, int(((now or datetime.now(UTC)) - observed).total_seconds()))
    if elapsed_seconds < _JUST_NOW_SECONDS:
        return "Just now"
    units = (
        (_SECONDS_PER_MINUTE, 1, "second"),
        (_SECONDS_PER_MINUTE * _MINUTES_PER_HOUR, _SECONDS_PER_MINUTE, "minute"),
        (
            _SECONDS_PER_MINUTE * _MINUTES_PER_HOUR * _HOURS_PER_DAY,
            _SECONDS_PER_MINUTE * _MINUTES_PER_HOUR,
            "hour",
        ),
    )
    for upper_bound, divisor, unit in units:
        if elapsed_seconds < upper_bound:
            return _relative_age(elapsed_seconds // divisor, unit)
    seconds_per_day = _SECONDS_PER_MINUTE * _MINUTES_PER_HOUR * _HOURS_PER_DAY
    return _relative_age(elapsed_seconds // seconds_per_day, "day")


def _relative_age(value: int, unit: str) -> str:
    """Pluralize one relative-time unit without adding a date-time dependency."""

    suffix = "" if value == 1 else "s"
    return f"{value} {unit}{suffix} ago"


def _format_duration(elapsed_ms: int) -> str:
    """Render stage durations compactly without hiding meaningful seconds."""

    total_seconds = max(0, round(elapsed_ms / 1000))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"
