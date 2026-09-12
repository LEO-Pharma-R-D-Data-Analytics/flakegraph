"""Deterministic, record-level evidence grounding for model observations."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

_SENTENCE_RE = re.compile(r"[^\n.!?]+(?:[.!?]+|(?=\n|$))", re.MULTILINE)
_MIN_SPLITTABLE_OCR_TOKEN_LENGTH = 2


@dataclass(frozen=True)
class GroundedSpan:
    """Represent an exact source substring and chunk-local character offsets.

    ``repair`` records when a provider quote was replaced by a source sentence,
    allowing traces to distinguish direct grounding from conservative recovery.
    """

    quote: str
    start_offset: int
    end_offset: int
    repair: str | None = None


@dataclass(frozen=True)
class SurfaceTextIndex:
    """Cache normalized source text and its exact original offset mapping."""

    normalized_source: str
    source_starts: list[int]
    source_ends: list[int]


def ground_evidence(
    source: str,
    proposed_quote: str,
    required_surface_groups: list[list[str]],
) -> GroundedSpan | None:
    """Ground a quote or repair it to the shortest supporting sentence.

    Each surface group represents one required endpoint and is satisfied when
    any configured name or alias occurs in the source span.  Invalid records
    are dropped independently; one hallucinated quote never rewrites or loses
    the other valid observations returned by the same provider call.
    """

    quote_span = _find_quote(source, proposed_quote)
    if quote_span is not None and _contains_groups(quote_span.quote, required_surface_groups):
        return quote_span
    return _supporting_span(source, required_surface_groups)


def surface_occurs(source: str, surfaces: list[str]) -> bool:
    """Return whether any complete, token-bounded entity surface occurs in text.

    The helper shares the exact matching policy used for evidence repair so alias
    validation and relation grounding cannot disagree about occurrence.
    """

    return bool(surface_spans(source, surfaces))


def build_surface_text_index(source: str) -> SurfaceTextIndex:
    """Normalize one source chunk once for repeated endpoint grounding checks."""

    normalized_source, source_starts, source_ends = _normalized_text_with_offsets(source)
    return SurfaceTextIndex(normalized_source, source_starts, source_ends)


def indexed_surface_occurs(index: SurfaceTextIndex, surfaces: list[str]) -> bool:
    """Check surfaces against a previously normalized source chunk."""

    return bool(_surface_spans_from_index(index, surfaces))


def indexed_surface_spans(
    index: SurfaceTextIndex,
    surfaces: list[str],
) -> list[tuple[int, int]]:
    """Return exact source offsets for surfaces using a precomputed text index."""

    return _surface_spans_from_index(index, surfaces)


def surface_spans(source: str, surfaces: list[str]) -> list[tuple[int, int]]:
    """Find token-bounded spans while tolerating punctuation between name tokens.

    Substring matching would incorrectly ground ``India`` in ``Indian`` and
    ``Greece`` in ``Greek``. Token boundaries are therefore part of the graph
    evidence contract, while spaces and hyphens remain interchangeable. A
    multi-token surface may also carry an attached numeric footnote marker, as
    PDF extractors commonly render an author and affiliation as ``Jane Doe1*``.
    One token may contain a first-letter OCR split such as ``M ETHOD``; allowing
    exactly one such split preserves identity while avoiding general fuzzy
    matching. Single-token surfaces keep the strict boundary to avoid matching a
    generic name such as ``Model`` inside ``Model2``.
    """

    return _surface_spans_from_index(build_surface_text_index(source), surfaces)


def _surface_spans_from_index(
    index: SurfaceTextIndex,
    surfaces: list[str],
) -> list[tuple[int, int]]:
    normalized_source = index.normalized_source
    source_starts = index.source_starts
    source_ends = index.source_ends
    spans: set[tuple[int, int]] = set()
    for surface in surfaces:
        normalized_surface = unicodedata.normalize("NFKC", surface).casefold()
        tokens = re.findall(r"\w+", normalized_surface, flags=re.UNICODE)
        if not tokens:
            continue
        trailing_marker = (
            r"(?:\d+[*†‡]*)?" if len(tokens) > 1 and not tokens[-1].isdigit() else ""
        )
        for pattern in _compiled_surface_patterns(tuple(tokens), trailing_marker):
            for match in pattern.finditer(normalized_source):
                spans.add((source_starts[match.start()], source_ends[match.end() - 1]))
    return sorted(spans)


@lru_cache(maxsize=8192)
def _compiled_surface_patterns(
    tokens: tuple[str, ...],
    trailing_marker: str,
) -> tuple[re.Pattern[str], ...]:
    """Compile one surface's patterns once and reuse them across every chunk.

    Grounding matches every accepted entity against every chunk of its window, so
    the same handful of patterns per entity is used hundreds of times. The re
    module's own cache is bounded and a corpus of a few hundred entities evicts
    faster than it reuses, leaving the work to be redone on nearly every call.
    """

    return tuple(
        re.compile(pattern) for pattern in _surface_patterns(list(tokens), trailing_marker)
    )


def _surface_patterns(tokens: list[str], trailing_marker: str) -> list[str]:
    """Build exact and single-prefix-split patterns for one entity surface.

    A separate alternative is emitted for each eligible token, which means one
    match can repair at most one PDF extraction artifact. The rest of the token
    sequence must remain exact and token-bounded.
    """

    token_patterns = [re.escape(token) for token in tokens]
    alternatives = [token_patterns]
    for index, token in enumerate(tokens):
        if len(token) < _MIN_SPLITTABLE_OCR_TOKEN_LENGTH:
            continue
        repaired = list(token_patterns)
        repaired[index] = re.escape(token[0]) + r"[\W_]+" + re.escape(token[1:])
        alternatives.append(repaired)
    return [
        r"(?<!\w)" + r"[\W_]+".join(parts) + trailing_marker + r"(?!\w)" for parts in alternatives
    ]


def sentence_spans(source: str) -> list[GroundedSpan]:
    """Return trimmed, offset-preserving sentence spans from normalized source text.

    Extraction and deterministic candidate discovery share this boundary so a
    cue-derived relation can retain the same exact evidence contract as an LLM-proposed
    relation. Line-oriented fragments without terminal punctuation remain supported.
    """

    spans: list[GroundedSpan] = []
    for match in _SENTENCE_RE.finditer(source):
        raw = match.group(0)
        left_trim = len(raw) - len(raw.lstrip())
        right_trimmed = raw.rstrip()
        start = match.start() + left_trim
        end = match.start() + len(right_trimmed)
        sentence = source[start:end]
        if sentence:
            spans.append(GroundedSpan(sentence, start, end))
    return spans


def _find_quote(source: str, quote: str) -> GroundedSpan | None:
    """Locate a proposed quote exactly or with whitespace-only normalization.

    Even when whitespace is repaired, the returned text and offsets always refer
    to the original source rather than a model-normalized representation.
    """

    stripped = quote.strip()
    if not stripped:
        return None
    exact_start = source.find(stripped)
    if exact_start >= 0:
        return GroundedSpan(stripped, exact_start, exact_start + len(stripped))

    # OCR and model serializers can normalize whitespace while preserving all
    # words. Match that representation but always return the exact source span.
    tokens = stripped.split()
    if not tokens:
        return None
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    match = re.search(pattern, source, flags=re.IGNORECASE)
    if match is not None:
        return GroundedSpan(source[match.start() : match.end()], match.start(), match.end())

    # PDF text commonly contains compatibility ligatures such as ``ﬁ`` while a
    # model returns ordinary ``fi``. Match in NFKC/casefold space, then map the
    # span back to the untouched source so evidence offsets and quotes remain
    # exact. No edit-distance matching is used: words still have to agree fully.
    normalized_source, source_starts, source_ends = _normalized_text_with_offsets(source)
    normalized_quote = unicodedata.normalize("NFKC", stripped).casefold()
    normalized_tokens = normalized_quote.split()
    if normalized_tokens:
        normalized_pattern = r"\s+".join(re.escape(token) for token in normalized_tokens)
        normalized_match = re.search(normalized_pattern, normalized_source)
        if normalized_match is not None:
            start = source_starts[normalized_match.start()]
            end = source_ends[normalized_match.end() - 1]
            return GroundedSpan(source[start:end], start, end)
    return None


def _normalized_text_with_offsets(source: str) -> tuple[str, list[int], list[int]]:
    """Build NFKC/casefold text and an exact normalized-to-source offset map.

    Compatibility characters can expand to several normalized code points. Each
    expanded character records the same original span, allowing callers to map a
    successful normalized regex match back without modifying source evidence.
    Visual line-break hyphens between word characters are then removed from the
    matching view while their source positions remain recoverable through the
    surrounding characters. Ordinary compounds without following whitespace are
    preserved.
    """

    normalized: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    index = 0
    while index < len(source):
        cluster_end = index + 1
        while cluster_end < len(source) and unicodedata.combining(source[cluster_end]):
            cluster_end += 1
        cluster = source[index:cluster_end]
        for normalized_character in unicodedata.normalize("NFKC", cluster).casefold():
            normalized.append(normalized_character)
            starts.append(index)
            ends.append(cluster_end)
        index = cluster_end
    return _remove_visual_line_break_hyphens("".join(normalized), starts, ends)


def _remove_visual_line_break_hyphens(
    normalized: str,
    starts: list[int],
    ends: list[int],
) -> tuple[str, list[int], list[int]]:
    """Remove PDF line-wrap hyphens from a normalized offset-mapped string."""

    removed_indices = {
        index
        for match in re.finditer(r"(?<=\w)-\s+(?=\w)", normalized)
        for index in range(match.start(), match.end())
    }
    if not removed_indices:
        return normalized, starts, ends
    kept_indices = [index for index in range(len(normalized)) if index not in removed_indices]
    return (
        "".join(normalized[index] for index in kept_indices),
        [starts[index] for index in kept_indices],
        [ends[index] for index in kept_indices],
    )


def _supporting_span(
    source: str,
    required_surface_groups: list[list[str]],
) -> GroundedSpan | None:
    """Select the shortest one-to-three-sentence span containing every surface group.

    Most evidence is sentence-local. A bounded adjacent span also covers explicit
    antecedents such as ``The model is X. It uses Y`` without allowing arbitrary
    document-level co-occurrence. Length and source offset provide deterministic
    tie-breaking when several spans can ground the same observation.
    """

    source_sentences = sentence_spans(source)
    candidates: list[GroundedSpan] = []
    for start_index, first in enumerate(source_sentences):
        for end_index in range(start_index, min(start_index + 3, len(source_sentences))):
            last = source_sentences[end_index]
            quote = source[first.start_offset : last.end_offset]
            if quote and _contains_groups(quote, required_surface_groups):
                candidates.append(
                    GroundedSpan(
                        quote=quote,
                        start_offset=first.start_offset,
                        end_offset=last.end_offset,
                        repair=(
                            "supporting_sentence"
                            if end_index == start_index
                            else "supporting_passage"
                        ),
                    )
                )
                # Longer spans from the same starting sentence cannot be the
                # shortest candidate once all endpoint groups are present.
                break
    return min(candidates, key=lambda item: (len(item.quote), item.start_offset), default=None)


def _contains_groups(source: str, groups: list[list[str]]) -> bool:
    """Require a distinct complete source span for every endpoint alias group.

    Merely finding each string is insufficient when one entity name is nested in
    another, such as ``gradient descent`` inside ``stochastic gradient descent``.
    Distinct graph endpoints must be witnessed by non-overlapping text spans so a
    single mention cannot masquerade as both source and target. Backtracking is
    bounded by the small endpoint-group count used by extraction contracts.
    """

    spans_by_group = [surface_spans(source, group) for group in groups]
    if any(not spans for spans in spans_by_group):
        return False
    return _has_non_overlapping_group_spans(spans_by_group, 0, [])


def _has_non_overlapping_group_spans(
    spans_by_group: list[list[tuple[int, int]]],
    group_index: int,
    selected: list[tuple[int, int]],
) -> bool:
    """Find one mutually non-overlapping span from each endpoint group."""

    if group_index == len(spans_by_group):
        return True
    for candidate in spans_by_group[group_index]:
        if all(
            candidate[1] <= existing[0] or existing[1] <= candidate[0] for existing in selected
        ) and _has_non_overlapping_group_spans(
            spans_by_group,
            group_index + 1,
            [*selected, candidate],
        ):
            return True
    return False
