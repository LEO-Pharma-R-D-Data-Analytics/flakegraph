# SPDX-License-Identifier: Apache-2.0
"""Recognize and parse values stated in text: limits, ranges, and results."""

from __future__ import annotations

import re

from kg_processor.domain.graph import Quantity

_RANGE_BOUNDS = 2
_UPPER_BOUND_COMPARATORS = {"<", "<=", "≤", "nmt", "not more than", "max", "maximum", "up to"}
_LOWER_BOUND_COMPARATORS = {">", ">=", "≥", "nlt", "not less than", "min", "minimum", "at least"}
_NUMBER_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")
_COMPARATOR_RE = re.compile(
    r"(<=|>=|≤|≥|<|>|\bnmt\b|\bnlt\b|\bnot more than\b|\bnot less than\b|\bmax(?:imum)?\b"
    r"|\bmin(?:imum)?\b|\bup to\b|\bat least\b)",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(r"\d\s*(?:-|–|—|to|\.\.)\s*[-+]?\d", re.IGNORECASE)
_UNIT_SYMBOL_RE = re.compile(r"[%‰°µμ]")
# A letter written straight before a digit: a model, part or project code.
_CODE_RE = re.compile(r"[^\W\d_]\d")
# What a value can be written with besides its number: units, comparators,
# range and list words. A name with nothing else is a value, not a thing.
_VALUE_WORD_RE = re.compile(
    r"\b(?:[kmµμnpcd]?(?:g|l|m|mol|s|pa|w|j|v|a|hz)|c|f|k|ppm|ppb|ppt|cfu|iu|u|meq|eq"
    r"|cps|cp|mpas|pas|rpm|rh|ph|bar|psi|min|h|hr|hrs|d|days?|weeks?|months?|years?|x"
    r"|max|min|nmt|nlt|not|more|less|than|to|and|or|up|at|least|approx|ca|about)\b",
    re.IGNORECASE,
)


def is_value_text(text: str) -> bool:
    """Whether text is only a stated value: numbers with units or comparators.

    "≤ 0.10 %", "154 - 162", "40 mg/kg max" and "0001271276" are values; "Tween 80",
    "Ph. Eur. 2.5.12" and "USP <711>" name things that merely contain numbers. A
    letter written straight before a digit makes a code - "S450", "U40",
    "J1-6746" - since a unit follows its number rather than leading it.
    """

    if not _NUMBER_RE.search(text) or _CODE_RE.search(text):
        return False
    remainder = _VALUE_WORD_RE.sub(" ", _UNIT_SYMBOL_RE.sub(" ", _NUMBER_RE.sub(" ", text)))
    return not re.search(r"[^\W\d_]", remainder)


def parse_quantity(
    text: str,
    *,
    kind: str | None = None,
    unit: str | None = None,
    comparator: str | None = None,
) -> Quantity:
    """Parse the bounds of a stated value while keeping its text as written.

    A range gives both bounds; "≤ 0.5" gives an upper bound; a single number gives
    both. Decimal commas are read as decimal points.
    """

    numbers = [float(match.replace(",", ".")) for match in _NUMBER_RE.findall(text)]
    # A model's comparator counts only when it is one; otherwise read the text.
    given = comparator.strip() if comparator else ""
    found = _COMPARATOR_RE.fullmatch(given) or _COMPARATOR_RE.search(text)
    symbol = found.group(1).lower() if found else None
    low: float | None = None
    high: float | None = None
    if numbers:
        if len(numbers) >= _RANGE_BOUNDS and _RANGE_RE.search(text):
            low, high = min(numbers[:2]), max(numbers[:2])
        elif symbol and symbol.strip() in _UPPER_BOUND_COMPARATORS:
            high = numbers[0]
        elif symbol and symbol.strip() in _LOWER_BOUND_COMPARATORS:
            low = numbers[0]
        else:
            low = high = numbers[0]
    return Quantity(
        text=text,
        kind=kind or None,
        comparator=symbol,
        unit=unit or None,
        low=low,
        high=high,
    )
