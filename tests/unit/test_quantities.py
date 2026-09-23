from __future__ import annotations

import pytest

from kg_processor.application.quantities import is_value_text, parse_quantity


@pytest.mark.parametrize(
    "text",
    [
        "≤ 0.10 %",
        "154 - 162",
        "0001271276",
        "40 mg/kg max",
        "NMT 5.0",
        "2-8 °C",
        "12 months",
        "pH 5.5 - 7.0",
    ],
)
def test_values_are_recognised_as_values(text: str) -> None:
    """A number with its unit, comparator, or range words names no thing."""

    assert is_value_text(text)


@pytest.mark.parametrize(
    "text",
    ["Tween 80", "PEG 400", "Ph. Eur. 2.5.12", "USP <711>", "Batch 123", "Water", "Model 2"],
)
def test_names_that_contain_numbers_are_not_values(text: str) -> None:
    """A name that merely contains a number still names something."""

    assert not is_value_text(text)


def test_bounds_follow_ranges_comparators_and_decimal_commas() -> None:
    """Parse what can be compared while keeping the value as written."""

    assert (parse_quantity("98.5 - 101.5 %").low, parse_quantity("98.5 - 101.5 %").high) == (
        98.5,
        101.5,
    )
    upper = parse_quantity("≤ 0,5 %", kind="limit", unit="%")
    assert (upper.low, upper.high, upper.comparator, upper.text) == (None, 0.5, "≤", "≤ 0,5 %")
    assert parse_quantity("NLT 99.0").low == 99.0
    assert parse_quantity("0.1 %").low == parse_quantity("0.1 %").high == 0.1


def test_a_comparator_that_is_not_one_is_read_from_the_value_instead() -> None:
    """Models put words like 'maximum' or 'result' where a comparator belongs."""

    assert parse_quantity("2,0", comparator="result").comparator is None
    assert parse_quantity("NMT 2,0", comparator="limit").comparator == "nmt"
    assert parse_quantity("2,0", comparator="maximum").high == 2.0
    assert parse_quantity("2,0", comparator="≤").high == 2.0


def test_a_letter_written_before_a_digit_makes_a_code_not_a_value() -> None:
    assert not any(is_value_text(text) for text in ("S450", "U40", "J1-6746", "MT50G2"))
    assert all(is_value_text(text) for text in ("≤ 0.5 %", "40 mg/kg", "0001271276", "20 s"))
