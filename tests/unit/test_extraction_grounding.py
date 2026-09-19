from __future__ import annotations

from kg_processor.application.extraction_grounding import ground_evidence, surface_occurs


def test_surface_occurs_requires_complete_tokens() -> None:
    """Prevent substring matches from grounding place names inside adjectival tokens.

    Complete token boundaries are part of the evidence contract.
    """

    source = "Greek wrestling and Indian kushti influenced regional practice."

    assert not surface_occurs(source, ["Greece"])
    assert not surface_occurs(source, ["India"])
    assert surface_occurs(source, ["Greek"])
    assert surface_occurs(source, ["Indian kushti"])


def test_surface_occurs_tolerates_space_and_hyphen_variants() -> None:
    """Allow harmless spacing and hyphen variation while preserving complete-token matching.

    OCR normalization should not lose genuine mentions.
    """

    source = "Brazilian Jiu-Jitsu developed in Brazil."

    assert surface_occurs(source, ["Brazilian Jiu Jitsu"])


def test_surface_occurs_accepts_attached_author_footnote_markers() -> None:
    """Ground multi-token byline names without weakening single-token boundaries."""

    source = "David Silver1*, Aja Huang1* and the Model2 baseline."

    assert surface_occurs(source, ["David Silver"])
    assert surface_occurs(source, ["Aja Huang"])
    assert not surface_occurs(source, ["Model"])


def test_surface_occurs_repairs_visual_line_break_hyphens() -> None:
    """Ground dehyphenated model surfaces against exact offset-preserving PDF text."""

    source = "stability with respect to small per-\nturbations to the inputs"

    assert surface_occurs(source, ["small perturbations to the inputs"])
    grounded = ground_evidence(
        source,
        "stability with respect to small perturbations to the inputs",
        [["small perturbations to the inputs"]],
    )
    assert grounded is not None
    assert grounded.quote == source
    assert grounded.start_offset == 0
    assert grounded.end_offset == len(source)


def test_surface_occurs_repairs_one_first_letter_ocr_split() -> None:
    """Ground a complete title despite one PDF extraction token split."""

    source = "ADAM: A M ETHOD FOR STOCHASTIC OPTIMIZATION"

    assert surface_occurs(source, ["Adam: A Method for Stochastic Optimization"])
    assert not surface_occurs(source, ["Adam: A Method for Stochastic Objectives"])


def test_ground_evidence_rejects_substring_only_endpoint() -> None:
    """Reject relation evidence when one required endpoint exists only as a substring."""

    source = "Indian wrestling developed through regional schools."

    grounded = ground_evidence(
        source,
        source,
        [["wrestling"], ["India"]],
    )

    assert grounded is None


def test_ground_evidence_requires_distinct_non_overlapping_endpoints() -> None:
    """Do not ground a shorter target inside the source entity's sole mention."""

    source = "Stochastic gradient descent is an optimization method."

    grounded = ground_evidence(
        source,
        source,
        [["stochastic gradient descent"], ["gradient descent"]],
    )

    assert grounded is None


def test_ground_evidence_repairs_nested_endpoint_to_distinct_comparison() -> None:
    """Select the sentence where both nested names have independent mentions."""

    unsupported = "Stochastic gradient descent is an optimization method."
    comparison = "Stochastic gradient descent outperforms gradient descent."
    source = f"{unsupported} {comparison}"

    grounded = ground_evidence(
        source,
        unsupported,
        [["stochastic gradient descent"], ["gradient descent"]],
    )

    assert grounded is not None
    assert grounded.quote == comparison
    assert grounded.repair == "supporting_sentence"
