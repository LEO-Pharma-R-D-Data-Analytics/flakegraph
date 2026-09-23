from __future__ import annotations

from kg_processor.application.extraction_grounding import (
    find_quote,
    ground_evidence,
    name_words_in_quote,
    sentence_spans,
    surface_occurs,
)


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


def test_a_quote_given_in_pieces_grounds_on_the_stretch_that_holds_them() -> None:
    """Ground a layout fact quoted as header and field joined by an ellipsis."""

    source = (
        "Certificate of Analysis\nL-Arginine Hydrochloride\nBulk Pharmaceutical Chemical\n"
        "Manufactured by a supplier\nMaterial No.: 4930-06\n"
        "Batch No.: 23D1061038\nAppearance: white powder"
    )

    grounded = ground_evidence(
        source,
        "Batch No.: 23D1061038 ... L-Arginine Hydrochloride",
        [["L-Arginine Hydrochloride"], ["23D1061038"]],
    )

    assert grounded is not None
    assert grounded.repair == "quote_pieces"
    assert grounded.quote == source[grounded.start_offset : grounded.end_offset]
    assert grounded.quote.startswith("L-Arginine Hydrochloride")
    assert grounded.quote.endswith("23D1061038")


def test_every_piece_of_a_pieced_quote_must_occur_exactly() -> None:
    """Refuse a pieced quote with a paraphrased part."""

    source = "L-Arginine Hydrochloride\n" + "filler line\n" * 5 + "Batch No.: 23D1061038"

    assert (
        ground_evidence(
            source,
            "Batch number 23D1061038 ... L-Arginine Hydrochloride",
            [["L-Arginine Hydrochloride"], ["23D1061038"]],
        )
        is None
    )


def test_letters_and_digits_of_a_name_match_with_or_without_a_separator() -> None:
    """Treat GG918, GG-918 and GG 918 as one name, but not Model inside Model2."""

    assert surface_occurs("the potent GG-918 was developed", ["GG918"])
    assert surface_occurs("the potent GG918 was developed", ["GG-918"])
    assert surface_occurs("Tween 80 and butanol", ["Tween80"])
    assert not surface_occurs("the Model2 baseline", ["Model"])


def test_sentences_keep_decimals_together_and_split_table_rows() -> None:
    """A decimal is not a sentence end, and each table row is its own sentence."""

    table = (
        "<table><tr><td>Assay</td><td>98.5 - 101.5 %</td></tr>"
        "<tr><td>Water</td><td>0.1 %</td></tr></table>"
    )

    assert [span.quote for span in sentence_spans("Limit is 0.10 %. Result conforms.")] == [
        "Limit is 0.10 %.",
        "Result conforms.",
    ]
    rows = [span.quote for span in sentence_spans(table)]
    assert rows[0].endswith("98.5 - 101.5 %</td></tr>")
    assert rows[1].startswith("<tr><td>Water")


def test_a_quote_reads_across_table_cells_as_prose() -> None:
    """A parser writes a table as HTML; a faithful quote of a row reads its cells as text."""

    source = (
        "<table><tr><td>Product Name:</td><td>POLAWAX NF-PA-(RB)</td>"
        "<td>Date of test:</td><td>23.04.2010</td></tr></table>"
    )

    span = find_quote(source, "Product Name: POLAWAX NF-PA-(RB)")

    assert span is not None
    assert span.quote == source[span.start_offset : span.end_offset]
    assert span.quote == "Product Name:</td><td>POLAWAX NF-PA-(RB)"
    # Cell bars read the same way, and words still have to agree.
    assert find_quote("| Batch No. | 0101206650 |", "Batch No. 0101206650") is not None
    assert find_quote(source, "Product Name: POLAWAX NF-PA-(RC)") is None


def test_a_name_matches_its_plural_and_genitive_but_not_another_word() -> None:
    assert surface_occurs(
        "Denne test kan udføres af LEO Pharmas ventilationsafdeling.", ["LEO Pharma"]
    )
    assert surface_occurs("The tablets were coated.", ["tablet"])
    assert surface_occurs("Two processes ran.", ["process"])
    assert not surface_occurs("The Indian team won.", ["India"])
    assert not surface_occurs("Model GG91 differs.", ["GG9"])


def test_an_underscore_separates_the_words_of_a_file_name() -> None:
    name = "GEA Small-scale-solid-dosage_Gral PMA Fluidbed_2015-06-EN.pdf"

    assert surface_occurs(name, ["Gral PMA Fluidbed"])
    assert surface_occurs("Funded by KFI_16-1-2017-0025.", ["KFI"])


def test_an_underscore_name_matches_as_written_or_spaced() -> None:
    assert surface_occurs("Kan noget fra SOP_004858 bruges?", ["SOP_004858"])
    assert surface_occurs("Kan noget fra SOP_004858 bruges?", ["SOP 004858"])


def test_a_quote_matches_text_a_parser_left_html_escaped() -> None:
    source = "<td><p>O&#x27;Hara coater</p></td><td>HMI &amp; PLC control</td>"

    coater = find_quote(source, "O'Hara coater")
    control = find_quote(source, "HMI & PLC control")

    assert coater is not None and coater.quote == "O&#x27;Hara coater"
    assert control is not None and control.quote == "HMI &amp; PLC control"


def test_a_quote_may_keep_or_drop_a_line_break_hyphen() -> None:
    source = "Both products significantly improved skin hydra-\ntion, reduced redness."

    kept = find_quote(source, "significantly improved skin hydra-tion")
    dropped = find_quote(source, "significantly improved skin hydration")

    assert kept is not None and kept.quote == "significantly improved skin hydra-\ntion"
    assert dropped is not None and dropped.quote == kept.quote


def test_a_name_containing_the_other_endpoint_grounds_only_when_allowed() -> None:
    source = "The 10% urea foam was applied twice daily."
    groups = [["10% urea foam"], ["foam"]]

    assert ground_evidence(source, source, groups) is None
    nested = ground_evidence(source, source, groups, allow_nested=True)
    assert nested is not None and nested.quote == source


def test_a_rephrased_name_keeps_every_word_of_its_quote() -> None:
    assert name_words_in_quote("pH measurement", "The pH of the foams was measured")
    assert name_words_in_quote(
        "scintigraphic examinations", "In scintigraphic and colonoscopic examinations"
    )
    assert not name_words_in_quote("pore size", "interconnected pores ranging from 10 to 50 µm")
    assert not name_words_in_quote("pH value", "The pH of the foams was measured")


def test_a_name_of_several_parts_may_touch_digits_ocr_glued_to_it() -> None:
    assert surface_occurs("15548-01SmatProtect splinger shield", ["SmatProtect splinger shield"])
    assert not surface_occurs("ItemsSmart shield", ["Smart shield"])
    assert not surface_occurs("Model2 is new", ["Model"])


def test_a_hyphen_after_a_stray_space_ends_a_line_but_a_spaced_dash_stays() -> None:
    assert surface_occurs("width, and diame -\nter / length", ["diameter"])
    assert not surface_occurs("Foam - a review", ["Foama"])
