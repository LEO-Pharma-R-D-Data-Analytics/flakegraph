from __future__ import annotations

from kg_processor.application.content_kinds import (
    DATASET_PROFILE_STAGE,
    annotate_documents,
    content_kind,
    dataset_windows,
    related_file_ids,
    relative_location,
    window_filter,
)


def test_files_are_classified_by_type() -> None:
    """Spreadsheets and tables are datasets, pictures are images, the rest documents."""

    assert content_kind("application/vnd.ms-excel", "s3://bucket/plan.XLSX") == "dataset"
    assert content_kind("text/csv", "file:///data/rows.txt") == "dataset"
    assert content_kind("image/tiff", "file:///lab/Batch%209.tif") == "image"
    assert content_kind("application/pdf", "file:///docs/report.pdf") == "document"


def test_a_dataset_profile_decides_its_extraction_passes() -> None:
    """Only measurements get relation windows; every profiled sheet keeps its entities."""

    trace = [
        {"stage": DATASET_PROFILE_STAGE, "document_id": "list", "kind": "catalogue"},
        {"stage": DATASET_PROFILE_STAGE, "document_id": "calc", "kind": "other"},
        {"stage": DATASET_PROFILE_STAGE, "document_id": "assay", "kind": "measurements"},
    ]
    passes = window_filter(trace)

    assert passes("list") == (True, False)
    assert passes("calc") == (True, False)
    assert passes("assay") == (True, True)
    assert passes("report") == (True, True)
    assert dataset_windows(None) == (True, True)


def test_images_and_datasets_are_related_to_the_documents_beside_them() -> None:
    """Shared identifiers in file names relate files; a small folder relates them all."""

    rows = [
        {"file_id": "coa", "kind": "document", "source_uri": "file:///lot/MB2018-5356%20CoA.pdf"},
        {"file_id": "img", "kind": "image", "source_uri": "file:///lot/MB2018-5356%20photo.jpg"},
        {"file_id": "other", "kind": "document", "source_uri": "file:///lot/unrelated%20memo.pdf"},
        {"file_id": "sheet", "kind": "dataset", "source_uri": "file:///elsewhere/data.xlsx"},
        {"file_id": "near", "kind": "document", "source_uri": "file:///elsewhere/notes.pdf"},
    ]

    related = related_file_ids(rows)

    assert related["img"] == ["coa"]
    assert related["sheet"] == ["near"]
    assert "coa" not in related


def test_documents_get_their_dataset_summary_and_related_files() -> None:
    """Annotation fills what extraction learnt without dropping other columns."""

    rows = [
        {"file_id": "sheet", "kind": "dataset", "source_uri": "file:///d/a.xlsx", "size_bytes": 3},
        {"file_id": "doc", "kind": "document", "source_uri": "file:///d/b.pdf", "summary": None},
    ]
    trace = [{"stage": DATASET_PROFILE_STAGE, "document_id": "sheet", "summary": "Excipient list."}]

    annotated = {row["file_id"]: row for row in annotate_documents(rows, trace)}

    assert annotated["sheet"]["summary"] == "Excipient list."
    assert annotated["sheet"]["related_file_ids"] == ["doc"]
    assert annotated["sheet"]["size_bytes"] == 3
    assert annotated["doc"]["summary"] is None


def test_a_location_is_the_path_below_its_source_root() -> None:
    """Folders under the root say where a file belongs; elsewhere its last folders do."""

    uri = "file:///data/uploads/run1/Lab%20A/Trial%203/Pictures/IMG_0392.jpg"
    assert relative_location(uri, ["/data/uploads/run1"]) == "Lab A/Trial 3/Pictures/IMG_0392.jpg"
    assert relative_location("s3://bucket/corpus/a/b.pdf", ["s3://bucket/corpus"]) == "a/b.pdf"
    assert relative_location("file:///x/y/z/w/v/file.pdf", ["/other"]) == "z/w/v/file.pdf"


def test_an_image_in_its_own_subfolder_is_related_to_the_documents_above_it() -> None:
    """A Pictures folder with no documents belongs to its parent folder's documents."""

    rows = [
        {"file_id": "spec", "kind": "document", "folder": "Lab/Coater", "source_uri": "a"},
        {"file_id": "photo", "kind": "image", "folder": "Lab/Coater/Pictures", "source_uri": "b"},
        {"file_id": "far", "kind": "document", "folder": "Other", "source_uri": "c"},
    ]

    assert related_file_ids(rows)["photo"] == ["spec"]
