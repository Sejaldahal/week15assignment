from backend.rag.chunking import chunk_text, clean_text


def test_clean_text_collapses_whitespace():
    dirty = "Hello   world.\n\n\n\nSecond paragraph."
    cleaned = clean_text(dirty)
    assert "\n\n\n" not in cleaned


def test_chunking_respects_overlap_and_size():
    text = "sentence. " * 500
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 220  # allow a little slack for boundary snapping


def test_chunking_empty_text():
    assert chunk_text("") == []


def test_chunk_overlap_must_be_smaller_than_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, chunk_overlap=10)
