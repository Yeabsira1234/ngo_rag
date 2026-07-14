import pytest

from src.retrieval import RetrievalResult


def test_retrieval_result_maps_chroma_metadata() -> None:
    result = RetrievalResult.from_chroma(
        chunk_text="Orientation is required.",
        metadata={
            "source": "handbook.pdf",
            "page_number": 7,
            "chunk_index": 31,
        },
        distance=0.42,
    )

    assert result.chunk_text == "Orientation is required."
    assert result.source == "handbook.pdf"
    assert result.page_number == 7
    assert result.chunk_index == 31
    assert result.distance == 0.42


def test_lower_distance_is_more_relevant() -> None:
    result = RetrievalResult(
        chunk_text="Relevant text",
        source="handbook.pdf",
        page_number=7,
        chunk_index=31,
        distance=0.8,
    )

    assert result.is_relevant(max_distance=0.8)
    assert not result.is_relevant(max_distance=0.79)


def test_retrieval_result_rejects_missing_metadata() -> None:
    with pytest.raises(ValueError, match="page_number"):
        RetrievalResult.from_chroma(
            chunk_text="Text",
            metadata={"source": "handbook.pdf", "chunk_index": 1},
            distance=0.5,
        )
