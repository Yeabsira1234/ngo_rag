from src.chunking.text_chunker import TextChunker
from src.documents import Document, DocumentMetadata


def test_split_documents_preserves_page_metadata() -> None:
    page_document = Document(
        page_content="abcdefghij",
        metadata=DocumentMetadata(
            source="handbook.pdf",
            page_number=3,
        ),
    )
    documents = [page_document]
    chunker = TextChunker(chunk_size=6, chunk_overlap=2)

    chunks = chunker.split_documents(documents)

    assert [chunk.page_content for chunk in chunks] == [
        "abcdef",
        "efghij",
        "ij",
    ]
    assert all(
        chunk.metadata.source == "handbook.pdf" for chunk in chunks
    )
    assert all(chunk.metadata.page_number == 3 for chunk in chunks)
    assert [chunk.metadata.chunk_index for chunk in chunks] == [0, 1, 2]
    assert page_document.metadata.chunk_index is None


def test_chunk_indices_are_unique_across_pages() -> None:
    documents = [
        Document(
            page_content="first page",
            metadata=DocumentMetadata(
                source="handbook.pdf",
                page_number=1,
            ),
        ),
        Document(
            page_content="second page",
            metadata=DocumentMetadata(
                source="handbook.pdf",
                page_number=2,
            ),
        ),
    ]

    chunks = TextChunker(
        chunk_size=100,
        chunk_overlap=0,
    ).split_documents(documents)

    assert [chunk.metadata.page_number for chunk in chunks] == [1, 2]
    assert [chunk.metadata.chunk_index for chunk in chunks] == [0, 1]


def test_split_documents_does_not_create_chunks_for_empty_pages() -> None:
    documents = [
        Document(
            page_content="   ",
            metadata=DocumentMetadata(
                source="handbook.pdf",
                page_number=1,
            ),
        )
    ]

    assert TextChunker().split_documents(documents) == []
