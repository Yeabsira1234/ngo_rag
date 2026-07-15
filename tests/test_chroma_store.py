from unittest.mock import Mock

import pytest

from src.documents import Document, DocumentMetadata
from src.vectorstore.chroma_store import ChromaVectorStore


def test_add_documents_sends_page_metadata_to_chroma() -> None:
    collection = Mock()
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.collection = collection
    document = Document(
        page_content="A policy passage",
        metadata=DocumentMetadata(
            source="handbook.pdf",
            page_number=5,
            chunk_index=2,
        ),
    )

    store.add_documents([document], [[0.1, 0.2]])

    collection.upsert.assert_called_once_with(
        ids=["handbook.pdf-page-5-chunk-2"],
        documents=["A policy passage"],
        embeddings=[[0.1, 0.2]],
        metadatas=[
            {
                "source": "handbook.pdf",
                    "page_number": 5,
                    "source_relative_path": "handbook.pdf",
                    "document_id": "handbook.pdf",
                    "chunk_index": 2,
            }
        ],
    )


def test_add_documents_requires_chunk_metadata() -> None:
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.collection = Mock()
    page_document = Document(
        page_content="A page",
        metadata=DocumentMetadata(
            source="handbook.pdf",
            page_number=1,
        ),
    )

    with pytest.raises(ValueError, match="chunk_index"):
        store.add_documents([page_document], [[0.1, 0.2]])


def test_chunk_ids_are_stable_and_unique_across_documents() -> None:
    first = Document("one", DocumentMetadata("guide.pdf", 1, 0, "north/guide.pdf", "doc-north"))
    second = Document("two", DocumentMetadata("guide.pdf", 1, 0, "south/guide.pdf", "doc-south"))
    assert ChromaVectorStore._document_id(first) == "doc-north-page-1-chunk-0"
    assert ChromaVectorStore._document_id(second) == "doc-south-page-1-chunk-0"
    assert ChromaVectorStore._document_id(first) == ChromaVectorStore._document_id(first)


def test_stale_documents_are_removed_but_active_documents_remain() -> None:
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.collection = Mock()
    store.collection.get.return_value = {
        "ids": ["active-id", "stale-id"],
        "metadatas": [{"document_id": "active"}, {"document_id": "removed"}],
    }
    assert store.remove_stale_documents({"active"}) == 1
    store.collection.delete.assert_called_once_with(ids=["stale-id"])


def test_search_maps_and_filters_results_by_distance() -> None:
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.collection = Mock()
    store.collection.query.return_value = {
        "documents": [["Relevant passage", "Weak passage"]],
        "metadatas": [[
            {
                "source": "handbook.pdf",
                "page_number": 7,
                "chunk_index": 31,
            },
            {
                "source": "handbook.pdf",
                "page_number": 20,
                "chunk_index": 90,
            },
        ]],
        "distances": [[0.4, 1.4]],
    }

    results = store.search(
        query_embedding=[0.1, 0.2],
        number_of_results=4,
        max_distance=0.9,
    )

    assert len(results) == 1
    assert results[0].chunk_text == "Relevant passage"
    assert results[0].distance == 0.4
    store.collection.query.assert_called_once_with(
        query_embeddings=[[0.1, 0.2]],
        n_results=4,
        include=["documents", "metadatas", "distances"],
    )


def test_search_can_return_multiple_documents() -> None:
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.collection = Mock()
    store.collection.query.return_value = {
        "documents": [["North", "South"]],
        "metadatas": [[
            {"source": "guide.pdf", "source_relative_path": "north/guide.pdf", "document_id": "north", "page_number": 1, "chunk_index": 0},
            {"source": "guide.pdf", "source_relative_path": "south/guide.pdf", "document_id": "south", "page_number": 1, "chunk_index": 0},
        ]],
        "distances": [[0.2, 0.3]],
    }
    results = store.search([0.1], number_of_results=2, max_distance=0.9)
    assert [result.document_id for result in results] == ["north", "south"]
