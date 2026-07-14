from pathlib import Path

import chromadb

from src.documents import Document
from src.retrieval import RetrievalResult


class ChromaVectorStore:
    """Persist embedded chunks and perform typed similarity searches."""

    def __init__(
        self,
        collection_name: str = "ngo_documents",
        persist_directory: str = "chroma_data",
    ) -> None:
        Path(persist_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=collection_name
            )
        )

    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> None:
        if len(documents) != len(embeddings):
            raise ValueError(
                "Each document must have one corresponding embedding."
            )

        if any(
            document.metadata.chunk_index is None
            for document in documents
        ):
            raise ValueError(
                "Each stored document must include a chunk_index."
            )

        ids = [
            self._document_id(document)
            for document in documents
        ]

        self.collection.upsert(
            ids=ids,
            documents=[document.page_content for document in documents],
            embeddings=embeddings,
            metadatas=[
                document.metadata.to_dict()
                for document in documents
            ],
        )

    @staticmethod
    def _document_id(document: Document) -> str:
        metadata = document.metadata
        return (
            f"{metadata.source}-page-{metadata.page_number}"
            f"-chunk-{metadata.chunk_index}"
        )

    def search(
        self,
        query_embedding: list[float],
        *,
        number_of_results: int = 4,
        max_distance: float,
    ) -> list[RetrievalResult]:
        query_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=number_of_results,
            include=["documents", "metadatas", "distances"],
        )

        documents = (query_results.get("documents") or [[]])[0]
        metadatas = (query_results.get("metadatas") or [[]])[0]
        distances = (query_results.get("distances") or [[]])[0]

        if not (
            len(documents) == len(metadatas) == len(distances)
        ):
            raise ValueError(
                "Chroma returned inconsistent document, metadata, and "
                "distance counts."
            )

        results = [
            RetrievalResult.from_chroma(
                chunk_text=document,
                metadata=metadata,
                distance=distance,
            )
            for document, metadata, distance in zip(
                documents,
                metadatas,
                distances,
                strict=True,
            )
        ]
        return [
            result
            for result in results
            if result.is_relevant(max_distance)
        ]
