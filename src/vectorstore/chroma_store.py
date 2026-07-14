from pathlib import Path

import chromadb

from src.documents import Document


class ChromaVectorStore:
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
        number_of_results: int = 4,
    ) -> dict:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=number_of_results,
        )
