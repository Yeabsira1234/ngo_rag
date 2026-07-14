from pathlib import Path

import chromadb


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

    def add_chunks(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        source: str,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Each chunk must have one corresponding embedding."
            )

        ids = [
            f"{source}-chunk-{index}"
            for index in range(len(chunks))
        ]

        metadatas = [
            {
                "source": source,
                "chunk_index": index,
            }
            for index in range(len(chunks))
        ]

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
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