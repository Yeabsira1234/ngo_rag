from src.chunking.text_chunker import TextChunker
from src.config import Settings
from src.embeddings.openai_embeddings import (
    OpenAIEmbeddingService,
)
from src.loaders.pdf_loader import PDFLoader
from src.vectorstore.chroma_store import ChromaVectorStore

def main() -> None:
    settings = Settings.from_env()
    loader = PDFLoader()
    chunker = TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embedding_service = OpenAIEmbeddingService(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
    )
    vector_store = ChromaVectorStore(
        collection_name=settings.chroma_collection_name,
        persist_directory=str(settings.chroma_persist_directory),
    )

    page_documents = loader.load(settings.document_path)
    chunks = chunker.split_documents(page_documents)

    print(f"Creating embeddings for {len(chunks)} chunks...")

    embeddings = embedding_service.embed_documents(
        [chunk.page_content for chunk in chunks]
    )

    vector_store.add_documents(
        documents=chunks,
        embeddings=embeddings,
    )

    print("Document successfully stored in ChromaDB.")

    question = input("\nAsk a question: ")

    question_embedding = embedding_service.embed_query(
        question
    )

    results = vector_store.search(
        query_embedding=question_embedding,
        number_of_results=settings.retrieval_result_count,
        max_distance=settings.retrieval_max_distance,
    )

    print("\nMost relevant passages:\n")

    for index, result in enumerate(results, start=1):
        print(f"Result {index}:")
        print(result.chunk_text)
        print("-" * 80)


if __name__ == "__main__":
    main()
