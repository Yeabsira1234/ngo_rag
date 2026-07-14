from src.chunking.text_chunker import TextChunker
from src.config import Settings
from src.embeddings.openai_embeddings import OpenAIEmbeddingService
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

    print("Loading PDF...")
    document_text = loader.load(str(settings.document_path))

    print("Creating chunks...")
    chunks = chunker.split(document_text)

    print(f"Creating embeddings for {len(chunks)} chunks...")
    embeddings = embedding_service.embed_documents(chunks)

    vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        source=settings.document_source_name,
    )

    print("Ingestion complete.")
    print(f"Stored {len(chunks)} chunks in ChromaDB.")


if __name__ == "__main__":
    main()
