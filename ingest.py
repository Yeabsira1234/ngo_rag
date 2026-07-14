from src.chunking.text_chunker import TextChunker
from src.embeddings.openai_embeddings import OpenAIEmbeddingService
from src.loaders.pdf_loader import PDFLoader
from src.vectorstore.chroma_store import ChromaVectorStore


PDF_PATH = "data/InternationalHandbook.pdf"
SOURCE_NAME = "InternationalHandbook.pdf"


def main() -> None:
    loader = PDFLoader()
    chunker = TextChunker(
        chunk_size=800,
        chunk_overlap=150,
    )
    embedding_service = OpenAIEmbeddingService()
    vector_store = ChromaVectorStore()

    print("Loading PDF...")
    document_text = loader.load(PDF_PATH)

    print("Creating chunks...")
    chunks = chunker.split(document_text)

    print(f"Creating embeddings for {len(chunks)} chunks...")
    embeddings = embedding_service.embed_documents(chunks)

    vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        source=SOURCE_NAME,
    )

    print("Ingestion complete.")
    print(f"Stored {len(chunks)} chunks in ChromaDB.")


if __name__ == "__main__":
    main()