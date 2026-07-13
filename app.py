from src.chunking.text_chunker import TextChunker
from src.loaders.pdf_loader import PDFLoader


PDF_PATH = "data/InternationalHandbook.pdf"


def main() -> None:
    loader = PDFLoader()
    document_text = loader.load(PDF_PATH)

    chunker = TextChunker(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = chunker.split(document_text)

    print(f"Document characters: {len(document_text)}")
    print(f"Chunks created: {len(chunks)}")

    print("\nFirst chunk:")
    print(chunks[0])

    print("\nSecond chunk:")
    print(chunks[1])


if __name__ == "__main__":
    main()