from src.config import Settings
from src.embeddings.openai_embeddings import OpenAIEmbeddingService
from src.llm.openai_llm import OpenAILLMService
from src.vectorstore.chroma_store import ChromaVectorStore


def main() -> None:
    settings = Settings.from_env()
    embedding_service = OpenAIEmbeddingService(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
    )
    vector_store = ChromaVectorStore(
        collection_name=settings.chroma_collection_name,
        persist_directory=str(settings.chroma_persist_directory),
    )
    llm_service = OpenAILLMService(
        api_key=settings.openai_api_key,
        model=settings.llm_model,
    )

    print("Document assistant is ready.")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("Ask a question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        question_embedding = embedding_service.embed_query(question)

        results = vector_store.search(
            query_embedding=question_embedding,
            number_of_results=settings.retrieval_result_count,
        )

        documents = results.get("documents", [[]])[0]

        if not documents:
            print("\nNo relevant document passages were found.\n")
            continue

        answer = llm_service.generate_answer(
            question=question,
            context_chunks=documents,
        )

        print("\nAnswer:")
        print(answer)

        print("\nSources retrieved:")
        metadatas = results.get("metadatas", [[]])[0]

        for index, metadata in enumerate(metadatas, start=1):
            source = metadata.get("source", "Unknown source")
            page_number = metadata.get("page_number", "Unknown")
            chunk_index = metadata.get("chunk_index", "Unknown")
            print(
                f"{index}. {source}, page {page_number}, "
                f"chunk {chunk_index}"
            )

        print()


if __name__ == "__main__":
    main()
