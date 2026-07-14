from src.config import Settings
from src.embeddings.openai_embeddings import OpenAIEmbeddingService
from src.llm.openai_llm import OpenAILLMService
from src.retrieval import RetrievalResult
from src.vectorstore.chroma_store import ChromaVectorStore

NO_RELEVANT_RESULTS_MESSAGE = (
    "I could not find enough relevant information in the documents "
    "to answer that question."
)


def answer_question(
    question: str,
    embedding_service: OpenAIEmbeddingService,
    vector_store: ChromaVectorStore,
    llm_service: OpenAILLMService,
    settings: Settings,
) -> tuple[str, list[RetrievalResult]]:
    """Retrieve relevant context and generate a grounded answer."""
    question_embedding = embedding_service.embed_query(question)
    results = vector_store.search(
        query_embedding=question_embedding,
        number_of_results=settings.retrieval_result_count,
        max_distance=settings.retrieval_max_distance,
    )

    if not results:
        return NO_RELEVANT_RESULTS_MESSAGE, []

    answer = llm_service.generate_answer(
        question=question,
        context_chunks=[result.chunk_text for result in results],
    )
    return answer, results


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

        answer, results = answer_question(
            question=question,
            embedding_service=embedding_service,
            vector_store=vector_store,
            llm_service=llm_service,
            settings=settings,
        )

        print("\nAnswer:")
        print(answer)

        if results:
            print("\nSources retrieved:")

        for index, result in enumerate(results, start=1):
            print(
                f"{index}. {result.source}, page {result.page_number}, "
                f"chunk {result.chunk_index}, distance {result.distance:.4f}"
            )

        print()


if __name__ == "__main__":
    main()
