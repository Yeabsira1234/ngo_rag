from src.config import Settings
from src.embeddings.openai_embeddings import OpenAIEmbeddingService
from src.llm.openai_llm import OpenAILLMService
from src.prompting import RAGPromptBuilder
from src.rag_service import RAGService
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
    rag_service = RAGService(
        embedding_provider=embedding_service,
        retriever=vector_store,
        answer_generator=llm_service,
        prompt_builder=RAGPromptBuilder(),
        retrieval_result_count=settings.retrieval_result_count,
        retrieval_max_distance=settings.retrieval_max_distance,
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

        response = rag_service.answer(question)

        print("\nAnswer:")
        print(response.answer)

        if response.citations:
            print("\nSources retrieved:")

        for index, citation in enumerate(response.citations, start=1):
            print(
                f"{index}. {citation.source}, page {citation.page_number}, "
                f"chunk {citation.chunk_index}, "
                f"distance {citation.distance:.4f}"
            )

        print()


if __name__ == "__main__":
    main()
