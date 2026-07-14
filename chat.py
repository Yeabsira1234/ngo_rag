from src.embeddings.openai_embeddings import OpenAIEmbeddingService
from src.llm.openai_llm import OpenAILLMService
from src.vectorstore.chroma_store import ChromaVectorStore


def main() -> None:
    embedding_service = OpenAIEmbeddingService()
    vector_store = ChromaVectorStore()
    llm_service = OpenAILLMService()

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
            number_of_results=4,
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
            chunk_index = metadata.get("chunk_index", "Unknown")
            print(f"{index}. {source}, chunk {chunk_index}")

        print()


if __name__ == "__main__":
    main()