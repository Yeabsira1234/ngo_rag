from unittest.mock import Mock

from src import application
from src.config import Settings


def test_build_rag_service_constructs_expected_dependencies(
    monkeypatch,
) -> None:
    settings = Settings(openai_api_key="test-key")
    embedding_service = object()
    vector_store = object()
    llm_service = object()
    prompt_builder = object()
    rag_service = object()

    embedding_class = Mock(return_value=embedding_service)
    vector_store_class = Mock(return_value=vector_store)
    llm_class = Mock(return_value=llm_service)
    prompt_builder_class = Mock(return_value=prompt_builder)
    rag_service_class = Mock(return_value=rag_service)
    monkeypatch.setattr(application, "OpenAIEmbeddingService", embedding_class)
    monkeypatch.setattr(application, "ChromaVectorStore", vector_store_class)
    monkeypatch.setattr(application, "OpenAILLMService", llm_class)
    monkeypatch.setattr(application, "RAGPromptBuilder", prompt_builder_class)
    monkeypatch.setattr(application, "RAGService", rag_service_class)

    result = application.build_rag_service(settings)

    assert result is rag_service
    embedding_class.assert_called_once_with(
        api_key="test-key",
        model=settings.embedding_model,
    )
    vector_store_class.assert_called_once_with(
        collection_name=settings.chroma_collection_name,
        persist_directory=str(settings.chroma_persist_directory),
    )
    llm_class.assert_called_once_with(
        api_key="test-key",
        model=settings.llm_model,
    )
    rag_service_class.assert_called_once_with(
        embedding_provider=embedding_service,
        retriever=vector_store,
        answer_generator=llm_service,
        prompt_builder=prompt_builder,
        retrieval_result_count=settings.retrieval_result_count,
        retrieval_max_distance=settings.retrieval_max_distance,
    )
