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


def test_build_agent_service_injects_model_and_document_tool(monkeypatch) -> None:
    settings = Settings(openai_api_key="test-key")
    rag_service = object()
    responses_client = object()
    openai_instance = Mock(responses=responses_client)
    model = object()
    document_tool = object()
    organization_tool = object()
    memory = object()
    agent_service = object()

    monkeypatch.setattr(application, "build_rag_service", Mock(return_value=rag_service))
    openai_class = Mock(return_value=openai_instance)
    model_class = Mock(return_value=model)
    tool_class = Mock(return_value=document_tool)
    organization_tool_class = Mock(return_value=organization_tool)
    agent_class = Mock(return_value=agent_service)
    memory_class = Mock(return_value=memory)
    monkeypatch.setattr(application, "OpenAI", openai_class)
    monkeypatch.setattr(application, "OpenAIAgentModel", model_class)
    monkeypatch.setattr(application, "DocumentSearchTool", tool_class)
    monkeypatch.setattr(
        application,
        "OrganizationInfoTool",
        organization_tool_class,
    )
    monkeypatch.setattr(application, "AgentService", agent_class)
    monkeypatch.setattr(application, "InMemoryConversationMemory", memory_class)

    result = application.build_agent_service(settings)

    assert result is agent_service
    openai_class.assert_called_once_with(api_key="test-key")
    model_class.assert_called_once_with(
        client=responses_client,
        model=settings.llm_model,
    )
    tool_class.assert_called_once_with(rag_service)
    organization_tool_class.assert_called_once_with()
    memory_class.assert_called_once_with(
        max_turns=settings.agent_memory_max_turns
    )
    agent_class.assert_called_once_with(
        model=model,
        tools=(document_tool, organization_tool),
        memory=memory,
        max_tool_iterations=settings.agent_max_tool_iterations,
    )
