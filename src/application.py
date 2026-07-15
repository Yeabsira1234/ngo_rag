from openai import OpenAI

from src.agent.openai_model import OpenAIAgentModel
from src.agent.memory import InMemoryConversationMemory
from src.agent.service import AgentService
from src.agent.tools import DocumentSearchTool, OrganizationInfoTool, SQLQueryTool
from src.config import Settings
from src.embeddings.openai_embeddings import OpenAIEmbeddingService
from src.llm.openai_llm import OpenAILLMService
from src.prompting import RAGPromptBuilder
from src.rag_service import RAGService
from src.vectorstore.chroma_store import ChromaVectorStore
from src.sql.connection import build_connection_factory
from src.sql.repository import SQLServerRepository


def build_rag_service(settings: Settings) -> RAGService:
    """Construct the RAG service and its infrastructure dependencies."""
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
    return RAGService(
        embedding_provider=embedding_service,
        retriever=vector_store,
        answer_generator=llm_service,
        prompt_builder=RAGPromptBuilder(),
        retrieval_result_count=settings.retrieval_result_count,
        retrieval_max_distance=settings.retrieval_max_distance,
    )


def build_agent_service(settings: Settings) -> AgentService:
    """Construct the agent and register its injected application tools."""
    rag_service = build_rag_service(settings)
    openai_client = OpenAI(api_key=settings.openai_api_key)
    model = OpenAIAgentModel(
        client=openai_client.responses,
        model=settings.llm_model,
    )
    return AgentService(
        model=model,
        tools=(
            DocumentSearchTool(rag_service),
            OrganizationInfoTool(),
            SQLQueryTool(
                SQLServerRepository(
                    build_connection_factory(settings),
                    timeout_seconds=settings.sql_query_timeout_seconds,
                    max_rows=settings.sql_max_rows,
                )
            ),
        ),
        memory=InMemoryConversationMemory(
            max_turns=settings.agent_memory_max_turns
        ),
        max_tool_iterations=settings.agent_max_tool_iterations,
    )
