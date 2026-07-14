from src.config import Settings
from src.embeddings.openai_embeddings import OpenAIEmbeddingService
from src.llm.openai_llm import OpenAILLMService
from src.prompting import RAGPromptBuilder
from src.rag_service import RAGService
from src.vectorstore.chroma_store import ChromaVectorStore


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
