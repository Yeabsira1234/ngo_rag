import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence

from src.prompting import Prompt
from src.retrieval import RetrievalResult


logger = logging.getLogger(__name__)


class RAGStatus(str, Enum):
    ANSWERED = "answered"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    INVALID_QUESTION = "invalid_question"


@dataclass(frozen=True, slots=True)
class SourceReference:
    source: str
    page_number: int
    chunk_index: int
    distance: float

    @classmethod
    def from_retrieval_result(
        cls,
        result: RetrievalResult,
    ) -> "SourceReference":
        return cls(
            source=result.source,
            page_number=result.page_number,
            chunk_index=result.chunk_index,
            distance=result.distance,
        )


@dataclass(frozen=True, slots=True)
class RAGResponse:
    answer: str
    citations: tuple[SourceReference, ...]
    llm_called: bool
    status: RAGStatus


class EmbeddingProvider(Protocol):
    def embed_query(self, query: str) -> list[float]: ...


class Retriever(Protocol):
    def search(
        self,
        query_embedding: list[float],
        *,
        number_of_results: int,
        max_distance: float,
    ) -> list[RetrievalResult]: ...


class AnswerGenerator(Protocol):
    def generate_answer(self, prompt: Prompt) -> str: ...


class PromptBuilder(Protocol):
    def build(
        self,
        question: str,
        results: Sequence[RetrievalResult],
    ) -> Prompt: ...


class RAGService:
    """Orchestrate validated, retrieval-grounded answer generation."""

    INVALID_QUESTION_MESSAGE = "Please enter a question."
    INSUFFICIENT_CONTEXT_MESSAGE = (
        "I could not find enough relevant information in the documents "
        "to answer that question."
    )

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        retriever: Retriever,
        answer_generator: AnswerGenerator,
        prompt_builder: PromptBuilder,
        retrieval_result_count: int,
        retrieval_max_distance: float,
    ) -> None:
        if retrieval_result_count <= 0:
            raise ValueError("retrieval_result_count must be greater than zero.")
        if (
            not math.isfinite(retrieval_max_distance)
            or retrieval_max_distance < 0
        ):
            raise ValueError(
                "retrieval_max_distance must be finite and non-negative."
            )

        self.embedding_provider = embedding_provider
        self.retriever = retriever
        self.answer_generator = answer_generator
        self.prompt_builder = prompt_builder
        self.retrieval_result_count = retrieval_result_count
        self.retrieval_max_distance = retrieval_max_distance

    def answer(self, question: str) -> RAGResponse:
        normalized_question = question.strip()
        if not normalized_question:
            logger.info("event=rag_invalid_question")
            return RAGResponse(
                answer=self.INVALID_QUESTION_MESSAGE,
                citations=(),
                llm_called=False,
                status=RAGStatus.INVALID_QUESTION,
            )

        question_length = len(normalized_question)
        try:
            query_embedding = self.embedding_provider.embed_query(
                normalized_question
            )
        except Exception as error:
            logger.exception(
                "event=rag_embedding_failed error_type=%s question_length=%d",
                type(error).__name__,
                question_length,
            )
            raise

        logger.info(
            "event=rag_retrieval_started requested_results=%d "
            "max_distance=%.4f question_length=%d",
            self.retrieval_result_count,
            self.retrieval_max_distance,
            question_length,
        )
        try:
            results = self.retriever.search(
                query_embedding=query_embedding,
                number_of_results=self.retrieval_result_count,
                max_distance=self.retrieval_max_distance,
            )
        except Exception as error:
            logger.exception(
                "event=rag_retrieval_failed error_type=%s question_length=%d",
                type(error).__name__,
                question_length,
            )
            raise

        relevant_results = [
            result
            for result in results
            if result.is_relevant(self.retrieval_max_distance)
        ]
        logger.info(
            "event=rag_retrieval_completed returned_results=%d "
            "relevant_results=%d",
            len(results),
            len(relevant_results),
        )

        if not relevant_results:
            logger.info(
                "event=rag_insufficient_context question_length=%d",
                question_length,
            )
            return RAGResponse(
                answer=self.INSUFFICIENT_CONTEXT_MESSAGE,
                citations=(),
                llm_called=False,
                status=RAGStatus.INSUFFICIENT_CONTEXT,
            )

        try:
            prompt = self.prompt_builder.build(
                normalized_question,
                relevant_results,
            )
        except Exception as error:
            logger.exception(
                "event=rag_prompt_build_failed error_type=%s",
                type(error).__name__,
            )
            raise

        logger.info(
            "event=rag_llm_invocation_started context_chunks=%d",
            len(relevant_results),
        )
        try:
            answer = self.answer_generator.generate_answer(prompt)
        except Exception as error:
            logger.exception(
                "event=rag_llm_invocation_failed error_type=%s",
                type(error).__name__,
            )
            raise

        logger.info("event=rag_llm_invocation_completed")
        citations = tuple(
            SourceReference.from_retrieval_result(result)
            for result in relevant_results
        )

        return RAGResponse(
            answer=answer,
            citations=citations,
            llm_called=True,
            status=RAGStatus.ANSWERED,
        )
