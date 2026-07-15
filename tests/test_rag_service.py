from unittest.mock import Mock

import pytest

from src.prompting import RAGPromptBuilder
from src.rag_service import RAGDependencyError, RAGService, RAGStatus
from src.retrieval import RetrievalResult


def _retrieval_result() -> RetrievalResult:
    return RetrievalResult(
        chunk_text="Orientation attendance is required.",
        source="handbook.pdf",
        page_number=7,
        chunk_index=31,
        distance=0.42,
    )


def _service(
    embedding_provider: Mock,
    retriever: Mock,
    answer_generator: Mock,
) -> RAGService:
    return RAGService(
        embedding_provider=embedding_provider,
        retriever=retriever,
        answer_generator=answer_generator,
        prompt_builder=RAGPromptBuilder(),
        retrieval_result_count=4,
        retrieval_max_distance=0.9,
    )


def test_relevant_result_causes_one_llm_call_with_correct_context() -> None:
    embedding_provider = Mock()
    embedding_provider.embed_query.return_value = [0.1, 0.2]
    retriever = Mock()
    retriever.search.return_value = [_retrieval_result()]
    answer_generator = Mock()
    answer_generator.generate_answer.return_value = (
        "Orientation attendance is required."
    )
    service = _service(embedding_provider, retriever, answer_generator)

    response = service.answer("When is orientation?")

    assert response.status is RAGStatus.ANSWERED
    assert response.llm_called is True
    retriever.search.assert_called_once_with(
        query_embedding=[0.1, 0.2],
        number_of_results=4,
        max_distance=0.9,
    )
    answer_generator.generate_answer.assert_called_once()
    prompt = answer_generator.generate_answer.call_args.args[0]
    assert "Orientation attendance is required." in prompt.user_input
    assert "When is orientation?" in prompt.user_input
    assert "handbook.pdf, page 7, chunk 31" in prompt.user_input


def test_no_relevant_results_cause_zero_llm_calls() -> None:
    embedding_provider = Mock()
    embedding_provider.embed_query.return_value = [0.1, 0.2]
    retriever = Mock()
    retriever.search.return_value = []
    answer_generator = Mock()
    service = _service(embedding_provider, retriever, answer_generator)

    response = service.answer("Unsupported question")

    assert response.status is RAGStatus.INSUFFICIENT_CONTEXT
    assert response.llm_called is False
    assert response.citations == ()
    assert response.answer == RAGService.INSUFFICIENT_CONTEXT_MESSAGE
    answer_generator.generate_answer.assert_not_called()


def test_service_filters_weak_results_returned_by_retriever() -> None:
    embedding_provider = Mock()
    embedding_provider.embed_query.return_value = [0.1, 0.2]
    retriever = Mock()
    retriever.search.return_value = [
        RetrievalResult(
            chunk_text="Weakly related text",
            source="handbook.pdf",
            page_number=20,
            chunk_index=90,
            distance=1.1,
        )
    ]
    answer_generator = Mock()
    service = _service(embedding_provider, retriever, answer_generator)

    response = service.answer("A question")

    assert response.status is RAGStatus.INSUFFICIENT_CONTEXT
    assert response.citations == ()
    answer_generator.generate_answer.assert_not_called()


def test_citations_preserve_all_retrieval_metadata() -> None:
    embedding_provider = Mock()
    embedding_provider.embed_query.return_value = [0.1, 0.2]
    retriever = Mock()
    retriever.search.return_value = [_retrieval_result()]
    answer_generator = Mock()
    answer_generator.generate_answer.return_value = "An answer"
    service = _service(embedding_provider, retriever, answer_generator)

    response = service.answer("A question")

    citation = response.citations[0]
    assert citation.source == "handbook.pdf"
    assert citation.page_number == 7
    assert citation.chunk_index == 31
    assert citation.distance == 0.42


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_empty_questions_are_handled_without_calling_dependencies(
    question: str,
) -> None:
    embedding_provider = Mock()
    retriever = Mock()
    answer_generator = Mock()
    service = _service(embedding_provider, retriever, answer_generator)

    response = service.answer(question)

    assert response.status is RAGStatus.INVALID_QUESTION
    assert response.llm_called is False
    assert response.answer == RAGService.INVALID_QUESTION_MESSAGE
    embedding_provider.embed_query.assert_not_called()
    retriever.search.assert_not_called()
    answer_generator.generate_answer.assert_not_called()


def test_dependency_failures_are_not_silently_swallowed() -> None:
    embedding_provider = Mock()
    embedding_provider.embed_query.return_value = [0.1, 0.2]
    retriever = Mock()
    retriever.search.side_effect = RuntimeError("retrieval unavailable")
    answer_generator = Mock()
    service = _service(embedding_provider, retriever, answer_generator)

    with pytest.raises(RAGDependencyError) as raised:
        service.answer("A question")

    assert isinstance(raised.value.__cause__, RuntimeError)
    answer_generator.generate_answer.assert_not_called()


def test_llm_failures_are_not_silently_swallowed() -> None:
    embedding_provider = Mock()
    embedding_provider.embed_query.return_value = [0.1, 0.2]
    retriever = Mock()
    retriever.search.return_value = [_retrieval_result()]
    answer_generator = Mock()
    answer_generator.generate_answer.side_effect = RuntimeError(
        "LLM unavailable"
    )
    service = _service(embedding_provider, retriever, answer_generator)

    with pytest.raises(RAGDependencyError) as raised:
        service.answer("A question")

    assert isinstance(raised.value.__cause__, RuntimeError)
