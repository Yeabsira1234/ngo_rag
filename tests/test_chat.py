from unittest.mock import Mock

from chat import NO_RELEVANT_RESULTS_MESSAGE, answer_question
from src.config import Settings


def test_answer_question_does_not_call_llm_without_relevant_results() -> None:
    embedding_service = Mock()
    embedding_service.embed_query.return_value = [0.1, 0.2]
    vector_store = Mock()
    vector_store.search.return_value = []
    llm_service = Mock()
    settings = Settings.from_env({"OPENAI_API_KEY": "test-key"})

    answer, sources = answer_question(
        question="An unsupported question",
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_service=llm_service,
        settings=settings,
    )

    assert answer == NO_RELEVANT_RESULTS_MESSAGE
    assert sources == []
    llm_service.generate_answer.assert_not_called()
