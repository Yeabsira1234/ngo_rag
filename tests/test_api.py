from unittest.mock import Mock

from fastapi.testclient import TestClient

from api import (
    INTERNAL_ERROR_MESSAGE,
    SERVICE_UNAVAILABLE_MESSAGE,
    create_app,
)
from src.rag_service import (
    RAGDependencyError,
    RAGResponse,
    RAGStatus,
    SourceReference,
)


def _client(service: Mock) -> TestClient:
    return TestClient(create_app(rag_service=service))


def test_health_endpoint_does_not_call_rag_service() -> None:
    service = Mock()

    with _client(service) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    service.assert_not_called()
    service.answer.assert_not_called()


def test_answered_response_serializes_citations() -> None:
    service = Mock()
    service.answer.return_value = RAGResponse(
        answer="The office is open from 9:00 a.m. to 5:00 p.m.",
        status=RAGStatus.ANSWERED,
        llm_called=True,
        citations=(
            SourceReference(
                source="sample_document.pdf",
                page_number=1,
                chunk_index=0,
                distance=0.7207,
            ),
        ),
    )

    with _client(service) as client:
        response = client.post(
            "/api/v1/ask",
            json={"question": "  What are the support office hours?  "},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "The office is open from 9:00 a.m. to 5:00 p.m.",
        "status": "answered",
        "llm_called": True,
        "citations": [
            {
                "source": "sample_document.pdf",
                "page_number": 1,
                "chunk_index": 0,
                    "distance": 0.7207,
                    "source_relative_path": "sample_document.pdf",
                    "document_id": "sample_document.pdf",
            }
        ],
    }
    service.answer.assert_called_once_with(
        "What are the support office hours?"
    )


def test_insufficient_context_is_a_successful_response() -> None:
    service = Mock()
    service.answer.return_value = RAGResponse(
        answer="Not enough relevant information.",
        status=RAGStatus.INSUFFICIENT_CONTEXT,
        llm_called=False,
        citations=(),
    )

    with _client(service) as client:
        response = client.post(
            "/api/v1/ask",
            json={"question": "An unsupported question"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_context"
    assert response.json()["llm_called"] is False
    assert response.json()["citations"] == []


def test_blank_question_returns_validation_error() -> None:
    service = Mock()

    with _client(service) as client:
        response = client.post(
            "/api/v1/ask",
            json={"question": "  \n  "},
        )

    assert response.status_code == 422
    service.answer.assert_not_called()


def test_dependency_failure_returns_safe_503() -> None:
    service = Mock()
    service.answer.side_effect = RAGDependencyError(
        "private dependency detail"
    )

    with _client(service) as client:
        response = client.post(
            "/api/v1/ask",
            json={"question": "A valid question"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": SERVICE_UNAVAILABLE_MESSAGE}
    assert "private dependency detail" not in response.text


def test_unexpected_failure_returns_safe_500() -> None:
    service = Mock()
    service.answer.side_effect = RuntimeError("private internal path")

    with _client(service) as client:
        response = client.post(
            "/api/v1/ask",
            json={"question": "A valid question"},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": INTERNAL_ERROR_MESSAGE}
    assert "private internal path" not in response.text
