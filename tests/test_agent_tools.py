from unittest.mock import Mock

import pytest

from src.agent.models import ToolExecutionStatus, ToolProvenance
from src.agent.tools import DocumentSearchTool, OrganizationInfoTool
from src.rag_service import RAGResponse, RAGStatus, SourceReference


def test_document_search_tool_maps_complete_rag_response() -> None:
    rag_service = Mock()
    rag_service.answer.return_value = RAGResponse(
        answer="The office is open from 9 to 5.",
        status=RAGStatus.ANSWERED,
        llm_called=True,
        citations=(
            SourceReference(
                source="sample_document.pdf",
                page_number=1,
                chunk_index=0,
                distance=0.72,
            ),
        ),
    )
    tool = DocumentSearchTool(rag_service)

    result = tool.execute({"question": "  What are the office hours?  "})

    rag_service.answer.assert_called_once_with("What are the office hours?")
    assert result.answer == "The office is open from 9 to 5."
    assert result.status is ToolExecutionStatus.ANSWERED
    assert result.rag_llm_called is True
    assert result.source == "document_search"
    assert result.provenance is ToolProvenance.DOCUMENT
    assert result.citations[0].source == "sample_document.pdf"
    assert result.citations[0].page_number == 1
    assert result.citations[0].chunk_index == 0
    assert result.citations[0].distance == 0.72


@pytest.mark.parametrize(
    "arguments",
    [{}, {"question": ""}, {"question": 42}, {"question": "q", "extra": 1}],
)
def test_document_search_tool_rejects_invalid_arguments(arguments) -> None:
    tool = DocumentSearchTool(Mock())

    with pytest.raises(ValueError):
        tool.execute(arguments)


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("organization_name", "Community Support Network"),
        ("general_contact_email", "hello@example.org"),
        ("main_office_location", "100 Example Avenue"),
        ("service_categories", "Case Management"),
    ],
)
def test_organization_info_returns_one_fictional_category(
    category: str,
    expected: str,
) -> None:
    result = OrganizationInfoTool().execute({"category": category})

    assert expected in result.answer
    assert result.status is ToolExecutionStatus.ANSWERED
    assert result.rag_llm_called is False
    assert result.citations == ()
    assert result.source == "organization_info"
    assert result.category == category
    assert result.to_model_output()["source"] == "organization_info"


def test_organization_info_unknown_category_returns_safe_result() -> None:
    result = OrganizationInfoTool().execute({"category": "private_records"})

    assert result.status is ToolExecutionStatus.NOT_FOUND
    assert result.answer == OrganizationInfoTool.NOT_FOUND_MESSAGE
    assert result.citations == ()
    assert result.rag_llm_called is False


@pytest.mark.parametrize(
    "arguments",
    [{}, {"category": 42}, {"category": "organization_name", "extra": 1}],
)
def test_organization_info_rejects_malformed_arguments(arguments) -> None:
    with pytest.raises(ValueError):
        OrganizationInfoTool().execute(arguments)
