from unittest.mock import Mock

import pytest

from src.agent.tools import DocumentSearchTool
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
    assert result.status is RAGStatus.ANSWERED
    assert result.rag_llm_called is True
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
