from unittest.mock import Mock

import pytest

from src.agent.models import (
    AgentCitation,
    AgentModelResponse,
    AgentStatus,
    ToolCall,
    ToolDefinition,
    ToolExecutionResult,
)
from src.agent.service import AgentDependencyError, AgentService
from src.rag_service import RAGStatus


DOCUMENT_TOOL = ToolDefinition(
    name="document_search",
    description="Search the indexed document.",
    parameters={
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "required": ["question"],
        "additionalProperties": False,
    },
)


def _model_response(
    *,
    text: str = "",
    tool_call: ToolCall | None = None,
) -> AgentModelResponse:
    calls = (tool_call,) if tool_call else ()
    continuation = (
        ({"type": "function_call", "call_id": tool_call.call_id},)
        if tool_call
        else ()
    )
    return AgentModelResponse(
        output_text=text,
        tool_calls=calls,
        continuation_items=continuation,
    )


def _tool() -> Mock:
    tool = Mock()
    tool.definition = DOCUMENT_TOOL
    return tool


def test_agent_can_answer_directly_without_calling_tool() -> None:
    model = Mock()
    model.respond.return_value = _model_response(text="Hello! How can I help?")
    tool = _tool()
    service = AgentService(model=model, tools=(tool,))

    response = service.answer("Say hello")

    assert response.status is AgentStatus.DIRECT_ANSWER
    assert response.answer == "Hello! How can I help?"
    assert response.document_tool_used is False
    assert response.citations == ()
    tool.execute.assert_not_called()


def test_document_tool_path_preserves_citations_and_returns_output_to_model() -> None:
    model = Mock()
    model.respond.side_effect = [
        _model_response(
            tool_call=ToolCall(
                call_id="call-1",
                name="document_search",
                arguments='{"question":"What are the office hours?"}',
            )
        ),
        _model_response(text="The office is open from 9 to 5."),
    ]
    citation = AgentCitation(
        source="sample_document.pdf",
        page_number=1,
        chunk_index=0,
        distance=0.72,
    )
    tool = _tool()
    tool.execute.return_value = ToolExecutionResult(
        answer="The office is open from 9 to 5.",
        status=RAGStatus.ANSWERED,
        citations=(citation,),
        rag_llm_called=True,
    )
    service = AgentService(model=model, tools=(tool,))

    response = service.answer("What are the office hours?")

    assert response.status is AgentStatus.DOCUMENT_ANSWER
    assert response.document_tool_used is True
    assert response.citations == (citation,)
    tool.execute.assert_called_once_with(
        {"question": "What are the office hours?"}
    )
    second_input = model.respond.call_args_list[1].kwargs["input_items"]
    assert second_input[-1]["type"] == "function_call_output"
    assert second_input[-1]["call_id"] == "call-1"
    assert '"llm_called": true' in second_input[-1]["output"]


def test_malformed_tool_arguments_are_rejected_safely() -> None:
    model = Mock()
    model.respond.return_value = _model_response(
        tool_call=ToolCall(
            call_id="call-1",
            name="document_search",
            arguments="not-json",
        )
    )
    tool = _tool()
    service = AgentService(model=model, tools=(tool,))

    response = service.answer("A question")

    assert response.status is AgentStatus.TOOL_ERROR
    assert response.answer == AgentService.TOOL_ERROR_MESSAGE
    tool.execute.assert_not_called()


def test_unknown_tool_name_is_rejected_safely() -> None:
    model = Mock()
    model.respond.return_value = _model_response(
        tool_call=ToolCall(
            call_id="call-1",
            name="delete_everything",
            arguments="{}",
        )
    )
    tool = _tool()
    service = AgentService(model=model, tools=(tool,))

    response = service.answer("A question")

    assert response.status is AgentStatus.TOOL_ERROR
    tool.execute.assert_not_called()


def test_tool_dependency_failure_is_not_silently_swallowed() -> None:
    model = Mock()
    model.respond.return_value = _model_response(
        tool_call=ToolCall(
            call_id="call-1",
            name="document_search",
            arguments='{"question":"A question"}',
        )
    )
    tool = _tool()
    tool.execute.side_effect = RuntimeError("private dependency detail")
    service = AgentService(model=model, tools=(tool,))

    with pytest.raises(AgentDependencyError) as raised:
        service.answer("A question")

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "private dependency detail" not in str(raised.value)


def test_model_dependency_failure_is_not_silently_swallowed() -> None:
    model = Mock()
    model.respond.side_effect = RuntimeError("provider unavailable")
    service = AgentService(model=model, tools=(_tool(),))

    with pytest.raises(AgentDependencyError):
        service.answer("A question")


def test_maximum_tool_iterations_prevents_infinite_loop() -> None:
    repeated_call = ToolCall(
        call_id="call-1",
        name="document_search",
        arguments='{"question":"A question"}',
    )
    model = Mock()
    model.respond.side_effect = [
        _model_response(tool_call=repeated_call),
        _model_response(
            tool_call=ToolCall(
                call_id="call-2",
                name="document_search",
                arguments='{"question":"A question again"}',
            )
        ),
    ]
    tool = _tool()
    tool.execute.return_value = ToolExecutionResult(
        answer="An answer",
        status=RAGStatus.ANSWERED,
        citations=(),
        rag_llm_called=True,
    )
    service = AgentService(model=model, tools=(tool,), max_tool_iterations=1)

    response = service.answer("A question")

    assert response.status is AgentStatus.MAX_ITERATIONS
    assert response.answer == AgentService.MAX_ITERATIONS_MESSAGE
    assert tool.execute.call_count == 1
    assert model.respond.call_count == 2


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_empty_question_does_not_call_model_or_tools(question: str) -> None:
    model = Mock()
    tool = _tool()
    service = AgentService(model=model, tools=(tool,))

    response = service.answer(question)

    assert response.status is AgentStatus.INVALID_QUESTION
    model.respond.assert_not_called()
    tool.execute.assert_not_called()
