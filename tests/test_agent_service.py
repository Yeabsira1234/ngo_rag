from unittest.mock import Mock

import pytest

from src.agent.memory import InMemoryConversationMemory
from src.agent.models import (
    AgentCitation,
    AgentModelResponse,
    AgentStatus,
    ToolCall,
    ToolDefinition,
    ToolExecutionResult,
    ToolExecutionStatus,
    ToolProvenance,
)
from src.agent.service import AgentDependencyError, AgentService


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
ORGANIZATION_TOOL = ToolDefinition(
    name="organization_info",
    description="Look up fictional structured organization information.",
    parameters={
        "type": "object",
        "properties": {"category": {"type": "string"}},
        "required": ["category"],
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


def _organization_tool() -> Mock:
    tool = Mock()
    tool.definition = ORGANIZATION_TOOL
    return tool


def _memory(max_turns: int = 10) -> InMemoryConversationMemory:
    return InMemoryConversationMemory(max_turns=max_turns)


def test_agent_can_answer_directly_without_calling_tool() -> None:
    model = Mock()
    model.respond.return_value = _model_response(text="Hello! How can I help?")
    tool = _tool()
    organization_tool = _organization_tool()
    service = AgentService(
        model=model,
        tools=(tool, organization_tool),
        memory=_memory(),
    )

    response = service.answer("Say hello")

    assert response.status is AgentStatus.DIRECT_ANSWER
    assert response.answer == "Hello! How can I help?"
    assert response.document_tool_used is False
    assert response.citations == ()
    tool.execute.assert_not_called()
    organization_tool.execute.assert_not_called()


def test_previous_turns_are_passed_with_follow_up_question() -> None:
    model = Mock()
    model.respond.side_effect = [
        _model_response(text="The office is open from 9 to 5."),
        _model_response(text="Friday uses those same hours."),
    ]
    memory = _memory()
    service = AgentService(model=model, tools=(_tool(),), memory=memory)

    service.answer("What are the support office hours?")
    service.answer("What about on Friday?")

    second_context = model.respond.call_args_list[1].kwargs["context"]
    assert second_context.items == (
        {
            "role": "user",
            "content": "What are the support office hours?",
        },
        {
            "role": "assistant",
            "content": "The office is open from 9 to 5.",
        },
        {"role": "user", "content": "What about on Friday?"},
    )


def test_direct_answer_turn_is_stored() -> None:
    model = Mock()
    model.respond.return_value = _model_response(text="Hello!")
    memory = _memory()
    service = AgentService(model=model, tools=(_tool(),), memory=memory)

    service.answer("Say hello")

    turn = memory.get_state().turns[0]
    assert turn.user_message.content == "Say hello"
    assert turn.assistant_message.content == "Hello!"
    assert turn.status is AgentStatus.DIRECT_ANSWER


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
    organization_tool = _organization_tool()
    tool.execute.return_value = ToolExecutionResult(
        answer="The office is open from 9 to 5.",
        status=ToolExecutionStatus.ANSWERED,
        citations=(citation,),
        rag_llm_called=True,
        source="document_search",
        provenance=ToolProvenance.DOCUMENT,
        category="document_answer",
    )
    memory = _memory()
    service = AgentService(
        model=model,
        tools=(tool, organization_tool),
        memory=memory,
    )

    response = service.answer("What are the office hours?")

    assert response.status is AgentStatus.DOCUMENT_ANSWER
    assert response.document_tool_used is True
    assert response.citations == (citation,)
    assert response.tool_sources == ("document_search",)
    tool.execute.assert_called_once_with(
        {"question": "What are the office hours?"}
    )
    organization_tool.execute.assert_not_called()
    first_tools = model.respond.call_args_list[0].kwargs["tools"]
    assert {definition.name for definition in first_tools} == {
        "document_search",
        "organization_info",
    }
    second_context = model.respond.call_args_list[1].kwargs["context"]
    assert second_context.items[-1]["type"] == "function_call_output"
    assert second_context.items[-1]["call_id"] == "call-1"
    assert '"llm_called": true' in second_context.items[-1]["output"]
    stored_turn = memory.get_state().turns[0]
    assert stored_turn.citations == (citation,)
    assert [item.get("type") for item in stored_turn.model_items] == [
        None,
        "function_call",
        "function_call_output",
        None,
    ]


def test_organization_question_selects_organization_info() -> None:
    model = Mock()
    model.respond.side_effect = [
        _model_response(
            tool_call=ToolCall(
                call_id="call-org",
                name="organization_info",
                arguments='{"category":"main_office_location"}',
            )
        ),
        _model_response(text="The main office is in Example City."),
    ]
    document_tool = _tool()
    organization_tool = _organization_tool()
    organization_tool.execute.return_value = ToolExecutionResult(
        answer="100 Example Avenue, Example City, EX 00000",
        status=ToolExecutionStatus.ANSWERED,
        citations=(),
        rag_llm_called=False,
        source="organization_info",
        provenance=ToolProvenance.STRUCTURED_ORGANIZATION_DATA,
        category="main_office_location",
    )
    service = AgentService(
        model=model,
        tools=(document_tool, organization_tool),
        memory=_memory(),
    )

    response = service.answer("Where is the organization's main office?")

    assert response.status is AgentStatus.ORGANIZATION_ANSWER
    assert response.tool_sources == ("organization_info",)
    assert response.citations == ()
    organization_tool.execute.assert_called_once_with(
        {"category": "main_office_location"}
    )
    document_tool.execute.assert_not_called()
    tool_output = model.respond.call_args_list[1].kwargs["context"].items[-1]
    assert '"source": "organization_info"' in tool_output["output"]


def test_organization_follow_up_receives_previous_tool_context() -> None:
    model = Mock()
    model.respond.side_effect = [
        _model_response(
            tool_call=ToolCall(
                call_id="call-org",
                name="organization_info",
                arguments='{"category":"support_office_hours"}',
            )
        ),
        _model_response(text="Support is available Monday through Friday."),
        _model_response(text="Friday hours are 9:00 a.m. to 5:00 p.m."),
    ]
    organization_tool = _organization_tool()
    organization_tool.execute.return_value = ToolExecutionResult(
        answer="Monday through Friday, 9:00 a.m. to 5:00 p.m.",
        status=ToolExecutionStatus.ANSWERED,
        citations=(),
        rag_llm_called=False,
        source="organization_info",
        provenance=ToolProvenance.STRUCTURED_ORGANIZATION_DATA,
        category="support_office_hours",
    )
    service = AgentService(
        model=model,
        tools=(_tool(), organization_tool),
        memory=_memory(),
    )

    service.answer("What are the support office hours?")
    service.answer("What about on Friday?")

    follow_up_context = model.respond.call_args_list[2].kwargs["context"].items
    assert follow_up_context[-1] == {
        "role": "user",
        "content": "What about on Friday?",
    }
    assert any(
        item.get("type") == "function_call_output"
        and '"source": "organization_info"' in item["output"]
        for item in follow_up_context
    )


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
    service = AgentService(model=model, tools=(tool,), memory=_memory())

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
    service = AgentService(model=model, tools=(tool,), memory=_memory())

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
    service = AgentService(model=model, tools=(tool,), memory=_memory())

    with pytest.raises(AgentDependencyError) as raised:
        service.answer("A question")

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "private dependency detail" not in str(raised.value)


def test_model_dependency_failure_is_not_silently_swallowed() -> None:
    model = Mock()
    model.respond.side_effect = RuntimeError("provider unavailable")
    service = AgentService(model=model, tools=(_tool(),), memory=_memory())

    with pytest.raises(AgentDependencyError):
        service.answer("A question")


def test_failed_turn_does_not_change_previous_memory() -> None:
    model = Mock()
    model.respond.side_effect = [
        _model_response(text="A valid answer"),
        RuntimeError("provider unavailable"),
    ]
    memory = _memory()
    service = AgentService(model=model, tools=(_tool(),), memory=memory)
    service.answer("First question")

    with pytest.raises(AgentDependencyError):
        service.answer("Failed follow-up")

    state = memory.get_state()
    assert len(state.turns) == 1
    assert state.turns[0].user_message.content == "First question"


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
        status=ToolExecutionStatus.ANSWERED,
        citations=(),
        rag_llm_called=True,
        source="document_search",
        provenance=ToolProvenance.DOCUMENT,
        category="document_answer",
    )
    service = AgentService(
        model=model,
        tools=(tool,),
        memory=_memory(),
        max_tool_iterations=1,
    )

    response = service.answer("A question")

    assert response.status is AgentStatus.MAX_ITERATIONS
    assert response.answer == AgentService.MAX_ITERATIONS_MESSAGE
    assert tool.execute.call_count == 1
    assert model.respond.call_count == 2


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_empty_question_does_not_call_model_or_tools(question: str) -> None:
    model = Mock()
    tool = _tool()
    service = AgentService(model=model, tools=(tool,), memory=_memory())

    response = service.answer(question)

    assert response.status is AgentStatus.INVALID_QUESTION
    model.respond.assert_not_called()
    tool.execute.assert_not_called()
