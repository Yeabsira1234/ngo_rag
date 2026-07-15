import json
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
from src.agent.tools import SQLQueryTool
from src.sql.models import SQLOperation, SQLQueryResult


def tool(name: str, provenance: ToolProvenance) -> Mock:
    instance = Mock()
    instance.definition = ToolDefinition(
        name=name, description=name, parameters={"type": "object"}
    )
    instance.execute.return_value = ToolExecutionResult(
        answer=f"{name} result",
        status=ToolExecutionStatus.ANSWERED,
        citations=(),
        rag_llm_called=False,
        source=name,
        provenance=provenance,
        category="answer",
    )
    return instance


def test_compiled_graph_exposes_required_nodes() -> None:
    service = AgentService(
        model=Mock(),
        tools=(tool("document_search", ToolProvenance.DOCUMENT),),
        memory=InMemoryConversationMemory(),
    )
    nodes = set(service.graph.compiled.get_graph().nodes)
    assert {
        "validate_input", "call_model", "route_model_output", "execute_tools",
        "record_tool_results", "check_iteration_limit", "finalize_response",
        "finalize_limit_reached", "handle_dependency_failure", "commit_memory",
    } <= nodes


def test_multiple_tool_calls_execute_and_return_outputs_in_order() -> None:
    first = tool("document_search", ToolProvenance.DOCUMENT)
    second = tool(
        "organization_info", ToolProvenance.STRUCTURED_ORGANIZATION_DATA
    )
    model = Mock()
    model.respond.side_effect = [
        AgentModelResponse(
            output_text="",
            tool_calls=(
                ToolCall("call-1", "document_search", '{"question":"q"}'),
                ToolCall("call-2", "organization_info", '{"category":"name"}'),
            ),
            continuation_items=(
                {"type": "function_call", "call_id": "call-1"},
                {"type": "function_call", "call_id": "call-2"},
            ),
        ),
        AgentModelResponse(
            output_text="Combined answer",
            tool_calls=(),
            continuation_items=({"role": "assistant", "content": "Combined answer"},),
        ),
    ]
    memory = InMemoryConversationMemory()
    response = AgentService(
        model=model, tools=(first, second), memory=memory
    ).answer("Use both")
    assert response.status is AgentStatus.DOCUMENT_ANSWER
    assert response.tool_sources == ("document_search", "organization_info")
    second_context = model.respond.call_args_list[1].kwargs["context"].items
    outputs = [item for item in second_context if item.get("type") == "function_call_output"]
    assert [item["call_id"] for item in outputs] == ["call-1", "call-2"]
    assert len(memory.get_state().turns) == 1
    assert model.respond.call_args_list[1].kwargs["tools"] == ()
    assert "Synthesize the completed tool outputs" in (
        model.respond.call_args_list[1].kwargs["instructions"]
    )


def test_malformed_empty_model_output_is_safe_and_not_committed() -> None:
    model = Mock()
    model.respond.return_value = AgentModelResponse("", (), ())
    memory = InMemoryConversationMemory()
    response = AgentService(
        model=model,
        tools=(tool("document_search", ToolProvenance.DOCUMENT),),
        memory=memory,
    ).answer("Question")
    assert response.status is AgentStatus.TOOL_ERROR
    assert memory.get_state().turns == ()


def test_most_recent_service_events_use_one_bounded_predefined_query() -> None:
    rows = tuple(
        {
            "ServiceEventID": event_id,
            "ClientCode": f"CLIENT-{event_id}",
            "ServiceType": "Case management",
            "ServiceDate": f"2026-07-{event_id:02d}",
            "DurationMinutes": 30,
            "Outcome": "Completed",
        }
        for event_id in range(5, 0, -1)
    )
    repository = Mock()
    repository.execute.return_value = SQLQueryResult(
        operation=SQLOperation.RECENT_SERVICE_EVENTS,
        rows=rows,
        truncated=False,
    )
    model = Mock()
    model.respond.side_effect = [
        AgentModelResponse(
            output_text="",
            tool_calls=(
                ToolCall(
                    "sql-1",
                    "sql_query",
                    '{"operation":"recent_service_events","office_name":null,'
                    '"language":null,"question":null}',
                ),
            ),
            continuation_items=(
                {"type": "function_call", "call_id": "sql-1"},
            ),
        ),
        AgentModelResponse(
            output_text="The five most recent service events are listed here.",
            tool_calls=(),
            continuation_items=(
                {
                    "role": "assistant",
                    "content": "The five most recent service events are listed here.",
                },
            ),
        ),
    ]

    response = AgentService(
        model=model,
        tools=(SQLQueryTool(repository),),
        memory=InMemoryConversationMemory(),
    ).answer("What are the five most recent service events?")

    repository.execute.assert_called_once_with(
        SQLOperation.RECENT_SERVICE_EVENTS, {}
    )
    second_context = model.respond.call_args_list[1].kwargs["context"].items
    outputs = [
        item for item in second_context
        if item.get("type") == "function_call_output"
    ]
    assert len(outputs) == 1
    recorded_tool_result = json.loads(outputs[0]["output"])
    recorded_sql_result = json.loads(recorded_tool_result["answer"])
    assert recorded_sql_result["operation"] == "recent_service_events"
    assert recorded_sql_result["row_count"] == 5
    assert recorded_sql_result["rows"] == list(rows)
    assert response.status is AgentStatus.SQL_ANSWER
    assert response.tool_sources == ("sql_query",)
    assert "restriction" not in response.answer.lower()


def _multi_model(*calls: ToolCall, answer: str = "Combined answer") -> Mock:
    model = Mock()
    model.respond.side_effect = [
        AgentModelResponse(
            output_text="",
            tool_calls=tuple(calls),
            continuation_items=tuple(
                {"type": "function_call", "call_id": item.call_id}
                for item in calls
            ),
        ),
        AgentModelResponse(
            output_text=answer,
            tool_calls=(),
            continuation_items=({"role": "assistant", "content": answer},),
        ),
    ]
    return model


def test_sql_then_document_plan_preserves_order_and_multiple_citations() -> None:
    execution_order: list[str] = []
    sql = tool("sql_query", ToolProvenance.STRUCTURED_SQL_DATA)
    document = tool("document_search", ToolProvenance.DOCUMENT)
    sql.execute.side_effect = lambda arguments: (
        execution_order.append("sql_query") or sql.execute.return_value
    )
    citations = (
        AgentCitation("policy-a.pdf", 1, 2, 0.2),
        AgentCitation("policy-b.pdf", 4, 1, 0.3),
    )
    document_result = ToolExecutionResult(
        answer="Housing assessments are required.",
        status=ToolExecutionStatus.ANSWERED,
        citations=citations,
        rag_llm_called=True,
        source="document_search",
        provenance=ToolProvenance.DOCUMENT,
        category="document_answer",
    )
    document.execute.side_effect = lambda arguments: (
        execution_order.append("document_search") or document_result
    )
    model = _multi_model(
        ToolCall("sql", "sql_query", '{"operation":"list_open_cases"}'),
        ToolCall("doc", "document_search", '{"question":"housing assessments"}'),
        answer="SQL reports open cases; the documents require assessments.",
    )
    memory = InMemoryConversationMemory()

    response = AgentService(
        model=model, tools=(sql, document), memory=memory
    ).answer("Count open housing cases and summarize assessment guidance")

    assert execution_order == ["sql_query", "document_search"]
    assert response.tool_sources == ("sql_query", "document_search")
    assert response.citations == citations
    assert len(memory.get_state().turns) == 1
    synthesis_items = model.respond.call_args_list[1].kwargs["context"].items
    outputs = [item for item in synthesis_items if item.get("type") == "function_call_output"]
    assert [item["call_id"] for item in outputs] == ["sql", "doc"]
    assert '"provenance": "structured_sql_data"' in outputs[0]["output"]
    assert '"provenance": "document"' in outputs[1]["output"]


def test_organization_then_sql_plan_executes_in_declared_order() -> None:
    order: list[str] = []
    organization = tool(
        "organization_info", ToolProvenance.STRUCTURED_ORGANIZATION_DATA
    )
    sql = tool("sql_query", ToolProvenance.STRUCTURED_SQL_DATA)
    organization.execute.side_effect = lambda arguments: (
        order.append("organization_info") or organization.execute.return_value
    )
    sql.execute.side_effect = lambda arguments: (
        order.append("sql_query") or sql.execute.return_value
    )
    model = _multi_model(
        ToolCall("org", "organization_info", '{"category":"location"}'),
        ToolCall("sql", "sql_query", '{"operation":"list_offices"}'),
    )
    AgentService(model=model, tools=(organization, sql), memory=InMemoryConversationMemory()).answer(
        "Compare the published location with database offices"
    )
    assert order == ["organization_info", "sql_query"]


def test_partial_tool_dependency_failure_keeps_successful_evidence() -> None:
    sql = tool("sql_query", ToolProvenance.STRUCTURED_SQL_DATA)
    document = tool("document_search", ToolProvenance.DOCUMENT)
    document.execute.side_effect = RuntimeError("vector service unavailable")
    model = _multi_model(
        ToolCall("sql", "sql_query", '{"operation":"list_open_cases"}'),
        ToolCall("doc", "document_search", '{"question":"housing policy"}'),
        answer="The database result is available, but document guidance was unavailable.",
    )
    memory = InMemoryConversationMemory()
    response = AgentService(
        model=model, tools=(sql, document), memory=memory
    ).answer("Give the database count and document policy")
    assert response.status is AgentStatus.DOCUMENT_ANSWER
    assert "database result is available" in response.answer
    outputs = [
        item for item in model.respond.call_args_list[1].kwargs["context"].items
        if item.get("type") == "function_call_output"
    ]
    assert '"status": "answered"' in outputs[0]["output"]
    assert '"status": "error"' in outputs[1]["output"]
    assert len(memory.get_state().turns) == 1


def test_weak_document_evidence_is_labeled_insufficient_not_strong() -> None:
    sql = tool("sql_query", ToolProvenance.STRUCTURED_SQL_DATA)
    document = tool("document_search", ToolProvenance.DOCUMENT)
    document.execute.return_value = ToolExecutionResult(
        answer="Not enough relevant document information was found.",
        status=ToolExecutionStatus.INSUFFICIENT_CONTEXT,
        citations=(),
        rag_llm_called=False,
        source="document_search",
        provenance=ToolProvenance.DOCUMENT,
        category="insufficient_context",
    )
    model = _multi_model(
        ToolCall("sql", "sql_query", '{"operation":"recent_service_events"}'),
        ToolCall("doc", "document_search", '{"question":"service policy"}'),
        answer="Recent events were found; indexed documents lacked relevant guidance.",
    )
    response = AgentService(
        model=model, tools=(sql, document), memory=InMemoryConversationMemory()
    ).answer("Show recent events and related guidance")
    assert response.citations == ()
    assert "lacked relevant guidance" in response.answer


def test_duplicate_calls_inside_plan_are_rejected_before_execution() -> None:
    document = tool("document_search", ToolProvenance.DOCUMENT)
    duplicate = ToolCall("one", "document_search", '{"question":"same policy"}')
    model = _multi_model(
        duplicate,
        ToolCall("two", "document_search", '{"question":"  same   policy  "}'),
    )
    memory = InMemoryConversationMemory()
    response = AgentService(
        model=model, tools=(document,), memory=memory
    ).answer("Search twice")
    assert response.status is AgentStatus.TOOL_ERROR
    assert response.answer == AgentService.REPEATED_TOOL_CALL_MESSAGE
    document.execute.assert_not_called()
    assert memory.get_state().turns == ()


def test_oversized_plan_is_rejected_before_any_tool_executes() -> None:
    first = tool("document_search", ToolProvenance.DOCUMENT)
    second = tool("organization_info", ToolProvenance.STRUCTURED_ORGANIZATION_DATA)
    model = _multi_model(
        ToolCall("one", "document_search", '{"question":"policy"}'),
        ToolCall("two", "organization_info", '{"category":"location"}'),
    )
    memory = InMemoryConversationMemory()
    response = AgentService(
        model=model,
        tools=(first, second),
        memory=memory,
        max_tool_calls_per_turn=1,
    ).answer("Use both")
    assert response.status is AgentStatus.TOOL_ERROR
    first.execute.assert_not_called()
    second.execute.assert_not_called()
    assert model.respond.call_count == 1
    assert memory.get_state().turns == ()


def test_unknown_tool_in_plan_is_rejected_without_committing_memory() -> None:
    known = tool("document_search", ToolProvenance.DOCUMENT)
    model = _multi_model(ToolCall("bad", "external_api", '{}'))
    memory = InMemoryConversationMemory()
    response = AgentService(model=model, tools=(known,), memory=memory).answer("Use unknown")
    assert response.status is AgentStatus.TOOL_ERROR
    known.execute.assert_not_called()
    assert memory.get_state().turns == ()


def test_follow_up_receives_prior_combined_turn_context() -> None:
    document = tool("document_search", ToolProvenance.DOCUMENT)
    sql = tool("sql_query", ToolProvenance.STRUCTURED_SQL_DATA)
    model = _multi_model(
        ToolCall("sql", "sql_query", '{"operation":"list_open_cases"}'),
        ToolCall("doc", "document_search", '{"question":"case policy"}'),
        answer="Combined first answer",
    )
    model.respond.side_effect = list(model.respond.side_effect) + [
        AgentModelResponse(
            output_text="Follow-up answer",
            tool_calls=(),
            continuation_items=({"role": "assistant", "content": "Follow-up answer"},),
        )
    ]
    service = AgentService(
        model=model, tools=(sql, document), memory=InMemoryConversationMemory()
    )
    service.answer("Combine database and policy")
    service.answer("What does that mean?")
    follow_up_items = model.respond.call_args_list[2].kwargs["context"].items
    assert any(item.get("content") == "Combined first answer" for item in follow_up_items)
    assert follow_up_items[-1]["content"] == "What does that mean?"


def test_fully_failed_multi_tool_turn_does_not_commit_memory() -> None:
    document = tool("document_search", ToolProvenance.DOCUMENT)
    sql = tool("sql_query", ToolProvenance.STRUCTURED_SQL_DATA)
    document.execute.side_effect = RuntimeError("document unavailable")
    sql.execute.side_effect = RuntimeError("database unavailable")
    model = _multi_model(
        ToolCall("sql", "sql_query", '{"operation":"list_open_cases"}'),
        ToolCall("doc", "document_search", '{"question":"case policy"}'),
        answer="Neither requested source was available.",
    )
    memory = InMemoryConversationMemory()
    response = AgentService(
        model=model, tools=(sql, document), memory=memory
    ).answer("Use both unavailable sources")
    assert "Neither requested source" in response.answer
    assert memory.get_state().turns == ()


def test_weather_question_selects_only_weather_tool() -> None:
    weather = tool("weather_information", ToolProvenance.EXTERNAL_API)
    document = tool("document_search", ToolProvenance.DOCUMENT)
    model = Mock()
    model.respond.side_effect = [
        AgentModelResponse(
            "",
            (ToolCall("weather", "weather_information", '{"city":"Arlington"}'),),
            ({"type": "function_call", "call_id": "weather"},),
        ),
        AgentModelResponse(
            "Arlington is warm today.", (),
            ({"role": "assistant", "content": "Arlington is warm today."},),
        ),
    ]
    response = AgentService(
        model=model, tools=(document, weather), memory=InMemoryConversationMemory()
    ).answer("What's the weather in Arlington today?")
    assert response.status is AgentStatus.WEATHER_ANSWER
    assert response.tool_sources == ("weather_information",)
    weather.execute.assert_called_once_with({"city": "Arlington"})
    document.execute.assert_not_called()


def test_sql_and_weather_are_combined_in_planned_order() -> None:
    order: list[str] = []
    sql = tool("sql_query", ToolProvenance.STRUCTURED_SQL_DATA)
    weather = tool("weather_information", ToolProvenance.EXTERNAL_API)
    sql.execute.side_effect = lambda arguments: (
        order.append("sql_query") or sql.execute.return_value
    )
    weather.execute.side_effect = lambda arguments: (
        order.append("weather_information") or weather.execute.return_value
    )
    model = _multi_model(
        ToolCall("sql", "sql_query", '{"operation":"list_offices"}'),
        ToolCall("weather", "weather_information", '{"city":"Alexandria"}'),
    )
    response = AgentService(
        model=model, tools=(sql, weather), memory=InMemoryConversationMemory()
    ).answer("List offices and give Alexandria weather")
    assert order == ["sql_query", "weather_information"]
    assert response.tool_sources == ("sql_query", "weather_information")


def test_organization_and_weather_are_combined_in_planned_order() -> None:
    organization = tool(
        "organization_info", ToolProvenance.STRUCTURED_ORGANIZATION_DATA
    )
    weather = tool("weather_information", ToolProvenance.EXTERNAL_API)
    model = _multi_model(
        ToolCall(
            "org", "organization_info", '{"category":"main_office_location"}'
        ),
        ToolCall("weather", "weather_information", '{"city":"Alexandria"}'),
    )
    response = AgentService(
        model=model,
        tools=(organization, weather),
        memory=InMemoryConversationMemory(),
    ).answer("Where is the office and what is the weather there?")
    assert response.tool_sources == ("organization_info", "weather_information")
    assert model.respond.call_args_list[1].kwargs["tools"] == ()


def test_document_question_does_not_call_weather_tool() -> None:
    document = tool("document_search", ToolProvenance.DOCUMENT)
    weather = tool("weather_information", ToolProvenance.EXTERNAL_API)
    model = Mock()
    model.respond.side_effect = [
        AgentModelResponse(
            "",
            (ToolCall("doc", "document_search", '{"question":"Train Dreams author"}'),),
            ({"type": "function_call", "call_id": "doc"},),
        ),
        AgentModelResponse(
            "The indexed document identifies the author.", (),
            ({"role": "assistant", "content": "The indexed document identifies the author."},),
        ),
    ]
    AgentService(
        model=model, tools=(document, weather), memory=InMemoryConversationMemory()
    ).answer("Who wrote Train Dreams?")
    document.execute.assert_called_once()
    weather.execute.assert_not_called()


def test_weather_dependency_failure_does_not_corrupt_existing_memory() -> None:
    weather = tool("weather_information", ToolProvenance.EXTERNAL_API)
    weather.execute.side_effect = RuntimeError("network detail")
    model = Mock()
    model.respond.side_effect = [
        AgentModelResponse(
            "A retained answer.", (),
            ({"role": "assistant", "content": "A retained answer."},),
        ),
        AgentModelResponse(
            "",
            (ToolCall("weather", "weather_information", '{"city":"Arlington"}'),),
            ({"type": "function_call", "call_id": "weather"},),
        ),
    ]
    memory = InMemoryConversationMemory()
    service = AgentService(model=model, tools=(weather,), memory=memory)
    service.answer("Remember this")
    with pytest.raises(AgentDependencyError):
        service.answer("What's the weather?")
    assert len(memory.get_state().turns) == 1
    assert memory.get_state().turns[0].assistant_message.content == "A retained answer."
