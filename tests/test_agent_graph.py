import json
from unittest.mock import Mock

from src.agent.memory import InMemoryConversationMemory
from src.agent.models import (
    AgentModelResponse,
    AgentStatus,
    ToolCall,
    ToolDefinition,
    ToolExecutionResult,
    ToolExecutionStatus,
    ToolProvenance,
)
from src.agent.service import AgentService
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
