import asyncio
import io
import json
import logging
from contextlib import redirect_stdout
from unittest.mock import Mock

import pytest

import mcp_server
from mcp.server.fastmcp.exceptions import ToolError
from mcp.shared.memory import create_connected_server_and_client_session
from src.agent.models import (
    AgentCitation,
    ToolDefinition,
    ToolExecutionResult,
    ToolExecutionStatus,
    ToolProvenance,
)
from src.agent.tools import SQLQueryTool
from src.mcp.adapters import MCPAdapters
from src.mcp.server import create_mcp_server
from src.mcp.utilities import configure_mcp_logging
from src.sql.validation import SQLPrivacyError


def fake_tool(
    name: str,
    provenance: ToolProvenance,
    *,
    answer: str = "answer",
    status: ToolExecutionStatus = ToolExecutionStatus.ANSWERED,
    citations: tuple[AgentCitation, ...] = (),
) -> Mock:
    instance = Mock()
    instance.definition = ToolDefinition(name, name, {"type": "object"})
    instance.execute.return_value = ToolExecutionResult(
        answer=answer,
        status=status,
        citations=citations,
        rag_llm_called=False,
        source=name,
        provenance=provenance,
        category="test_category",
    )
    return instance


def adapters_with(*, sql_tool=None) -> tuple[MCPAdapters, dict[str, Mock]]:
    tools = {
        "document_search": fake_tool(
            "document_search", ToolProvenance.DOCUMENT
        ),
        "organization_info": fake_tool(
            "organization_info", ToolProvenance.STRUCTURED_ORGANIZATION_DATA
        ),
        "sql_query": sql_tool
        or fake_tool(
            "sql_query",
            ToolProvenance.STRUCTURED_SQL_DATA,
            answer=json.dumps(
                {
                    "operation": "list_offices",
                    "row_count": 1,
                    "rows": [{"OfficeName": "Alexandria Community Office"}],
                }
            ),
        ),
        "weather_information": fake_tool(
            "weather_information",
            ToolProvenance.EXTERNAL_API,
            answer=json.dumps(
                {
                    "provider": "Open-Meteo",
                    "location": "Arlington, Virginia, United States",
                    "current": {"temperature_c": 24.0},
                }
            ),
        ),
    }
    return MCPAdapters.from_tools(tuple(tools.values()), 2_000), tools


def run(coroutine):
    return asyncio.run(coroutine)


def structured(result):
    return result[1]


def test_tool_list_contains_only_four_documented_read_only_tools() -> None:
    adapters, _ = adapters_with()
    tools = run(create_mcp_server(adapters).list_tools())
    assert [tool.name for tool in tools] == [
        "search_documents",
        "organization_information",
        "sql_information",
        "weather_information",
    ]
    for tool in tools:
        assert tool.inputSchema["type"] == "object"
        assert tool.outputSchema["type"] == "object"
        assert tool.annotations.readOnlyHint is True
        assert tool.annotations.destructiveHint is False
    sql_schema = next(tool for tool in tools if tool.name == "sql_information")
    assert "sql" not in sql_schema.inputSchema["properties"]
    assert "operation" in sql_schema.inputSchema["required"]
    assert "delete" not in json.dumps(sql_schema.inputSchema).lower()


def test_document_call_preserves_full_citations_and_reuses_existing_tool() -> None:
    citation = AgentCitation(
        source="guide.pdf",
        page_number=3,
        chunk_index=7,
        distance=0.21,
        source_relative_path="uploads/guide.pdf",
        document_id="doc-7",
    )
    adapters, tools = adapters_with()
    tools["document_search"].execute.return_value = ToolExecutionResult(
        answer="Document answer",
        status=ToolExecutionStatus.ANSWERED,
        citations=(citation,),
        rag_llm_called=True,
        source="document_search",
        provenance=ToolProvenance.DOCUMENT,
        category="document_answer",
    )
    result = structured(run(
        create_mcp_server(adapters).call_tool(
            "search_documents", {"question": "What is the policy?"}
        )
    ))
    tools["document_search"].execute.assert_called_once_with(
        {"question": "What is the policy?"}
    )
    assert result["status"] == "answered"
    assert result["citations"] == [
        {
            "source": "guide.pdf",
            "page_number": 3,
            "chunk_index": 7,
            "distance": 0.21,
            "source_relative_path": "uploads/guide.pdf",
            "document_id": "doc-7",
        }
    ]


def test_organization_and_weather_outputs_are_structured() -> None:
    adapters, tools = adapters_with()
    server = create_mcp_server(adapters)
    organization = structured(run(
        server.call_tool(
            "organization_information", {"category": "organization_name"}
        )
    ))
    weather = structured(run(
        server.call_tool("weather_information", {"city": "Arlington"})
    ))
    assert organization["source"] == "organization_info"
    assert weather["data"]["provider"] == "Open-Meteo"
    assert weather["data"]["current"]["temperature_c"] == 24.0
    tools["weather_information"].execute.assert_called_once_with(
        {"city": "Arlington"}
    )


def test_sql_privacy_rejection_uses_existing_validator_boundary() -> None:
    repository = Mock()
    natural_language = Mock()
    natural_language.query.side_effect = SQLPrivacyError("private generated SQL")
    sql_tool = SQLQueryTool(repository, natural_language)
    adapters, _ = adapters_with(sql_tool=sql_tool)
    result = structured(run(
        create_mcp_server(adapters).call_tool(
            "sql_information",
            {
                "operation": "natural_language_query",
                "office_name": None,
                "language": None,
                "question": "Return private client details",
            },
        )
    ))
    assert result["status"] == "error"
    assert result["failure_category"] == "sql_policy_rejection"
    assert result["data"] is None
    assert "private generated SQL" not in result["answer"]
    repository.execute.assert_not_called()


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("search_documents", {}),
        ("search_documents", {"question": 4}),
        ("organization_information", {"category": "private_records"}),
        ("sql_information", {"operation": "delete"}),
        ("weather_information", {"city": 4}),
    ],
)
def test_malformed_inputs_are_rejected_by_generated_schemas(
    tool_name: str, arguments: dict
) -> None:
    adapters, tools = adapters_with()
    with pytest.raises(Exception):
        run(create_mcp_server(adapters).call_tool(tool_name, arguments))
    for application_tool in tools.values():
        application_tool.execute.assert_not_called()


def test_configured_input_limit_rejects_oversized_text_without_service_call() -> None:
    adapters, tools = adapters_with()
    adapters.max_input_length = 5
    with pytest.raises(ToolError, match="size limit"):
        run(
            create_mcp_server(adapters).call_tool(
                "search_documents", {"question": "sixsix"}
            )
        )
    tools["document_search"].execute.assert_not_called()


def test_in_process_server_operations_do_not_print_to_stdout() -> None:
    adapters, _ = adapters_with()
    output = io.StringIO()
    with redirect_stdout(output):
        run(create_mcp_server(adapters).list_tools())
        run(
            create_mcp_server(adapters).call_tool(
                "weather_information", {"city": "Arlington"}
            )
        )
    assert output.getvalue() == ""


def test_entry_point_runs_official_stdio_transport_without_stdout(monkeypatch) -> None:
    settings = Mock(log_level="INFO", openai_api_key="secret")
    server = Mock()
    monkeypatch.setattr(mcp_server.Settings, "from_env", Mock(return_value=settings))
    monkeypatch.setattr(mcp_server, "build_server", Mock(return_value=server))
    monkeypatch.setattr(mcp_server, "configure_mcp_logging", Mock())
    output = io.StringIO()
    with redirect_stdout(output):
        result = mcp_server.run()
    assert result == 0
    server.run.assert_called_once_with(transport="stdio")
    assert output.getvalue() == ""


def test_official_client_session_initializes_lists_and_calls_server() -> None:
    adapters, _ = adapters_with()

    async def scenario():
        async with create_connected_server_and_client_session(
            create_mcp_server(adapters), raise_exceptions=True
        ) as session:
            listed = await session.list_tools()
            called = await session.call_tool(
                "weather_information", {"city": "Arlington"}
            )
            return listed, called

    listed, called = run(scenario())
    assert [tool.name for tool in listed.tools] == [
        "search_documents",
        "organization_information",
        "sql_information",
        "weather_information",
    ]
    assert called.isError is False
    assert called.structuredContent["data"]["provider"] == "Open-Meteo"


def test_production_server_builder_reuses_shared_application_tool_factory(
    monkeypatch,
) -> None:
    adapters, tools = adapters_with()
    settings = Mock(mcp_max_input_length=123)
    factory = Mock(return_value=tuple(tools.values()))
    monkeypatch.setattr(mcp_server, "build_application_tools", factory)
    server = mcp_server.build_server(settings)
    factory.assert_called_once_with(settings)
    assert len(run(server.list_tools())) == 4


def test_mcp_logging_writes_stderr_and_application_file_not_stdout(
    tmp_path, capsys
) -> None:
    log_path = tmp_path / "application.log"
    configure_mcp_logging("INFO", log_path=log_path)
    logging.getLogger("src.mcp.adapters").info(
        "event=mcp_tool_completed tool=search_documents duration_ms=1 "
        "status=answered failure_category=none"
    )
    for handler in logging.getLogger().handlers:
        handler.flush()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "tool=search_documents" in captured.err
    assert "tool=search_documents" in log_path.read_text(encoding="utf-8")
