from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from src.agent.organization_data import OrganizationInfoCategory
from src.mcp.adapters import MCPAdapters
from src.mcp.schemas import (
    DocumentSearchOutput,
    MAX_SCHEMA_INPUT_LENGTH,
    OrganizationInformationOutput,
    SQLInformationOutput,
    WeatherInformationOutput,
)
from src.sql.models import SQLOperation


InputText = Annotated[
    str,
    Field(min_length=1, max_length=MAX_SCHEMA_INPUT_LENGTH),
]
READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
WEATHER_READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
)


def create_mcp_server(adapters: MCPAdapters) -> FastMCP:
    """Create the stdio MCP interface without importing LangGraph internals."""
    server = FastMCP(
        "NGO RAG Read-Only Tools",
        instructions=(
            "Read-only access to existing document, organization, structured SQL, "
            "and live weather application capabilities. No resources or prompts."
        ),
        log_level="WARNING",
    )

    @server.tool(
        name="search_documents",
        description=(
            "Search indexed documents through the existing relevance-filtered RAG "
            "service and return its answer, status, and full citations."
        ),
        annotations=READ_ONLY,
    )
    def search_documents(question: InputText) -> DocumentSearchOutput:
        return adapters.search_documents(question)

    @server.tool(
        name="organization_information",
        description=(
            "Return one approved category from the existing fictional organization "
            "information service."
        ),
        annotations=READ_ONLY,
    )
    def organization_information(
        category: OrganizationInfoCategory,
    ) -> OrganizationInformationOutput:
        return adapters.organization_information(category.value)

    @server.tool(
        name="sql_information",
        description=(
            "Run an existing predefined or validated natural-language SQL operation. "
            "All execution remains read-only, private-field restricted, bounded, and "
            "timeout controlled. This tool never accepts or returns raw SQL."
        ),
        annotations=READ_ONLY,
    )
    def sql_information(
        operation: SQLOperation,
        office_name: InputText | None = None,
        language: InputText | None = None,
        question: InputText | None = None,
    ) -> SQLInformationOutput:
        return adapters.sql_information(
            operation.value,
            office_name,
            language,
            question,
        )

    @server.tool(
        name="weather_information",
        description=(
            "Return current weather and today's forecast for a validated city through "
            "the existing fixed-endpoint Open-Meteo client."
        ),
        annotations=WEATHER_READ_ONLY,
    )
    def weather_information(city: InputText) -> WeatherInformationOutput:
        return adapters.weather_information(city)

    return server
