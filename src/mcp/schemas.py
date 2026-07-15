from typing import Any

from pydantic import BaseModel, ConfigDict


MAX_SCHEMA_INPUT_LENGTH = 10_000


class MCPOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CitationOutput(MCPOutput):
    source: str
    page_number: int
    chunk_index: int
    distance: float
    source_relative_path: str = ""
    document_id: str = ""


class DocumentSearchOutput(MCPOutput):
    answer: str
    status: str
    citations: list[CitationOutput]


class OrganizationInformationOutput(MCPOutput):
    answer: str
    status: str
    category: str | None
    source: str


class SQLInformationOutput(MCPOutput):
    answer: str
    status: str
    operation: str | None
    data: dict[str, Any] | None = None
    failure_category: str | None = None


class WeatherInformationOutput(MCPOutput):
    answer: str
    status: str
    data: dict[str, Any] | None = None
    failure_category: str | None = None
