import json
import logging
from typing import Any, Protocol

from src.agent.models import (
    AgentCitation,
    AgentModelResponse,
    AgentResponse,
    AgentStatus,
    ToolDefinition,
)
from src.agent.tools import AgentTool


logger = logging.getLogger(__name__)


class AgentServiceError(RuntimeError):
    """Base exception for agent execution failures."""


class AgentDependencyError(AgentServiceError):
    """Raised when the model or an application tool fails."""


class AgentModel(Protocol):
    def respond(
        self,
        *,
        instructions: str,
        input_items: str | list[Any],
        tools: tuple[ToolDefinition, ...],
    ) -> AgentModelResponse: ...


class AgentService:
    """Coordinate model-directed tool calls separately from direct RAG."""

    INVALID_QUESTION_MESSAGE = "Please enter a question."
    TOOL_ERROR_MESSAGE = "The agent could not safely execute the selected tool."
    MAX_ITERATIONS_MESSAGE = (
        "The agent stopped because it reached its tool-call safety limit."
    )
    DEPENDENCY_ERROR_MESSAGE = "The agent could not complete the request."
    INSTRUCTIONS = (
        "You are a helpful assistant with access to an indexed-document search "
        "tool. Use document_search for questions about the organization's indexed "
        "document, policies, procedures, or facts. Answer directly only when "
        "document retrieval is unnecessary. Do not invent document facts."
    )

    def __init__(
        self,
        model: AgentModel,
        tools: tuple[AgentTool, ...],
        max_tool_iterations: int = 2,
    ) -> None:
        if max_tool_iterations <= 0:
            raise ValueError("max_tool_iterations must be greater than zero.")
        tool_map = {tool.definition.name: tool for tool in tools}
        if len(tool_map) != len(tools):
            raise ValueError("Agent tool names must be unique.")

        self.model = model
        self.tools = tool_map
        self.tool_definitions = tuple(tool.definition for tool in tools)
        self.max_tool_iterations = max_tool_iterations

    def answer(self, question: str) -> AgentResponse:
        normalized_question = question.strip()
        if not normalized_question:
            logger.info("event=agent_invalid_question")
            return AgentResponse(
                answer=self.INVALID_QUESTION_MESSAGE,
                status=AgentStatus.INVALID_QUESTION,
                citations=(),
                document_tool_used=False,
            )

        input_items: str | list[Any] = [
            {"role": "user", "content": normalized_question}
        ]
        citations: list[AgentCitation] = []
        document_tool_used = False

        for iteration in range(self.max_tool_iterations + 1):
            logger.info("event=agent_model_invocation iteration=%d", iteration)
            try:
                model_response = self.model.respond(
                    instructions=self.INSTRUCTIONS,
                    input_items=input_items,
                    tools=self.tool_definitions,
                )
            except Exception as error:
                logger.exception(
                    "event=agent_model_failed error_type=%s iteration=%d",
                    type(error).__name__,
                    iteration,
                )
                raise AgentDependencyError(self.DEPENDENCY_ERROR_MESSAGE) from error

            if not model_response.tool_calls:
                status = (
                    AgentStatus.DOCUMENT_ANSWER
                    if document_tool_used
                    else AgentStatus.DIRECT_ANSWER
                )
                return AgentResponse(
                    answer=model_response.output_text,
                    status=status,
                    citations=tuple(citations),
                    document_tool_used=document_tool_used,
                )

            if iteration >= self.max_tool_iterations:
                logger.warning(
                    "event=agent_max_iterations_reached max_iterations=%d",
                    self.max_tool_iterations,
                )
                return AgentResponse(
                    answer=self.MAX_ITERATIONS_MESSAGE,
                    status=AgentStatus.MAX_ITERATIONS,
                    citations=tuple(citations),
                    document_tool_used=document_tool_used,
                )

            next_input = list(input_items)
            next_input.extend(model_response.continuation_items)
            for tool_call in model_response.tool_calls:
                logger.info(
                    "event=agent_tool_selected tool_name=%s iteration=%d",
                    tool_call.name,
                    iteration + 1,
                )
                tool = self.tools.get(tool_call.name)
                if tool is None:
                    return self._tool_error(
                        tool_call.name,
                        "unknown_tool",
                        citations,
                        document_tool_used,
                    )

                try:
                    arguments = json.loads(tool_call.arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be a JSON object.")
                    result = tool.execute(arguments)
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    logger.warning(
                        "event=agent_tool_arguments_invalid tool_name=%s "
                        "error_type=%s",
                        tool_call.name,
                        type(error).__name__,
                    )
                    return self._tool_error(
                        tool_call.name,
                        "invalid_arguments",
                        citations,
                        document_tool_used,
                    )
                except Exception as error:
                    logger.exception(
                        "event=agent_tool_failed tool_name=%s error_type=%s",
                        tool_call.name,
                        type(error).__name__,
                    )
                    raise AgentDependencyError(
                        self.DEPENDENCY_ERROR_MESSAGE
                    ) from error

                document_tool_used = (
                    document_tool_used or tool_call.name == "document_search"
                )
                citations.extend(result.citations)
                logger.info(
                    "event=agent_tool_completed tool_name=%s citation_count=%d",
                    tool_call.name,
                    len(result.citations),
                )
                next_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call.call_id,
                        "output": json.dumps(result.to_model_output()),
                    }
                )

            input_items = next_input

        raise AssertionError("Agent loop exited unexpectedly.")

    def _tool_error(
        self,
        tool_name: str,
        reason: str,
        citations: list[AgentCitation],
        document_tool_used: bool,
    ) -> AgentResponse:
        logger.warning(
            "event=agent_tool_rejected tool_name=%s reason=%s",
            tool_name,
            reason,
        )
        return AgentResponse(
            answer=self.TOOL_ERROR_MESSAGE,
            status=AgentStatus.TOOL_ERROR,
            citations=tuple(citations),
            document_tool_used=document_tool_used,
        )
