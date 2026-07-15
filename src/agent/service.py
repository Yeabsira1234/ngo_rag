import json
import logging
from typing import Protocol

from src.agent.memory import (
    ConversationMemory,
    ConversationMessage,
    ConversationRole,
    ConversationTurn,
)
from src.agent.models import (
    AgentCitation,
    AgentModelInput,
    AgentModelResponse,
    AgentResponse,
    AgentStatus,
    ToolDefinition,
    ToolProvenance,
)
from src.agent.tools import AgentTool, ToolRegistry


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
        context: AgentModelInput,
        tools: tuple[ToolDefinition, ...],
    ) -> AgentModelResponse: ...


class AgentService:
    """Coordinate model-directed tool calls separately from direct RAG."""

    INVALID_QUESTION_MESSAGE = "Please enter a question."
    TOOL_ERROR_MESSAGE = "The agent could not safely execute the selected tool."
    MAX_ITERATIONS_MESSAGE = (
        "The agent stopped because it reached its tool-call safety limit."
    )
    REPEATED_TOOL_CALL_MESSAGE = (
        "The agent stopped because it repeated an equivalent tool request."
    )
    DEPENDENCY_ERROR_MESSAGE = "The agent could not complete the request."
    INSTRUCTIONS = (
        "You are a helpful assistant with three tools. Use document_search for "
        "questions about contents, policies, procedures, or facts in the indexed "
        "document. Use organization_info for structured facts in the fictional "
        "sample organization directory, such as its name, support hours, contact "
        "email, location, or service categories. Use sql_query for structured "
        "questions about offices, programs, staff, clients, cases, or service "
        "events. Always use sql_query with natural_language_query for comparisons, "
        "rankings, grouped counts, 'most common', 'most', or other aggregate questions "
        "that are not answered by one exact predefined operation. Pass the complete "
        "user question and do not compose multiple predefined SQL calls. Predefined "
        "operations remain available only when one operation fits exactly. Answer "
        "directly only when no tool is "
        "necessary. Do not invent facts."
    )

    def __init__(
        self,
        model: AgentModel,
        tools: tuple[AgentTool, ...],
        memory: ConversationMemory,
        max_tool_iterations: int = 2,
    ) -> None:
        if max_tool_iterations <= 0:
            raise ValueError("max_tool_iterations must be greater than zero.")
        self.model = model
        self.memory = memory
        self.tool_registry = ToolRegistry(tools)
        self.tool_definitions = self.tool_registry.definitions
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

        state = self.memory.get_state()
        user_message = ConversationMessage(
            role=ConversationRole.USER,
            content=normalized_question,
        )
        history_items = list(state.to_model_items())
        turn_items = [user_message.to_model_item()]
        citations: list[AgentCitation] = []
        tool_sources: list[str] = []
        used_provenance: set[ToolProvenance] = set()
        executed_tool_calls: set[tuple[str, str]] = set()
        natural_language_sql_executed = False
        logger.info(
            "event=agent_turn_started retained_turns=%d retained_messages=%d",
            len(state.turns),
            state.message_count,
        )

        for iteration in range(self.max_tool_iterations + 1):
            logger.info("event=agent_model_invocation iteration=%d", iteration)
            try:
                model_response = self.model.respond(
                    instructions=self.INSTRUCTIONS,
                    context=AgentModelInput(
                        items=tuple(history_items + turn_items)
                    ),
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
                if ToolProvenance.DOCUMENT in used_provenance:
                    status = AgentStatus.DOCUMENT_ANSWER
                elif (
                    ToolProvenance.STRUCTURED_ORGANIZATION_DATA
                    in used_provenance
                ):
                    status = AgentStatus.ORGANIZATION_ANSWER
                elif ToolProvenance.STRUCTURED_SQL_DATA in used_provenance:
                    status = AgentStatus.SQL_ANSWER
                else:
                    status = AgentStatus.DIRECT_ANSWER
                response = AgentResponse(
                    answer=model_response.output_text,
                    status=status,
                    citations=tuple(citations),
                    document_tool_used=(
                        ToolProvenance.DOCUMENT in used_provenance
                    ),
                    tool_sources=tuple(tool_sources),
                )
                turn_items.extend(model_response.continuation_items)
                if not model_response.continuation_items:
                    turn_items.append(
                        {
                            "role": "assistant",
                            "content": response.answer,
                        }
                    )
                self.memory.append_turn(
                    ConversationTurn(
                        user_message=user_message,
                        assistant_message=ConversationMessage(
                            role=ConversationRole.ASSISTANT,
                            content=response.answer,
                        ),
                        model_items=tuple(turn_items),
                        status=response.status,
                        citations=response.citations,
                    )
                )
                logger.info(
                    "event=agent_turn_stored model_item_count=%d "
                    "citation_count=%d",
                    len(turn_items),
                    len(response.citations),
                )
                return response

            if iteration >= self.max_tool_iterations:
                logger.warning(
                    "event=agent_max_iterations_reached max_iterations=%d",
                    self.max_tool_iterations,
                )
                return AgentResponse(
                    answer=self.MAX_ITERATIONS_MESSAGE,
                    status=AgentStatus.MAX_ITERATIONS,
                    citations=tuple(citations),
                    document_tool_used=(
                        ToolProvenance.DOCUMENT in used_provenance
                    ),
                    tool_sources=tuple(tool_sources),
                )

            turn_items.extend(model_response.continuation_items)
            for tool_call in model_response.tool_calls:
                logger.info(
                    "event=agent_tool_selected tool_name=%s iteration=%d",
                    tool_call.name,
                    iteration + 1,
                )
                tool = self.tool_registry.get(tool_call.name)
                if tool is None:
                    return self._tool_error(
                        tool_call.name,
                        "unknown_tool",
                        citations,
                        tool_sources,
                        used_provenance,
                    )

                try:
                    arguments = json.loads(tool_call.arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be a JSON object.")
                    call_fingerprint = (
                        tool_call.name,
                        json.dumps(
                            {
                                key: " ".join(value.split())
                                if isinstance(value, str)
                                else value
                                for key, value in arguments.items()
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    )
                    if call_fingerprint in executed_tool_calls:
                        logger.warning(
                            "event=agent_repeated_tool_call tool_name=%s",
                            tool_call.name,
                        )
                        return AgentResponse(
                            answer=self.REPEATED_TOOL_CALL_MESSAGE,
                            status=AgentStatus.TOOL_ERROR,
                            citations=tuple(citations),
                            document_tool_used=(
                                ToolProvenance.DOCUMENT in used_provenance
                            ),
                            tool_sources=tuple(tool_sources),
                        )
                    is_natural_sql = (
                        tool_call.name == "sql_query"
                        and arguments.get("operation") == "natural_language_query"
                    )
                    if is_natural_sql and natural_language_sql_executed:
                        logger.warning("event=agent_repeated_natural_sql_call")
                        return AgentResponse(
                            answer=self.REPEATED_TOOL_CALL_MESSAGE,
                            status=AgentStatus.TOOL_ERROR,
                            citations=tuple(citations),
                            document_tool_used=False,
                            tool_sources=tuple(tool_sources),
                        )
                    result = tool.execute(arguments)
                    executed_tool_calls.add(call_fingerprint)
                    natural_language_sql_executed = (
                        natural_language_sql_executed or is_natural_sql
                    )
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
                        tool_sources,
                        used_provenance,
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

                used_provenance.add(result.provenance)
                if result.source not in tool_sources:
                    tool_sources.append(result.source)
                citations.extend(result.citations)
                logger.info(
                    "event=agent_tool_completed tool_name=%s result_status=%s "
                    "result_category=%s citation_count=%d",
                    tool_call.name,
                    result.status.value,
                    result.category or "none",
                    len(result.citations),
                )
                turn_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call.call_id,
                        "output": json.dumps(result.to_model_output()),
                    }
                )

        raise AssertionError("Agent loop exited unexpectedly.")

    def clear_memory(self) -> None:
        """Clear retained conversation history for the current session."""
        previous_turn_count = len(self.memory.get_state().turns)
        self.memory.clear()
        logger.info(
            "event=agent_memory_cleared previous_turn_count=%d",
            previous_turn_count,
        )

    def _tool_error(
        self,
        tool_name: str,
        reason: str,
        citations: list[AgentCitation],
        tool_sources: list[str],
        used_provenance: set[ToolProvenance],
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
            document_tool_used=(ToolProvenance.DOCUMENT in used_provenance),
            tool_sources=tuple(tool_sources),
        )
