import logging

from src.agent.graph import (
    AgentDependencyError,
    AgentModel,
    AgentServiceError,
    AgentTurnGraph,
)
from src.agent.memory import ConversationMemory
from src.agent.models import AgentResponse
from src.agent.tools import AgentTool, ToolRegistry

logger = logging.getLogger(__name__)


class AgentService:
    """Stable application interface backed by one-turn LangGraph execution."""

    INVALID_QUESTION_MESSAGE = AgentTurnGraph.INVALID_QUESTION_MESSAGE
    TOOL_ERROR_MESSAGE = AgentTurnGraph.TOOL_ERROR_MESSAGE
    MAX_ITERATIONS_MESSAGE = AgentTurnGraph.MAX_ITERATIONS_MESSAGE
    MAX_TOOL_CALLS_MESSAGE = AgentTurnGraph.MAX_TOOL_CALLS_MESSAGE
    REPEATED_TOOL_CALL_MESSAGE = AgentTurnGraph.REPEATED_TOOL_CALL_MESSAGE
    DEPENDENCY_ERROR_MESSAGE = AgentTurnGraph.DEPENDENCY_ERROR_MESSAGE
    INSTRUCTIONS = (
        "You are a helpful assistant with three tools. Use document_search for "
        "questions about contents, policies, procedures, or facts in the indexed "
        "document. Use organization_info for structured facts in the fictional "
        "sample organization directory, such as its name, support hours, contact "
        "email, location, or service categories. Use sql_query for structured "
        "questions about offices, programs, staff, clients, cases, or service "
        "events. Always use sql_query with natural_language_query for comparisons, "
        "rankings, grouped counts, 'most common', 'most', or other aggregate questions "
        "that are not answered by one exact predefined operation. Temporal requests "
        "for recent, latest, or 'most recent' service events must use the predefined "
        "recent_service_events operation, not natural_language_query. Pass the complete "
        "user question and do not compose multiple predefined SQL calls. Predefined "
        "operations remain available only when one operation fits exactly. Answer "
        "directly only when no tool is necessary. When a question genuinely asks for "
        "facts from multiple sources, select all required tools together in the order "
        "their evidence should be gathered; do not add unrelated tools. The application "
        "will validate the plan and synthesize the labeled results. Do not invent facts."
    )

    def __init__(
        self,
        model: AgentModel,
        tools: tuple[AgentTool, ...],
        memory: ConversationMemory,
        max_tool_iterations: int = 2,
        max_tool_calls_per_turn: int = 3,
    ) -> None:
        if max_tool_iterations <= 0:
            raise ValueError("max_tool_iterations must be greater than zero.")
        if max_tool_calls_per_turn <= 0:
            raise ValueError("max_tool_calls_per_turn must be greater than zero.")
        self.model = model
        self.memory = memory
        self.tool_registry = ToolRegistry(tools)
        self.tool_definitions = self.tool_registry.definitions
        self.max_tool_iterations = max_tool_iterations
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self.graph = AgentTurnGraph(
            model=model,
            tool_registry=self.tool_registry,
            memory=memory,
            instructions=self.INSTRUCTIONS,
            max_tool_iterations=max_tool_iterations,
            max_tool_calls_per_turn=max_tool_calls_per_turn,
        )

    def answer(self, question: str) -> AgentResponse:
        return self.graph.run(question)

    def clear_memory(self) -> None:
        previous_turn_count = len(self.memory.get_state().turns)
        self.memory.clear()
        logger.info(
            "event=agent_memory_cleared previous_turn_count=%d",
            previous_turn_count,
        )


__all__ = ["AgentDependencyError", "AgentService", "AgentServiceError"]
