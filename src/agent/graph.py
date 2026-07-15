import json
import logging
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

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
    ToolExecutionResult,
    ToolProvenance,
)
from src.agent.tools import ToolRegistry

logger = logging.getLogger(__name__)


class AgentServiceError(RuntimeError):
    pass


class AgentDependencyError(AgentServiceError):
    pass


class AgentModel(Protocol):
    def respond(
        self,
        *,
        instructions: str,
        context: AgentModelInput,
        tools: tuple[ToolDefinition, ...],
    ) -> AgentModelResponse: ...


class AgentTurnState(TypedDict, total=False):
    question: str
    normalized_question: str
    history_items: list[dict[str, Any]]
    turn_items: list[dict[str, Any]]
    user_message: ConversationMessage
    model_response: AgentModelResponse
    pending_tool_calls: tuple[Any, ...]
    tool_output_items: list[dict[str, Any]]
    citations: list[AgentCitation]
    tool_sources: list[str]
    used_provenance: list[ToolProvenance]
    executed_call_fingerprints: list[str]
    natural_language_sql_executed: bool
    iteration: int
    route: str
    status: AgentStatus
    final_answer: str
    failure_category: str
    response: AgentResponse
    should_commit: bool


class AgentTurnGraph:
    """One-turn LangGraph workflow; session memory remains externally owned."""

    INVALID_QUESTION_MESSAGE = "Please enter a question."
    TOOL_ERROR_MESSAGE = "The agent could not safely execute the selected tool."
    MAX_ITERATIONS_MESSAGE = (
        "The agent stopped because it reached its tool-call safety limit."
    )
    REPEATED_TOOL_CALL_MESSAGE = (
        "The agent stopped because it repeated an equivalent tool request."
    )
    MALFORMED_MODEL_MESSAGE = "The agent returned an unsupported response."
    DEPENDENCY_ERROR_MESSAGE = "The agent could not complete the request."

    def __init__(
        self,
        *,
        model: AgentModel,
        tool_registry: ToolRegistry,
        memory: ConversationMemory,
        instructions: str,
        max_tool_iterations: int,
    ) -> None:
        self.model = model
        self.tool_registry = tool_registry
        self.tool_definitions = tool_registry.definitions
        self.memory = memory
        self.instructions = instructions
        self.max_tool_iterations = max_tool_iterations
        self._dependency_error: Exception | None = None
        self.compiled = self._build().compile()

    def run(self, question: str) -> AgentResponse:
        memory_state = self.memory.get_state()
        self._dependency_error = None
        initial: AgentTurnState = {
            "question": question,
            "history_items": list(memory_state.to_model_items()),
            "turn_items": [],
            "citations": [],
            "tool_sources": [],
            "used_provenance": [],
            "executed_call_fingerprints": [],
            "natural_language_sql_executed": False,
            "iteration": 0,
            "should_commit": False,
        }
        logger.info(
            "event=agent_graph_started retained_turns=%d retained_messages=%d",
            len(memory_state.turns),
            memory_state.message_count,
        )
        try:
            result = self.compiled.invoke(initial)
        except AgentServiceError:
            raise
        except Exception as error:
            self._dependency_error = error
            logger.exception(
                "event=agent_graph_unexpected_failure error_type=%s",
                type(error).__name__,
            )
            raise AgentServiceError(self.DEPENDENCY_ERROR_MESSAGE) from error
        return result["response"]

    def _build(self) -> StateGraph:
        graph = StateGraph(AgentTurnState)
        graph.add_node("validate_input", self.validate_input)
        graph.add_node("call_model", self.call_model)
        graph.add_node("route_model_output", self.route_model_output)
        graph.add_node("execute_tools", self.execute_tools)
        graph.add_node("record_tool_results", self.record_tool_results)
        graph.add_node("check_iteration_limit", self.check_iteration_limit)
        graph.add_node("finalize_response", self.finalize_response)
        graph.add_node("finalize_limit_reached", self.finalize_limit_reached)
        graph.add_node("handle_dependency_failure", self.handle_dependency_failure)
        graph.add_node("commit_memory", self.commit_memory)
        graph.add_edge(START, "validate_input")
        graph.add_conditional_edges(
            "validate_input",
            lambda state: state["route"],
            {"model": "call_model", "finalize": "finalize_response"},
        )
        graph.add_edge("call_model", "route_model_output")
        graph.add_conditional_edges(
            "route_model_output",
            lambda state: state["route"],
            {
                "direct": "finalize_response",
                "tools": "execute_tools",
                "limit": "finalize_limit_reached",
                "dependency": "handle_dependency_failure",
                "malformed": "finalize_response",
            },
        )
        graph.add_conditional_edges(
            "execute_tools",
            lambda state: state["route"],
            {
                "record": "record_tool_results",
                "finalize": "finalize_response",
                "dependency": "handle_dependency_failure",
            },
        )
        graph.add_edge("record_tool_results", "check_iteration_limit")
        graph.add_edge("check_iteration_limit", "call_model")
        graph.add_conditional_edges(
            "finalize_response",
            lambda state: "commit" if state.get("should_commit") else "end",
            {"commit": "commit_memory", "end": END},
        )
        graph.add_edge("finalize_limit_reached", END)
        graph.add_edge("commit_memory", END)
        return graph

    @staticmethod
    def _entered(node: str, state: AgentTurnState) -> None:
        logger.info(
            "event=agent_graph_node node=%s iteration=%d",
            node,
            state.get("iteration", 0),
        )

    def validate_input(self, state: AgentTurnState) -> dict[str, Any]:
        self._entered("validate_input", state)
        normalized = state["question"].strip()
        if not normalized:
            return {
                "route": "finalize",
                "status": AgentStatus.INVALID_QUESTION,
                "final_answer": self.INVALID_QUESTION_MESSAGE,
                "failure_category": "invalid_input",
            }
        user_message = ConversationMessage(ConversationRole.USER, normalized)
        return {
            "normalized_question": normalized,
            "user_message": user_message,
            "turn_items": [user_message.to_model_item()],
            "route": "model",
        }

    def call_model(self, state: AgentTurnState) -> dict[str, Any]:
        self._entered("call_model", state)
        try:
            response = self.model.respond(
                instructions=self.instructions,
                context=AgentModelInput(
                    items=tuple(state["history_items"] + state["turn_items"])
                ),
                tools=self.tool_definitions,
            )
        except Exception as error:
            logger.exception(
                "event=agent_graph_model_failed error_type=%s iteration=%d",
                type(error).__name__,
                state["iteration"],
            )
            return {
                "failure_category": "model_dependency_failure",
                "route": "dependency",
            }
        return {
            "model_response": response,
            "turn_items": state["turn_items"] + list(response.continuation_items),
        }

    def route_model_output(self, state: AgentTurnState) -> dict[str, Any]:
        self._entered("route_model_output", state)
        if state.get("failure_category") == "model_dependency_failure":
            return {"route": "dependency"}
        response = state.get("model_response")
        if response is None:
            return self._malformed("missing_model_response")
        if response.tool_calls:
            if state["iteration"] >= self.max_tool_iterations:
                return {"route": "limit"}
            return {"pending_tool_calls": response.tool_calls, "route": "tools"}
        if not response.output_text.strip():
            return self._malformed("empty_model_output")
        return {
            "route": "direct",
            "final_answer": response.output_text,
            "should_commit": True,
        }

    def _malformed(self, category: str) -> dict[str, Any]:
        return {
            "route": "malformed",
            "status": AgentStatus.TOOL_ERROR,
            "final_answer": self.MALFORMED_MODEL_MESSAGE,
            "failure_category": category,
        }

    def execute_tools(self, state: AgentTurnState) -> dict[str, Any]:
        self._entered("execute_tools", state)
        outputs: list[dict[str, Any]] = []
        citations = list(state["citations"])
        sources = list(state["tool_sources"])
        provenances = list(state["used_provenance"])
        fingerprints = list(state["executed_call_fingerprints"])
        natural_executed = state["natural_language_sql_executed"]
        safe_failure_category = state.get("failure_category", "")
        for call in state["pending_tool_calls"]:
            tool = self.tool_registry.get(call.name)
            if tool is None:
                return self._tool_failure(state, "unknown_tool", citations, sources, provenances)
            try:
                arguments = json.loads(call.arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be a JSON object.")
            except (json.JSONDecodeError, TypeError, ValueError):
                return self._tool_failure(state, "invalid_arguments", citations, sources, provenances)
            normalized = {
                key: " ".join(value.split()) if isinstance(value, str) else value
                for key, value in arguments.items()
            }
            fingerprint = f"{call.name}:{json.dumps(normalized, sort_keys=True, separators=(',', ':'))}"
            is_natural = (
                call.name == "sql_query"
                and arguments.get("operation") == "natural_language_query"
            )
            if fingerprint in fingerprints or (is_natural and natural_executed):
                return self._tool_failure(state, "repeated_tool_call", citations, sources, provenances,
                                          self.REPEATED_TOOL_CALL_MESSAGE)
            try:
                result: ToolExecutionResult = tool.execute(arguments)
            except (TypeError, ValueError):
                return self._tool_failure(state, "invalid_arguments", citations, sources, provenances)
            except Exception as error:
                self._dependency_error = error
                logger.exception(
                    "event=agent_graph_tool_failed tool_name=%s error_type=%s",
                    call.name,
                    type(error).__name__,
                )
                return {
                    "failure_category": "tool_dependency_failure",
                    "route": "dependency",
                }
            fingerprints.append(fingerprint)
            natural_executed = natural_executed or is_natural
            if result.source not in sources:
                sources.append(result.source)
            if result.provenance not in provenances:
                provenances.append(result.provenance)
            citations.extend(result.citations)
            if result.failure_category:
                safe_failure_category = result.failure_category
            outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result.to_model_output()),
                }
            )
            logger.info(
                "event=agent_graph_tool_completed tool_name=%s iteration=%d status=%s",
                call.name,
                state["iteration"] + 1,
                result.status.value,
            )
        return {
            "tool_output_items": outputs,
            "citations": citations,
            "tool_sources": sources,
            "used_provenance": provenances,
            "executed_call_fingerprints": fingerprints,
            "natural_language_sql_executed": natural_executed,
            "failure_category": safe_failure_category,
            "route": "record",
        }

    def _tool_failure(
        self,
        state: AgentTurnState,
        category: str,
        citations: list[AgentCitation],
        sources: list[str],
        provenances: list[ToolProvenance],
        answer: str | None = None,
    ) -> dict[str, Any]:
        logger.warning(
            "event=agent_graph_tool_rejected category=%s iteration=%d",
            category,
            state["iteration"],
        )
        return {
            "route": "finalize",
            "status": AgentStatus.TOOL_ERROR,
            "final_answer": answer or self.TOOL_ERROR_MESSAGE,
            "failure_category": category,
            "citations": citations,
            "tool_sources": sources,
            "used_provenance": provenances,
            "should_commit": False,
        }

    def record_tool_results(self, state: AgentTurnState) -> dict[str, Any]:
        self._entered("record_tool_results", state)
        return {
            "turn_items": state["turn_items"] + state["tool_output_items"],
            "iteration": state["iteration"] + 1,
        }

    def check_iteration_limit(self, state: AgentTurnState) -> dict[str, Any]:
        self._entered("check_iteration_limit", state)
        return {}

    def finalize_limit_reached(self, state: AgentTurnState) -> dict[str, Any]:
        self._entered("finalize_limit_reached", state)
        return {
            "response": self._response(
                state,
                self.MAX_ITERATIONS_MESSAGE,
                AgentStatus.MAX_ITERATIONS,
            ),
            "failure_category": "maximum_iterations",
        }

    def finalize_response(self, state: AgentTurnState) -> dict[str, Any]:
        self._entered("finalize_response", state)
        status = state.get("status")
        if status is None:
            provenance = set(state["used_provenance"])
            if ToolProvenance.DOCUMENT in provenance:
                status = AgentStatus.DOCUMENT_ANSWER
            elif ToolProvenance.STRUCTURED_ORGANIZATION_DATA in provenance:
                status = AgentStatus.ORGANIZATION_ANSWER
            elif ToolProvenance.STRUCTURED_SQL_DATA in provenance:
                status = AgentStatus.SQL_ANSWER
            else:
                status = AgentStatus.DIRECT_ANSWER
        return {
            "response": self._response(state, state["final_answer"], status),
        }

    @staticmethod
    def _response(
        state: AgentTurnState, answer: str, status: AgentStatus
    ) -> AgentResponse:
        provenance = set(state["used_provenance"])
        return AgentResponse(
            answer=answer,
            status=status,
            citations=tuple(state["citations"]),
            document_tool_used=ToolProvenance.DOCUMENT in provenance,
            tool_sources=tuple(state["tool_sources"]),
        )

    def commit_memory(self, state: AgentTurnState) -> dict[str, Any]:
        self._entered("commit_memory", state)
        response = state["response"]
        turn_items = list(state["turn_items"])
        model_response = state["model_response"]
        if not model_response.continuation_items:
            turn_items.append({"role": "assistant", "content": response.answer})
        self.memory.append_turn(
            ConversationTurn(
                user_message=state["user_message"],
                assistant_message=ConversationMessage(
                    ConversationRole.ASSISTANT, response.answer
                ),
                model_items=tuple(turn_items),
                status=response.status,
                citations=response.citations,
            )
        )
        logger.info(
            "event=agent_graph_completed status=%s tool_count=%d iteration=%d",
            response.status.value,
            len(response.tool_sources),
            state["iteration"],
        )
        return {}

    def handle_dependency_failure(self, state: AgentTurnState) -> dict[str, Any]:
        self._entered("handle_dependency_failure", state)
        raise AgentDependencyError(self.DEPENDENCY_ERROR_MESSAGE) from self._dependency_error
