from typing import Any, Protocol

from src.agent.models import (
    AgentModelInput,
    AgentModelResponse,
    ToolCall,
    ToolDefinition,
)


class ResponsesClient(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class OpenAIAgentModel:
    """Translate between the agent model contract and OpenAI Responses."""

    def __init__(self, client: ResponsesClient, model: str) -> None:
        if not model.strip():
            raise ValueError("model cannot be empty.")
        self.client = client
        self.model = model

    def respond(
        self,
        *,
        instructions: str,
        context: AgentModelInput,
        tools: tuple[ToolDefinition, ...],
    ) -> AgentModelResponse:
        response = self.client.create(
            model=self.model,
            instructions=instructions,
            input=list(context.items),
            tools=[tool.to_openai() for tool in tools],
            store=False,
            include=["reasoning.encrypted_content"],
        )

        tool_calls = tuple(
            ToolCall(
                call_id=item.call_id,
                name=item.name,
                arguments=item.arguments,
            )
            for item in response.output
            if item.type == "function_call"
        )
        continuation_items = tuple(
            item.model_dump(exclude_none=True) for item in response.output
        )
        return AgentModelResponse(
            output_text=response.output_text or "",
            tool_calls=tool_calls,
            continuation_items=continuation_items,
        )
