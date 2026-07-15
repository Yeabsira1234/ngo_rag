from types import SimpleNamespace
from unittest.mock import Mock

from src.agent.models import ToolDefinition
from src.agent.openai_model import OpenAIAgentModel


def test_openai_agent_model_normalizes_function_calls() -> None:
    output_item = SimpleNamespace(
        type="function_call",
        call_id="call-1",
        name="document_search",
        arguments='{"question":"Office hours?"}',
    )
    response = SimpleNamespace(
        output=[output_item],
        output_text="",
    )
    client = Mock()
    client.create.return_value = response
    definition = ToolDefinition(
        name="document_search",
        description="Search documents.",
        parameters={
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
            "additionalProperties": False,
        },
    )
    model = OpenAIAgentModel(client=client, model="test-model")

    result = model.respond(
        instructions="Choose a tool when needed.",
        input_items="Office hours?",
        tools=(definition,),
    )

    assert result.tool_calls[0].name == "document_search"
    assert result.tool_calls[0].call_id == "call-1"
    assert result.continuation_items == (output_item,)
    sent_tool = client.create.call_args.kwargs["tools"][0]
    assert sent_tool["type"] == "function"
    assert sent_tool["strict"] is True
