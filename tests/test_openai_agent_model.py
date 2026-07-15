from types import SimpleNamespace
from unittest.mock import Mock

from src.agent.models import AgentModelInput, ToolDefinition
from src.agent.openai_model import OpenAIAgentModel


def test_openai_agent_model_normalizes_function_calls() -> None:
    output_item = SimpleNamespace(
        type="function_call",
        call_id="call-1",
        name="document_search",
        arguments='{"question":"Office hours?"}',
        model_dump=Mock(
            return_value={
                "type": "function_call",
                "call_id": "call-1",
                "name": "document_search",
                "arguments": '{"question":"Office hours?"}',
            }
        ),
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
        context=AgentModelInput(
            items=({"role": "user", "content": "Office hours?"},)
        ),
        tools=(definition,),
    )

    assert result.tool_calls[0].name == "document_search"
    assert result.tool_calls[0].call_id == "call-1"
    assert result.continuation_items[0]["type"] == "function_call"
    sent_tool = client.create.call_args.kwargs["tools"][0]
    assert sent_tool["type"] == "function"
    assert sent_tool["strict"] is True
    assert client.create.call_args.kwargs["store"] is False
    assert client.create.call_args.kwargs["include"] == [
        "reasoning.encrypted_content"
    ]
