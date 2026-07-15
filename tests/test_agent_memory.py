from src.agent.memory import (
    ConversationMessage,
    ConversationRole,
    ConversationTurn,
    InMemoryConversationMemory,
)
from src.agent.models import AgentStatus


def _turn(label: str, model_items: tuple[dict, ...]) -> ConversationTurn:
    return ConversationTurn(
        user_message=ConversationMessage(
            role=ConversationRole.USER,
            content=f"user-{label}",
        ),
        assistant_message=ConversationMessage(
            role=ConversationRole.ASSISTANT,
            content=f"assistant-{label}",
        ),
        model_items=model_items,
        status=AgentStatus.DIRECT_ANSWER,
    )


def test_history_limit_removes_oldest_complete_turn() -> None:
    memory = InMemoryConversationMemory(max_turns=1)
    first_items = (
        {"role": "user", "content": "first"},
        {"type": "function_call", "call_id": "call-1"},
        {"type": "function_call_output", "call_id": "call-1"},
        {"role": "assistant", "content": "first answer"},
    )
    second_items = (
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "second answer"},
    )

    memory.append_turn(_turn("first", first_items))
    memory.append_turn(_turn("second", second_items))

    state = memory.get_state()
    assert len(state.turns) == 1
    assert state.turns[0].model_items == second_items
    assert state.to_model_items() == second_items


def test_clear_removes_session_history() -> None:
    memory = InMemoryConversationMemory(max_turns=2)
    memory.append_turn(
        _turn(
            "one",
            (
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "answer"},
            ),
        )
    )

    memory.clear()

    assert memory.get_state().turns == ()


def test_separate_memory_instances_do_not_share_history() -> None:
    first = InMemoryConversationMemory(max_turns=2)
    second = InMemoryConversationMemory(max_turns=2)
    first.append_turn(
        _turn(
            "one",
            (
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "answer"},
            ),
        )
    )

    assert len(first.get_state().turns) == 1
    assert second.get_state().turns == ()
