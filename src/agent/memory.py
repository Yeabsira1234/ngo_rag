from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from src.agent.models import AgentCitation, AgentStatus


class ConversationRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    """A user-visible message retained in an agent conversation."""

    role: ConversationRole
    content: str

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("Conversation message content cannot be empty.")

    def to_model_item(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """One complete user/assistant exchange and its ordered model context."""

    user_message: ConversationMessage
    assistant_message: ConversationMessage
    model_items: tuple[dict[str, Any], ...]
    status: AgentStatus
    citations: tuple[AgentCitation, ...] = ()

    def __post_init__(self) -> None:
        if self.user_message.role is not ConversationRole.USER:
            raise ValueError("A conversation turn must start with a user message.")
        if self.assistant_message.role is not ConversationRole.ASSISTANT:
            raise ValueError(
                "A conversation turn must end with an assistant message."
            )
        if not self.model_items:
            raise ValueError("A conversation turn must contain model input items.")


@dataclass(frozen=True, slots=True)
class ConversationState:
    """Immutable snapshot of complete retained conversation turns."""

    turns: tuple[ConversationTurn, ...]

    @property
    def message_count(self) -> int:
        return len(self.turns) * 2

    def to_model_items(self) -> tuple[dict[str, Any], ...]:
        return tuple(item for turn in self.turns for item in turn.model_items)


class ConversationMemory(Protocol):
    def get_state(self) -> ConversationState: ...

    def append_turn(self, turn: ConversationTurn) -> None: ...

    def clear(self) -> None: ...


class InMemoryConversationMemory:
    """Process-local conversation memory trimmed by complete turns."""

    def __init__(self, max_turns: int = 10) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than zero.")
        self.max_turns = max_turns
        self._turns: list[ConversationTurn] = []

    def get_state(self) -> ConversationState:
        return ConversationState(turns=tuple(self._turns))

    def append_turn(self, turn: ConversationTurn) -> None:
        self._turns.append(turn)
        overflow = len(self._turns) - self.max_turns
        if overflow > 0:
            del self._turns[:overflow]

    def clear(self) -> None:
        self._turns.clear()
