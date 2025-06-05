from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Message:
    role: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextManager:
    """Tracks conversation history and tool results."""

    def __init__(self) -> None:
        self.history: List[Message] = []

    def add_user_message(self, content: str) -> None:
        self.history.append(Message(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        self.history.append(Message(role="assistant", content=content))

    def to_prompt(self) -> str:
        return "\n".join(f"{m.role}: {m.content}" for m in self.history)
