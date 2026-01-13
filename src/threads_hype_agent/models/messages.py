"""Message and conversation schemas."""

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in a conversation."""

    role: Literal["system", "user", "assistant"] = Field(
        description="The role of the message sender"
    )
    content: str = Field(description="The content of the message")

    def to_openai_format(self) -> dict[str, str]:
        """Convert to OpenAI-compatible message format."""
        return {"role": self.role, "content": self.content}


class Conversation(BaseModel):
    """A conversation consisting of multiple messages."""

    messages: list[Message] = Field(default_factory=list)

    def add_message(self, role: Literal["system", "user", "assistant"], content: str) -> None:
        """Add a message to the conversation."""
        self.messages.append(Message(role=role, content=content))

    def to_openai_format(self) -> list[dict[str, str]]:
        """Convert all messages to OpenAI-compatible format."""
        return [msg.to_openai_format() for msg in self.messages]

    def get_last_user_message(self) -> str | None:
        """Get the content of the last user message."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None
