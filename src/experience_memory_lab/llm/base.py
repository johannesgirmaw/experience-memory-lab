"""LLM port. All providers implement this; nothing else talks to HTTP."""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str


class LLMResponse(BaseModel):
    content: str
    model: str
    finish_reason: str = "stop"
    usage: dict[str, int] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None


@runtime_checkable
class LLM(Protocol):
    model_id: str

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        seed: int | None = None,
    ) -> LLMResponse: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...
