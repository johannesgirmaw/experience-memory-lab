"""Thin OpenAI-compatible chat + embeddings client.

Not used by Milestone 1 default runs. Kept so the port is real, not aspirational.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from experience_memory_lab.llm.base import LLMResponse, Message


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        embed_model: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.model_id = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.embed_model = embed_model or os.environ.get(
            "OPENAI_EMBED_MODEL", "text-embedding-3-small"
        )
        self.timeout_s = timeout_s

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        seed: int | None = None,
    ) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAICompatibleClient")
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        data = self._post("/chat/completions", payload)
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return LLMResponse(
            content=choice["message"]["content"] or "",
            model=data.get("model", self.model_id),
            finish_reason=choice.get("finish_reason") or "stop",
            usage={
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
            },
            raw=data,
            request_id=data.get("id"),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAICompatibleClient")
        data = self._post(
            "/embeddings",
            {"model": self.embed_model, "input": texts},
        )
        sorted_rows = sorted(data["data"], key=lambda row: row["index"])
        return [row["embedding"] for row in sorted_rows]

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(f"{self.base_url}{path}", headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()
