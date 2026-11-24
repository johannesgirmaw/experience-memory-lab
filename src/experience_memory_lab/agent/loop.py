"""Shared observe–retrieve–act–write loop. No condition branching."""

from __future__ import annotations

import json
import re
from typing import Any

from experience_memory_lab.agent.prompts import ACT_SYSTEM_PROMPT, build_user_prompt
from experience_memory_lab.env.base import Environment
from experience_memory_lab.llm.base import LLM, Message
from experience_memory_lab.memory.base import MemorySystem
from experience_memory_lab.runlog.jsonl import JsonlRunLogger, estimate_tokens
from experience_memory_lab.types.contracts import (
    Action,
    EpisodeOutcome,
    MemoryBudget,
    StepRecord,
    Trajectory,
)


def parse_action(raw: str) -> Action:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return Action(type="done", args={}, raw=raw)
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return Action(type="done", args={}, raw=raw)
    if not isinstance(data, dict):
        return Action(type="done", args={}, raw=raw)
    atype = str(data.get("type", "done"))
    args = data.get("args") if isinstance(data.get("args"), dict) else {}
    # Allow flat {"type":"pick_key","color":"red"} as a convenience.
    if not args:
        args = {k: v for k, v in data.items() if k != "type"}
    return Action(type=atype, args=args, raw=raw)


class AgentLoop:
    def __init__(
        self,
        *,
        env: Environment,
        memory: MemorySystem,
        llm: LLM,
        logger: JsonlRunLogger,
        budget: MemoryBudget,
        temperature: float = 0.0,
        seed: int,
    ) -> None:
        self.env = env
        self.memory = memory
        self.llm = llm
        self.logger = logger
        self.budget = budget
        self.temperature = temperature
        self.seed = seed

    def run_episode(self, *, episode_index: int, episode_id: str) -> EpisodeOutcome:
        obs = self.env.reset(self.seed, episode_index=episode_index)
        self.logger.emit(
            "episode_start",
            episode_id=episode_id,
            step=0,
            phase=self.env.phase(),
            hidden_rule_id=self.env.hidden_rule_id(),
            goal=obs.goal,
        )

        trajectory = Trajectory(episode_id=episode_id)
        total_reward = 0.0
        terminated = False
        truncated = False
        last_info: dict[str, Any] = {}
        applied_ids: list[str] = []

        while not terminated and not truncated:
            retrieved = self.memory.retrieve(obs, obs.goal, self.budget.k)
            if retrieved.token_estimate > self.budget.memory_tokens:
                retrieved.truncated = True
                self.logger.emit(
                    "budget_fault",
                    episode_id=episode_id,
                    step=obs.step,
                    kind="memory_tokens",
                    token_estimate=retrieved.token_estimate,
                    budget=self.budget.memory_tokens,
                )
                # Enforce budget: truncate text rather than grant extra context.
                retrieved.text = retrieved.text[: self.budget.memory_tokens * 4]
                retrieved.token_estimate = estimate_tokens(retrieved.text)

            self.logger.emit(
                "retrieve",
                episode_id=episode_id,
                step=obs.step,
                record_ids=retrieved.record_ids,
                token_estimate=retrieved.token_estimate,
                truncated=retrieved.truncated,
                meta=retrieved.meta,
            )
            applied_ids = list(retrieved.record_ids)

            user_prompt = build_user_prompt(obs, retrieved)
            messages = [
                Message(role="system", content=ACT_SYSTEM_PROMPT),
                Message(role="user", content=user_prompt),
            ]
            response = self.llm.complete(
                messages,
                temperature=self.temperature,
                max_tokens=128,
                seed=self.seed,
            )
            self.logger.emit(
                "llm_call",
                episode_id=episode_id,
                step=obs.step,
                model=response.model,
                request_id=response.request_id,
                usage=response.usage,
                prompt_hash=_sha(user_prompt),
                completion=response.content,
            )

            action = parse_action(response.content)
            self.logger.emit(
                "action",
                episode_id=episode_id,
                step=obs.step,
                action=action.model_dump(),
            )

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            last_info = info
            self.logger.emit(
                "env_step",
                episode_id=episode_id,
                step=obs.step,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info={k: v for k, v in info.items() if k != "unlock_ok"},
                observation_visible=next_obs.visible,
            )

            trajectory.steps.append(
                StepRecord(
                    step=obs.step,
                    observation=obs,
                    action=action,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
            )
            obs = next_obs

        success = bool(last_info.get("success", False))
        outcome = EpisodeOutcome(
            success=success,
            total_reward=total_reward,
            n_steps=len(trajectory.steps),
            terminated=terminated,
            truncated=truncated,
        )

        written = self.memory.write(episode_id, trajectory, outcome)
        self.logger.emit(
            "write",
            episode_id=episode_id,
            step=obs.step,
            record_ids=written,
            outcome=outcome.model_dump(),
        )
        self.memory.notify_outcome(applied_ids, outcome)
        self.logger.emit(
            "outcome",
            episode_id=episode_id,
            step=obs.step,
            applied_record_ids=applied_ids,
            outcome=outcome.model_dump(),
        )
        self.logger.emit(
            "episode_end",
            episode_id=episode_id,
            step=obs.step,
            success=success,
            total_reward=total_reward,
        )
        return outcome


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
