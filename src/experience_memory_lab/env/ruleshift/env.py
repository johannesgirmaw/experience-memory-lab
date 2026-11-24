"""RuleShift-v0: Access family (Milestone 1 ships matching rule only).

Deterministic given (seed, episode_index, rule_id). Hidden rule_id is never
placed in Observation.visible or Observation.text.
"""

from __future__ import annotations

import hashlib
from typing import Any

from experience_memory_lab.env.ruleshift.rules import ACCESS_MATCH_V1, RULES_BY_ID, RuleCard
from experience_memory_lab.types.contracts import Action, Observation, PhaseId

COLORS = ("red", "blue", "green")
ENV_VERSION = "ruleshift-v0.1-access"
FORBIDDEN_OBS_KEYS = frozenset(
    {
        "rule_id",
        "hidden_rule_id",
        "rule",
        "unlock_ok",
        "rule_card",
    }
)


class RuleShiftEnv:
    """Minimal door/key world with a hidden color-lock rule."""

    def __init__(
        self,
        *,
        rule_id: str = ACCESS_MATCH_V1.rule_id,
        phase: PhaseId = "P1",
        max_steps: int = 6,
        context_cue: str | None = None,
    ) -> None:
        if rule_id not in RULES_BY_ID:
            raise ValueError(f"Unknown rule_id: {rule_id}")
        self._rule: RuleCard = RULES_BY_ID[rule_id]
        self._phase: PhaseId = phase
        self._max_steps = max_steps
        self._context_cue = context_cue
        self._seed = 0
        self._episode_index = 0
        self._step = 0
        self._door_color = "red"
        self._inventory: str | None = None
        self._door_open = False
        self._available_keys = list(COLORS)

    @property
    def env_version(self) -> str:
        return ENV_VERSION

    def reset(self, seed: int, *, episode_index: int = 0) -> Observation:
        self._seed = seed
        self._episode_index = episode_index
        self._step = 0
        self._inventory = None
        self._door_open = False
        self._door_color = self._pick_door_color(seed, episode_index)
        self._available_keys = list(COLORS)
        return self._observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        self._step += 1
        reward = -0.05
        terminated = False
        truncated = False
        info: dict[str, Any] = {
            "rule_id": self._rule.rule_id,
            "phase": self._phase,
            "action_type": action.type,
        }

        atype = action.type.strip().lower()
        if atype == "pick_key":
            color = str(action.args.get("color", "")).lower()
            if color in self._available_keys:
                self._inventory = color
                reward = 0.0
                info["picked"] = color
            else:
                reward = -0.2
                info["error"] = "unknown_or_unavailable_key"
        elif atype == "unlock":
            door = str(action.args.get("door", self._door_color)).lower()
            ok = door == self._door_color and self._rule.unlock_ok(self._inventory, door)
            if ok:
                self._door_open = True
                reward = 0.5
                info["unlocked"] = True
            else:
                reward = -0.3
                info["unlocked"] = False
        elif atype == "done":
            terminated = True
            if self._door_open:
                reward = 1.0
                info["success"] = True
            else:
                reward = -1.0
                info["success"] = False
        else:
            reward = -0.2
            info["error"] = "unknown_action"

        if self._step >= self._max_steps and not terminated:
            truncated = True
            info["success"] = bool(self._door_open)

        obs = self._observation()
        info["door_open"] = self._door_open
        # Ground-truth for eval only — never copy these into Observation.
        info["gt_needed_key"] = self._needed_key()
        return obs, reward, terminated, truncated, info

    def hidden_rule_id(self) -> str:
        return self._rule.rule_id

    def phase(self) -> PhaseId:
        return self._phase

    def applicability_truth(self, record: dict[str, Any]) -> bool | None:
        """Eval-only. Returns None if the record has no scorable applicability."""
        appl = record.get("applicability")
        if not isinstance(appl, dict):
            return None
        rule_eq = appl.get("rule_id_eq")
        if isinstance(rule_eq, str):
            return rule_eq == self._rule.rule_id
        cue_eq = appl.get("context_cue_eq")
        if isinstance(cue_eq, str):
            return cue_eq == self._context_cue
        return None

    def _needed_key(self) -> str | None:
        door = self._door_color
        for color in COLORS:
            if self._rule.unlock_ok(color, door):
                return color
        return None

    def _pick_door_color(self, seed: int, episode_index: int) -> str:
        digest = hashlib.sha256(f"{seed}:{episode_index}:{self._rule.rule_id}".encode()).digest()
        return COLORS[digest[0] % len(COLORS)]

    def _observation(self) -> Observation:
        visible = {
            "door_color": self._door_color,
            "inventory": self._inventory,
            "door_open": self._door_open,
            "available_keys": list(self._available_keys),
        }
        if self._context_cue is not None:
            visible["context_cue"] = self._context_cue

        leak = FORBIDDEN_OBS_KEYS.intersection(visible)
        if leak:
            raise RuntimeError(f"Observation leaked hidden keys: {sorted(leak)}")

        goal = f"Open the {self._door_color} door, then done."
        text = (
            f"goal: {goal}\n"
            f"door_color: {self._door_color}\n"
            f"inventory: {self._inventory}\n"
            f"door_open: {str(self._door_open).lower()}\n"
            f"available_keys: {list(self._available_keys)}\n"
            f"actions: pick_key(color), unlock(door), done\n"
            "Respond with a single JSON object: "
            '{"type":"...","args":{...}}'
        )
        if self._context_cue is not None:
            text += f"\ncontext_cue: {self._context_cue}"

        return Observation(
            episode_index=self._episode_index,
            step=self._step,
            goal=goal,
            visible=visible,
            text=text,
            phase=self._phase,
            context_cue=self._context_cue,
        )
