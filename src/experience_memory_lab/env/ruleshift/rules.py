"""Hidden rule cards. Never serialized into agent observations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class RuleCard:
    rule_id: str
    family: str
    description: str  # for researchers / logs only
    unlock_ok: Callable[[str | None, str], bool]


def _match(inventory: str | None, door: str) -> bool:
    return inventory is not None and inventory == door


def _reversed(inventory: str | None, door: str) -> bool:
    """Conflict rule for later phases: needs the 'other' of red/blue."""
    if inventory is None:
        return False
    pair = {"red": "blue", "blue": "red"}
    needed = pair.get(door, door)
    return inventory == needed


ACCESS_MATCH_V1 = RuleCard(
    rule_id="access_match_v1",
    family="access",
    description="Unlock succeeds iff inventory key color matches door color.",
    unlock_ok=_match,
)

ACCESS_REVERSED_V1 = RuleCard(
    rule_id="access_reversed_v1",
    family="access",
    description="Unlock for red/blue requires the opposite color key.",
    unlock_ok=_reversed,
)

RULES_BY_ID: dict[str, RuleCard] = {
    ACCESS_MATCH_V1.rule_id: ACCESS_MATCH_V1,
    ACCESS_REVERSED_V1.rule_id: ACCESS_REVERSED_V1,
}
