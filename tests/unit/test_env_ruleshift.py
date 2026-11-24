from experience_memory_lab.env.ruleshift import RuleShiftEnv
from experience_memory_lab.env.ruleshift.env import FORBIDDEN_OBS_KEYS
from experience_memory_lab.types.contracts import Action


def test_reset_is_deterministic_for_seed_and_episode() -> None:
    a = RuleShiftEnv()
    b = RuleShiftEnv()
    oa = a.reset(7, episode_index=2)
    ob = b.reset(7, episode_index=2)
    assert oa.visible == ob.visible
    assert oa.goal == ob.goal
    assert a.hidden_rule_id() == "access_match_v1"


def test_observation_does_not_leak_rule_id() -> None:
    env = RuleShiftEnv()
    obs = env.reset(0, episode_index=0)
    blob = obs.model_dump()
    text = obs.text.lower()
    assert "rule_id" not in blob["visible"]
    assert FORBIDDEN_OBS_KEYS.isdisjoint(blob["visible"].keys())
    assert "access_match" not in text
    assert "rule_id" not in text
    assert env.hidden_rule_id() == "access_match_v1"


def test_matching_policy_succeeds() -> None:
    env = RuleShiftEnv()
    obs = env.reset(0, episode_index=0)
    door = obs.visible["door_color"]
    _, _, _, _, _ = env.step(Action(type="pick_key", args={"color": door}))
    _, r_unlock, _, _, info_u = env.step(Action(type="unlock", args={"door": door}))
    assert info_u["unlocked"] is True
    assert r_unlock == 0.5
    _, r_done, term, _, info_d = env.step(Action(type="done", args={}))
    assert term is True
    assert info_d["success"] is True
    assert r_done == 1.0


def test_wrong_key_fails_unlock() -> None:
    env = RuleShiftEnv()
    obs = env.reset(1, episode_index=0)
    door = obs.visible["door_color"]
    wrong = next(c for c in obs.visible["available_keys"] if c != door)
    env.step(Action(type="pick_key", args={"color": wrong}))
    _, _, _, _, info = env.step(Action(type="unlock", args={"door": door}))
    assert info["unlocked"] is False
