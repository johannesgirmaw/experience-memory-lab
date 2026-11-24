from experience_memory_lab.llm.base import Message
from experience_memory_lab.llm.mock import MockLLM


def test_mock_llm_is_seed_stable() -> None:
    llm = MockLLM()
    messages = [
        Message(role="system", content="sys"),
        Message(
            role="user",
            content=(
                "goal: Open the red door\ndoor_color: red\ninventory: null\ndoor_open: false\n"
            ),
        ),
    ]
    a = llm.complete(messages, seed=0)
    b = llm.complete(messages, seed=99)
    assert a.content == b.content
    assert a.content == '{"args":{"color":"red"},"type":"pick_key"}'


def test_mock_embed_is_deterministic() -> None:
    llm = MockLLM()
    e1 = llm.embed(["hello", "world"])
    e2 = llm.embed(["hello", "world"])
    assert e1 == e2
    assert len(e1[0]) == 8


def test_mock_ignores_door_open_inside_memory_block() -> None:
    """Regression: retrieved notes must not make the policy emit done early."""
    llm = MockLLM()
    messages = [
        Message(
            role="user",
            content=(
                "goal: Open the red door\n"
                "door_color: red\n"
                "inventory: null\n"
                "door_open: false\n"
                "\n--- memory ---\n"
                "final_state: end_door_open=True\n"
                "door_open: true\n"
                "--- end memory ---\n"
            ),
        ),
    ]
    out = llm.complete(messages)
    assert out.content == '{"args":{"color":"red"},"type":"pick_key"}'
