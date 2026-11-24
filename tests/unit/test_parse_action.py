from experience_memory_lab.agent.loop import parse_action


def test_parse_nested_and_flat_json() -> None:
    a = parse_action('{"type":"pick_key","args":{"color":"blue"}}')
    assert a.type == "pick_key"
    assert a.args["color"] == "blue"

    b = parse_action('{"type":"unlock","door":"red"}')
    assert b.type == "unlock"
    assert b.args["door"] == "red"


def test_parse_falls_back_to_done() -> None:
    a = parse_action("not json at all")
    assert a.type == "done"
