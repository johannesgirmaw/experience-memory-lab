from experience_memory_lab.memory.store import InMemoryVectorStore, StoredNote, cosine_similarity


def test_cosine_identical_is_one() -> None:
    v = [1.0, 0.0, 0.0]
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9


def test_search_orders_by_score_then_id() -> None:
    store = InMemoryVectorStore()
    store.add(StoredNote("b", "beta", [1.0, 0.0], {}))
    store.add(StoredNote("a", "alpha", [1.0, 0.0], {}))
    store.add(StoredNote("c", "gamma", [0.0, 1.0], {}))
    hits = store.search([1.0, 0.0], k=2)
    assert [n.record_id for n, _ in hits] == ["a", "b"]
