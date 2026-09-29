from llm_gate.cache import Cache


def test_put_get_roundtrip(tmp_path):
    c = Cache(tmp_path / "cache.db")
    key = Cache.key("model", "prompt", 0.0, 42)
    assert c.get(key) is None
    c.put(key, {"text": "hello", "cost_usd": 0.001})
    hit = c.get(key)
    assert hit == {"text": "hello", "cost_usd": 0.001}
    assert (c.hits, c.misses) == (1, 1)
    c.close()


def test_key_changes_with_any_input():
    k1 = Cache.key("m", "p", 0.0, 1)
    k2 = Cache.key("m", "p", 0.0, 2)
    k3 = Cache.key("m", "p2", 0.0, 1)
    assert len({k1, k2, k3}) == 3


def test_persistence_across_instances(tmp_path):
    path = tmp_path / "cache.db"
    c1 = Cache(path)
    c1.put("k", {"v": 1})
    c1.close()
    c2 = Cache(path)
    assert c2.get("k") == {"v": 1}
    c2.close()
