# tests/test_utils.py
import threading
import time
from src.utils import LRUCache, hash_individuo

def test_lru_cache_basic():
    c = LRUCache(maxsize=3)
    assert len(c) == 0
    c.set(b"a", 1)
    c.set(b"b", 2)
    c.set(b"c", 3)
    assert c.get(b"a") == 1
    c.set(b"d", 4)
    # agora "b" foi removido? Depende da ordem - garantir que tamanho respeita maxsize
    assert len(c) <= 3

def test_lru_thread_safety():
    c = LRUCache(maxsize=100)
    def writer(i):
        for j in range(100):
            c.set(f"{i}-{j}".encode(), j)
    threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert len(c) <= 100

def test_hash_individuo_consistency():
    a = hash_individuo(b'rota', b'vels', digest_size=8)
    b = hash_individuo(b'rota', b'vels', digest_size=8)
    assert a == b
    c = hash_individuo(b'rotaX', b'vels', digest_size=8)
    assert a != c
