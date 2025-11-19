# src/utils.py
import threading
from collections import OrderedDict
import hashlib
from typing import Tuple

class LRUCache:
    """Thread-safe LRU cache based on OrderedDict."""
    def __init__(self, maxsize: int = 200_000):
        self.maxsize = int(maxsize)
        self._od = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            try:
                val = self._od.pop(key)
                self._od[key] = val
                return val
            except KeyError:
                return None

    def set(self, key, value):
        with self._lock:
            if key in self._od:
                self._od.pop(key)
            self._od[key] = value
            if len(self._od) > self.maxsize:
                self._od.popitem(last=False)

    def clear(self):
        with self._lock:
            self._od.clear()

    def __len__(self):
        with self._lock:
            return len(self._od)

def hash_individuo(rota_bytes: bytes, vels_bytes: bytes, digest_size: int = 8) -> bytes:
    h = hashlib.blake2b(digest_size=digest_size)
    h.update(rota_bytes)
    h.update(b'|')
    h.update(vels_bytes)
    return h.digest()
