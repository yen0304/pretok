"""Caching utilities for pretok."""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with value and metadata."""

    value: T
    created_at: float
    hits: int = 0

    def is_expired(self, ttl: int) -> bool:
        """Check if entry is expired.

        Args:
            ttl: Time-to-live in seconds (0 = no expiry)

        Returns:
            True if entry is expired
        """
        if ttl == 0:
            return False
        return time.time() - self.created_at > ttl


class Cache(ABC):
    """Abstract base class for caches."""

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        ...

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries from cache."""
        ...

    @abstractmethod
    def size(self) -> int:
        """Return number of entries in cache."""
        ...


class MemoryCache(Cache):
    """In-memory LRU cache.

    Thread-safe in-memory cache with configurable max size and TTL.

    Example:
        >>> cache = MemoryCache(max_size=1000, ttl=3600)
        >>> cache.set("key", "value")
        >>> cache.get("key")
        'value'
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl: int = 3600,
    ) -> None:
        """Initialize memory cache.

        Args:
            max_size: Maximum number of entries
            ttl: Time-to-live in seconds (0 = no expiry)
        """
        self._max_size = max_size
        self._ttl = ttl
        self._cache: OrderedDict[str, CacheEntry[Any]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        if entry.is_expired(self._ttl):
            del self._cache[key]
            self._misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.hits += 1
        self._hits += 1

        return entry.value

    def set(self, key: str, value: Any) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        # Remove if exists to update position
        if key in self._cache:
            del self._cache[key]

        # Evict oldest if at capacity
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        self._cache[key] = CacheEntry(
            value=value,
            created_at=time.time(),
        )

    def delete(self, key: str) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all entries from cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def size(self) -> int:
        """Return number of entries in cache."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    @property
    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        return {
            "size": self.size(),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


def make_cache_key(
    text: str,
    source_language: str | None,
    target_language: str,
    translator: str,
) -> str:
    """Create cache key for translation.

    Args:
        text: Source text
        source_language: Source language code
        target_language: Target language code
        translator: Translator name

    Returns:
        Unique cache key
    """
    key_parts = [
        text,
        source_language or "auto",
        target_language,
        translator,
    ]
    key_str = "|".join(key_parts)
    return hashlib.sha256(key_str.encode()).hexdigest()[:32]
