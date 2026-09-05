from backend.cache.base import Cache, build_cache_key
from backend.cache.memory_cache import MemoryCache
from backend.cache.redis_cache import RedisCache
from backend.config import Settings


def build_cache(settings: Settings) -> Cache:
    if settings.redis_url:
        return RedisCache(settings.redis_url)
    return MemoryCache()


__all__ = ["Cache", "build_cache_key", "MemoryCache", "RedisCache", "build_cache"]
