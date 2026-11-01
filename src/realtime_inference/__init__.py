from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from queue import Queue
from typing import Any, Deque, Dict, Iterable, List, Optional

from realtime_inference.model_registry import MultiDomainModelRegistry


@dataclass
class StreamEvent:
    domain: str
    payload: Dict[str, Any]
    event_id: str


class StreamBackend(ABC):
    @abstractmethod
    def emit(self, domain: str, payload: Dict[str, Any], event_id: Optional[str] = None) -> StreamEvent:
        raise NotImplementedError

    @abstractmethod
    def drain(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError


class InMemoryStreamBackend(StreamBackend):
    def __init__(self, registry: Optional[MultiDomainModelRegistry] = None) -> None:
        self.registry = registry or MultiDomainModelRegistry()
        self.queue: Queue[StreamEvent] = Queue()
        self.history: Deque[Dict[str, Any]] = deque(maxlen=1000)

    def emit(self, domain: str, payload: Dict[str, Any], event_id: Optional[str] = None) -> StreamEvent:
        event = StreamEvent(domain=domain, payload=payload, event_id=event_id or f"evt-{len(self.history)}")
        self.queue.put(event)
        return event

    def drain(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        count = 0
        while not self.queue.empty() and (limit is None or count < limit):
            event = self.queue.get()
            pred = self.registry.predict(event.domain, event.payload)
            record = {"event_id": event.event_id, "domain": event.domain, **pred}
            self.history.append(record)
            results.append(record)
            count += 1
        return results


class RedisStreamBackend(StreamBackend):
    """Optional Redis-backed stream backend for production-style ingestion."""

    def __init__(self, redis_url: Optional[str] = None, stream_name: str = "realtime_inference") -> None:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("redis package is required for Redis stream backend") from exc

        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.stream_name = stream_name
        self.client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        self.history: Deque[Dict[str, Any]] = deque(maxlen=1000)

    def emit(self, domain: str, payload: Dict[str, Any], event_id: Optional[str] = None) -> StreamEvent:
        event = StreamEvent(domain=domain, payload=payload, event_id=event_id or f"evt-{len(self.history)}")
        payload_raw = json.dumps(payload)
        self.client.xadd(self.stream_name, {"event_id": event.event_id, "domain": event.domain, "payload": payload_raw})
        return event

    def drain(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        entries = self.client.xread(streams={self.stream_name: 0}, count=limit or 10, block=1000)
        results: List[Dict[str, Any]] = []
        for _, items in entries:
            for _, values in items:
                data = values or {}
                payload = json.loads(data.get("payload", "{}"))
                results.append({"event_id": data.get("event_id"), "domain": data.get("domain"), "payload": payload})
        self.history.extend(results)
        return results


class InMemoryStreamProcessor:
    def __init__(self, registry: Optional[MultiDomainModelRegistry] = None, backend: Optional[StreamBackend] = None) -> None:
        self.backend = backend or InMemoryStreamBackend(registry=registry)
        self.registry = registry or getattr(self.backend, "registry", MultiDomainModelRegistry())
        self.history: Deque[Dict[str, Any]] = self.backend.history

    def emit(self, domain: str, payload: Dict[str, Any], event_id: Optional[str] = None) -> StreamEvent:
        return self.backend.emit(domain, payload, event_id=event_id)

    def drain(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.backend.drain(limit)

    def process_batch(self, events: Iterable[StreamEvent]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for event in events:
            pred = self.registry.predict(event.domain, event.payload)
            record = {"event_id": event.event_id, "domain": event.domain, **pred}
            self.history.append(record)
            results.append(record)
        return results


class RedisStreamProcessor(InMemoryStreamProcessor):
    def __init__(self, registry: Optional[MultiDomainModelRegistry] = None, redis_url: Optional[str] = None) -> None:
        backend = RedisStreamBackend(redis_url=redis_url)
        super().__init__(registry=registry, backend=backend)


__all__ = [
    "StreamEvent",
    "StreamBackend",
    "InMemoryStreamBackend",
    "RedisStreamBackend",
    "InMemoryStreamProcessor",
    "RedisStreamProcessor",
]

