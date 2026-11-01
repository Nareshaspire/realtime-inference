from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Deque, Dict, Iterable, List, Optional

from realtime_inference.model_registry import MultiDomainModelRegistry


@dataclass
class StreamEvent:
    domain: str
    payload: Dict[str, Any]
    event_id: str


class InMemoryStreamProcessor:
    """Simple in-memory stream processor that mimics ingestion from a message bus."""

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
            results.append(record)
            self.history.append(record)
            count += 1
        return results

    def process_batch(self, events: Iterable[StreamEvent]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for event in events:
            pred = self.registry.predict(event.domain, event.payload)
            record = {"event_id": event.event_id, "domain": event.domain, **pred}
            self.history.append(record)
            records.append(record)
        return records


__all__ = ["InMemoryStreamProcessor", "StreamEvent"]

