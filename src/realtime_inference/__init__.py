from __future__ import annotations

from typing import Any, Dict, List

from realtime_inference.api import app
from realtime_inference.model_registry import MultiDomainModelRegistry
from realtime_inference.streaming import InMemoryStreamProcessor, RedisStreamProcessor, StreamEvent

__all__ = [
    "app",
    "MultiDomainModelRegistry",
    "InMemoryStreamProcessor",
    "RedisStreamProcessor",
    "StreamEvent",
]


def main() -> None:
    """Run the demo FastAPI app entry point."""
    import uvicorn

    uvicorn.run("realtime_inference.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
