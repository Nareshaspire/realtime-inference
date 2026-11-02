"""Dynamic micro-batcher module re-exported for root access."""

from realtime_inference.batcher import (
    BatchItem,
    BatcherStats,
    DynamicMicroBatcher,
    QueueFullError,
)

__all__ = [
    "BatchItem",
    "BatcherStats",
    "DynamicMicroBatcher",
    "QueueFullError",
]
