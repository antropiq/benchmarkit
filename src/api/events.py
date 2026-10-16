"""In-memory, seq-numbered event bus powering the SSE run stream.

Events are buffered for replay so reconnecting clients (with
``Last-Event-ID``) never miss anything from the current run. ``publish``
is thread-safe: the benchmark loop runs in a worker thread while SSE
subscribers live on the event loop.
"""

import asyncio
import json
import threading
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

Event = Dict[str, Any]


def sse_format(seq: int, event: Event) -> str:
    """Serialize one buffered event as an SSE frame."""
    name = event.get("event", "message")
    data = json.dumps(event, ensure_ascii=False)
    return f"id: {seq}\nevent: {name}\ndata: {data}\n\n"


class EventBus:
    """Replayable event buffer with async subscribers."""

    def __init__(self) -> None:
        self._events: List[Tuple[int, Event]] = []
        self._subscribers: List[asyncio.Queue] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._closed = False
        self._lock = threading.Lock()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the running loop so worker threads can publish to it."""
        self._loop = loop

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._events[-1][0] if self._events else -1

    def snapshot(self, after_seq: int = -1) -> List[Tuple[int, Event]]:
        """Return buffered events with ``seq > after_seq``."""
        with self._lock:
            return [(seq, event) for seq, event in self._events if seq > after_seq]

    def publish(self, event: Event) -> int:
        """Buffer one event and notify subscribers. Safe from any thread."""
        with self._lock:
            if self._closed:
                return -1
            seq = len(self._events)
            self._events.append((seq, event))
            subscribers = list(self._subscribers)
        loop = self._loop
        if loop is not None:
            for queue in subscribers:
                loop.call_soon_threadsafe(queue.put_nowait, (seq, event))
        return seq

    def close(self) -> None:
        """Mark the bus closed and release all subscribers."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            subscribers = list(self._subscribers)
        loop = self._loop
        if loop is not None:
            for queue in subscribers:
                loop.call_soon_threadsafe(queue.put_nowait, None)

    async def subscribe(self, last_seq: int = -1) -> AsyncIterator[Tuple[int, Event]]:
        """Yield replayed events after ``last_seq``, then live events.

        Terminates when the bus closes.
        """
        queue: asyncio.Queue = asyncio.Queue()
        with self._lock:
            replay = [(seq, event) for seq, event in self._events if seq > last_seq]
            closed = self._closed
            if not closed:
                self._subscribers.append(queue)
        try:
            for item in replay:
                yield item
            if closed:
                return
            while True:
                item = await queue.get()
                if item is None:
                    return
                yield item
        finally:
            with self._lock:
                if queue in self._subscribers:
                    self._subscribers.remove(queue)
