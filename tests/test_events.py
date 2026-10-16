import asyncio

from src.api.events import EventBus, sse_format


def _collect(bus: EventBus, last_seq: int = -1):
    async def run():
        bus.bind_loop(asyncio.get_running_loop())
        return [event async for _, event in bus.subscribe(last_seq)]

    return asyncio.run(run())


class TestEventBusBuffer:
    def test_publish_buffers_with_seq(self):
        bus = EventBus()
        assert bus.publish({"event": "a"}) == 0
        assert bus.publish({"event": "b"}) == 1
        assert bus.last_seq == 1
        assert [event["event"] for _, event in bus.snapshot()] == ["a", "b"]

    def test_snapshot_after_seq_filters(self):
        bus = EventBus()
        for name in ("a", "b", "c"):
            bus.publish({"event": name})
        assert [event["event"] for _, event in bus.snapshot(0)] == ["b", "c"]

    def test_publish_after_close_is_ignored(self):
        bus = EventBus()
        bus.publish({"event": "a"})
        bus.close()
        assert bus.publish({"event": "b"}) == -1
        assert bus.last_seq == 0


class TestEventBusSubscribe:
    def test_replays_buffered_events_then_stops_on_close(self):
        bus = EventBus()
        for name in ("a", "b", "c"):
            bus.publish({"event": name})
        bus.close()
        assert [event["event"] for event in _collect(bus)] == ["a", "b", "c"]

    def test_subscribe_after_last_seq_replays_only_newer(self):
        bus = EventBus()
        for name in ("a", "b", "c"):
            bus.publish({"event": name})
        bus.close()
        assert [event["event"] for event in _collect(bus, last_seq=1)] == ["c"]

    def test_live_events_reach_subscriber(self):
        bus = EventBus()

        async def scenario():
            bus.bind_loop(asyncio.get_running_loop())
            received = []

            async def consume():
                async for _, event in bus.subscribe():
                    received.append(event)

            task = asyncio.create_task(consume())
            await asyncio.sleep(0)
            bus.publish({"event": "x"})
            bus.publish({"event": "y"})
            await asyncio.sleep(0)
            bus.close()
            await task
            return received

        received = asyncio.run(scenario())
        assert [event["event"] for event in received] == ["x", "y"]


class TestSseFormat:
    def test_frame_shape(self):
        frame = sse_format(3, {"event": "token", "text": "hi"})
        assert frame == 'id: 3\nevent: token\ndata: {"event": "token", "text": "hi"}\n\n'

    def test_non_ascii_preserved(self):
        frame = sse_format(0, {"event": "token", "text": "héllo"})
        assert "héllo" in frame
