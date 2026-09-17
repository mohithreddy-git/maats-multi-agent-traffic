import asyncio

from backend.agents.message_bus import MessageBus


def test_publish_delivers_to_subscriber():
    async def scenario():
        bus = MessageBus()
        queue = bus.subscribe("topic.a")
        await bus.publish("topic.a", sender="tester", payload={"x": 1})
        message = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert message.topic == "topic.a"
        assert message.sender == "tester"
        assert message.payload == {"x": 1}

    asyncio.run(scenario())


def test_publish_fans_out_to_all_subscribers_independently():
    async def scenario():
        bus = MessageBus()
        q1 = bus.subscribe("metrics")
        q2 = bus.subscribe("metrics")
        await bus.publish("metrics", sender="simulator", payload=[1, 2, 3])

        m1 = await asyncio.wait_for(q1.get(), timeout=1.0)
        m2 = await asyncio.wait_for(q2.get(), timeout=1.0)
        assert m1.payload == [1, 2, 3]
        assert m2.payload == [1, 2, 3]
        assert q1.empty() and q2.empty()

    asyncio.run(scenario())


def test_subscribers_to_different_topics_do_not_cross_talk():
    async def scenario():
        bus = MessageBus()
        q_metrics = bus.subscribe("metrics")
        q_signal = bus.subscribe("signal_command")
        await bus.publish("metrics", sender="simulator", payload="m")

        message = await asyncio.wait_for(q_metrics.get(), timeout=1.0)
        assert message.payload == "m"
        assert q_signal.empty()

    asyncio.run(scenario())


def test_no_subscribers_on_topic_does_not_raise():
    async def scenario():
        bus = MessageBus()
        await bus.publish("nobody.listening", sender="tester", payload=None)

    asyncio.run(scenario())
