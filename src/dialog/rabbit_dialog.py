import asyncio
import json
from typing import Awaitable, Callable, Optional
from dialog.dialog_contract import DialogContract
import aio_pika


class RabbitDialog(DialogContract):
    PUB = "raw_text"
    SUB = "speech"

    def __init__(self, amqp_url: str, source: str):
        self.amqp_url = amqp_url
        self.source = source

        # callbacks (события)
        self.on_say: Optional[Callable[[str], None]] = None
        self.on_ask: Optional[Callable[[str], Awaitable[str]]] = None

        self.connection: Optional[aio_pika.RobustConnection] = None
        self.channel: Optional[aio_pika.RobustChannel] = None
        self.queue: Optional[aio_pika.Queue] = None
        self.pub_exchange: Optional[aio_pika.Exchange] = None

    # ========= PUBLISH =========
    def pub_input(self, text: str) -> None:
        if self.channel is None:
            raise RuntimeError("Channel not initialized. Call start first.")
        asyncio.run_coroutine_threadsafe(
                self._publish("input", {"session_id": 0, "text": text}),
                self.loop
            )

    async def _publish(self, event: str, payload: dict) -> None:
        if self.channel is None:
            raise RuntimeError("Channel not initialized. Call start first.")
        message = aio_pika.Message(
            body=json.dumps(payload).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT
        )
        await self.pub_exchange.publish(
            message,
            routing_key=f"{self.PUB}.{event}.{self.source}"
        )

    # ========= START =========
    async def start(self):
        await self._connect()

        self.pub_exchange = await self.channel.declare_exchange(
            self.PUB, aio_pika.ExchangeType.TOPIC, durable=True
        )

        sub_exchange = await self.channel.declare_exchange(
            self.SUB, aio_pika.ExchangeType.TOPIC, durable=True
        )
        self.queue = await self.channel.declare_queue(self.SUB, durable=True)
        await self.queue.bind(sub_exchange, routing_key=f"{self.SUB}.*.{self.source}")

        self.loop = asyncio.get_running_loop()
        asyncio.create_task(self._listen_rabbit())
        print("[*] Listen RabbitMQ ...")

    async def _connect(self):
        self.connection = await aio_pika.connect_robust(self.amqp_url)
        self.channel = await self.connection.channel()
        await self.channel.set_qos(prefetch_count=1)

        print("[*] Connected to RabbitMQ")

    async def _listen_rabbit(self):
        async with self.queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    await self._handle_message(message)

    async def _handle_message(self, message: aio_pika.IncomingMessage):
        payload = json.loads(message.body)
        event = message.routing_key.split(".")[1]

        if event == "say" and self.on_say:
            asyncio.create_task(asyncio.to_thread(self.on_say, payload["text"]))

        elif event == "ask" and self.on_ask:
            async def ask_publish():
                result = await self.on_ask(payload["text"])
                await self._publish("response", {"session_id": 0, "text": result})

            asyncio.create_task(ask_publish())

    # ========= CLOSE =========
    async def close(self):
        if self.connection:
            await self.connection.close()
            print("[*] RabbitMQ connection closed")
