import asyncio
import inspect
import logging
from typing import Any

from dotenv import load_dotenv
import instructor
import litellm
from pydantic import BaseModel

from qe.models.envelope import Envelope
from qe.models.genome import Blueprint
from qe.runtime.context_manager import ContextManager
from qe.runtime.router import AutoRouter

load_dotenv()

log = logging.getLogger(__name__)


class BaseService:
    def __init__(self, blueprint: Blueprint, bus: Any, substrate: Any) -> None:
        self.blueprint = blueprint
        self.bus = bus
        self.substrate = substrate
        self.context_manager = ContextManager(blueprint)
        self.router = AutoRouter(blueprint.model_preference)
        self._turn_count = 0
        self._running = False
        self._heartbeat_task: asyncio.Task | None = None

    async def _maybe_await(self, result: Any) -> Any:
        if inspect.isawaitable(result):
            return await result
        return result

    async def start(self) -> None:
        self._running = True
        for topic in self.blueprint.capabilities.bus_topics_subscribe:
            await self._maybe_await(self.bus.subscribe(topic, self._handle_envelope))
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        for topic in self.blueprint.capabilities.bus_topics_subscribe:
            await self._maybe_await(self.bus.unsubscribe(topic, self._handle_envelope))

    async def _handle_envelope(self, envelope: Envelope) -> None:
        assert envelope.topic in self.blueprint.capabilities.bus_topics_subscribe
        messages = self.context_manager.build_messages(envelope, self._turn_count)
        model = self.router.select(envelope)
        response = await self._call_llm(model, messages, self.get_response_schema(envelope.topic))

        self._turn_count += 1
        if self._turn_count % self.blueprint.reinforcement_interval_turns == 0:
            self.context_manager.reinforce()

        await self.handle_response(envelope, response)

    async def handle_response(self, envelope: Envelope, response: Any) -> None:
        raise NotImplementedError

    def get_response_schema(self, topic: str) -> type[BaseModel]:
        raise NotImplementedError

    async def _call_llm(self, model: str, messages: list[dict], schema: type[BaseModel]) -> Any:
        client = instructor.from_litellm(litellm.acompletion)
        return await client.chat.completions.create(
            model=model,
            messages=messages,
            response_model=schema,
        )

    async def reconfigure(self, new_config: dict[str, Any]) -> None:
        self.config = new_config

    async def _heartbeat_loop(self) -> None:
        while self._running:
            self.bus.publish(
                Envelope(
                    topic="system.heartbeat",
                    source_service_id=self.blueprint.service_id,
                    payload={"turn_count": self._turn_count, "status": "alive"},
                )
            )
            await asyncio.sleep(30)
