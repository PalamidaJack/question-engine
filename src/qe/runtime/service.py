import asyncio
import logging
from typing import Any, Callable, Awaitable

from pydantic import BaseModel
import instructor
import litellm
from dotenv import load_dotenv

from qe.models.envelope import Envelope
from qe.models.genome import Blueprint
from qe.runtime.context_manager import ContextManager
from qe.runtime.router import AutoRouter

# Load environment variables from .env file if present
load_dotenv()

log = logging.getLogger(__name__)


class BaseService:
    def __init__(
        self,
        blueprint: Blueprint,
        bus: Any,  # Duck-typed: has subscribe, unsubscribe, publish
        substrate: Any,  # Duck-typed: has commit_claim, get_claims, etc.
    ) -> None:
        self.blueprint = blueprint
        self.bus = bus
        self.substrate = substrate
        self.context_manager = ContextManager(blueprint)
        self.router = AutoRouter(blueprint.model_preference)
        self._turn_count = 0
        self._running = False
        self._heartbeat_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Subscribe to all declared topics and start heartbeat loop."""
        self._running = True

        # Subscribe to all declared topics
        for topic in self.blueprint.capabilities.bus_topics_subscribe:
            await self.bus.subscribe(topic, self._handle_envelope)

        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        """Unsubscribe and stop heartbeat loop."""
        self._running = False

        # Cancel heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Unsubscribe from all topics
        for topic in self.blueprint.capabilities.bus_topics_subscribe:
            await self.bus.unsubscribe(topic, self._handle_envelope)

    async def _handle_envelope(self, envelope: Envelope) -> None:
        """The 6-step service runtime loop."""
        # 1. Validate envelope schema (bus does this, but assert here)
        assert envelope.topic in self.blueprint.capabilities.bus_topics_subscribe

        # 2. Build LLM messages
        messages = self.context_manager.build_messages(envelope, self._turn_count)

        # 3. Select model
        model = self.router.select(envelope)

        # 4. Make LLM call via Instructor for structured output
        schema = self.get_response_schema(envelope.topic)
        response = await self._call_llm(model, messages, schema)

        # 5. Increment turn counter; trigger reinforcement if needed
        self._turn_count += 1
        if self._turn_count % self.blueprint.reinforcement_interval_turns == 0:
            self.context_manager.reinforce()

        # 6. Let subclass handle the response
        await self.handle_response(envelope, response)

    async def handle_response(self, envelope: Envelope, response: Any) -> None:
        """Override in subclass. Default: do nothing."""
        raise NotImplementedError("Subclasses must implement handle_response")

    def get_response_schema(self, topic: str) -> type[BaseModel]:
        """
        Override in subclass to return the Pydantic model for this topic's expected response.
        """
        raise NotImplementedError("Subclasses must implement get_response_schema")

    async def _call_llm(
        self,
        model: str,
        messages: list[dict],
        schema: type[BaseModel],
    ) -> Any:
        """
        Use Instructor client. Let exceptions propagate — supervisor handles them.
        """
        client = instructor.from_litellm(litellm.acompletion)
        return await client.chat.completions.create(
            model=model,
            messages=messages,
            response_model=schema,
        )

    async def _heartbeat_loop(self) -> None:
        """Publish heartbeat every 30 seconds."""
        while self._running:
            envelope = Envelope(
                topic="system.heartbeat",
                source_service_id=self.blueprint.service_id,
                payload={
                    "turn_count": self._turn_count,
                    "status": "alive"
                }
            )
            self.bus.publish(envelope)
            await asyncio.sleep(30)
