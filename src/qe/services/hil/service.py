# TODO: integration test
import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from qe.models.envelope import Envelope
from qe.runtime.service import BaseService


class HILService(BaseService):
    def __init__(self, blueprint, bus, substrate) -> None:
        super().__init__(blueprint, bus, substrate)
        self.pending_dir = Path("data/hil_queue/pending")
        self.completed_dir = Path("data/hil_queue/completed")
        self._poll_tasks: dict[str, asyncio.Task] = {}

    async def start(self) -> None:
        self._running = True
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.completed_dir.mkdir(parents=True, exist_ok=True)
        await self._maybe_await(
            self.bus.subscribe("hil.approval_required", self._handle_hil_request)
        )

    async def stop(self) -> None:
        self._running = False
        await self._maybe_await(
            self.bus.unsubscribe("hil.approval_required", self._handle_hil_request)
        )
        for task in self._poll_tasks.values():
            task.cancel()

    async def _handle_hil_request(self, envelope: Envelope) -> None:
        timeout_seconds = int(envelope.payload.get("timeout_seconds", 3600))
        now = datetime.now(UTC)
        expires_at = now + timedelta(seconds=timeout_seconds)

        proposal = {
            "envelope_id": envelope.envelope_id,
            "timestamp": now.isoformat(),
            "reason": envelope.payload.get("reason", "approval_required"),
            "proposal_summary": envelope.payload.get("proposal_summary", ""),
            "full_payload": envelope.payload,
            "timeout_seconds": timeout_seconds,
            "expires_at": expires_at.isoformat(),
        }

        pending_file = self.pending_dir / f"{envelope.envelope_id}.json"
        pending_file.write_text(json.dumps(proposal, indent=2), encoding="utf-8")

        task = asyncio.create_task(self._poll_for_decision(envelope, expires_at))
        self._poll_tasks[envelope.envelope_id] = task

    async def _poll_for_decision(self, original: Envelope, expires_at: datetime) -> None:
        pending_file = self.pending_dir / f"{original.envelope_id}.json"
        completed_file = self.completed_dir / f"{original.envelope_id}.json"

        while self._running:
            if completed_file.exists():
                data = json.loads(completed_file.read_text(encoding="utf-8"))
                decision = data.get("decision")
                if decision == "approved":
                    self.bus.publish(
                        Envelope(
                            topic="hil.approved",
                            source_service_id=self.blueprint.service_id,
                            correlation_id=original.envelope_id,
                            causation_id=original.envelope_id,
                            payload={"decision": "approved", "decided_at": data.get("decided_at")},
                        )
                    )
                else:
                    self.bus.publish(
                        Envelope(
                            topic="hil.rejected",
                            source_service_id=self.blueprint.service_id,
                            correlation_id=original.envelope_id,
                            causation_id=original.envelope_id,
                            payload={
                                "decision": "rejected",
                                "reason": data.get("reason", "rejected"),
                                "decided_at": data.get("decided_at"),
                            },
                        )
                    )
                return

            if datetime.now(UTC) > expires_at:
                timeout_payload = {
                    "decision": "rejected",
                    "reason": "timeout",
                    "decided_at": datetime.now(UTC).isoformat(),
                }
                if pending_file.exists():
                    completed_file.write_text(
                        json.dumps(timeout_payload, indent=2), encoding="utf-8"
                    )
                    pending_file.unlink(missing_ok=True)
                self.bus.publish(
                    Envelope(
                        topic="hil.rejected",
                        source_service_id=self.blueprint.service_id,
                        correlation_id=original.envelope_id,
                        causation_id=original.envelope_id,
                        payload=timeout_payload,
                    )
                )
                return

            await asyncio.sleep(10)

    async def handle_response(self, envelope: Envelope, response: Any) -> None:
        return None

    def get_response_schema(self, topic: str):
        raise NotImplementedError("HIL service does not use LLM calls in Phase 0")
