"""Outbound webhook notifications for goal lifecycle events.

Sends HTTP POST to configured URLs when goals complete or fail.
Supports HMAC-SHA256 signing, retry, and event filtering.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger(__name__)


@dataclass
class WebhookTarget:
    """A configured webhook destination."""

    url: str
    secret: str | None = None  # for HMAC-SHA256 signing
    events: list[str] = field(default_factory=lambda: ["goal.completed", "goal.failed"])
    max_retries: int = 3
    enabled: bool = True


@dataclass
class WebhookDelivery:
    """Record of a webhook delivery attempt."""

    target_url: str
    event_type: str
    payload: dict[str, Any]
    status_code: int | None = None
    error: str | None = None
    attempts: int = 0
    delivered_at: float = field(default_factory=time.time)
    success: bool = False


class WebhookNotifier:
    """Sends outbound webhook notifications for system events.

    Subscribes to bus topics and forwards matching events
    to registered webhook targets.

    Usage:
        notifier = WebhookNotifier()
        notifier.add_target(WebhookTarget(
            url="https://example.com/hooks/qe",
            secret="my-secret",
            events=["goal.completed", "goal.failed"],
        ))
        await notifier.notify("goal.completed", {"goal_id": "g-123", ...})
    """

    def __init__(self) -> None:
        self._targets: list[WebhookTarget] = []
        self._history: list[WebhookDelivery] = []
        self._max_history = 500

    def add_target(self, target: WebhookTarget) -> None:
        """Register a webhook target."""
        self._targets.append(target)
        log.info(
            "webhook.target_added url=%s events=%s",
            target.url,
            target.events,
        )

    def remove_target(self, url: str) -> bool:
        """Remove a webhook target by URL. Returns True if found."""
        before = len(self._targets)
        self._targets = [t for t in self._targets if t.url != url]
        return len(self._targets) < before

    def list_targets(self) -> list[dict[str, Any]]:
        """Return all registered targets."""
        return [
            {
                "url": t.url,
                "events": t.events,
                "enabled": t.enabled,
                "has_secret": t.secret is not None,
            }
            for t in self._targets
        ]

    async def notify(
        self,
        event_type: str,
        payload: dict[str, Any],
    ) -> list[WebhookDelivery]:
        """Send webhook to all matching targets.

        Returns list of delivery records.
        """
        deliveries: list[WebhookDelivery] = []

        for target in self._targets:
            if not target.enabled:
                continue
            if event_type not in target.events:
                continue

            delivery = await self._deliver(target, event_type, payload)
            deliveries.append(delivery)
            self._record_delivery(delivery)

        return deliveries

    async def _deliver(
        self,
        target: WebhookTarget,
        event_type: str,
        payload: dict[str, Any],
    ) -> WebhookDelivery:
        """Deliver a webhook with retry."""
        body = {
            "event": event_type,
            "timestamp": time.time(),
            "data": payload,
        }
        body_bytes = json.dumps(body, sort_keys=True).encode()

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if target.secret:
            signature = hmac.new(
                target.secret.encode(),
                body_bytes,
                hashlib.sha256,
            ).hexdigest()
            headers["X-QE-Signature"] = f"sha256={signature}"

        delivery = WebhookDelivery(
            target_url=target.url,
            event_type=event_type,
            payload=payload,
        )

        for attempt in range(1, target.max_retries + 1):
            delivery.attempts = attempt
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        target.url,
                        content=body_bytes,
                        headers=headers,
                    )
                delivery.status_code = resp.status_code
                if 200 <= resp.status_code < 300:
                    delivery.success = True
                    log.info(
                        "webhook.delivered url=%s event=%s status=%d",
                        target.url,
                        event_type,
                        resp.status_code,
                    )
                    return delivery

                log.warning(
                    "webhook.http_error url=%s status=%d attempt=%d/%d",
                    target.url,
                    resp.status_code,
                    attempt,
                    target.max_retries,
                )
            except Exception as exc:
                delivery.error = str(exc)
                log.warning(
                    "webhook.send_error url=%s error=%s attempt=%d/%d",
                    target.url,
                    exc,
                    attempt,
                    target.max_retries,
                )

        return delivery

    def _record_delivery(self, delivery: WebhookDelivery) -> None:
        """Record delivery in history, trimming old entries."""
        self._history.append(delivery)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def delivery_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent delivery history."""
        entries = self._history[-limit:]
        return [
            {
                "target_url": d.target_url,
                "event_type": d.event_type,
                "status_code": d.status_code,
                "success": d.success,
                "attempts": d.attempts,
                "error": d.error,
                "delivered_at": d.delivered_at,
            }
            for d in reversed(entries)
        ]

    def stats(self) -> dict[str, Any]:
        """Return webhook notifier statistics."""
        total = len(self._history)
        successes = sum(1 for d in self._history if d.success)
        return {
            "targets": len(self._targets),
            "enabled_targets": sum(1 for t in self._targets if t.enabled),
            "total_deliveries": total,
            "successful_deliveries": successes,
            "success_rate": successes / total if total > 0 else 0.0,
        }


# ── Singleton ──────────────────────────────────────────────────────────────

_webhook_notifier: WebhookNotifier | None = None


def get_webhook_notifier() -> WebhookNotifier:
    global _webhook_notifier
    if _webhook_notifier is None:
        _webhook_notifier = WebhookNotifier()
    return _webhook_notifier
