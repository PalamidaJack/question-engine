"""Notification routing across registered channels."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from qe.channels.base import ChannelAdapter

log = logging.getLogger(__name__)


class NotificationPriority(Enum):
    """Priority levels for outgoing notifications."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class NotificationPreferences:
    """Per-user notification routing preferences."""

    user_id: str
    channels: dict[str, list[str]] = field(default_factory=dict)
    """Mapping of event_type -> list of channel names."""

    quiet_hours: tuple[int, int] | None = None
    """Optional (start_hour, end_hour) during which LOW/NORMAL
    notifications are suppressed.  Hours are in 24-hour format."""


class NotificationRouter:
    """Route notifications to the right channels based on user preferences.

    Channels are registered by name; each user can choose which event
    types are delivered to which channels.
    """

    def __init__(self) -> None:
        self._channels: dict[str, ChannelAdapter] = {}
        self._preferences: dict[str, NotificationPreferences] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_channel(self, name: str, adapter: ChannelAdapter) -> None:
        """Register a channel adapter under *name*."""
        self._channels[name] = adapter
        log.info("notification_router.channel_registered name=%s", name)

    def set_preferences(
        self, user_id: str, prefs: NotificationPreferences
    ) -> None:
        """Store notification preferences for *user_id*."""
        self._preferences[user_id] = prefs
        log.debug(
            "notification_router.preferences_set user=%s channels=%s",
            user_id,
            list(prefs.channels.keys()),
        )

    def get_registered_channels(self) -> list[str]:
        """Return the names of all registered channels."""
        return list(self._channels.keys())

    # ------------------------------------------------------------------
    # Notification delivery
    # ------------------------------------------------------------------

    async def notify(
        self,
        user_id: str,
        event_type: str,
        message: str,
        urgency: str = "normal",
    ) -> list[str]:
        """Send a notification to the appropriate channels.

        Returns a list of channel names that were successfully notified.
        """
        prefs = self._preferences.get(user_id)
        if prefs is None:
            log.debug(
                "notification_router.no_prefs user=%s, broadcasting to all",
                user_id,
            )
            target_channels = list(self._channels.keys())
        else:
            target_channels = self._should_notify(prefs, event_type)

        # Respect quiet hours for low/normal urgency
        if prefs and prefs.quiet_hours and urgency in ("low", "normal"):
            now_hour = datetime.now(UTC).hour
            start, end = prefs.quiet_hours
            if start <= end:
                in_quiet = start <= now_hour < end
            else:
                # Wraps midnight, e.g. (22, 6)
                in_quiet = now_hour >= start or now_hour < end
            if in_quiet:
                log.debug(
                    "notification_router.quiet_hours user=%s suppressed",
                    user_id,
                )
                return []

        notified: list[str] = []
        for ch_name in target_channels:
            adapter = self._channels.get(ch_name)
            if adapter is None:
                log.warning(
                    "notification_router.channel_not_found name=%s", ch_name
                )
                continue
            try:
                await adapter.send(user_id, message)
                notified.append(ch_name)
            except Exception:
                log.exception(
                    "notification_router.delivery_failed channel=%s user=%s",
                    ch_name,
                    user_id,
                )

        log.info(
            "notification_router.notified user=%s event=%s channels=%s",
            user_id,
            event_type,
            notified,
        )
        return notified

    async def deliver_result(
        self,
        user_id: str,
        goal_id: str,
        result: dict[str, Any],
    ) -> list[str]:
        """Format and deliver a goal result to the user.

        Returns a list of channel names that received the result.
        """
        status = result.get("status", "completed")
        summary = result.get("summary", "No summary available.")
        message = (
            f"Goal {goal_id} -- {status}\n"
            f"---\n"
            f"{summary}"
        )

        return await self.notify(
            user_id=user_id,
            event_type="goal_result",
            message=message,
            urgency="high",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_notify(
        self,
        prefs: NotificationPreferences,
        event_type: str,
    ) -> list[str]:
        """Determine which channels to use for this event type.

        Falls back to all registered channels if the event type has no
        explicit mapping in the user's preferences.
        """
        channels = prefs.channels.get(event_type)
        if channels:
            return channels

        # Wildcard / default
        default = prefs.channels.get("*")
        if default:
            return default

        return list(self._channels.keys())
