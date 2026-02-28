"""Communication & integration layer — channel adapters and notification routing."""

from qe.channels.base import ChannelAdapter, ChannelMessage
from qe.channels.email_channel import EmailAdapter
from qe.channels.notifications import (
    NotificationPreferences,
    NotificationPriority,
    NotificationRouter,
)
from qe.channels.slack import SlackAdapter
from qe.channels.telegram import TelegramAdapter
from qe.channels.webhook import WebhookAdapter

__all__ = [
    "ChannelAdapter",
    "ChannelMessage",
    "EmailAdapter",
    "NotificationPreferences",
    "NotificationPriority",
    "NotificationRouter",
    "SlackAdapter",
    "TelegramAdapter",
    "WebhookAdapter",
]
