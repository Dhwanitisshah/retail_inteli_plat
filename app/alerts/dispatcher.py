"""
Real-Time Event Broker & Alert Dispatch (Module 4).

Provides:
  - In-process pub/sub so the FastAPI WebSocket layer can push live alerts to
    the dashboard without polling the database.
  - Optional outbound channels: a generic webhook (ALERT_WEBHOOK_URL) and a
    Telegram bot (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID), matching the
    spec's "Local Floor WebSockets / Telegram Bot API" alert channel.

Both outbound channels are best-effort: a failed HTTP call is logged and
swallowed so a flaky/offline WAN link never blocks the edge pipeline
(Section 5.3 Offline Survivability & Store-and-Forward Engine).
"""
import json
import logging
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List

from config import AlertConfig

logger = logging.getLogger("alerts")

PRIORITY_P1 = "P1"  # Critical: OOS on high-velocity SKU, flash queue surge
PRIORITY_P2 = "P2"  # High: low-stock/pusher breach, predictive queue warning
PRIORITY_P3 = "P3"  # Medium: planogram drift / misplacement
PRIORITY_P4 = "P4"  # Low: traffic aggregation / hourly summaries


@dataclass
class AlertEvent:
    event_type: str
    priority: str
    message: str
    payload: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self):
        return {
            "event_type": self.event_type,
            "priority": self.priority,
            "message": self.message,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }


class AlertDispatcher:
    def __init__(self, config: AlertConfig):
        self.config = config
        self._listeners: List[Callable[[AlertEvent], None]] = []

    def register_listener(self, callback: Callable[[AlertEvent], None]):
        self._listeners.append(callback)

    def dispatch(self, event: AlertEvent):
        logger.info("[%s] %s: %s", event.priority, event.event_type, event.message)
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:
                logger.exception("alert listener failed")
        self._send_telegram(event)
        self._send_webhook(event)

    def _send_telegram(self, event: AlertEvent):
        if not (self.config.telegram_bot_token and self.config.telegram_chat_id):
            return
        url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
        data = json.dumps({
            "chat_id": self.config.telegram_chat_id,
            "text": f"[{event.priority}] {event.event_type}\n{event.message}",
        }).encode("utf-8")
        self._post(url, data)

    def _send_webhook(self, event: AlertEvent):
        if not self.config.webhook_url:
            return
        data = json.dumps(event.to_dict()).encode("utf-8")
        self._post(self.config.webhook_url, data)

    @staticmethod
    def _post(url: str, data: bytes):
        try:
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as exc:
            logger.warning("alert delivery to %s failed: %s", url, exc)
