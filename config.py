"""
Central configuration for the Edge Retail Intelligence Platform MVP.

Threshold values below are taken directly from the platform specification
(Section 3.2 Predictive Queue Intelligence, Section 4.1 Core Queue Alerting
Heuristic, and Section 3.3 Class-Agnostic Shelf Void Detection).
"""
import os
from dataclasses import dataclass, field


@dataclass
class QueueConfig:
    threshold: int = 4                 # QUEUE_THRESHOLD: alert if >= N people qualify
    dwell_time_seconds: float = 60.0   # DWELL_TIME_SECONDS: filters shoppers walking past
    confirmation_window: float = 120.0 # CONFIRMATION_WINDOW: congestion must persist this long
    surge_rate: float = 0.20           # >20% inbound footfall surge also triggers an alert


@dataclass
class ShelfConfig:
    fill_ratio_warn: float = 0.30      # Fill Ratio < 0.30 -> Low-Stock Warning
    void_ratio_alert: float = 0.65     # Void Ratio > 0.65 -> restock notification
    consecutive_frames_required: int = 3
    evaluation_interval_seconds: float = 5.0  # ~0.1-0.2 FPS shelf gating from spec


@dataclass
class FootfallConfig:
    surge_window_seconds: float = 300.0  # 5-minute rolling window for surge rate calc


@dataclass
class TrackerConfig:
    backend: str = os.environ.get("TRACKER_BACKEND", "motion")  # "motion" or "yolo"
    yolo_model_path: str = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")
    confidence_threshold: float = 0.5
    max_track_age_seconds: float = 2.0   # ByteTrack-style: drop lost tracks after this long
    min_track_area: int = 900            # discard tiny blobs (noise) for the motion backend


@dataclass
class AlertConfig:
    telegram_bot_token: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.environ.get("TELEGRAM_CHAT_ID", "")
    webhook_url: str = os.environ.get("ALERT_WEBHOOK_URL", "")


@dataclass
class AppConfig:
    store_id: str = os.environ.get("STORE_ID", "IN_STORE_001")
    gateway_id: str = os.environ.get("GATEWAY_ID", "edge_node_01")
    db_path: str = os.environ.get("DB_PATH", os.path.join("data", "retail_edge.db"))
    layout_path: str = os.environ.get("STORE_LAYOUT_PATH", os.path.join("sample_config", "store_layout.json"))
    api_host: str = os.environ.get("API_HOST", "0.0.0.0")
    api_port: int = int(os.environ.get("API_PORT", "8000"))
    queue: QueueConfig = field(default_factory=QueueConfig)
    shelf: ShelfConfig = field(default_factory=ShelfConfig)
    footfall: FootfallConfig = field(default_factory=FootfallConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)


CONFIG = AppConfig()
