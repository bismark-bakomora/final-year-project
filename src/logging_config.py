"""Structured logging setup for pipeline runs."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config import RunSettings


class _StageFormatter(logging.Formatter):
    """Consistent timestamped log lines."""

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "stage"):
            record.stage = "-"
        return super().format(record)


def setup_logging(
    settings: RunSettings,
    run_id: str,
    run_log_path: Path | None = None,
) -> logging.Logger:
    """
    Configure root logging with console + rotating file handlers.

    Returns the pipeline logger (``heart_disease.pipeline``).
    """
    settings.logs_dir.mkdir(parents=True, exist_ok=True)

    if run_log_path is None:
        run_log_path = settings.logs_dir / f"{run_id}.log"

    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s | "
        "[%(stage)s] %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = _StageFormatter(log_format, datefmt=date_format)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        run_log_path,
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    logger = logging.getLogger("heart_disease.pipeline")
    logger.extra = {"stage": "init"}  # type: ignore[attr-defined]
    return logger


def get_stage_logger(stage: str) -> logging.LoggerAdapter:
    """Return a logger adapter that tags every message with a pipeline stage."""
    base = logging.getLogger("heart_disease.pipeline")
    return logging.LoggerAdapter(base, {"stage": stage})
