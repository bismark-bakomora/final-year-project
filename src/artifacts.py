"""Persist models, hyperparameters, and run manifests to disk."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("heart_disease.pipeline")

MANIFEST_FILE = "manifest.json"
LATEST_POINTER = "latest_run.txt"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def resolve_run_dir(runs_dir: Path, run_id: str | None) -> Path:
    if run_id in (None, "latest"):
        pointer = runs_dir / LATEST_POINTER
        if not pointer.exists():
            raise FileNotFoundError(
                f"No runs found. Expected pointer at {pointer}"
            )
        run_id = pointer.read_text(encoding="utf-8").strip()
    return runs_dir / run_id


def init_run(runs_dir: Path, run_id: str, preset: str, mode: str) -> Path:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "preset": preset,
        "mode": mode,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "stages": {},
        "artifacts": {},
    }
    _write_json(run_dir / MANIFEST_FILE, manifest)
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / LATEST_POINTER).write_text(run_id, encoding="utf-8")
    logger.info("Run directory: %s", run_dir)
    return run_dir


def update_manifest(run_dir: Path, **updates: Any) -> None:
    path = run_dir / MANIFEST_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(updates)
    _write_json(path, data)


def record_stage(
    run_dir: Path,
    stage: str,
    *,
    status: str,
    duration_sec: float,
    details: dict[str, Any] | None = None,
) -> None:
    path = run_dir / MANIFEST_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    entry = {
        "status": status,
        "duration_sec": round(duration_sec, 2),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if details:
        entry["details"] = details
    data.setdefault("stages", {})[stage] = entry
    _write_json(path, data)


def save_hyperparameters(run_dir: Path, model_key: str, hp: dict) -> Path:
    model_dir = run_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    out = model_dir / "hyperparameters.json"
    _write_json(out, hp)
    return out


def save_convergence(run_dir: Path, model_key: str, curve: list) -> Path:
    model_dir = run_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    out = model_dir / "convergence.json"
    _write_json(out, {"curve": curve})
    return out


def save_keras_model(run_dir: Path, model_key: str, model) -> Path:
    if model is None:
        raise ValueError(f"Cannot save null model for {model_key}")
    model_dir = run_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    out = model_dir / "model.keras"
    model.save(out)
    logger.info("Saved model -> %s", out)
    return out


def load_keras_model(run_dir: Path, model_key: str):
    import tensorflow as tf
    path = run_dir / model_key / "model.keras"
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}")
    return tf.keras.models.load_model(path)


def model_artifact_exists(run_dir: Path, model_key: str) -> bool:
    """True if a completed model was saved for this stage."""
    return (run_dir / model_key / "model.keras").exists()


def list_saved_models(run_dir: Path) -> list[str]:
    return sorted(
        p.name
        for p in run_dir.iterdir()
        if p.is_dir() and (p / "model.keras").exists()
    )


def register_artifact(run_dir: Path, model_key: str) -> None:
    path = run_dir / MANIFEST_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    artifacts = data.setdefault("artifacts", {})
    artifacts[model_key] = str(run_dir / model_key)
    _write_json(path, data)


def finalize_run(run_dir: Path, status: str = "completed") -> None:
    update_manifest(
        run_dir,
        status=status,
        finished_at=datetime.now(timezone.utc).isoformat(),
    )


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, default=_json_default),
        encoding="utf-8",
    )
