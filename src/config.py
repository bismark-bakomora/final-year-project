"""Load pipeline configuration from config.yaml with preset overrides."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class RunSettings:
    """Resolved settings for a single pipeline run."""

    preset: str
    population_size: int
    hybrid_iterations_per_stage: int
    standalone_iterations: int
    train_attempts_final: int
    train_attempts_standalone: int
    early_stopping_patience: int
    random_seed: int
    search_seeds: tuple[int, ...]
    cv_folds: int
    ensemble_top_k: int
    persist_search_checkpoints: bool
    smote_mode: str
    validation_gap_penalty: float
    reuse_search_best_model: bool
    raw_data: Path
    processed_dir: Path
    models_dir: Path
    outputs_dir: Path
    runs_dir: Path
    logs_dir: Path
    shap_n_background: int
    shap_n_explain: int
    smote_n_attempts: int
    log_level: str
    log_max_bytes: int
    log_backup_count: int
    fitness_eval_log_interval: int
    gc_after_fitness: bool


def _load_yaml() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration not found: {CONFIG_PATH}")
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_settings(preset: str = "paper") -> RunSettings:
    """Return merged settings for the given preset (paper | quick)."""
    cfg = _load_yaml()
    if preset not in cfg:
        raise ValueError(f"Unknown preset '{preset}'. Use: paper, quick")

    preset_cfg = copy.deepcopy(cfg[preset])
    paths = cfg["paths"]
    shap = cfg.get("shap", {})
    smote = cfg.get("smote", {})
    logging_cfg = cfg.get("logging", {})
    memory = cfg.get("memory", {})
    training = cfg.get("training", {})

    root = PROJECT_ROOT
    search_seeds = tuple(int(s) for s in training.get("search_seeds", [42]))
    cv_folds = int(training.get("cv_folds", 0))
    ensemble_top_k = int(training.get("ensemble_top_k", 3))
    smote_mode = str(training.get("smote_mode", "none"))

    # Faster iteration when using the quick preset
    if preset == "quick":
        search_seeds = (42,)
        cv_folds = 0
        ensemble_top_k = 2
        smote_mode = "balanced"

    return RunSettings(
        preset=preset,
        population_size=int(preset_cfg["population_size"]),
        hybrid_iterations_per_stage=int(
            preset_cfg["hybrid_iterations_per_stage"]
        ),
        standalone_iterations=int(preset_cfg["standalone_iterations"]),
        train_attempts_final=int(preset_cfg["train_attempts_final"]),
        train_attempts_standalone=int(preset_cfg["train_attempts_standalone"]),
        early_stopping_patience=int(preset_cfg["early_stopping_patience"]),
        random_seed=int(training.get("random_seed", 42)),
        search_seeds=search_seeds,
        cv_folds=cv_folds,
        ensemble_top_k=ensemble_top_k,
        persist_search_checkpoints=bool(
            training.get("persist_search_checkpoints", True)
        ),
        smote_mode=smote_mode,
        validation_gap_penalty=float(
            training.get("validation_gap_penalty", 0.0)
        ),
        reuse_search_best_model=bool(
            training.get("reuse_search_best_model", True)
        ),
        raw_data=root / paths["raw_data"],
        processed_dir=root / paths["processed_dir"],
        models_dir=root / paths["models_dir"],
        outputs_dir=root / paths["outputs_dir"],
        runs_dir=root / paths["runs_dir"],
        logs_dir=root / paths["logs_dir"],
        shap_n_background=int(shap.get("n_background", 50)),
        shap_n_explain=int(shap.get("n_explain", 100)),
        smote_n_attempts=int(smote.get("n_attempts", 3)),
        log_level=str(logging_cfg.get("level", "INFO")),
        log_max_bytes=int(logging_cfg.get("max_bytes", 10_485_760)),
        log_backup_count=int(logging_cfg.get("backup_count", 5)),
        fitness_eval_log_interval=int(
            memory.get("fitness_eval_log_interval", 25)
        ),
        gc_after_fitness=bool(memory.get("gc_after_fitness", True)),
    )
