"""
Paper-aligned pipeline runner.

Each command runs one logical experiment at a time, saves artifacts to disk,
and frees memory before the next stage — matching how the paper timed
individual optimizer runs (Section 4.2).
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np

from src.artifacts import (
    finalize_run,
    init_run,
    load_ensemble_models,
    load_keras_model,
    list_saved_models,
    model_artifact_exists,
    new_run_id,
    record_stage,
    register_artifact,
    resolve_run_dir,
    save_convergence,
    save_ensemble_models,
    save_hyperparameters,
    save_keras_model,
    ensemble_manifest_exists,
)
from src.cnn_model import (
    LOWER_BOUNDS,
    UPPER_BOUNDS,
    decode_hyperparameters,
    train_model_with_retries,
)
from src.config import RunSettings, load_settings
from src.evaluate import run_full_evaluation, evaluate_ensemble, evaluate_model
from src.fitness import (
    configure_fitness,
    configure_fitness_logging,
    reset_history,
    reset_search_checkpoints,
    set_data,
)
from src.hybrid_optimizer import HybridOptimizer
from src.logging_config import get_stage_logger, setup_logging
from src.memory_utils import cleanup_after_stage, release_model
from src.preprocess import run_preprocessing
from src.shap_analysis import run_shap_analysis
from src.smote_analysis import (
    prepare_balanced,
    prepare_double_balanced,
    run_smote_analysis,
)
from src.standalone_optimizers import (
    run_standalone_aoa,
    run_standalone_gwo,
    run_standalone_rime,
    run_standalone_woa,
)
from src.tf_config import configure_tensorflow_runtime, set_random_seeds

STANDALONE_ALGOS = {
    "gwo": ("GWO-CNN", run_standalone_gwo, "GWO"),
    "woa": ("WOA-CNN", run_standalone_woa, "WOA"),
    "aoa": ("AOA-CNN", run_standalone_aoa, "AOA"),
    "rime": ("RIME-CNN", run_standalone_rime, "RIME"),
}

COMPARE_STAGES: list[tuple[str, str, str | None]] = [
    ("NO-CNN", "baseline", None),
    ("RIME-CNN", "standalone", "rime"),
    ("AOA-CNN", "standalone", "aoa"),
    ("WOA-CNN", "standalone", "woa"),
    ("GWO-CNN", "standalone", "gwo"),
    ("GWO-WOA-AOA-CNN", "hybrid", None),
]

TABLE7_ORDER = [
    "NO-CNN",
    "RIME-CNN",
    "AOA-CNN",
    "WOA-CNN",
    "GWO-CNN",
    "GWO-WOA-AOA-CNN",
]


@dataclass
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    y_test_raw: np.ndarray
    y_train_raw: np.ndarray
    y_val_raw: np.ndarray


@dataclass
class PipelineRunner:
    settings: RunSettings
    run_id: str
    run_dir: Path
    log: Any = field(repr=False)

    @classmethod
    def create(
        cls,
        preset: str = "paper",
        run_id: str | None = None,
        mode: str = "hybrid",
        resume: bool = False,
    ) -> PipelineRunner:
        settings = load_settings(preset)

        if resume:
            rid = run_id
            if rid in (None, "latest"):
                rid = resolve_run_dir(settings.runs_dir, "latest").name
            run_dir = settings.runs_dir / rid
            if not run_dir.exists():
                raise FileNotFoundError(
                    f"Cannot resume: run directory not found at {run_dir}"
                )
            settings.logs_dir.mkdir(parents=True, exist_ok=True)
            run_log = settings.logs_dir / f"{rid}.log"
            setup_logging(settings, rid, run_log)
            log = get_stage_logger("runner")
            configure_fitness_logging(settings.fitness_eval_log_interval)
            configure_tensorflow_runtime()
            set_random_seeds(settings.random_seed)
            log.info(
                "Resuming pipeline | preset=%s mode=%s run_id=%s",
                preset,
                mode,
                rid,
            )
            return cls(settings=settings, run_id=rid, run_dir=run_dir, log=log)

        rid = run_id or new_run_id()
        settings.logs_dir.mkdir(parents=True, exist_ok=True)
        run_log = settings.logs_dir / f"{rid}.log"
        setup_logging(settings, rid, run_log)
        log = get_stage_logger("runner")
        run_dir = init_run(settings.runs_dir, rid, preset, mode)
        configure_fitness_logging(settings.fitness_eval_log_interval)
        configure_tensorflow_runtime()
        set_random_seeds(settings.random_seed)
        log.info(
            "Pipeline started | preset=%s mode=%s run_id=%s",
            preset,
            mode,
            rid,
        )
        return cls(settings=settings, run_id=rid, run_dir=run_dir, log=log)

    # ── data ──────────────────────────────────────────────

    def ensure_preprocessed(self, force: bool = False) -> None:
        with self._stage("preprocess"):
            required = [
                self.settings.processed_dir / "X_train.npy",
                self.settings.processed_dir / "y_train.npy",
            ]
            if force or not all(p.exists() for p in required):
                self.log.info("Running preprocessing pipeline")
                run_preprocessing(str(self.settings.raw_data))
            else:
                self.log.info("Processed data already present — skipping")

    def load_data(self) -> DatasetBundle:
        p = self.settings.processed_dir
        return DatasetBundle(
            X_train=np.load(p / "X_train.npy"),
            y_train=np.load(p / "y_train.npy"),
            X_val=np.load(p / "X_val.npy"),
            y_val=np.load(p / "y_val.npy"),
            X_test=np.load(p / "X_test.npy"),
            y_test=np.load(p / "y_test.npy"),
            y_test_raw=np.load(p / "y_test_raw.npy"),
            y_train_raw=np.load(p / "y_train_raw.npy"),
            y_val_raw=np.load(p / "y_val_raw.npy"),
        )

    def _bind_fitness_data(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        y_train_raw=None,
    ) -> None:
        set_data(X_train, y_train, X_val, y_val, y_train_raw)

    def _configure_fitness_runtime(self, seed: int, model_key: str) -> None:
        ckpt_dir = self.run_dir / model_key / f"search_checkpoints_seed{seed}"
        configure_fitness(
            cv_folds=self.settings.cv_folds,
            persist_checkpoints=self.settings.persist_search_checkpoints,
            validation_gap_penalty=self.settings.validation_gap_penalty,
            early_stopping_patience=self.settings.early_stopping_patience,
            ensemble_top_k=self.settings.ensemble_top_k,
            checkpoint_dir=ckpt_dir,
            fitness_seed=seed,
        )

    def _prepare_hybrid_training_data(
        self, data: DatasetBundle
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Optionally apply SMOTE to training split only (paper Section 4.5)."""
        mode = self.settings.smote_mode.lower()
        if mode in ("none", ""):
            self.log.info("Hybrid training data: original (no SMOTE)")
            y_raw = np.argmax(data.y_train, axis=1)
            return data.X_train, data.y_train, y_raw

        scaler = joblib.load(self.settings.models_dir / "scaler.pkl")
        self.log.info("Hybrid training data: SMOTE mode=%s", mode)

        if mode == "balanced":
            X_tr, y_tr, _, _, _, _, _ = prepare_balanced(
                data.X_train,
                data.y_train_raw,
                data.X_val,
                data.y_val_raw,
                data.X_test,
                data.y_test_raw,
                scaler,
            )
        elif mode == "double_balanced":
            X_tr, y_tr, _, _, _, _, _ = prepare_double_balanced(
                data.X_train,
                data.y_train_raw,
                data.X_val,
                data.y_val_raw,
                data.X_test,
                data.y_test_raw,
                scaler,
            )
        else:
            raise ValueError(
                f"Unknown smote_mode '{mode}'. Use: none, balanced, double_balanced"
            )

        y_raw = np.argmax(y_tr, axis=1)
        return X_tr, y_tr, y_raw

    # ── training stages ─────────────────────────────────────

    def run_baseline(self, data: DatasetBundle) -> str:
        """NO-CNN baseline (Table 7)."""
        key = "NO-CNN"
        with self._stage(key):
            hp = decode_hyperparameters(
                [1, 1, 1, 2, 0.3, 0.001, 2, 0, 50]
            )
            model = train_model_with_retries(
                hp,
                data.X_train,
                data.y_train,
                data.X_val,
                data.y_val,
                n_attempts=self.settings.train_attempts_standalone,
                verbose=True,
                label=key,
                patience=self.settings.early_stopping_patience,
                base_seed=self.settings.random_seed,
            )
            self._persist_model(key, model, hp, curve=None)
            release_model(model, label=key)
        return key

    def run_standalone(self, data: DatasetBundle, algorithm: str) -> str:
        """One standalone optimizer — paper Section 4.2 (30 iterations)."""
        algo = algorithm.lower()
        if algo not in STANDALONE_ALGOS:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Choose: {', '.join(STANDALONE_ALGOS)}"
            )
        model_key, runner, curve_key = STANDALONE_ALGOS[algo]
        with self._stage(model_key):
            self._configure_fitness_runtime(self.settings.random_seed, model_key)
            self._bind_fitness_data(
                data.X_train,
                data.y_train,
                data.X_val,
                data.y_val,
                data.y_train_raw,
            )
            reset_history()
            reset_search_checkpoints(clear_disk=True)
            self.log.info(
                "Optimising %s | pop=%d iter=%d",
                model_key,
                self.settings.population_size,
                self.settings.standalone_iterations,
            )
            best_pos, best_fit, curve = runner(
                self.settings.population_size,
                self.settings.standalone_iterations,
                LOWER_BOUNDS,
                UPPER_BOUNDS,
                verbose=True,
            )
            hp = decode_hyperparameters(best_pos)
            self.log.info(
                "%s best fitness=%.4f val_acc=%.2f%%",
                model_key,
                best_fit,
                (1 - best_fit) * 100,
            )
            model = train_model_with_retries(
                hp,
                data.X_train,
                data.y_train,
                data.X_val,
                data.y_val,
                n_attempts=self.settings.train_attempts_standalone,
                verbose=True,
                label=model_key,
                patience=self.settings.early_stopping_patience,
                base_seed=self.settings.random_seed,
            )
            self._persist_model(model_key, model, hp, curve, curve_key)
            release_model(model, label=model_key)
            cleanup_after_stage(model_key)
        return model_key

    def run_hybrid(self, data: DatasetBundle) -> str:
        """GWO→WOA→AOA hybrid with SMOTE, multi-seed search, and ensemble."""
        key = "GWO-WOA-AOA-CNN"
        with self._stage(key):
            X_train, y_train, y_train_raw = self._prepare_hybrid_training_data(
                data
            )
            it = self.settings.hybrid_iterations_per_stage
            seed_results: list[dict] = []

            for seed in self.settings.search_seeds:
                self.log.info(
                    "Hybrid seed=%d | pop=%d stages=%d+%d+%d | cv_folds=%d",
                    seed,
                    self.settings.population_size,
                    it,
                    it,
                    it,
                    self.settings.cv_folds,
                )
                set_random_seeds(seed)
                reset_history()
                reset_search_checkpoints(clear_disk=True)
                self._configure_fitness_runtime(seed, key)
                self._bind_fitness_data(
                    X_train, y_train, data.X_val, data.y_val, y_train_raw
                )

                hybrid = HybridOptimizer(
                    population_size=self.settings.population_size,
                    gwo_iterations=it,
                    woa_iterations=it,
                    aoa_iterations=it,
                    lower_bounds=LOWER_BOUNDS,
                    upper_bounds=UPPER_BOUNDS,
                )
                best_hp, best_fitness, curve = hybrid.optimize(
                    X_train,
                    y_train,
                    data.X_val,
                    data.y_val,
                    verbose=True,
                    y_train_raw=y_train_raw,
                )
                self.log.info(
                    "Seed %d hybrid best fitness=%.4f val_acc=%.2f%%",
                    seed,
                    best_fitness,
                    (1 - best_fitness) * 100,
                )

                model = hybrid.train_final_model(
                    X_train,
                    y_train,
                    data.X_val,
                    data.y_val,
                    verbose=True,
                    n_attempts=self.settings.train_attempts_final,
                    patience=self.settings.early_stopping_patience,
                    base_seed=seed,
                    reuse_search_best=self.settings.reuse_search_best_model,
                )
                _, val_acc = model.evaluate(
                    data.X_val, data.y_val, verbose=0
                )
                seed_results.append(
                    {
                        "seed": seed,
                        "model": model,
                        "hyperparams": best_hp,
                        "fitness": best_fitness,
                        "curve": curve,
                        "val_accuracy": float(val_acc),
                        "checkpoints": hybrid.get_ensemble_checkpoints(),
                    }
                )

            best_run = max(seed_results, key=lambda r: r["val_accuracy"])
            model = best_run["model"]
            best_hp = best_run["hyperparams"]
            curve = best_run["curve"]

            self.log.info(
                "Selected seed %d for primary model (val_acc=%.2f%%)",
                best_run["seed"],
                best_run["val_accuracy"] * 100,
            )

            self._persist_model(key, model, best_hp, curve, "GWO-WOA-AOA")

            ensemble_models: list = []
            ensemble_meta: list[dict] = []
            extra_ensemble_models: list = []
            for run in seed_results:
                ensemble_models.append(run["model"])
                ensemble_meta.append(
                    {
                        "seed": run["seed"],
                        "val_accuracy": run["val_accuracy"],
                        "source": "final_model",
                    }
                )

            for run in seed_results:
                for ckpt in run.get("checkpoints", []):
                    try:
                        from src.cnn_model import load_model_from_checkpoint

                        ens_model = load_model_from_checkpoint(
                            ckpt["hyperparams"],
                            ckpt["weights_path"],
                        )
                        ensemble_models.append(ens_model)
                        extra_ensemble_models.append(ens_model)
                        ensemble_meta.append(
                            {
                                "seed": run["seed"],
                                "val_accuracy": ckpt.get("val_accuracy"),
                                "source": "search_checkpoint",
                            }
                        )
                    except Exception as exc:
                        self.log.warning(
                            "Skipping ensemble checkpoint: %s", exc
                        )

            if len(ensemble_models) > 1:
                save_ensemble_models(
                    self.run_dir,
                    key,
                    ensemble_models,
                    metadata=ensemble_meta,
                )
                self.log.info(
                    "Saved ensemble with %d members", len(ensemble_models)
                )

            for run in seed_results:
                if run["model"] is not model:
                    release_model(run["model"], label=f"{key}-seed{run['seed']}")
            for ens_model in extra_ensemble_models:
                release_model(ens_model, label=f"{key}-ensemble-extra")
            release_model(model, label=key)
            cleanup_after_stage(key)
        return key

    def run_compare(self, data: DatasetBundle, *, resume: bool = False) -> list[str]:
        """
        Table 7 comparison — each optimizer run sequentially with cleanup.
        Paper Section 4.2: standalone 30 iter each + hybrid.

        With resume=True, skips models that already have model.keras saved.
        """
        completed: list[str] = []
        for model_key, mode, algo in COMPARE_STAGES:
            if resume and model_artifact_exists(self.run_dir, model_key):
                self.log.info(
                    "Skipping %s — artifact already exists (resume)",
                    model_key,
                )
                completed.append(model_key)
                continue

            if mode == "baseline":
                self.run_baseline(data)
            elif mode == "standalone":
                assert algo is not None
                self.run_standalone(data, algo)
            elif mode == "hybrid":
                self.run_hybrid(data)
            completed.append(model_key)
        return completed

    # ── post-training analysis (separate commands) ──────────

    def evaluate_run(self, data: DatasetBundle | None = None) -> dict:
        """Load saved models from this run and produce Table 7 outputs."""
        with self._stage("evaluate"):
            if data is None:
                data = self.load_data()
            models_dict = {}
            hp_dict = {}
            convergence_curves = {}

            for key in list_saved_models(self.run_dir):
                self.log.info("Loading saved model: %s", key)
                models_dict[key] = load_keras_model(self.run_dir, key)
                hp_path = self.run_dir / key / "hyperparameters.json"
                if hp_path.exists():
                    hp_dict[key] = json.loads(
                        hp_path.read_text(encoding="utf-8")
                    )
                conv_path = self.run_dir / key / "convergence.json"
                if conv_path.exists():
                    payload = json.loads(
                        conv_path.read_text(encoding="utf-8")
                    )
                    curve_key = key.replace("-CNN", "").replace(
                        "GWO-WOA-AOA", "GWO-WOA-AOA"
                    )
                    if key == "GWO-WOA-AOA-CNN":
                        curve_key = "GWO-WOA-AOA"
                    elif key.endswith("-CNN"):
                        curve_key = key.replace("-CNN", "")
                    convergence_curves[curve_key] = payload.get(
                        "curve", []
                    )

            ordered = {
                k: models_dict[k]
                for k in TABLE7_ORDER
                if k in models_dict
            }
            if not ordered:
                raise RuntimeError(
                    f"No models found in {self.run_dir}. Run training first."
                )

            results = run_full_evaluation(
                models_dict=ordered,
                X_test=data.X_test,
                y_test_cat=data.y_test,
                y_test_raw=data.y_test_raw,
                convergence_curves=convergence_curves or None,
                hp_dict=hp_dict or None,
            )

            hybrid_key = "GWO-WOA-AOA-CNN"
            if ensemble_manifest_exists(self.run_dir, hybrid_key):
                ensemble_models = load_ensemble_models(
                    self.run_dir, hybrid_key
                )
                if len(ensemble_models) > 1:
                    self.log.info(
                        "Evaluating %s with ensemble (%d members)",
                        hybrid_key,
                        len(ensemble_models),
                    )
                    _, _, ens_metrics = evaluate_ensemble(
                        ensemble_models,
                        data.X_test,
                        data.y_test,
                        data.y_test_raw,
                    )
                    single_acc = results.get(hybrid_key, {}).get(
                        "accuracy", 0
                    )
                    ens_acc = ens_metrics.get("accuracy", 0)
                    results[hybrid_key] = ens_metrics
                    self.log.info(
                        "Hybrid single=%.2f%% ensemble=%.2f%% (using ensemble)",
                        single_acc,
                        ens_acc,
                    )
                    for ens_model in ensemble_models:
                        release_model(ens_model, label="ensemble-member")

            out = self.settings.outputs_dir / "results" / (
                f"metrics_{self.run_id}.json"
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            serializable = {
                k: {mk: mv for mk, mv in v.items() if mk != "cm"}
                for k, v in results.items()
            }
            out.write_text(
                json.dumps(serializable, indent=2, default=str),
                encoding="utf-8",
            )
            self.log.info("Metrics saved -> %s", out)

            for model in models_dict.values():
                release_model(model)
            cleanup_after_stage("evaluate")
            return results

    def run_shap(self, data: DatasetBundle | None = None) -> None:
        """SHAP analysis on hybrid model — paper Section 4.4."""
        with self._stage("shap"):
            if data is None:
                data = self.load_data()
            key = "GWO-WOA-AOA-CNN"
            model = load_keras_model(self.run_dir, key)
            self.log.info(
                "SHAP | background=%d explain=%d",
                self.settings.shap_n_background,
                self.settings.shap_n_explain,
            )
            run_shap_analysis(
                model=model,
                X_train=data.X_train,
                X_test=data.X_test,
                n_background=self.settings.shap_n_background,
                n_explain=self.settings.shap_n_explain,
            )
            release_model(model, label=key)
            cleanup_after_stage("shap")

    def run_smote(self, data: DatasetBundle | None = None) -> None:
        """SMOTE augmentation study — paper Section 4.5."""
        with self._stage("smote"):
            if data is None:
                data = self.load_data()
            key = "GWO-WOA-AOA-CNN"
            hp_path = self.run_dir / key / "hyperparameters.json"
            if not hp_path.exists():
                raise FileNotFoundError(
                    f"Hybrid hyperparameters not found at {hp_path}"
                )
            best_hp = json.loads(hp_path.read_text(encoding="utf-8"))
            scaler = joblib.load(self.settings.models_dir / "scaler.pkl")
            run_smote_analysis(
                best_hyperparams=best_hp,
                X_train_raw=data.X_train,
                y_train_raw=data.y_train_raw,
                X_val_raw=data.X_val,
                y_val_raw=data.y_val_raw,
                X_test_raw=data.X_test,
                y_test_raw=data.y_test_raw,
                scaler=scaler,
                framingham_path=None,
                n_attempts=self.settings.smote_n_attempts,
            )
            cleanup_after_stage("smote")

    def finish(self, status: str = "completed") -> None:
        finalize_run(self.run_dir, status=status)
        self.log.info("Pipeline finished | status=%s run_id=%s", status, self.run_id)

    # ── helpers ─────────────────────────────────────────────

    def _persist_model(
        self,
        model_key: str,
        model,
        hp: dict,
        curve: list | None,
        curve_key: str | None = None,
    ) -> None:
        save_hyperparameters(self.run_dir, model_key, hp)
        if curve is not None:
            save_convergence(self.run_dir, model_key, curve)
        save_keras_model(self.run_dir, model_key, model)
        register_artifact(self.run_dir, model_key)

    @contextmanager
    def _stage(self, name: str):
        self.log = get_stage_logger(name)
        self.log.info("Stage started: %s", name)
        t0 = time.perf_counter()
        error = None
        try:
            yield
            status = "completed"
        except Exception as exc:
            error = exc
            status = "failed"
            self.log.exception("Stage failed: %s — %s", name, exc)
            record_stage(
                self.run_dir,
                name,
                status=status,
                duration_sec=time.perf_counter() - t0,
                details={"error": str(exc)},
            )
            finalize_run(self.run_dir, status="failed")
            raise
        else:
            duration = time.perf_counter() - t0
            record_stage(
                self.run_dir,
                name,
                status=status,
                duration_sec=duration,
            )
            self.log.info(
                "Stage completed: %s in %.1f s (%.1f min)",
                name,
                duration,
                duration / 60,
            )


def load_runner_for_eval(
    preset: str, run_id: str | None
) -> tuple[PipelineRunner, Path]:
    settings = load_settings(preset)
    run_dir = resolve_run_dir(settings.runs_dir, run_id)
    rid = run_dir.name
    setup_logging(settings, rid, settings.logs_dir / f"{rid}.log")
    return (
        PipelineRunner(
            settings=settings,
            run_id=rid,
            run_dir=run_dir,
            log=get_stage_logger("evaluate"),
        ),
        run_dir,
    )
