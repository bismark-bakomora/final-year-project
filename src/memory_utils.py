"""TensorFlow / Python memory management between pipeline stages."""

from __future__ import annotations

import gc
import logging

logger = logging.getLogger("heart_disease.pipeline")

_FITNESS_EVALS_SINCE_GC = 0


def clear_tf_session() -> None:
    """Release Keras/TensorFlow graph state."""
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception as exc:
        logger.debug("clear_session skipped: %s", exc)


def aggressive_memory_cleanup(*, label: str = "cleanup") -> int:
    """
    Full cleanup after a fitness evaluation or failed training.

    Returns number of objects collected by gc.
    """
    global _FITNESS_EVALS_SINCE_GC
    clear_tf_session()
    collected = gc.collect()
    _FITNESS_EVALS_SINCE_GC += 1
    if _FITNESS_EVALS_SINCE_GC % 10 == 0:
        collected += gc.collect()
        logger.debug(
            "Aggressive memory cleanup (%s); gc collected %d objects",
            label,
            collected,
        )
    return collected


def reset_fitness_gc_counter() -> None:
    global _FITNESS_EVALS_SINCE_GC
    _FITNESS_EVALS_SINCE_GC = 0


def release_model(model, *, label: str = "model") -> None:
    """Delete a Keras model and free backend memory."""
    if model is None:
        return
    try:
        del model
    except Exception:
        pass
    aggressive_memory_cleanup(label=label)


def cleanup_after_stage(stage: str) -> None:
    """Standard post-stage memory cleanup."""
    collected = aggressive_memory_cleanup(label=stage)
    logger.info(
        "Memory cleanup after '%s' (gc collected %d objects)",
        stage,
        collected,
    )
