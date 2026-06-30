"""
TensorFlow environment and runtime tuning for long CPU training runs.

Must call configure_tensorflow_env() before TensorFlow is first imported.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("heart_disease.pipeline")

_CONFIGURED_ENV = False
_CONFIGURED_RUNTIME = False


def configure_tensorflow_env(
    *,
    disable_onednn: bool = True,
    log_level: str = "2",
) -> None:
    """
    Set process environment variables for TensorFlow.

    Disabling oneDNN avoids MKL 'could not create a memory object' crashes
  on Windows after hundreds of sequential CNN trainings.
    """
    global _CONFIGURED_ENV
    if _CONFIGURED_ENV:
        return

    if disable_onednn:
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", log_level)
    # Reduce allocator fragmentation on long runs
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

    _CONFIGURED_ENV = True
    logger.debug(
        "TensorFlow env: TF_ENABLE_ONEDNN_OPTS=%s",
        os.environ.get("TF_ENABLE_ONEDNN_OPTS", "unset"),
    )


def configure_tensorflow_runtime(
    *,
    intra_op_threads: int = 2,
    inter_op_threads: int = 2,
) -> None:
    """Limit TF thread pools to lower peak RAM during Conv2D training."""
    global _CONFIGURED_RUNTIME
    if _CONFIGURED_RUNTIME:
        return

    import tensorflow as tf

    try:
        tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
    except Exception as exc:
        logger.debug("TF threading config skipped: %s", exc)

    _CONFIGURED_RUNTIME = True
    logger.info(
        "TensorFlow runtime: intra_op=%d inter_op=%d",
        intra_op_threads,
        inter_op_threads,
    )
