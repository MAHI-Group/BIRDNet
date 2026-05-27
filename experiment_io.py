"""IO utilities for BIRDNet experiments.

Layout:
    saved_models/<Model>/<timestamp>_<dataset>/
        config.json
        run.log
        summary.json
        fold_<k>/
            model.pt | model.joblib
            history.json
            metrics.json
            bir_stats.json   (BIRDNet only)
"""

import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def make_run_dir(model_name, dataset_name, base="saved_models"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / model_name / f"{ts}_{dataset_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logger(run_dir, name="run"):
    log_path = Path(run_dir) / "run.log"
    logger = logging.getLogger(f"{name}.{run_dir}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def save_torch_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def save_sklearn_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_fold(run_dir, fold, model=None, history=None, metrics=None,
              extras=None, framework="torch"):
    fold_dir = Path(run_dir) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    if model is not None:
        if framework == "torch":
            save_torch_model(model, fold_dir / "model.pt")
        else:
            save_sklearn_model(model, fold_dir / "model.joblib")
    if history is not None:
        save_json(history, fold_dir / "history.json")
    if metrics is not None:
        save_json(metrics, fold_dir / "metrics.json")
    if extras:
        for k, v in extras.items():
            save_json(v, fold_dir / f"{k}.json")


def save_birdnet(fold_dir, model, architecture, history=None, metrics=None,
                 bir_stats=None):
    """Save BIRDNet weights + architecture spec needed for reconstruction.

    architecture: dict of constructor kwargs for BIRDNN. Must be sufficient
    for `BIRDNN(**architecture)` to instantiate an empty network of the same
    topology, after which load_state_dict() restores weights and mask buffers.
    """
    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    save_torch_model(model, fold_dir / "model.pt")
    save_json(architecture, fold_dir / "architecture.json")
    if history is not None:
        save_json(history, fold_dir / "history.json")
    if metrics is not None:
        save_json(metrics, fold_dir / "metrics.json")
    if bir_stats is not None:
        save_json(bir_stats, fold_dir / "bir_stats.json")


MODEL_NAMES = ("BIRDNet", "MLP", "LogRegL1", "RandomForest")


def make_run_dirs(dataset_name, models=MODEL_NAMES, base="saved_models"):
    return {m: make_run_dir(m, dataset_name, base=base) for m in models}


def set_global_seed(seed, deterministic_cudnn=True):
    """Seed every RNG that could affect training reproducibility.

    Seeds: Python `random`, NumPy, PyTorch (CPU and CUDA), PYTHONHASHSEED.
    When deterministic_cudnn=True, additionally forces CuDNN into
    deterministic mode and enables PyTorch's deterministic-algorithms
    flag. Costs ~10-20% on GPU but produces bitwise-identical training
    across runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cuBLAS deterministic mode requires this env var (PyTorch docs).
    # Must be set before the first CUDA call; safe to set repeatedly.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)


def torch_generator(seed):
    """A seeded torch.Generator for use with DataLoader(generator=...)."""
    if torch is None:
        return None
    g = torch.Generator()
    g.manual_seed(seed)
    return g
