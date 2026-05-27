"""Reload saved runs for downstream inference / explainability.

Conventions assumed (set by experiment_io.save_birdnet / run_experiments.py):

    saved_models/<Model>/<timestamp>_<dataset>/
        config.json
        run.log
        cv_summary.json
        fold_<k>/
            model.pt | model.joblib
            architecture.json
            scaler.joblib
            metrics.json
            history.json           (BIRDNet, MLP)
            bir_stats.json         (BIRDNet)
        final_8020/                (BIRDNet only, used for paper figures)
            model.pt
            architecture.json
            scaler.joblib
            metrics.json
"""

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from data_loaders import LOADERS
from model import BIRDNet
from baselines import MatchedMLP


def latest_run(model_name, dataset_name, base="saved_models"):
    parent = Path(base) / model_name
    if not parent.exists():
        raise FileNotFoundError(f"no runs under {parent}")
    runs = sorted(p for p in parent.iterdir()
                  if p.is_dir() and p.name.endswith(f"_{dataset_name}"))
    if not runs:
        raise FileNotFoundError(f"no runs for {model_name}/{dataset_name}")
    return runs[-1]


def load_birdnet(fold_dir, device="cpu"):
    fold_dir = Path(fold_dir)
    arch = json.loads((fold_dir / "architecture.json").read_text())
    model = BIRDNet.from_arch(**arch)
    state = torch.load(fold_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_matched_mlp(fold_dir, device="cpu"):
    fold_dir = Path(fold_dir)
    arch = json.loads((fold_dir / "architecture.json").read_text())
    model = MatchedMLP.from_arch(**arch)
    state = torch.load(fold_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_sklearn(fold_dir, fname="model.joblib"):
    return joblib.load(Path(fold_dir) / fname)


def load_scaler(fold_dir):
    p = Path(fold_dir) / "scaler.joblib"
    return joblib.load(p) if p.exists() else None


def reproduce_split(dataset_name, fold, n_folds=5, seed=42, data_dir="data"):
    X, y, feats, classes = LOADERS[dataset_name].load(Path(data_dir))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y))
    tr, te = splits[fold]
    return (X[tr], y[tr]), (X[te], y[te]), feats, classes


def load_run(run_dir, fold=0, device="cpu", data_dir="data"):
    """Reload a fold from a BIRDNet run directory.

    Note: features-selection (top-K F-test) used during training is NOT
    reproduced here. If n_features < total in config, the saved model
    expects the F-test-selected feature subset; you'll need to re-select
    before inference. For most Tier A datasets n_features=2000 default
    is enough that selection isn't triggered.
    """
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    fold_dir = run_dir / f"fold_{fold}"

    model = load_birdnet(fold_dir, device=device)
    scaler = load_scaler(fold_dir)
    (Xtr, ytr), (Xte, yte), feats, classes = reproduce_split(
        config["dataset_key"], fold,
        n_folds=config["n_folds"], seed=config["seed"],
        data_dir=data_dir,
    )

    if scaler is not None:
        Xtr = scaler.transform(Xtr).astype(np.float32)
        Xte = scaler.transform(Xte).astype(np.float32)

    metrics = json.loads((fold_dir / "metrics.json").read_text())
    bir_stats_path = fold_dir / "bir_stats.json"
    bir_stats = json.loads(bir_stats_path.read_text()) if bir_stats_path.exists() else None
    return {
        "model": model,
        "scaler": scaler,
        "X_train": Xtr, "y_train": ytr,
        "X_test": Xte, "y_test": yte,
        "feature_names": feats,
        "class_names": classes,
        "config": config,
        "metrics": metrics,
        "bir_stats": bir_stats,
    }


def load_final(run_dir, device="cpu"):
    """Load the final 80/20 BIRDNet used for paper figures."""
    fold_dir = Path(run_dir) / "final_8020"
    return load_birdnet(fold_dir, device=device)


def predict(model, X, device="cpu", batch_size=256):
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            logits = model(xb)
            p = torch.softmax(logits, dim=-1)
            probs.append(p.cpu().numpy())
            preds.append(p.argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", default="BIRDNet")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--run_dir", default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run(args.model, args.dataset)
    print(f"loading {run_dir} fold {args.fold}")
    out = load_run(run_dir, fold=args.fold)
    yhat, _ = predict(out["model"], out["X_test"])
    acc = (yhat == out["y_test"]).mean()
    print(f"reloaded test acc: {acc:.4f} (saved metrics: {out['metrics']})")
