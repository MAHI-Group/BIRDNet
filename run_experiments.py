"""
BIRDNet experiments on GEO + Tier A datasets.

GEO datasets:
  GSE39582 - Colon cancer molecular subtypes (Marisa et al., PLoS Med 2013)
  GSE91061 - Melanoma anti-PD-1 response (Riaz et al., Cell 2017)
  GSE78220 - Melanoma anti-PD-1 response (Hugo et al., Cell 2016)
  GSE72056 - Melanoma scRNA-seq cell types (Tirosh et al., Science 2016)

Tier A datasets:
  metabric             - PAM50 subtype (Curtis et al., Nature 2012)
  tcga_rppa            - Pan-cancer RPPA (Akbani et al., Nat Commun 2014)
  tcga_rnaseq          - Pan-cancer RNA-seq (Weinstein et al., Nat Genet 2013)
  uci_gene_expression  - UCI 401 (Fiorini, UCI 2016)
  uci_mice_protein     - UCI 342 (Higuera et al., PLOS ONE 2015)

Usage:
  python run_experiments.py --dataset gse39582
  python run_experiments.py --dataset metabric
  python run_experiments.py --dataset all
"""

import argparse
import os

# Must be set before any CUDA call for cuBLAS determinism.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import time
import joblib
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from bir import binarize, compute_birs, BIR_NAMES
from model import greedy_build_birdnet, _select_birs
from train import train_birdnn, full_evaluation, EarlyStopping
from data_loaders import LOADERS
from experiment_io import (
    make_run_dir, setup_logger, save_json, save_birdnet,
    save_torch_model, save_sklearn_model, set_global_seed,
)
from baselines import MatchedMLP, count_params_dense, count_birdnet_params


DATASET_DESCRIPTIONS = {
    "gse39582": "Colon cancer molecular subtypes (Marisa et al., 2013)",
    "gse91061": "Melanoma anti-PD-1 response (Riaz et al., 2017)",
    "gse78220": "Melanoma anti-PD-1 response (Hugo et al., 2016)",
    "gse72056": "Melanoma scRNA-seq cell types (Tirosh et al., 2016)",
    "metabric": "Breast cancer PAM50 subtype (Curtis et al., 2012)",
    "tcga_rppa": "TCGA pan-cancer RPPA (Akbani et al., 2014)",
    "tcga_rnaseq": "TCGA pan-cancer RNA-seq (Weinstein et al., 2013)",
    "uci_gene_expression": "UCI 401 gene expression (Fiorini, 2016)",
    "uci_mice_protein": "UCI 342 mice protein (Higuera et al., 2015)",
}


# ------------------------------------------------------------------ #
#  MLP baseline (matched architecture)                                #
# ------------------------------------------------------------------ #

def train_matched_mlp(model, X_train, y_train, X_val, y_val, n_classes,
                      device, epochs=200):
    from torch.utils.data import DataLoader, TensorDataset
    model = model.to(device)
    criterion = (nn.BCEWithLogitsLoss() if n_classes == 2
                 else nn.CrossEntropyLoss())
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    early = EarlyStopping(patience=20)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=32, shuffle=True, drop_last=True,
    )
    best_state, best_val = None, float("inf")
    xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.long, device=device)
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            if n_classes == 2:
                loss = criterion(logits.squeeze(-1), yb.float())
            else:
                loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            logits = model(xv)
            if n_classes == 2:
                vl = criterion(logits.squeeze(-1), yv.float()).item()
            else:
                vl = criterion(logits, yv).item()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if early.step(vl):
            break
    if best_state:
        model.load_state_dict(best_state)
    return model.to(device)


def eval_torch_model(model, X_test, y_test, n_classes, device):
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X_test, dtype=torch.float32, device=device)
        logits = model(xt)
        if n_classes == 2:
            p = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            preds = (p > 0.5).astype(int)
            probs = np.stack([1 - p, p], axis=1)
        else:
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    try:
        if n_classes == 2:
            auroc = roc_auc_score(y_test, probs[:, 1])
        else:
            auroc = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")
    except ValueError:
        auroc = None
    return {"accuracy": acc, "f1_macro": f1, "auroc": auroc}


def select_features(X, y, feature_names, n_features):
    if X.shape[1] <= n_features:
        return X, feature_names
    scores, _ = f_classif(X, y)
    scores = np.nan_to_num(scores, nan=0.0)
    top_idx = np.sort(np.argsort(scores)[-n_features:])
    return X[:, top_idx], [feature_names[i] for i in top_idx]


def build_birdnet(X_train_sc, X_train_raw, y_train, n_classes,
                  n_layers, device, p_threshold, sparse_frac,
                  max_birs_per_layer, fold_idx=None, log=print):
    if fold_idx is not None:
        log(f"  [fold {fold_idx}] discovering BIRs on training set...")
    model, build_timings = greedy_build_birdnet(
        X_train=X_train_sc, y_train=y_train, n_classes=n_classes,
        n_bir_layers=n_layers,
        p_threshold=p_threshold, sparse_frac=sparse_frac,
        max_birs_per_layer=max_birs_per_layer,
        classifier_hidden=[64], activation="relu", dropout=0.3,
        device=device, verbose=False, bir_verbose=False,
        return_timings=True,
    )
    if len(model.bir_layers) > 0:
        if fold_idx is not None:
            n_units = sum(len(bl) for bl in model._layer_birs)
            log(f"  [fold {fold_idx}] BIRDNet built: {n_units} units, training...")
        return model, build_timings

    X_bin, _ = binarize(X_train_raw)
    birs, _ = compute_birs(X_bin, p_threshold=p_threshold,
                           sparse_frac=sparse_frac, verbose=False)
    primary = [(i, j, bt, pv) for i, j, bt, pv in birs if bt < 4]
    deduped = _select_birs(primary, max_birs_per_layer)
    if len(deduped) >= 5:
        model.add_bir_layer(deduped)
        model.build_classifier(hidden_dims=[64])
        model.to(device)
    return model, build_timings


# ------------------------------------------------------------------ #
#  Main experiment                                                    #
# ------------------------------------------------------------------ #

def run_experiment(X, y, feature_names, class_names, dataset_key,
                   dataset_name, n_features, n_folds, n_layers,
                   p_threshold, sparse_frac, max_birs_per_layer,
                   device, run_dirs=None, logger=None, seed=42):
    log = logger.info if logger else print
    n_classes = len(class_names)

    log(f"\n{'#'*72}")
    log(f"# {dataset_name}")
    log(f"# Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {n_classes}")
    for c, name in enumerate(class_names):
        log(f"#   {name}: {int(np.sum(y == c))}")
    log(f"# BIR params: p<{p_threshold}, sparse<{sparse_frac}, "
        f"max_birs={max_birs_per_layer}")
    log(f"{'#'*72}")

    if X.shape[1] > n_features:
        log(f"Selecting top {n_features} features by F-test...")
        X, feature_names = select_features(X, y, feature_names, n_features)
        log(f"Reduced to {X.shape[1]} features")

    config = {
        "dataset_key": dataset_key,
        "dataset_name": dataset_name,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": n_classes,
        "class_names": list(class_names),
        "n_folds": n_folds,
        "n_layers": n_layers,
        "p_threshold": p_threshold,
        "sparse_frac": sparse_frac,
        "max_birs_per_layer": max_birs_per_layer,
        "seed": seed,
    }
    if run_dirs:
        for d in run_dirs.values():
            save_json(config, Path(d) / "config.json")

    log(f"\n--- Global BIR Discovery Overview ---")
    X_bin_full, _ = binarize(X)
    birs_full, _ = compute_birs(
        X_bin_full, p_threshold=p_threshold, sparse_frac=sparse_frac,
        max_birs=max_birs_per_layer * 2, verbose=True,
    )
    primary_full = [(i, j, bt, pv) for i, j, bt, pv in birs_full if bt < 4]
    if primary_full:
        primary_full.sort(key=lambda r: r[3])
        log(f"Top 8 strongest primary BIRs (by p-value):")
        for i, j, bt, pv in primary_full[:8]:
            fi = feature_names[i] if i < len(feature_names) else f"f{i}"
            fj = feature_names[j] if j < len(feature_names) else f"f{j}"
            log(f"  p={pv:.2e}  {fi:>20s} --[{BIR_NAMES[bt]:>10s}]--> {fj}")

    log(f"\n--- {n_folds}-Fold Cross-Validation ---")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results = {"BIRDNet": [], "MLP": [], "LogReg_L1": [], "RandomForest": []}
    timings = {"BIRDNet": [], "MLP": [], "LogReg_L1": [], "RandomForest": []}
    birdnet_breakdown = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        set_global_seed(seed + fold)
        t_fold = time.time()
        X_tr_full, X_te = X[tr_idx], X[te_idx]
        y_tr_full, y_te = y[tr_idx], y[te_idx]
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_tr_full, y_tr_full, test_size=0.15,
            stratify=y_tr_full, random_state=seed,
        )
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        X_te_sc = scaler.transform(X_te)

        # ---------- BIRDNet ----------
        t0 = time.time()
        birdnet, build_t = build_birdnet(
            X_tr_sc, X_tr, y_tr, n_classes, n_layers, device,
            p_threshold, sparse_frac, max_birs_per_layer,
            fold_idx=fold + 1, log=log,
        )
        t_build = time.time() - t0

        if len(birdnet.bir_layers) == 0:
            results["BIRDNet"].append({"accuracy": 0, "f1_macro": 0, "auroc": None})
            t_train = 0.0
            n_units = 0
        else:
            t0 = time.time()
            history = train_birdnn(
                model=birdnet, X_train=X_tr_sc, y_train=y_tr,
                X_val=X_va_sc, y_val=y_va,
                epochs=200, batch_size=32, lr=1e-3, patience=20,
                device=device, verbose=False, logger=logger,
            )
            t_train = time.time() - t0
            r = full_evaluation(birdnet, X_te_sc, y_te, device=device)
            results["BIRDNet"].append(r)
            n_units = sum(len(bl) for bl in birdnet._layer_birs)
            bird_params = count_birdnet_params(birdnet)
            if run_dirs:
                bir_stats = {
                    "n_units": int(n_units),
                    "n_layers": len(birdnet._layer_birs),
                    "units_per_layer": [len(bl) for bl in birdnet._layer_birs],
                    "params_bir_active": bird_params["bir_active"],
                    "params_bir_nominal": bird_params["bir_nominal"],
                    "params_classifier": bird_params["other"],
                    "params_active": bird_params["active"],
                    "params_nominal": bird_params["nominal"],
                    "bir_sparsity": bird_params["bir_sparsity"],
                    "total_sparsity": bird_params["sparsity"],
                }
                save_birdnet(
                    Path(run_dirs["BIRDNet"]) / f"fold_{fold}",
                    model=birdnet,
                    architecture=birdnet.to_arch_dict(),
                    history=history,
                    metrics=r,
                    bir_stats=bir_stats,
                )
                joblib.dump(scaler, Path(run_dirs["BIRDNet"]) /
                            f"fold_{fold}" / "scaler.joblib")
            if device == "cuda":
                torch.cuda.empty_cache()

        timings["BIRDNet"].append(t_build + t_train)
        bn_break = {
            "fold": fold + 1, "n_units": n_units,
            "build_total": t_build, "train": t_train,
            "binarize": sum(build_t.get("binarize_per_layer", [])),
            "bir_discovery": sum(build_t.get("bir_discovery_per_layer", [])),
            "select": sum(build_t.get("select_per_layer", [])),
            "forward": sum(build_t.get("forward_per_layer", [])),
        }
        if len(birdnet.bir_layers) > 0:
            bp = count_birdnet_params(birdnet)
            bn_break["params_bir_active"] = bp["bir_active"]
            bn_break["params_bir_nominal"] = bp["bir_nominal"]
            bn_break["params_classifier"] = bp["other"]
            bn_break["params_active"] = bp["active"]
            bn_break["params_nominal"] = bp["nominal"]
            bn_break["bir_sparsity"] = bp["bir_sparsity"]
            bn_break["total_sparsity"] = bp["sparsity"]
        birdnet_breakdown.append(bn_break)

        # ---------- MLP (architecture matched to BIRDNet) ----------
        t0 = time.time()
        if len(birdnet.bir_layers) > 0:
            mlp_layer_dims = [bl.out_features for bl in birdnet.bir_layers]
            mlp_classifier_hidden = birdnet.classifier_hidden
        else:
            mlp_layer_dims = [128, 64]
            mlp_classifier_hidden = None
        mlp = MatchedMLP(
            in_features=X_tr_sc.shape[1], n_classes=n_classes,
            layer_dims=mlp_layer_dims,
            classifier_hidden=mlp_classifier_hidden,
            activation="relu", dropout=0.3, use_batchnorm=True,
        )
        mlp = train_matched_mlp(mlp, X_tr_sc, y_tr, X_va_sc, y_va,
                                n_classes, device)
        timings["MLP"].append(time.time() - t0)
        mlp_metrics = eval_torch_model(mlp, X_te_sc, y_te, n_classes, device)
        results["MLP"].append(mlp_metrics)
        if run_dirs:
            mlp_dir = Path(run_dirs["MLP"]) / f"fold_{fold}"
            mlp_dir.mkdir(parents=True, exist_ok=True)
            save_torch_model(mlp, mlp_dir / "model.pt")
            save_json(mlp.to_arch_dict(), mlp_dir / "architecture.json")
            save_json(mlp_metrics, mlp_dir / "metrics.json")
            joblib.dump(scaler, mlp_dir / "scaler.joblib")
        mlp_params = count_params_dense(mlp)
        if "params" not in birdnet_breakdown[-1]:
            birdnet_breakdown[-1]["mlp_params"] = mlp_params

        # ---------- Logistic Regression L1 ----------
        t0 = time.time()
        try:
            lr = LogisticRegression(solver="saga", l1_ratio=1.0, max_iter=5000,
                                    C=1.0, random_state=seed)
            lr.fit(X_tr_sc, y_tr)
        except Exception:
            lr = LogisticRegression(penalty="l1", solver="saga",
                                    max_iter=5000, C=1.0, random_state=seed)
            lr.fit(X_tr_sc, y_tr)
        timings["LogReg_L1"].append(time.time() - t0)
        preds = lr.predict(X_te_sc)
        probs = lr.predict_proba(X_te_sc)
        try:
            auroc = (roc_auc_score(y_te, probs[:, 1]) if n_classes == 2
                     else roc_auc_score(y_te, probs, multi_class="ovr",
                                         average="macro"))
        except ValueError:
            auroc = None
        lr_metrics = {
            "accuracy": accuracy_score(y_te, preds),
            "f1_macro": f1_score(y_te, preds, average="macro", zero_division=0),
            "auroc": auroc,
        }
        results["LogReg_L1"].append(lr_metrics)
        if run_dirs:
            lr_dir = Path(run_dirs["LogReg_L1"]) / f"fold_{fold}"
            lr_dir.mkdir(parents=True, exist_ok=True)
            save_sklearn_model(lr, lr_dir / "model.joblib")
            save_json(lr_metrics, lr_dir / "metrics.json")
            joblib.dump(scaler, lr_dir / "scaler.joblib")

        # ---------- Random Forest ----------
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        rf.fit(X_tr_sc, y_tr)
        timings["RandomForest"].append(time.time() - t0)
        preds = rf.predict(X_te_sc)
        probs = rf.predict_proba(X_te_sc)
        try:
            auroc = (roc_auc_score(y_te, probs[:, 1]) if n_classes == 2
                     else roc_auc_score(y_te, probs, multi_class="ovr",
                                         average="macro"))
        except ValueError:
            auroc = None
        rf_metrics = {
            "accuracy": accuracy_score(y_te, preds),
            "f1_macro": f1_score(y_te, preds, average="macro", zero_division=0),
            "auroc": auroc,
        }
        results["RandomForest"].append(rf_metrics)
        if run_dirs:
            rf_dir = Path(run_dirs["RandomForest"]) / f"fold_{fold}"
            rf_dir.mkdir(parents=True, exist_ok=True)
            save_sklearn_model(rf, rf_dir / "model.joblib")
            save_json(rf_metrics, rf_dir / "metrics.json")
            joblib.dump(scaler, rf_dir / "scaler.joblib")

        elapsed = time.time() - t_fold
        b = results["BIRDNet"][-1]
        log(f"  Fold {fold+1}: "
            f"BIRDNet {b['accuracy']:.3f} ({n_units} units, "
            f"build {t_build:.1f}s + train {t_train:.1f}s) | "
            f"MLP {results['MLP'][-1]['accuracy']:.3f} "
            f"({timings['MLP'][-1]:.1f}s) | "
            f"LR {results['LogReg_L1'][-1]['accuracy']:.3f} "
            f"({timings['LogReg_L1'][-1]:.1f}s) | "
            f"RF {results['RandomForest'][-1]['accuracy']:.3f} "
            f"({timings['RandomForest'][-1]:.1f}s) | "
            f"total {elapsed:.1f}s")

    log(f"\n{'='*72}")
    log(f"RESULTS: {dataset_name}")
    log(f"{'='*72}")
    log(f"{'Method':<15s} {'Accuracy':>16s} {'F1 (macro)':>16s} "
        f"{'AUROC':>16s} {'Time (s)':>12s}")
    log("-" * 72)

    summary = {}
    for method, res in results.items():
        if not res:
            continue
        accs = [r["accuracy"] for r in res]
        f1s = [r["f1_macro"] for r in res]
        aurocs = [r["auroc"] for r in res if r.get("auroc") is not None]
        ts = timings.get(method, [])
        acc_s = f"{np.mean(accs):.4f} +/- {np.std(accs):.4f}"
        f1_s = f"{np.mean(f1s):.4f} +/- {np.std(f1s):.4f}"
        auroc_s = (f"{np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}"
                   if aurocs else "N/A")
        time_s = f"{np.mean(ts):.1f}" if ts else "N/A"
        log(f"{method:<15s} {acc_s:>16s} {f1_s:>16s} "
            f"{auroc_s:>16s} {time_s:>12s}")
        summary[method] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "f1_macro_mean": float(np.mean(f1s)),
            "f1_macro_std": float(np.std(f1s)),
            "auroc_mean": float(np.mean(aurocs)) if aurocs else None,
            "auroc_std": float(np.std(aurocs)) if aurocs else None,
            "time_mean": float(np.mean(ts)) if ts else None,
            "fold_metrics": res,
        }

    if birdnet_breakdown:
        log(f"\nBIRDNet timing breakdown (mean over folds):")
        for key in ("binarize", "bir_discovery", "select", "forward",
                    "build_total", "train"):
            vals = [b[key] for b in birdnet_breakdown]
            log(f"  {key:>15s}: {np.mean(vals):.2f}s "
                f"(min {np.min(vals):.2f}, max {np.max(vals):.2f})")
        n_units = [b["n_units"] for b in birdnet_breakdown]
        log(f"  {'BIR units':>15s}: mean {np.mean(n_units):.0f}, "
            f"range {np.min(n_units)}-{np.max(n_units)}")
        active = [b.get("params_active", 0) for b in birdnet_breakdown]
        nominal = [b.get("params_nominal", 0) for b in birdnet_breakdown]
        bir_active = [b.get("params_bir_active", 0) for b in birdnet_breakdown]
        bir_nominal = [b.get("params_bir_nominal", 0) for b in birdnet_breakdown]
        classifier = [b.get("params_classifier", 0) for b in birdnet_breakdown]
        bir_sparsities = [b.get("bir_sparsity", 0.0) for b in birdnet_breakdown]
        total_sparsities = [b.get("total_sparsity", 0.0) for b in birdnet_breakdown]
        mlp_params_list = [b.get("mlp_params", 0) for b in birdnet_breakdown]
        if any(active):
            log(f"\nParameter counts (mean over folds):")
            log(f"  {'BIRDNet active (BIR layers)':>32s}: {np.mean(bir_active):,.0f}")
            log(f"  {'BIRDNet nominal (BIR layers)':>32s}: {np.mean(bir_nominal):,.0f}")
            log(f"  {'Classifier head (dense)':>32s}: {np.mean(classifier):,.0f}")
            log(f"  {'BIRDNet total active':>32s}: {np.mean(active):,.0f}")
            log(f"  {'BIRDNet total nominal':>32s}: {np.mean(nominal):,.0f}")
            log(f"  {'MatchedMLP total':>32s}: {np.mean(mlp_params_list):,.0f}")
            log(f"  {'BIR-layer sparsity':>32s}: {np.mean(bir_sparsities):.4f} "
                f"({(1-np.mean(bir_sparsities))*100:.2f}% active)")
            log(f"  {'Total sparsity':>32s}: {np.mean(total_sparsities):.4f} "
                f"({(1-np.mean(total_sparsities))*100:.2f}% active)")
            if np.mean(active) > 0:
                ratio = np.mean(mlp_params_list) / np.mean(active)
                log(f"  {'MLP/BIRDNet ratio':>32s}: {ratio:.1f}x")
        summary["BIRDNet_timings"] = birdnet_breakdown

    if run_dirs:
        for d in run_dirs.values():
            save_json(summary, Path(d) / "cv_summary.json")

    # ---------- Final model on 80/20 split for explainability ---------- #
    log(f"\n--- Final Model for Explainability (80/20 split, seed={seed}) ---")
    set_global_seed(seed)
    from sklearn.model_selection import train_test_split as tts
    from rules import per_class_rules, explain_instance, format_class_rules_text

    X_tr, X_te, y_tr, y_te = tts(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    final_model, _ = build_birdnet(
        X_tr_sc, X_tr, y_tr, n_classes, n_layers, device,
        p_threshold, sparse_frac, max_birs_per_layer,
        fold_idx="final", log=log,
    )
    if len(final_model.bir_layers) == 0:
        log("  Final model has no BIR layers; skipping rule extraction.")
        return results, timings, birdnet_breakdown, summary

    train_birdnn(
        model=final_model, X_train=X_tr_sc, y_train=y_tr,
        X_val=X_te_sc, y_val=y_te,
        epochs=200, batch_size=32, lr=1e-3, patience=20,
        device=device, verbose=False, logger=logger,
    )
    final_eval = full_evaluation(final_model, X_te_sc, y_te, device=device)
    log(f"  Final model: Acc={final_eval['accuracy']:.4f}, "
        f"F1={final_eval['f1_macro']:.4f}")

    if run_dirs:
        final_dir = Path(run_dirs["BIRDNet"]) / "final_8020"
        save_birdnet(
            final_dir, model=final_model,
            architecture=final_model.to_arch_dict(),
            metrics=final_eval,
        )
        joblib.dump(scaler, final_dir / "scaler.joblib")

    log(f"\n--- Per-Class Decision Rules (DRM-style propositionalisation) ---")
    class_rules = per_class_rules(
        final_model, X_te_sc, y_te,
        feature_names=feature_names,
        class_names=class_names,
        layer_idx=0, top_k=5, min_support=0.05,
        device=device,
    )
    log(format_class_rules_text(class_rules, max_per_class=5))

    log(f"\n--- Per-Instance Explanation (DRM-style, test sample 0) ---")
    expl = explain_instance(
        final_model, X_te_sc[0],
        feature_names=feature_names, class_names=class_names,
        device=device, top_k=5,
    )
    log(f"  Predicted: {expl['prediction']}  "
        f"(true: {class_names[y_te[0]]})")
    log(f"  Probabilities: " + ", ".join(
        f"{c}={p:.3f}" for c, p in expl["probabilities"].items()))
    log(f"  Rules fired: {expl['n_fired_rules']}/{expl['n_total_rules']}")
    log(f"  Top {len(expl['top_rules'])} contributing rules:")
    for r in expl["top_rules"]:
        fired_marker = "[FIRED]" if r["fired"] else "[      ]"
        log(f"    {fired_marker} {r['lhs']:60s}  "
            f"act={r['activation']:.3f}  "
            f"contrib={r['contribution_proxy']:.3f}")

    from lrp import build_explanation_tree, format_tree_text
    log(f"\n--- Per-Instance LRP Explanation Tree (test sample 0) ---")
    tree, pred_info = build_explanation_tree(
        final_model, X_te_sc[0],
        target_class=None,
        top_k_per_node=2,
        feature_names=feature_names,
        class_names=class_names,
        device=device,
    )
    log(f"  Target class: {pred_info['target_class']} "
        f"(true: {class_names[y_te[0]]})")
    log(format_tree_text(tree, max_depth=4))
    log(f"\n  Note: tree truncated at depth 4. Full tree available "
        f"via tree_to_dict() for offline analysis.")

    return results, timings, birdnet_breakdown, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=list(LOADERS.keys()) + ["all"])
    parser.add_argument("--n_features", type=int, default=2000)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--p_threshold", type=float, default=1e-6)
    parser.add_argument("--sparse_frac", type=float, default=0.05)
    parser.add_argument("--max_birs", type=int, default=1000)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--save_models", action="store_true", default=True,
                        help="Save trained models to saved_models/<Model>/<ts>_<dataset>/")
    parser.add_argument("--no_save_models", dest="save_models",
                        action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, PyTorch: {torch.__version__}")

    set_global_seed(args.seed)
    print(f"Global seed set to {args.seed}; CuDNN deterministic enabled.")

    datasets = (list(LOADERS.keys()) if args.dataset == "all"
                else [args.dataset])

    model_names = ("BIRDNet", "MLP", "LogReg_L1", "RandomForest")

    for ds_key in datasets:
        module = LOADERS[ds_key]
        description = DATASET_DESCRIPTIONS.get(ds_key, ds_key)
        print(f"\n\nLoading {ds_key} ({description})...")
        try:
            X, y, feat_names, class_names = module.load(Path(args.data_dir))
        except Exception as e:
            print(f"FAILED to load {ds_key}: {e}")
            import traceback
            traceback.print_exc()
            continue

        run_dirs = None
        logger = None
        if args.save_models:
            run_dirs = {m: make_run_dir(m, ds_key) for m in model_names}
            logger = setup_logger(run_dirs["BIRDNet"], name=ds_key)
            for m, d in run_dirs.items():
                if m != "BIRDNet":
                    (Path(d) / "run.log").write_text(
                        f"See: {run_dirs['BIRDNet']}/run.log\n"
                    )

        run_experiment(
            X=X, y=y, feature_names=feat_names, class_names=class_names,
            dataset_key=ds_key,
            dataset_name=f"{ds_key}: {description}",
            n_features=args.n_features,
            n_folds=args.n_folds,
            n_layers=args.n_layers,
            p_threshold=args.p_threshold,
            sparse_frac=args.sparse_frac,
            max_birs_per_layer=args.max_birs,
            device=device,
            run_dirs=run_dirs,
            logger=logger,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
