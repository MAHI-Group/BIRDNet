"""
Demo: BIRDNet on UCI / sklearn datasets (Breast Cancer, Wine).
Quick sanity check before running on GEO data.
"""

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from bir import binarize, compute_birs, BIR_NAMES
from model import BIRDNet, greedy_build_birdnet, _select_birs
from train import train_birdnn, full_evaluation


def _inject_raw_birs(model, X_raw, n_classes, device,
                     p_threshold=1e-4, sparse_frac=0.10,
                     max_birs=2000):
    """Fallback: discover BIRs on raw data and inject if greedy yielded none."""
    X_bin, _ = binarize(X_raw)
    birs, _ = compute_birs(X_bin, p_threshold=p_threshold,
                           sparse_frac=sparse_frac, verbose=False)
    primary = [(i, j, bt, pv) for i, j, bt, pv in birs if bt < 4]
    deduped = _select_birs(primary, max_birs)
    if len(deduped) < 5:
        return False
    model.add_bir_layer(deduped)
    model.build_classifier(hidden_dims=[32])
    model.to(device)
    return True


def run_dataset(X, y, feature_names, class_names, dataset_name, device="cpu"):
    n_classes = len(class_names)
    print(f"\n{'='*70}\nDataset: {dataset_name}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {n_classes}")
    for c, name in enumerate(class_names):
        print(f"  {name}: {np.sum(y == c)}")
    print('='*70)

    # Looser thresholds for these small UCI datasets (n=178-569, p=13-30)
    p_thr, sparse = 1e-4, 0.10

    X_bin, _ = binarize(X)
    birs_all, _ = compute_birs(X_bin, p_threshold=p_thr,
                               sparse_frac=sparse, verbose=True)
    primary = [(i, j, bt, pv) for i, j, bt, pv in birs_all if bt < 4]
    if not primary:
        print("No BIRs found.")
        return
    primary.sort(key=lambda r: r[3])
    print(f"\nTop 8 strongest BIRs:")
    for i, j, bt, pv in primary[:8]:
        print(f"  p={pv:.2e}  {feature_names[i]} --[{BIR_NAMES[bt]}]--> "
              f"{feature_names[j]}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        X_tr_full, X_te = X[tr], X[te]
        y_tr_full, y_te = y[tr], y[te]
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_tr_full, y_tr_full, test_size=0.15,
            stratify=y_tr_full, random_state=42
        )
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        X_te_sc = scaler.transform(X_te)

        model = greedy_build_birdnet(
            X_train=X_tr_sc, y_train=y_tr, n_classes=n_classes,
            n_bir_layers=2, p_threshold=p_thr, sparse_frac=sparse,
            max_birs_per_layer=2000,
            classifier_hidden=[32], device=device, verbose=False,
        )
        if len(model.bir_layers) == 0:
            if not _inject_raw_birs(model, X_tr, n_classes, device,
                                    p_threshold=p_thr, sparse_frac=sparse):
                continue

        train_birdnn(model, X_tr_sc, y_tr, X_va_sc, y_va,
                     epochs=200, batch_size=32, lr=1e-3, patience=20,
                     device=device, verbose=False)
        r = full_evaluation(model, X_te_sc, y_te, device=device)
        fold_results.append(r)
        n_units = sum(len(bl) for bl in model._layer_birs)
        auroc_s = f"{r['auroc']:.4f}" if r.get("auroc") else "N/A"
        print(f"  Fold {fold+1}: Acc={r['accuracy']:.4f}, "
              f"F1={r['f1_macro']:.4f}, AUROC={auroc_s}, "
              f"BIR units={n_units}")

    if fold_results:
        accs = [r["accuracy"] for r in fold_results]
        f1s = [r["f1_macro"] for r in fold_results]
        print(f"\nAccuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        print(f"F1 macro: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    bc = load_breast_cancer()
    run_dataset(bc.data, bc.target, list(bc.feature_names),
                list(bc.target_names),
                "Breast Cancer Wisconsin (Diagnostic)", device)
    wine = load_wine()
    run_dataset(wine.data, wine.target, list(wine.feature_names),
                [str(c) for c in wine.target_names],
                "Wine (3-class)", device)


if __name__ == "__main__":
    main()
