"""Training pipeline for BIR-DNN."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Dict, Tuple
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def step(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def _emit(logger, verbose, msg):
    if logger is not None:
        logger.info(msg)
    elif verbose:
        print(msg)


def train_birdnn(model, X_train, y_train, X_val=None, y_val=None,
                 epochs=100, batch_size=64, lr=1e-3, weight_decay=1e-4,
                 patience=15, device="cpu", verbose=True, logger=None):
    """Train BIR-DNN end-to-end with early stopping."""
    model = model.to(device)
    n_classes = model.n_classes

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(X_t, y_t),
                              batch_size=batch_size, shuffle=True,
                              drop_last=True)

    if X_val is not None:
        X_v = torch.tensor(X_val, dtype=torch.float32)
        y_v = torch.tensor(y_val, dtype=torch.long)
        val_loader = DataLoader(TensorDataset(X_v, y_v),
                                batch_size=batch_size, shuffle=False)
    else:
        val_loader = None

    criterion = (nn.BCEWithLogitsLoss() if n_classes == 2
                 else nn.CrossEntropyLoss())
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    early_stop = EarlyStopping(patience=patience)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_state = None
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            if n_classes == 2:
                loss = criterion(logits.squeeze(-1), yb.float())
            else:
                loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        history["train_loss"].append(avg_loss)

        if val_loader is not None:
            val_loss, val_acc, val_f1 = evaluate(
                model, val_loader, criterion, n_classes, device
            )
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            if (epoch + 1) % 10 == 0:
                _emit(logger, verbose,
                      f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train: {avg_loss:.4f} | Val: {val_loss:.4f} | "
                      f"Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
            if early_stop.step(val_loss):
                _emit(logger, verbose, f"Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
    return history


def evaluate(model, loader, criterion, n_classes, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            if n_classes == 2:
                loss = criterion(logits.squeeze(-1), yb.float())
                preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).long()
            else:
                loss = criterion(logits, yb)
                preds = logits.argmax(dim=-1)
            total_loss += loss.item()
            n_batches += 1
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    avg_loss = total_loss / max(n_batches, 1)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1


def predict(model, X, device="cpu", batch_size=256):
    model.eval()
    model = model.to(device)
    n_classes = model.n_classes
    X_t = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)
    all_probs = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb)
            if n_classes == 2:
                p = torch.sigmoid(logits.squeeze(-1))
                probs = torch.stack([1 - p, p], dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu())
    all_probs = torch.cat(all_probs, dim=0).numpy()
    preds = all_probs.argmax(axis=-1)
    return preds, all_probs


def full_evaluation(model, X_test, y_test, device="cpu"):
    preds, probs = predict(model, X_test, device)
    results = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, preds, average="weighted",
                                 zero_division=0),
        "report": classification_report(y_test, preds, zero_division=0),
    }
    n_classes = model.n_classes
    try:
        if n_classes == 2:
            results["auroc"] = roc_auc_score(y_test, probs[:, 1])
        else:
            results["auroc"] = roc_auc_score(
                y_test, probs, multi_class="ovr", average="macro"
            )
    except ValueError:
        results["auroc"] = None
    return results
