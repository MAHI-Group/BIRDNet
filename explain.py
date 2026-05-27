"""Intrinsic explainability for BIR-DNN."""

import torch
import numpy as np
from typing import Dict, List, Optional
from bir import BIR_NAMES


def bir_feature_importance(model, layer_idx=0):
    """Aggregate absolute weights per input feature."""
    if len(model.bir_layers) == 0:
        return {}
    layer = model.bir_layers[layer_idx]
    weights = (layer.weight * layer.mask).detach().cpu().numpy()
    importance = np.abs(weights).sum(axis=0)
    if importance.max() > 0:
        importance = importance / importance.max()
    return {i: float(importance[i]) for i in range(len(importance))}


def bir_unit_importance(model, X, y, device="cpu"):
    """Gradient-based importance per BIR unit."""
    model.eval()
    model = model.to(device)
    if len(model.bir_layers) == 0:
        return []

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)

    layer = model.bir_layers[0]
    x = layer(X_t)
    x.retain_grad()
    x_act = model.bn_layers[0](x)
    x_act = model._act_fn(x_act)

    h = x_act
    for k in range(1, len(model.bir_layers)):
        h = model.bir_layers[k](h)
        h = model.bn_layers[k](h)
        h = model._act_fn(h)

    logits = model.classifier(h)
    if model.n_classes == 2:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.squeeze(-1), y_t.float()
        )
    else:
        loss = torch.nn.functional.cross_entropy(logits, y_t)
    loss.backward()

    grad = x.grad.abs().mean(dim=0).cpu().numpy()
    results = []
    for k, (i, j, bt) in enumerate(model._layer_birs[0]):
        results.append({
            "unit_idx": k,
            "feature_i": i, "feature_j": j,
            "bir_type": BIR_NAMES[bt],
            "importance": float(grad[k]),
        })
    results.sort(key=lambda r: r["importance"], reverse=True)
    return results


def extract_bir_graph(model, layer_idx=0):
    """Extract BIR graph edges for visualisation."""
    if len(model.bir_layers) == 0:
        return []
    layer = model.bir_layers[layer_idx]
    weights = (layer.weight * layer.mask).detach().cpu().numpy()
    birs = model._layer_birs[layer_idx]
    edges = []
    for k, (i, j, bt) in enumerate(birs):
        edges.append({
            "source": int(i), "target": int(j),
            "bir_type": BIR_NAMES[bt],
            "weight_source": float(weights[k, i]),
            "weight_target": float(weights[k, j]),
            "unit_idx": k,
        })
    return edges


def per_sample_explanation(model, x, feature_names=None, top_k=10, device="cpu"):
    """Top BIR units activated for a single sample."""
    model.eval()
    x_t = torch.tensor(x.reshape(1, -1), dtype=torch.float32, device=device)
    if len(model.bir_layers) == 0:
        return []

    layer = model.bir_layers[0]
    w = layer.weight * layer.mask
    activations = torch.nn.functional.linear(x_t, w, layer.bias)
    act_np = activations.detach().cpu().numpy().flatten()

    results = []
    for k, (i, j, bt) in enumerate(model._layer_birs[0]):
        fi = feature_names[i] if feature_names else f"feat_{i}"
        fj = feature_names[j] if feature_names else f"feat_{j}"
        results.append({
            "unit_idx": k,
            "feature_i": fi, "feature_j": fj,
            "value_i": float(x[i]), "value_j": float(x[j]),
            "bir_type": BIR_NAMES[bt],
            "activation": float(act_np[k]),
        })
    results.sort(key=lambda r: abs(r["activation"]), reverse=True)
    return results[:top_k]


def get_active_subnetwork(model, X, y, class_idx, top_k=50, device="cpu"):
    """Class-specific BIR subnetwork from differential mean activation."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    if len(model.bir_layers) == 0:
        return []

    layer = model.bir_layers[0]
    w = layer.weight * layer.mask
    activations = torch.nn.functional.linear(X_t, w, layer.bias)
    act_np = activations.detach().cpu().numpy()

    mask_class = (y == class_idx)
    mean_class = act_np[mask_class].mean(axis=0)
    mean_other = act_np[~mask_class].mean(axis=0)
    diff = mean_class - mean_other

    results = []
    for k, (i, j, bt) in enumerate(model._layer_birs[0]):
        results.append({
            "unit_idx": k,
            "feature_i": int(i), "feature_j": int(j),
            "bir_type": BIR_NAMES[bt],
            "class_mean_activation": float(mean_class[k]),
            "other_mean_activation": float(mean_other[k]),
            "differential": float(diff[k]),
        })
    results.sort(key=lambda r: abs(r["differential"]), reverse=True)
    return results[:top_k]
