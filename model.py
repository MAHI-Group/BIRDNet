"""BIRDNet: Boolean Implication Relationship-Derived Deep Network."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from bir import binarize, compute_birs, BIR_NAMES


class BIRLayer(nn.Module):
    """Sparse linear layer with connectivity defined by BIRs.

    Each output unit corresponds to one Boolean implication and connects
    only to the two input features participating in that implication.
    A binary mask enforces sparsity throughout training.
    """

    def __init__(self, in_features, bir_list, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = len(bir_list)
        self.bir_list = bir_list

        self.weight = nn.Parameter(torch.zeros(self.out_features, in_features))
        self.bias = (nn.Parameter(torch.zeros(self.out_features))
                     if bias else None)

        mask = torch.zeros(self.out_features, in_features)
        for k, (i, j, bt) in enumerate(bir_list):
            mask[k, i] = 1.0
            mask[k, j] = 1.0
        self.register_buffer("mask", mask)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            nn.init.xavier_uniform_(self.weight)
            for k, (i, j, bt) in enumerate(self.bir_list):
                if bt == 0:
                    self.weight[k, i] = 0.5; self.weight[k, j] = 0.5
                elif bt == 1:
                    self.weight[k, i] = -0.5; self.weight[k, j] = -0.5
                elif bt == 2:
                    self.weight[k, i] = 0.5; self.weight[k, j] = -0.5
                elif bt == 3:
                    self.weight[k, i] = -0.5; self.weight[k, j] = 0.5
                elif bt == 4:
                    self.weight[k, i] = 0.5; self.weight[k, j] = 0.5
                elif bt == 5:
                    self.weight[k, i] = 0.5; self.weight[k, j] = -0.5
            self.weight.mul_(self.mask)

    def forward(self, x):
        w = self.weight * self.mask
        return F.linear(x, w, self.bias)


_ACTIVATIONS = {
    "relu": nn.ReLU, "gelu": nn.GELU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1), "tanh": nn.Tanh,
}


class BIRDNet(nn.Module):
    """BIRDNet: deep network with BIR-defined sparse layers."""

    def __init__(self, in_features, n_classes, activation="relu",
                 dropout=0.3, use_batchnorm=True):
        super().__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.activation_name = activation
        self.dropout_rate = dropout
        self.use_batchnorm = use_batchnorm

        self.bir_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.drop_layers = nn.ModuleList()
        self.classifier = None
        self.classifier_hidden = None

        self._act_fn = _ACTIVATIONS[activation]()
        self._current_dim = in_features
        self._layer_birs = []

    def add_bir_layer(self, bir_list):
        layer = BIRLayer(self._current_dim, bir_list)
        self.bir_layers.append(layer)
        self.bn_layers.append(
            nn.BatchNorm1d(layer.out_features) if self.use_batchnorm
            else nn.Identity()
        )
        self.drop_layers.append(nn.Dropout(self.dropout_rate))
        self._layer_birs.append(bir_list)
        self._current_dim = layer.out_features

    def build_classifier(self, hidden_dims=None):
        self.classifier_hidden = list(hidden_dims) if hidden_dims else None
        layers = []
        dim = self._current_dim
        if hidden_dims:
            for hd in hidden_dims:
                layers.extend([nn.Linear(dim, hd), nn.ReLU(),
                               nn.Dropout(self.dropout_rate)])
                dim = hd
        layers.append(nn.Linear(dim, 1 if self.n_classes == 2
                                else self.n_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        for bir_layer, bn, drop in zip(self.bir_layers, self.bn_layers,
                                        self.drop_layers):
            x = bir_layer(x)
            x = bn(x)
            x = self._act_fn(x)
            x = drop(x)
        if self.classifier is not None:
            x = self.classifier(x)
        return x

    def get_bir_activations(self, x, layer_idx):
        with torch.no_grad():
            x = self.bir_layers[layer_idx](x)
            x = self.bn_layers[layer_idx](x)
            x = self._act_fn(x)
        return x

    def summary(self):
        print(f"BIRDNet: {self.in_features} -> {self.n_classes} classes")
        print(f"  BIR Layers: {len(self.bir_layers)}")
        for k, layer in enumerate(self.bir_layers):
            print(f"    Layer {k}: {layer.in_features} -> "
                  f"{layer.out_features} ({len(self._layer_birs[k])} BIRs)")
        total = sum(p.numel() for p in self.parameters())
        print(f"  Total params: {total}")

    def to_arch_dict(self):
        """Serializable architecture spec for round-trip via from_arch()."""
        return {
            "in_features": self.in_features,
            "n_classes": self.n_classes,
            "activation": self.activation_name,
            "dropout": self.dropout_rate,
            "use_batchnorm": self.use_batchnorm,
            "bir_lists": [
                [[int(i), int(j), int(bt)] for (i, j, bt) in bl]
                for bl in self._layer_birs
            ],
            "classifier_hidden": self.classifier_hidden,
        }

    @classmethod
    def from_arch(cls, in_features, n_classes, bir_lists,
                  classifier_hidden=None, activation="relu",
                  dropout=0.3, use_batchnorm=True):
        """Reconstruct an empty BIRDNet from to_arch_dict() output.

        After construction call load_state_dict() to restore weights and
        mask buffers from torch.save(state_dict).
        """
        model = cls(in_features=in_features, n_classes=n_classes,
                    activation=activation, dropout=dropout,
                    use_batchnorm=use_batchnorm)
        for bl in bir_lists:
            model.add_bir_layer([tuple(b) for b in bl])
        model.build_classifier(hidden_dims=classifier_hidden)
        return model


def _select_birs(birs_with_pval, max_birs):
    # Sort input deterministically (pval, i, j, type) so dedup and
    # truncation are reproducible across runs even when multiple BIRs
    # share the same p-value at numerical precision.
    sorted_birs = sorted(birs_with_pval,
                         key=lambda r: (r[3], r[0], r[1], r[2]))
    by_pair = {}
    for i, j, bt, pval in sorted_birs:
        key = (min(i, j), max(i, j))
        if key not in by_pair:  # first (== lowest pval) wins
            by_pair[key] = (i, j, bt, pval)
    deduped = list(by_pair.values())  # already in pval order from sort above
    if max_birs is not None and len(deduped) > max_birs:
        deduped = deduped[:max_birs]
    return [(i, j, bt) for i, j, bt, _ in deduped]


def greedy_build_birdnet(X_train, y_train, n_classes, n_bir_layers=2,
                         p_threshold=1e-6, sparse_frac=0.05,
                         max_pairs=None, max_birs_per_layer=5000,
                         min_birs=10, classifier_hidden=None,
                         activation="relu", dropout=0.3,
                         device="cpu", verbose=True,
                         bir_verbose=None, return_timings=False):
    import time
    if bir_verbose is None:
        bir_verbose = verbose

    timings = {"binarize_per_layer": [], "bir_discovery_per_layer": [],
               "select_per_layer": [], "forward_per_layer": []}

    n_features = X_train.shape[1]
    model = BIRDNet(in_features=n_features, n_classes=n_classes,
                    activation=activation, dropout=dropout)

    current_repr = X_train.copy()
    for layer_idx in range(n_bir_layers):
        if verbose:
            print(f"\n--- BIR Layer {layer_idx} ---")
            print(f"  Input dim: {current_repr.shape[1]}")

        t0 = time.time()
        X_bin, _ = binarize(current_repr)
        timings["binarize_per_layer"].append(time.time() - t0)

        t0 = time.time()
        birs, _ = compute_birs(
            X_bin, p_threshold=p_threshold, sparse_frac=sparse_frac,
            max_pairs=max_pairs, max_birs=None, verbose=bir_verbose,
        )
        timings["bir_discovery_per_layer"].append(time.time() - t0)

        primary = [(i, j, bt, pval) for i, j, bt, pval in birs if bt < 4]

        if len(primary) < min_birs:
            if verbose:
                print(f"  Only {len(primary)} primary BIRs (< {min_birs}). "
                      f"Stopping.")
            break

        t0 = time.time()
        deduped = _select_birs(primary, max_birs_per_layer)
        timings["select_per_layer"].append(time.time() - t0)
        if verbose:
            print(f"  Using {len(deduped)} BIR units "
                  f"(deduplicated, capped at {max_birs_per_layer})")
        if len(deduped) < min_birs:
            break

        model.add_bir_layer(deduped)

        t0 = time.time()
        X_tensor = torch.tensor(current_repr, dtype=torch.float32).to(device)
        model = model.to(device)
        activations = model.get_bir_activations(X_tensor, layer_idx)
        current_repr = activations.cpu().numpy()
        timings["forward_per_layer"].append(time.time() - t0)

    model.build_classifier(hidden_dims=classifier_hidden)
    model = model.to(device)
    if verbose:
        model.summary()

    if return_timings:
        return model, timings
    return model


BIRDNN = BIRDNet
greedy_build_birdnn = greedy_build_birdnet
