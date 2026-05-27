"""Baselines and parameter accounting.

MatchedMLP: dense counterpart of BIRDNet — same depth, same per-layer
widths, same BN/activation/dropout, same classifier head. Only the BIR
mask is removed (full connectivity in each linear layer). This is the
principled control for connectivity vs topology.

build_random_birdnet: ablation that keeps BIRDNet's at-most-two-
incoming-edges topology but samples edges uniformly at random instead
of mining them from data. Isolates the contribution of the BIR-derived
prior from the structural sparsity constraint alone.
"""

import numpy as np
import torch
import torch.nn as nn


_ACTIVATIONS = {
    "relu": nn.ReLU, "gelu": nn.GELU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1), "tanh": nn.Tanh,
}


class MatchedMLP(nn.Module):
    def __init__(self, in_features, n_classes, layer_dims,
                 classifier_hidden=None, activation="relu",
                 dropout=0.3, use_batchnorm=True):
        super().__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.layer_dims = list(layer_dims)
        self.classifier_hidden = (list(classifier_hidden)
                                  if classifier_hidden else None)
        self.dropout_rate = dropout
        self.use_batchnorm = use_batchnorm
        self.activation_name = activation

        act_cls = _ACTIVATIONS[activation]
        layers = []
        dim = in_features
        for h in self.layer_dims:
            layers.append(nn.Linear(dim, h))
            layers.append(nn.BatchNorm1d(h) if use_batchnorm else nn.Identity())
            layers.append(act_cls())
            layers.append(nn.Dropout(dropout))
            dim = h
        if self.classifier_hidden:
            for hd in self.classifier_hidden:
                layers.append(nn.Linear(dim, hd))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                dim = hd
        out_dim = 1 if n_classes == 2 else n_classes
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def to_arch_dict(self):
        return {
            "in_features": self.in_features,
            "n_classes": self.n_classes,
            "layer_dims": self.layer_dims,
            "classifier_hidden": self.classifier_hidden,
            "activation": self.activation_name,
            "dropout": self.dropout_rate,
            "use_batchnorm": self.use_batchnorm,
        }

    @classmethod
    def from_arch(cls, **kwargs):
        return cls(**kwargs)


def count_params_dense(model):
    """Total trainable parameters (no mask accounting)."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def count_birdnet_params(model):
    """Effective vs nominal parameter count for BIRDNet.

    Returns BIR-layer-only counts (the meaningful sparsity from masks)
    and aggregate counts including the dense classifier head.
    'Nominal' = same architecture without the BIR mask (== MatchedMLP).
    """
    bir_active = 0
    bir_nominal = 0
    bir_layer_names = set()
    for k, bir_layer in enumerate(model.bir_layers):
        bir_layer_names.add(f"bir_layers.{k}")
        bir_nominal += bir_layer.weight.numel()
        bir_active += int(bir_layer.mask.sum().item())
        if bir_layer.bias is not None:
            b = bir_layer.bias.numel()
            bir_nominal += b
            bir_active += b

    other = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        prefix = ".".join(name.split(".")[:2])
        if prefix in bir_layer_names:
            continue
        other += p.numel()

    return {
        "bir_active": int(bir_active),
        "bir_nominal": int(bir_nominal),
        "other": int(other),
        "active": int(bir_active + other),
        "nominal": int(bir_nominal + other),
        "bir_sparsity": float(1.0 - bir_active / max(bir_nominal, 1)),
        "sparsity": float(1.0 - (bir_active + other) / max(bir_nominal + other, 1)),
    }


def build_random_birdnet(in_features, n_classes,
                         units_per_layer,
                         classifier_hidden=None,
                         activation="relu", dropout=0.3,
                         use_batchnorm=True,
                         device="cpu", seed=42, verbose=False):
    """Build a BIRDNet with random degree-2 connectivity (ablation).

    Identical to greedy_build_birdnet in every architectural detail
    except the source of edges: pairs are sampled uniformly without
    replacement; types in {0,1,2,3} are sampled uniformly. The fixed
    seed makes the construction reproducible.

    units_per_layer: list of ints giving the number of units in each
    BIR layer. Must match the corresponding BIRDNet's layer widths
    exactly so that capacity is controlled.
    """
    from model import BIRDNet
    rng = np.random.default_rng(seed)
    model = BIRDNet(in_features=in_features, n_classes=n_classes,
                    activation=activation, dropout=dropout,
                    use_batchnorm=use_batchnorm)

    current_dim = in_features
    for layer_idx, h_target in enumerate(units_per_layer):
        max_pairs = current_dim * (current_dim - 1) // 2
        h = min(h_target, max_pairs)
        if h < 10:
            break

        seen = set()
        i_arr = np.zeros(h, dtype=np.int64)
        j_arr = np.zeros(h, dtype=np.int64)
        filled = 0
        while filled < h:
            need = h - filled
            cand_i = rng.integers(0, current_dim, size=need * 2)
            cand_j = rng.integers(0, current_dim, size=need * 2)
            for ci, cj in zip(cand_i, cand_j):
                if ci == cj:
                    continue
                a, b = (int(ci), int(cj)) if ci < cj else (int(cj), int(ci))
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                i_arr[filled], j_arr[filled] = a, b
                filled += 1
                if filled == h:
                    break

        types = rng.integers(0, 4, size=h)
        bir_list = [(int(i_arr[p]), int(j_arr[p]), int(types[p]))
                    for p in range(h)]

        model.add_bir_layer(bir_list)
        current_dim = h
        if verbose:
            print(f"  random layer {layer_idx}: {h} units")

    model.build_classifier(hidden_dims=classifier_hidden)
    model = model.to(device)
    return model
