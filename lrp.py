"""
Layer-wise Relevance Propagation for BIRDNet, with explanation-tree
extraction.

Reference:
  Bach S, Binder A, Montavon G, Klauschen F, Muller K-R, Samek W.
  "On Pixel-Wise Explanations for Non-Linear Classifier Decisions by
  Layer-Wise Relevance Propagation." PLoS ONE 10(7):e0130140, 2015.
  https://doi.org/10.1371/journal.pone.0130140

  Montavon G, Binder A, Lapuschkin S, Samek W, Muller K-R.
  "Layer-wise relevance propagation: an overview." in Explainable AI:
  Interpreting, Explaining and Visualizing Deep Learning, Springer LNCS
  vol 11700, 2019.

Why LRP fits BIRDNet:
  Each BIR unit has exactly 2 non-zero incoming weights (the two features
  participating in the implication). LRP redistributes a unit's relevance
  to its inputs proportional to their (positive) contribution. For a BIR
  unit, this means relevance flows ONLY to those 2 features. Conservation
  R_input = R_output is preserved at every layer.

This module provides:
  - lrp_relevance(model, x, target_class): per-feature relevance at every
    layer, per LRP-epsilon rule.
  - build_explanation_tree(model, x, target_class, top_k_per_node): a
    rooted tree starting at the output, branching to the top-k most
    relevant units at each layer, terminating at input genes.
  - format_tree_text(tree): pretty-print the tree.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from bir import BIR_NAMES


# -------------------------------------------------------------------- #
#  Forward pass that captures activations at every layer               #
# -------------------------------------------------------------------- #

def _forward_capture(model, x_t):
    """
    Run forward pass capturing activations at the input of each linear
    operation, plus the final logits.

    Returns:
        layers_info: list of dicts ordered from input -> output:
          { 'name', 'type' in {'bir', 'dense'},
            'input': pre-activation input (n_in,),
            'output': post-activation output (n_out,),
            'weight': (n_out, n_in) effective weight (mask*W for BIR layers),
            'bias': (n_out,) or None,
            'bir_list': triples for BIR layers, else None }
        logits: final output (n_classes,).
    """
    model.eval()
    info = []
    with torch.no_grad():
        h = x_t

        # Each BIR layer: [BIRLayer -> BN -> activation -> Dropout]
        # We capture (input, weight, bias, output_after_activation).
        for k, (bir_layer, bn, drop) in enumerate(zip(
            model.bir_layers, model.bn_layers, model.drop_layers
        )):
            inp = h.clone()
            w_eff = (bir_layer.weight * bir_layer.mask).detach()
            b_eff = (bir_layer.bias.detach()
                     if bir_layer.bias is not None else None)
            preact = bir_layer(h)
            postbn = bn(preact) if not isinstance(bn, nn.Identity) else preact
            out = model._act_fn(postbn)

            # Fold BN into the linear weights so LRP sees a single linear map
            # followed by activation. BN(y) = (y - mu)/sqrt(var+eps) * gamma + beta
            if isinstance(bn, nn.BatchNorm1d):
                gamma = bn.weight.detach()
                beta = bn.bias.detach()
                mu = bn.running_mean.detach()
                var = bn.running_var.detach()
                eps = bn.eps
                scale = gamma / torch.sqrt(var + eps)
                w_folded = w_eff * scale.unsqueeze(1)
                b_orig = b_eff if b_eff is not None else torch.zeros_like(mu)
                b_folded = scale * (b_orig - mu) + beta
            else:
                w_folded = w_eff
                b_folded = b_eff if b_eff is not None else torch.zeros(
                    w_eff.shape[0], device=w_eff.device
                )

            info.append({
                "name": f"bir_layer_{k}",
                "type": "bir",
                "input": inp.squeeze(0).cpu().numpy(),
                "output": out.squeeze(0).cpu().numpy(),
                "weight": w_folded.cpu().numpy(),
                "bias": b_folded.cpu().numpy(),
                "bir_list": list(model._layer_birs[k]),
            })
            h = out  # next layer takes the post-activation as input

        # Dropout at eval is identity, ignore.

        # Classifier: a Sequential of (Linear, ReLU, Dropout, ..., Linear)
        if model.classifier is not None:
            for m in model.classifier:
                if isinstance(m, nn.Linear):
                    inp = h.clone()
                    out_pre = m(h)
                    info.append({
                        "name": f"dense_{len([d for d in info if d['type']=='dense'])}",
                        "type": "dense",
                        "input": inp.squeeze(0).cpu().numpy(),
                        "output": out_pre.squeeze(0).cpu().numpy(),
                        "weight": m.weight.detach().cpu().numpy(),
                        "bias": (m.bias.detach().cpu().numpy()
                                 if m.bias is not None else None),
                        "bir_list": None,
                    })
                    h = out_pre
                elif isinstance(m, (nn.ReLU, nn.GELU, nn.LeakyReLU, nn.Tanh)):
                    h = m(h)
                # Skip Dropout in eval

        logits = h.squeeze(0).cpu().numpy()
    return info, logits


# -------------------------------------------------------------------- #
#  LRP-epsilon rule                                                     #
# -------------------------------------------------------------------- #

def _lrp_eps_step(R_out, layer_info, epsilon=1e-6):
    """
    Backward step for LRP-epsilon rule:

      R_i = sum_j  (a_i * w_{ji}) / (sum_k a_k * w_{jk} + b_j + eps*sign(.))  *  R_j

    This is the original LRP-eps from Bach et al. 2015.

    Args:
        R_out: (n_out,) relevance of layer outputs.
        layer_info: dict from _forward_capture for THIS layer.
        epsilon: numerical stabiliser.

    Returns:
        R_in: (n_in,) relevance of layer inputs.
        contrib_matrix: (n_out, n_in) per-edge contributions (a_i * w_ji
            normalised) so caller can rank top-k input parents per output.
    """
    a = layer_info["input"]              # (n_in,)
    w = layer_info["weight"]             # (n_out, n_in)
    b = layer_info["bias"]               # (n_out,)
    if b is None:
        b = np.zeros(w.shape[0])

    # z_j = sum_i a_i * w_{ji} + b_j  (preactivation for output j)
    z = w @ a + b                        # (n_out,)
    sign = np.where(z >= 0, 1.0, -1.0)
    z_stab = z + epsilon * sign

    # ratios (n_out,): R_j / z_stab_j
    ratio = R_out / z_stab

    # contribution of input i to output j: (a_i * w_{ji}) * ratio_j
    # We want R_i = sum_j (a_i * w_{ji}) * ratio_j
    contrib = w * a[None, :] * ratio[:, None]   # (n_out, n_in)
    R_in = contrib.sum(axis=0)
    return R_in, contrib


def lrp_relevance(model, x, target_class, epsilon=1e-6, device="cpu"):
    """
    Compute LRP-epsilon relevance scores at every layer for a single
    sample x and target class.

    Args:
        model: trained BIRDNet.
        x: (n_features,) input.
        target_class: int class index.
        epsilon: LRP stabiliser.

    Returns:
        per_layer: list of dicts (input -> output order):
          { 'name', 'type', 'R_input', 'R_output', 'contrib' }
        logits: model's output logits.
    """
    model.eval()
    x_t = torch.tensor(x.reshape(1, -1), dtype=torch.float32, device=device)
    layers, logits = _forward_capture(model, x_t)

    # Initialise relevance at the output: select target class only.
    R = np.zeros_like(logits)
    if model.n_classes == 2:
        # Binary case: classifier outputs a single logit (the "positive class")
        # If target is class 1, relevance is +logit; if class 0, -logit.
        R[0] = logits[0] if target_class == 1 else -logits[0]
    else:
        R[target_class] = logits[target_class]

    per_layer = []
    R_out = R.copy()

    for layer in reversed(layers):
        R_in, contrib = _lrp_eps_step(R_out, layer, epsilon=epsilon)
        per_layer.append({
            "name": layer["name"],
            "type": layer["type"],
            "R_input": R_in,
            "R_output": R_out.copy(),
            "contrib": contrib,
            "bir_list": layer["bir_list"],
        })
        R_out = R_in

    per_layer.reverse()  # now ordered input -> output
    return per_layer, logits


# -------------------------------------------------------------------- #
#  Explanation tree                                                     #
# -------------------------------------------------------------------- #

def build_explanation_tree(model, x, target_class=None,
                           top_k_per_node=2, epsilon=1e-6,
                           feature_names=None, class_names=None,
                           device="cpu"):
    """
    Build a rooted explanation tree for a single instance via LRP.

    The tree starts at the predicted (or target) class and branches
    backward through the network. At each node we keep the top-k input
    units by relevance contribution.

    Tree node format:
      {
        'label':      human-readable description
        'layer':      'output' / 'dense_N' / 'bir_layer_N' / 'input'
        'unit_idx':   index in that layer
        'relevance':  total relevance at this node
        'contrib':    contribution from parent (only for non-root)
        'activation': layer activation value (None for input)
        'bir':        (i, j, type) triple for BIR units, else None
        'children':   list of child nodes
      }

    Args:
        top_k_per_node: branching factor at each internal node.
            Note BIR units can only have at most 2 meaningful children
            (their two input features), so top_k_per_node > 2 has no
            effect at BIR layers.

    Returns:
        root: tree node dict.
        prediction_info: dict with predicted class and probabilities.
    """
    model.eval()
    x_t = torch.tensor(x.reshape(1, -1), dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(x_t).cpu().numpy().flatten()
    if model.n_classes == 2:
        prob_pos = float(1 / (1 + np.exp(-logits[0])))
        probs = np.array([1 - prob_pos, prob_pos])
    else:
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
    pred = int(probs.argmax())
    if target_class is None:
        target_class = pred

    per_layer, _ = lrp_relevance(model, x, target_class, epsilon=epsilon,
                                  device=device)

    cls_name = (class_names[target_class] if class_names
                else f"class_{target_class}")

    # Build the tree top-down (output to input).
    # Layer ordering returned by lrp_relevance is input->output. Reverse for
    # convenient indexing from the output.
    layers_rev = list(reversed(per_layer))  # output -> input

    # Root node: the chosen output class
    root = {
        "label": f"Predicted class: {cls_name}",
        "layer": "output",
        "unit_idx": target_class,
        "relevance": float(layers_rev[0]["R_output"][target_class]),
        "contrib": None,
        "activation": float(logits[target_class]),
        "bir": None,
        "children": [],
    }

    # Recursively expand children.
    # At each layer-info dict, contrib[j, i] is contribution of input i to
    # output j. To find children of a node sitting at output j, we take
    # the top-k input indices by contrib[j, :].
    def expand(parent_layer_idx, parent_unit, parent_node):
        """parent_layer_idx is index into layers_rev (0 = closest to output)."""
        if parent_layer_idx >= len(layers_rev):
            return
        layer = layers_rev[parent_layer_idx]
        contrib_row = layer["contrib"][parent_unit, :]   # (n_in,)
        n_in = contrib_row.shape[0]

        # For BIR layers, only 2 inputs have non-zero contribution (the
        # mask zeros the rest). top_k_per_node beyond 2 is wasted.
        k = top_k_per_node
        if layer["type"] == "bir":
            k = min(k, 2)
        # Take top-k by absolute contribution
        order = np.argsort(-np.abs(contrib_row))
        selected = [int(i) for i in order[:k] if abs(contrib_row[i]) > 1e-12]

        for child_unit in selected:
            child_relevance = float(layer["R_input"][child_unit])
            child_contrib = float(contrib_row[child_unit])
            child_input_value = float(layer["contrib"].shape[1]
                                       and 0.0)  # filled below
            # Layer just BELOW (closer to input) gives this child's activation.
            # If parent_layer_idx is the LAST entry (i.e., closest to input),
            # the child is an input feature.
            is_input = (parent_layer_idx == len(layers_rev) - 1)

            if is_input:
                fname = (feature_names[child_unit]
                         if feature_names and child_unit < len(feature_names)
                         else f"feat_{child_unit}")
                child_node = {
                    "label": f"input: {fname}",
                    "layer": "input",
                    "unit_idx": child_unit,
                    "relevance": child_relevance,
                    "contrib": child_contrib,
                    "activation": float(x[child_unit]),
                    "bir": None,
                    "children": [],
                }
            else:
                # Child is a unit in layer (parent_layer_idx + 1) of layers_rev
                # = previous BIR or dense layer
                child_layer = layers_rev[parent_layer_idx + 1]
                act_val = (float(child_layer["R_output"][child_unit]
                                  / max(abs(child_layer["R_output"][child_unit]),
                                        1e-12))
                           if False else None)
                if child_layer["type"] == "bir" and child_layer["bir_list"]:
                    i, j, bt = child_layer["bir_list"][child_unit]
                    fi = (feature_names[i] if feature_names and i < len(feature_names)
                          else f"feat_{i}")
                    fj = (feature_names[j] if feature_names and j < len(feature_names)
                          else f"feat_{j}")
                    label = f"BIR_{BIR_NAMES[bt]}({fi}, {fj})"
                    bir_meta = (i, j, bt)
                else:
                    label = f"{child_layer['name']}[{child_unit}]"
                    bir_meta = None

                child_node = {
                    "label": label,
                    "layer": child_layer["name"],
                    "unit_idx": child_unit,
                    "relevance": child_relevance,
                    "contrib": child_contrib,
                    "activation": None,
                    "bir": bir_meta,
                    "children": [],
                }
                expand(parent_layer_idx + 1, child_unit, child_node)

            parent_node["children"].append(child_node)

    expand(0, target_class, root)

    pred_info = {
        "predicted_class": (class_names[pred] if class_names else f"class_{pred}"),
        "target_class": cls_name,
        "probabilities": (
            {class_names[c]: float(probs[c]) for c in range(len(probs))}
            if class_names else {f"class_{c}": float(probs[c])
                                 for c in range(len(probs))}
        ),
    }
    return root, pred_info


def format_tree_text(node, indent=0, max_depth=None, _depth=0):
    """
    Pretty-print an explanation tree.
    """
    pad = "  " * indent
    contrib_str = (f"  contrib={node['contrib']:+.4f}"
                   if node.get("contrib") is not None else "")
    rel_str = f"R={node['relevance']:+.4f}"
    act_str = (f"  val={node['activation']:.3f}"
               if node.get("activation") is not None else "")
    line = f"{pad}- {node['label']}  [{rel_str}{contrib_str}{act_str}]"
    lines = [line]
    if max_depth is None or _depth < max_depth:
        for child in node.get("children", []):
            lines.append(format_tree_text(child, indent + 1,
                                          max_depth=max_depth,
                                          _depth=_depth + 1))
    return "\n".join(lines)


def tree_to_dict(node):
    """Convert tree to a JSON-serialisable nested dict."""
    return {
        "label": node["label"],
        "layer": node["layer"],
        "unit_idx": node["unit_idx"],
        "relevance": node["relevance"],
        "contrib": node["contrib"],
        "activation": node["activation"],
        "bir": (list(node["bir"]) if node["bir"] else None),
        "children": [tree_to_dict(c) for c in node["children"]],
    }
