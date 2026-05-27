"""
Decision rule extraction from a trained BIRDNet.

Each BIR unit is a Boolean predicate over two input features (a discovered
implication relation). After training, we propositionalise the BIR-layer
activations and read off per-class decision rules.

This is conceptually aligned with the propositionalisation-then-explain
approach used for Deep Relational Machines (DRMs):

  Srinivasan A, Vig L, Bain M. "Logical Explanations for Deep Relational
  Machines Using Relevance Information." Journal of Machine Learning
  Research 20(130):1-47, 2019.

In a DRM, the input layer is a vector of Boolean-valued features defined
by relational background knowledge; logical/symbolic explanations are then
constructed from those features. In BIRDNet, the analogue is the BIR
layer: each unit corresponds to a discovered Boolean implication, so
propositionalising its activation gives a Boolean predicate that can be
used to build symbolic rules.

Two extraction modes:
  1) per-instance rules: which BIR units fire for sample x and how they
     contribute to the predicted class.
  2) per-class rules: globally compute the top-k BIR units most predictive
     of class c (by class-conditional firing rate vs base rate).
"""

import numpy as np
import torch
from typing import List, Dict, Optional
from bir import binarize, BIR_NAMES


def _bir_layer_activations(model, X, device="cpu"):
    """Forward pass returning per-BIR-layer activations (post-act, post-BN)."""
    model.eval()
    if len(model.bir_layers) == 0:
        return []
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32, device=device)
        outs = []
        for layer, bn in zip(model.bir_layers, model.bn_layers):
            x = layer(x)
            x = bn(x)
            x = model._act_fn(x)
            outs.append(x.cpu().numpy())
    return outs


def propositionalise(model, X, device="cpu"):
    """
    Convert continuous BIR-layer activations to Boolean predicates
    via per-unit StepMiner thresholds.

    Args:
        model: trained BIRDNet.
        X: (n, d) input. StepMiner thresholds are derived from this set.

    Returns:
        list of (n, h_l) bool arrays, one per BIR layer.
        list of (h_l,) threshold arrays, one per BIR layer.
    """
    activations = _bir_layer_activations(model, X, device=device)
    propositions = []
    thresholds_per_layer = []
    for act in activations:
        bin_act, thr = binarize(act)
        propositions.append(bin_act.astype(bool))
        thresholds_per_layer.append(thr)
    return propositions, thresholds_per_layer


def per_class_rules(model, X, y, feature_names, class_names,
                    layer_idx=0, top_k=10, min_support=0.05,
                    device="cpu"):
    """
    Extract top-k BIR predicates most associated with each class.

    A predicate P_k is scored for class c by:
        precision = P(y=c | P_k=1)
        recall    = P(P_k=1 | y=c)
        lift      = precision / P(y=c)

    We rank by lift (most class-specific firing) subject to
    a minimum support threshold (overall firing rate >= min_support).

    Args:
        layer_idx: which BIR layer to extract rules from (default 0).

    Returns:
        dict: class_name -> list of rule dicts.
            Each rule has: lhs (description), bir_type, p_class,
            p_overall, lift, n_fired, n_class_fired.
    """
    if layer_idx >= len(model.bir_layers):
        return {}

    propositions, _ = propositionalise(model, X, device=device)
    P = propositions[layer_idx]  # (n, h)
    bir_list = model._layer_birs[layer_idx]

    n = len(y)
    overall_firing = P.mean(axis=0)  # P(P_k=1) per unit

    rules = {}
    for c, cname in enumerate(class_names):
        mask_c = (y == c)
        n_c = mask_c.sum()
        if n_c == 0:
            rules[cname] = []
            continue
        p_class_base = n_c / n

        # P(P_k=1 | y=c)
        firing_in_class = P[mask_c].mean(axis=0)
        # P(y=c | P_k=1)
        n_fired = P.sum(axis=0)
        n_class_fired = P[mask_c].sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(n_fired > 0, n_class_fired / n_fired, 0.0)
            lift = np.where(p_class_base > 0,
                            precision / p_class_base, 0.0)

        # Filter by support
        valid = overall_firing >= min_support
        ranked = np.argsort(-lift)
        top_units = [k for k in ranked if valid[k]][:top_k]

        rules_c = []
        for k in top_units:
            i, j, bt = bir_list[k]
            fi = feature_names[i] if i < len(feature_names) else f"f{i}"
            fj = feature_names[j] if j < len(feature_names) else f"f{j}"
            rules_c.append({
                "unit_idx": int(k),
                "lhs": f"BIR_{BIR_NAMES[bt]}({fi}, {fj}) fires",
                "feature_i": fi, "feature_j": fj,
                "bir_type": BIR_NAMES[bt],
                "support": float(overall_firing[k]),
                "p_class_given_fired": float(precision[k]),
                "recall": float(firing_in_class[k]),
                "lift": float(lift[k]),
                "n_fired": int(n_fired[k]),
                "n_class_fired": int(n_class_fired[k]),
            })
        rules[cname] = rules_c
    return rules


def explain_instance(model, x, feature_names, class_names,
                      device="cpu", top_k=5):
    """
    Per-instance explanation: which BIR predicates fire for this sample,
    and how each contributes to the predicted class.

    For each BIR unit k in layer 0:
      - is the rule satisfied? (Boolean predicate)
      - what is its raw activation?
      - what is its weighted contribution to each class logit?
        (sum over downstream layers; here approximated by the
         classifier's first weight magnitude on unit k as a proxy)
    """
    model.eval()
    if len(model.bir_layers) == 0:
        return {"prediction": None, "fired_rules": [], "top_rules": []}

    with torch.no_grad():
        x_t = torch.tensor(x.reshape(1, -1), dtype=torch.float32, device=device)
        # BIR layer 0 raw output
        layer0 = model.bir_layers[0]
        raw0 = layer0(x_t)
        # Post-BN, post-act
        post0 = model._act_fn(model.bn_layers[0](raw0))
        post0_np = post0.cpu().numpy().flatten()

        # Forward through full model for the prediction
        logits = model(x_t).cpu().numpy().flatten()

    if model.n_classes == 2:
        prob = float(1 / (1 + np.exp(-logits[0])))
        probs = np.array([1 - prob, prob])
    else:
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
    pred = int(probs.argmax())

    # Estimate per-unit Boolean firing via post-activation > 0 threshold.
    # (Post-ReLU outputs are >= 0; >0 means non-trivially active.)
    fired = post0_np > 1e-6

    # Rule contribution proxy: |classifier first-layer weight on unit k|
    # weighted by activation
    classifier_first = None
    if model.classifier is not None:
        for m in model.classifier:
            if isinstance(m, torch.nn.Linear):
                classifier_first = m.weight.detach().cpu().numpy()
                break

    bir_list = model._layer_birs[0]
    rule_records = []
    for k, (i, j, bt) in enumerate(bir_list):
        fi = feature_names[i] if i < len(feature_names) else f"f{i}"
        fj = feature_names[j] if j < len(feature_names) else f"f{j}"
        if classifier_first is not None:
            # Single output: use absolute weight as proxy contribution
            contrib_proxy = (post0_np[k]
                             * float(np.abs(classifier_first[:, k]).sum()))
        else:
            contrib_proxy = post0_np[k]
        rule_records.append({
            "unit_idx": k,
            "lhs": f"BIR_{BIR_NAMES[bt]}({fi}, {fj}) fires",
            "feature_i": fi, "feature_j": fj,
            "value_i": float(x[i]),
            "value_j": float(x[j]),
            "bir_type": BIR_NAMES[bt],
            "fired": bool(fired[k]),
            "activation": float(post0_np[k]),
            "contribution_proxy": float(contrib_proxy),
        })

    fired_rules = [r for r in rule_records if r["fired"]]
    top_rules = sorted(rule_records,
                       key=lambda r: abs(r["contribution_proxy"]),
                       reverse=True)[:top_k]

    return {
        "prediction": class_names[pred],
        "probabilities": {class_names[c]: float(p)
                          for c, p in enumerate(probs)},
        "n_fired_rules": int(fired.sum()),
        "n_total_rules": len(bir_list),
        "fired_rules": fired_rules,
        "top_rules": top_rules,
    }


def format_class_rules_text(rules_per_class, max_per_class=5):
    """Pretty-print class rules in CRM-style text format."""
    lines = []
    for cname, rules in rules_per_class.items():
        lines.append(f"\nClass: {cname}")
        lines.append("-" * 60)
        if not rules:
            lines.append("  (no rules above support threshold)")
            continue
        for rank, r in enumerate(rules[:max_per_class], 1):
            lines.append(
                f"  R{rank}. IF {r['lhs']}\n"
                f"        THEN class={cname}  "
                f"(precision={r['p_class_given_fired']:.3f}, "
                f"recall={r['recall']:.3f}, lift={r['lift']:.2f}, "
                f"support={r['support']:.3f})"
            )
    return "\n".join(lines)
