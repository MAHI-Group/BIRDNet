"""Generate Figure 1 for the paper: 6 BIR types as quadrant scatters
plus a small fragment of a mined implication knowledge graph.

Usage:
    python scripts/make_method_figure.py \\
        --run_dir saved_models/BIRDNet/<timestamp>_<dataset> \\
        --output figures/bir_types_and_kg.pdf

Style targets R/grid-graphics quality: serif body font, light hairline
axes, no chart-junk, vector PDF output.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import joblib
import torch
import networkx as nx

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loaders import LOADERS
from model import BIRDNet
from bir import BIR_NAMES, stepmine

# Tableau 10 colorblind palette
PALETTE = {
    "T0": "#006BA4", "T1": "#FF800E", "T2": "#ABABAB", "T3": "#595959",
    "T4": "#5F9ED1", "T5": "#C85200",
}
TYPE_LABEL = {
    0: r"$T_0$: $A \to B$",
    1: r"$T_1$: $\neg A \to \neg B$",
    2: r"$T_2$: $A \to \neg B$",
    3: r"$T_3$: $\neg A \to B$",
    4: r"$T_4$: $A \leftrightarrow B$",
    5: r"$T_5$: $A \leftrightarrow \neg B$",
}
TYPE_COLOR = {0: PALETTE["T0"], 1: PALETTE["T1"], 2: PALETTE["T2"],
              3: PALETTE["T3"], 4: PALETTE["T4"], 5: PALETTE["T5"]}


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#333333",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 0.8,
        "patch.linewidth": 0.6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_run(run_dir):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    final_dir = run_dir / "final_8020"
    arch = json.loads((final_dir / "architecture.json").read_text())
    model = BIRDNet.from_arch(**arch)
    state = torch.load(final_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    scaler_path = final_dir / "scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    return model, scaler, config


def get_layer0_birs(model):
    if len(model._layer_birs) == 0:
        raise RuntimeError("model has no BIR layers")
    return model._layer_birs[0]


def find_representative_per_type(birs, target_types=(0, 1, 2, 3, 4, 5)):
    """First BIR encountered for each type. Order in birs is by
    construction-time priority (smallest p-value first after dedup),
    so the first one is also the strongest."""
    by_type = {}
    for entry in birs:
        i, j, t = entry[0], entry[1], entry[2]
        if t in target_types and t not in by_type:
            by_type[t] = (int(i), int(j), int(t))
        if len(by_type) == len(target_types):
            break
    return by_type


def plot_bir_quadrants(ax, x, y, x_thr, y_thr, t, x_name, y_name):
    color = TYPE_COLOR[t]
    rng = np.random.default_rng(0)
    jitter_x = rng.normal(0, 0.005 * (x.max() - x.min() + 1e-9), size=len(x))
    jitter_y = rng.normal(0, 0.005 * (y.max() - y.min() + 1e-9), size=len(y))
    ax.scatter(x + jitter_x, y + jitter_y,
               s=4.5, alpha=0.55, color=color,
               edgecolors="none", rasterized=True)
    ax.axvline(x_thr, color="#777777", linewidth=0.4, linestyle="--",
               zorder=0)
    ax.axhline(y_thr, color="#777777", linewidth=0.4, linestyle="--",
               zorder=0)
    ax.set_title(TYPE_LABEL[t], pad=2, loc="left")
    ax.set_xlabel(x_name, labelpad=1.5)
    ax.set_ylabel(y_name, labelpad=1.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5)


def panel_a(fig, gs, X_raw, feature_names, by_type):
    sub = gs.subgridspec(2, 3, hspace=0.55, wspace=0.45)
    for cell, t in enumerate(sorted(by_type.keys())):
        ax = fig.add_subplot(sub[cell // 3, cell % 3])
        i, j, _ = by_type[t]
        x = X_raw[:, i]
        y = X_raw[:, j]
        x_thr = stepmine(x)
        y_thr = stepmine(y)
        plot_bir_quadrants(ax, x, y, x_thr, y_thr, t,
                           feature_names[i], feature_names[j])


def panel_b(fig, gs, birs, feature_names, n_edges=14):
    ax = fig.add_subplot(gs)
    edges = [(int(b[0]), int(b[1]), int(b[2])) for b in birs[:n_edges]]
    nodes = sorted({i for i, _, _ in edges} | {j for _, j, _ in edges})
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n, label=feature_names[n])
    for i, j, t in edges:
        G.add_edge(i, j, type=t)

    pos = nx.spring_layout(G, seed=2, k=1.0 / np.sqrt(len(nodes) + 1),
                           iterations=200)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=380,
                           node_color="white",
                           edgecolors="#333333", linewidths=0.7)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={n: feature_names[n] for n in nodes},
                            font_size=7, font_family="serif")

    for t in sorted({d["type"] for _, _, d in G.edges(data=True)}):
        edgelist = [(u, v) for u, v, d in G.edges(data=True)
                    if d["type"] == t]
        style = "solid" if t in (0, 1, 4) else "dashed"
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=edgelist,
            edge_color=TYPE_COLOR[t], width=0.9,
            style=style, arrows=True, arrowsize=8,
            connectionstyle="arc3,rad=0.08",
            node_size=380,
        )

    handles = [plt.Line2D([0], [0],
                          color=TYPE_COLOR[t],
                          linestyle=("solid" if t in (0, 1, 4) else "dashed"),
                          linewidth=1.0,
                          label=TYPE_LABEL[t])
               for t in sorted({d["type"] for _, _, d in G.edges(data=True)})]
    ax.legend(handles=handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.05), ncol=3,
              handlelength=2.0, columnspacing=0.8, borderpad=0)
    ax.set_axis_off()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--output", default="figures/bir_types_and_kg.pdf")
    p.add_argument("--dataset", default=None,
                   help="override dataset key from config")
    p.add_argument("--width", type=float, default=7.16,
                   help="figure width in inches (ACM 2-col text width)")
    p.add_argument("--height", type=float, default=3.4)
    args = p.parse_args()

    setup_style()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    model, scaler, config = load_run(args.run_dir)
    dataset_key = args.dataset or config.get("dataset_key")
    if dataset_key is None:
        raise RuntimeError("dataset_key not in config; pass --dataset")

    X, y, feat_names, classes = LOADERS[dataset_key].load(Path("data"))
    if X.shape[1] != model.in_features:
        raise RuntimeError(
            f"feature dim mismatch: model expects {model.in_features}, "
            f"loaded {X.shape[1]}. Pre-selection differs; rerun the "
            f"final_8020 model with the same selected features."
        )

    birs = get_layer0_birs(model)
    by_type = find_representative_per_type(birs)
    if len(by_type) < 6:
        print(f"warning: only {len(by_type)} types found in layer-0 BIRs; "
              f"missing: {set(range(6)) - set(by_type.keys())}")

    fig = plt.figure(figsize=(args.width, args.height))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.18)
    panel_a(fig, gs[0, 0], X, feat_names, by_type)
    panel_b(fig, gs[0, 1], birs, feat_names)

    fig.text(0.005, 0.95, "(a)", fontsize=10, fontweight="bold",
             family="serif")
    fig.text(0.515, 0.95, "(b)", fontsize=10, fontweight="bold",
             family="serif")

    fig.savefig(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
