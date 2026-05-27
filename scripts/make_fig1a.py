"""Generate panel (a) of Figure 1: 6 BIR types as quadrant scatters
on simulated bimodal data. Pure pedagogical figure, no real data.

Output: figures/fig1a_bir_quadrants.pdf
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    0: "#006BA4",  # T0 high->high
    1: "#FF800E",  # T1 low->low
    2: "#A2C8EC",  # T2 high->low
    3: "#FFBC79",  # T3 low->high
    4: "#5F9ED1",  # T4 equivalent
    5: "#C85200",  # T5 opposite
}

TITLE = {
    0: r"$T_0:\ A \to B$",
    1: r"$T_1:\ \neg A \to \neg B$",
    2: r"$T_2:\ A \to \neg B$",
    3: r"$T_3:\ \neg A \to B$",
    4: r"$T_4:\ A \leftrightarrow B$",
    5: r"$T_5:\ A \leftrightarrow \neg B$",
}


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "axes.linewidth": 0.5,
        "axes.edgecolor": "#333333",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def bimodal(n, low_mean=-1.5, high_mean=1.5, sd=0.5, p_high=0.5, rng=None):
    rng = rng or np.random.default_rng(0)
    is_high = rng.random(n) < p_high
    vals = np.where(is_high,
                    rng.normal(high_mean, sd, n),
                    rng.normal(low_mean, sd, n))
    return vals, is_high


def simulate_pair_v0(t, n=320, exception_frac=0.04, rng=None):
    """Sample (a, b) pairs that exhibit BIR type t with a small exception
    fraction. Quadrants are: Q1=(H,H), Q2=(L,H), Q3=(L,L), Q4=(H,L).
    Sparse-quadrant rules:
      T0 forbids Q4   T1 forbids Q2   T2 forbids Q1   T3 forbids Q3
      T4 forbids Q2 and Q4   T5 forbids Q1 and Q3
    """
    rng = rng or np.random.default_rng(0)

    forbidden = {
        0: {(True, False)},
        1: {(False, True)},
        2: {(True, True)},
        3: {(False, False)},
        4: {(True, False), (False, True)},
        5: {(True, True), (False, False)},
    }[t]

    a_high = rng.random(n) < 0.5
    b_high = rng.random(n) < 0.5

    for k in range(n):
        if (a_high[k], b_high[k]) in forbidden and rng.random() > exception_frac:
            valid = [c for c in [(True, True), (True, False),
                                 (False, True), (False, False)]
                     if c not in forbidden]
            choice = valid[rng.integers(len(valid))]
            a_high[k], b_high[k] = choice

    a = np.where(a_high, rng.normal(1.5, 0.45, n),
                 rng.normal(-1.5, 0.45, n))
    b = np.where(b_high, rng.normal(1.5, 0.45, n),
                 rng.normal(-1.5, 0.45, n))
    return a, b


def simulate_pair(t, n=320, exception_frac=0.04, edge_frac=0.1, rng=None):
    rng = rng or np.random.default_rng(0)
    threshold = 0.5

    forbidden = {
        0: {(True, False)},
        1: {(False, True)},
        2: {(True, True)},
        3: {(False, False)},
        4: {(True, False), (False, True)},
        5: {(True, True), (False, False)},
    }[t]

    # 1. Logic for Quadrants
    a_high = rng.random(n) < 0.5
    b_high = rng.random(n) < 0.5

    for k in range(n):
        if (a_high[k], b_high[k]) in forbidden and rng.random() > exception_frac:
            valid = [c for c in [(True, True), (True, False),
                                 (False, True), (False, False)]
                     if c not in forbidden]
            choice = valid[rng.integers(len(valid))]
            a_high[k], b_high[k] = choice

    # 2. Value Generation
    def get_values(is_high):
        # Create the standard clusters at 1.5 and -1.5
        vals = np.where(is_high, 
                        rng.normal(1.5, 0.45, n), 
                        rng.normal(-1.5, 0.45, n))
        
        # 3. "Edge Injection": Pick a few indices to move to the 0.5 line
        edge_indices = rng.choice(n, size=int(n * edge_frac), replace=False)
        
        for idx in edge_indices:
            if is_high[idx]:
                # Place just ABOVE 0.5 (e.g., 0.5 to 0.65)
                vals[idx] = rng.uniform(threshold, threshold + 0.15)
            else:
                # Place just BELOW 0.5 (e.g., 0.35 to 0.5)
                vals[idx] = rng.uniform(threshold - 0.15, threshold)
        return vals

    a = get_values(a_high)
    b = get_values(b_high)

    return a, b


def plot_one(ax, t, rng):
    a, b = simulate_pair(t, n=320, exception_frac=0.04, rng=rng)
    color = PALETTE[t]
    ax.scatter(a, b, s=4.0, alpha=0.9, color=color,
               edgecolors="none", rasterized=True)
    ax.axvline(0, color="#888888", linewidth=0.4, linestyle="--", zorder=0)
    ax.axhline(0, color="#888888", linewidth=0.4, linestyle="--", zorder=0)
    ax.set_title(TITLE[t], pad=2.5, loc="left")
    ax.set_xlabel(r"$X_a$", labelpad=2)
    ax.set_ylabel(r"$X_b$", labelpad=0.5)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    setup_style()
    out = Path("figures/fig1a_bir_quadrants.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 6, figsize=(7.16, 1.45),
                             gridspec_kw={"wspace": 0.42})
    rng = np.random.default_rng(7)
    for t, ax in enumerate(axes):
        plot_one(ax, t, rng)
    fig.savefig(out)
    fig.savefig(f'{out}.png')
    print(f"wrote {out}")


if __name__ == "__main__":
    main()