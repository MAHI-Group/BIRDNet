"""
Boolean Implication Relationships (BIR) computation.

Based on:
  Sahoo et al., "Boolean implication networks derived from large scale,
  whole genome microarray datasets", Genome Biology, 2008.
  Sahoo et al., Frontiers in Physiology, 2012.

BIR Types:
  0: High->High, 1: Low->Low, 2: High->Low, 3: Low->High,
  4: Equivalent, 5: Opposite
"""

import numpy as np
from scipy import stats
from itertools import combinations
from typing import Optional
import multiprocessing as mp
import os

BIR_NAMES = {
    0: "High->High", 1: "Low->Low", 2: "High->Low",
    3: "Low->High", 4: "Equivalent", 5: "Opposite",
}


def stepmine(x: np.ndarray) -> float:
    """StepMiner: optimal step-function threshold minimising SSR."""
    xs = np.sort(x)
    n = len(xs)
    if n < 4:
        return np.median(xs)
    best_ssr = np.inf
    best_thr = xs[n // 2]
    cumsum = np.cumsum(xs)
    total = cumsum[-1]
    for k in range(1, n - 1):
        mean_lo = cumsum[k] / (k + 1)
        mean_hi = (total - cumsum[k]) / (n - k - 1)
        ssr_lo = np.sum((xs[: k + 1] - mean_lo) ** 2)
        ssr_hi = np.sum((xs[k + 1 :] - mean_hi) ** 2)
        ssr = ssr_lo + ssr_hi
        if ssr < best_ssr:
            best_ssr = ssr
            best_thr = (xs[k] + xs[k + 1]) / 2.0
    return best_thr


def binarize(X, thresholds=None):
    """Binarise feature matrix via StepMiner."""
    n_samples, n_features = X.shape
    if thresholds is None:
        thresholds = np.array([stepmine(X[:, j]) for j in range(n_features)])
    X_bin = (X > thresholds[np.newaxis, :]).astype(np.int8)
    return X_bin, thresholds


# ----------------------------------------------------------------- #
#  Vectorized BIR discovery via contingency tables                    #
# ----------------------------------------------------------------- #

def _compute_birs_vectorized_block(args):
    """
    Worker: test all BIRs for feature i against features j > i.

    Uses vectorized numpy contingency table computation.
    Returns list of (i, j, bir_type, pval) tuples.
    """
    i, X_bin, p_threshold, sparse_frac = args
    n_samples, n_features = X_bin.shape
    if i >= n_features - 1:
        return []

    a = X_bin[:, i].astype(np.int32)
    B = X_bin[:, i+1:].astype(np.int32)
    js = np.arange(i+1, n_features)

    n = n_samples
    # n11[k] = sum(a==1 & B[:,k]==1) etc.
    n11 = (a[:, None] * B).sum(axis=0)
    a_sum = a.sum()
    b_sum = B.sum(axis=0)
    n10 = a_sum - n11
    n01 = b_sum - n11
    n00 = n - a_sum - b_sum + n11

    pa = (n10 + n11) / n
    pb = (n01 + n11) / n

    # Filter out features with extreme marginals
    valid = (pa >= 0.05) & (pa <= 0.95) & (pb >= 0.05) & (pb <= 0.95)
    if not valid.any():
        return []

    results = []
    # Test each of the 4 directional implications
    # For each, compute exception count and expected probability,
    # then binomial CDF.
    sparse_count_thresh = sparse_frac * n
    test_specs = [
        (n10, pa * (1 - pb), 0),         # High->High
        (n01, (1 - pa) * pb,  1),        # Low->Low
        (n11, pa * pb,        2),        # High->Low
        (n00, (1 - pa) * (1 - pb), 3),   # Low->High
    ]

    holds_mask = {bt: np.zeros(len(js), dtype=bool) for bt in [0, 1, 2, 3]}
    pvals_arr = {bt: np.full(len(js), 1.0) for bt in [0, 1, 2, 3]}

    for exc_count, exp_prob, bt in test_specs:
        cand = valid & (exc_count <= sparse_count_thresh) & (exp_prob > 0)
        if not cand.any():
            continue
        # Binomial CDF for each candidate
        idx = np.where(cand)[0]
        for k in idx:
            pval = stats.binom.cdf(exc_count[k], n, exp_prob[k])
            if pval < p_threshold:
                holds_mask[bt][k] = True
                pvals_arr[bt][k] = pval

    # Emit BIRs
    for k in range(len(js)):
        j = int(js[k])
        for bt in [0, 1, 2, 3]:
            if holds_mask[bt][k]:
                results.append((i, j, bt, float(pvals_arr[bt][k])))
        # Composites
        if holds_mask[0][k] and holds_mask[1][k]:
            results.append((i, j, 4,
                            max(float(pvals_arr[0][k]),
                                float(pvals_arr[1][k]))))
        if holds_mask[2][k] and holds_mask[3][k]:
            results.append((i, j, 5,
                            max(float(pvals_arr[2][k]),
                                float(pvals_arr[3][k]))))
    return results


def compute_birs(X_bin, p_threshold=1e-6, sparse_frac=0.05,
                 max_pairs=None, max_birs=None, verbose=True,
                 n_workers=None):
    """
    Compute all pairwise BIRs (parallelised + vectorised).

    Args:
        X_bin: (n_samples, n_features) binary matrix.
        p_threshold: significance level (default 1e-6).
        sparse_frac: max exception fraction (default 0.05).
        max_pairs: ignored if n_workers in use; kept for compatibility.
        max_birs: hard cap on returned BIRs (kept by smallest p-value).
        verbose: print progress.
        n_workers: number of parallel processes (default: cpu_count - 1).

    Returns:
        birs: list of (i, j, bir_type, pval) tuples.
        adjacency: dict (i, j) -> list of (bir_type, pval).
    """
    n_features = X_bin.shape[1]
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 1)

    if max_pairs is not None:
        # Subsampled mode: fall back to simple loop (rarely needed now)
        return _compute_birs_subsampled(
            X_bin, p_threshold, sparse_frac, max_pairs, max_birs, verbose
        )

    # Distribute by feature index i. Each worker handles all (i, j) with j>i.
    args_list = [(i, X_bin, p_threshold, sparse_frac)
                 for i in range(n_features - 1)]

    birs = []
    adjacency = {}

    if verbose:
        print(f"  BIR discovery: {n_features} features, "
              f"{n_features*(n_features-1)//2} pairs, "
              f"{n_workers} workers")

    if n_workers == 1:
        # Single-threaded path (debugging)
        for idx, args in enumerate(args_list):
            results = _compute_birs_vectorized_block(args)
            for i, j, bt, pv in results:
                birs.append((i, j, bt, pv))
                adjacency.setdefault((i, j), []).append((bt, pv))
            if verbose and (idx + 1) % 200 == 0:
                print(f"  {idx+1}/{len(args_list)} features processed, "
                      f"{len(birs)} BIRs")
    else:
        with mp.Pool(processes=n_workers) as pool:
            done = 0
            # imap (ordered) instead of imap_unordered so worker results
            # are appended in deterministic args_list order, independent
            # of scheduler timing. Determinism cost is negligible since
            # BIR mining is CPU-bound and workers stay saturated.
            for results in pool.imap(
                _compute_birs_vectorized_block, args_list, chunksize=8
            ):
                for i, j, bt, pv in results:
                    birs.append((i, j, bt, pv))
                    adjacency.setdefault((i, j), []).append((bt, pv))
                done += 1
                if verbose and done % 200 == 0:
                    print(f"  {done}/{len(args_list)} features processed, "
                          f"{len(birs)} BIRs")

    # Deterministic sort with full tiebreak key (pval, i, j, type).
    # Required so dedup downstream (model._select_birs) is reproducible
    # across runs even when multiple BIRs share the same p-value at
    # numerical precision (common at p=0.0).
    birs.sort(key=lambda r: (r[3], r[0], r[1], r[2]))

    if verbose:
        print(f"Found {len(birs)} BIRs ({n_features} features)")
        for bt, name in BIR_NAMES.items():
            count = sum(1 for _, _, t, _ in birs if t == bt)
            if count > 0:
                print(f"  {name}: {count}")

    if max_birs is not None and len(birs) > max_birs:
        birs = birs[:max_birs]
        if verbose:
            print(f"  Capped at {max_birs} strongest BIRs by p-value")

    return birs, adjacency


def _compute_birs_subsampled(X_bin, p_threshold, sparse_frac, max_pairs,
                             max_birs, verbose):
    """Sequential subsampled path (for very large n_features)."""
    n_features = X_bin.shape[1]
    rng = np.random.default_rng(42)
    all_pairs = list(combinations(range(n_features), 2))
    idx = rng.choice(len(all_pairs), size=min(max_pairs, len(all_pairs)),
                     replace=False)
    all_pairs = [all_pairs[i] for i in idx]
    if verbose:
        print(f"  Subsampled {len(all_pairs)} pairs")

    birs = []
    adjacency = {}
    for k, (i, j) in enumerate(all_pairs):
        if verbose and k % 200000 == 0 and k > 0:
            print(f"  {k}/{len(all_pairs)} tested, {len(birs)} BIRs")
        results = test_bir(X_bin[:, i], X_bin[:, j], p_threshold, sparse_frac)
        for bt, pv in results:
            birs.append((i, j, bt, pv))
            adjacency.setdefault((i, j), []).append((bt, pv))

    if max_birs is not None and len(birs) > max_birs:
        birs.sort(key=lambda r: r[3])
        birs = birs[:max_birs]

    return birs, adjacency


def test_bir(a, b, p_threshold=1e-6, sparse_frac=0.05):
    """
    Test which BIR types hold between two binary features (single pair).

    Returns:
        list of (bir_type, pval) tuples.
    """
    a = np.asarray(a, dtype=np.int32)
    b = np.asarray(b, dtype=np.int32)
    n11 = int(((a == 1) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n00 = int(((a == 0) & (b == 0)).sum())
    total = n00 + n01 + n10 + n11
    if total < 10:
        return []
    pa = (n10 + n11) / total
    pb = (n01 + n11) / total
    if pa < 0.05 or pa > 0.95 or pb < 0.05 or pb > 0.95:
        return []

    holds = []
    pvals = {}
    tests = [
        (n10, pa * (1 - pb)),
        (n01, (1 - pa) * pb),
        (n11, pa * pb),
        (n00, (1 - pa) * (1 - pb)),
    ]
    for bir_type, (exc_count, exp_prob) in enumerate(tests):
        if exp_prob <= 0 or exc_count / total > sparse_frac:
            continue
        pval = stats.binom.cdf(exc_count, total, exp_prob)
        if pval < p_threshold:
            holds.append(bir_type)
            pvals[bir_type] = pval

    result = [(bt, pvals[bt]) for bt in holds]
    if 0 in holds and 1 in holds:
        result.append((4, max(pvals[0], pvals[1])))
    if 2 in holds and 3 in holds:
        result.append((5, max(pvals[2], pvals[3])))
    return result
