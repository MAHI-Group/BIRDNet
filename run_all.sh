#!/usr/bin/env bash
# Sequential runs across all Tier A datasets.
# Order: small/fast first, big/slow last. Each dataset writes its own
# per-dataset log under logs/, and combined output goes to logs/all_runs.log.
#
# Run via:
#     nohup ./run_all.sh > logs/all_runs.log 2>&1 &
#     tail -f logs/all_runs.log
#
# Detach safely; the script will keep going.

set -u  # do NOT use -e; we want to continue past per-dataset failures

mkdir -p logs

DATASETS=(
    "tcga_rppa"
    "uci_mice_protein"
    "uci_gene_expression"
    "gse39582"
    "metabric"
    "tcga_rnaseq"
)

MAX_BIRS=5000
N_FOLDS=5

# ---------------------------------------------------------------------------
# Print dataset properties table by loading each dataset.
# After this step every dataset is cached on disk; actual runs skip download.
# ---------------------------------------------------------------------------
echo "=========================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Dataset properties"
echo "=========================================================================="

python <<'PYEOF'
from pathlib import Path
import sys
import numpy as np
from data_loaders import LOADERS

datasets = [
    "tcga_rppa", "uci_mice_protein", "uci_gene_expression",
    "gse39582", "metabric", "tcga_rnaseq",
]

print(f"\n{'Dataset':<22s} {'Samples':>10s} {'Features':>12s} {'Classes':>10s}  Class counts")
print("-" * 100)
for name in datasets:
    try:
        X, y, feats, classes = LOADERS[name].load(Path("data"))
        counts = np.bincount(y).tolist()
        counts_str = ", ".join(f"{c}:{counts[i]}" for i, c in enumerate(classes))
        if len(counts_str) > 60:
            counts_str = counts_str[:57] + "..."
        print(f"{name:<22s} {X.shape[0]:>10d} {X.shape[1]:>12d} {len(classes):>10d}  {counts_str}")
    except Exception as e:
        print(f"{name:<22s} FAILED to load: {e}", file=sys.stderr)
print()
PYEOF

# ---------------------------------------------------------------------------
# Sequential runs
# ---------------------------------------------------------------------------
for name in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: $name"
    echo "=========================================================================="
    python run_experiments.py \
        --dataset "$name" \
        --max_birs "$MAX_BIRS" \
        --n_folds "$N_FOLDS" \
        2>&1 | tee "logs/${name}.log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $name FAILED (exit $rc), continuing..." 1>&2
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $name DONE"
    fi
done

echo ""
echo "=========================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All runs complete."
echo "=========================================================================="
