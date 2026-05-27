#!/bin/bash

cd ~/dash/BI-DNN

# uci_mice_protein (small dataset); for sanity check
python run_experiments.py --dataset uci_mice_protein 2>&1 | tee logs/uci_mice_protein.log

# ~801 samples, 20K features → F-test selects 2000
python run_experiments.py --dataset uci_gene_expression 2>&1 | tee logs/uci_gene_expression.log

# ~600 samples, ~20K features → already known to work
python run_experiments.py --dataset gse39582 2>&1 | tee logs/gse39582.log

# ~2000 samples, ~25K features → 2000 selected
python run_experiments.py --dataset metabric 2>&1 | tee logs/metabric.log

# ~7000 samples, ~200 features → no selection, full feature set
python run_experiments.py --dataset tcga_rppa 2>&1 | tee logs/tcga_rppa.log

# ~11K samples, top-5000 variable genes → 2000 selected
python run_experiments.py --dataset tcga_rnaseq 2>&1 | tee logs/tcga_rnaseq.log
