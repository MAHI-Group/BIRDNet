#!/bin/bash

for d in tcga_rppa uci_mice_protein uci_gene_expression gse39582 metabric tcga_rnaseq; do
  echo "########## $d ##########"
  sed -n '/RESULTS:/,/MLP\/BIRDNet ratio/p' logs/${d}.log
  echo
done > logs/paper_summary.txt

sed -n '/--- Per-Class Decision Rules/,/--- Per-Instance Explanation/p' logs/metabric.log > logs/paper_rules_metabric.txt
sed -n '/--- Per-Class Decision Rules/,/--- Per-Instance Explanation/p' logs/tcga_rnaseq.log > logs/paper_rules_tcga.txt

sed -n '/Per-Instance LRP Explanation Tree/,/Note: tree truncated/p' logs/metabric.log > logs/paper_lrp_metabric.txt

