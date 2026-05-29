## BIRDNet

Code for the paper:

> Tirtharaj Dash. *BIRDNet: Mining and Encoding Boolean Implication Knowledge Graphs as Interpretable Deep Neural Networks.* 2026.

BIRDNet mines Boolean Implication Relationships (BIRs) from tabular data via a sparse-exception binomial test and encodes them as the connectivity of a layered neural network. Each hidden unit corresponds to one mined 2-literal rule and binds only to its two input features, giving a sparse-by-construction model whose rules can be read directly off the trained weights.

### Conda environment

Use the conda environment supplied `env.yml` file. Note that this is my major conda environment for several projects so it might install several packages that you may not need. The standard thing would be to start with `torch`, `scikit-learn`, `numpy` etc. (just verify their versions in the `env.yml`).

### Datasets and how to run

2 proteomics datasets; 4 transcriptomics datasets (see the paper). Due to large size of the datasets, it is difficult to upload these here. We will provide them somewhere later.

To run (this will not run at the moment without data):
```bash
./run.sh                                       # all six datasets
python run_experiments.py --dataset gse39582   # single dataset
```

Each call writes to `saved_models/<Model>/<timestamp>_<dataset>/` and a per-dataset log to `logs/`. Each run contains per-fold `metrics.json`, `architecture.json`, `model.pt`, and an aggregate `cv_summary.json`.

Reload a saved run for inference or explanation:
```bash
python load_run.py --dataset gse39582 --fold 0
```
(Loading is a bit tricky due to the nature of the trained model. I got bugs sometimes. Use sparingly.)

In these experiments, we used an Ubuntu machine with a 12-core AMD processor, 64 GB RAM, and 24 GB GPU (NVIDIA RTX 4500 Ada Generation).

### Repository layout

I will update this later.

### Citation

If you find this paper useful and the repository helpful, please cite our work.
