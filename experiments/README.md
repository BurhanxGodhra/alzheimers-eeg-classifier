# Experiments

This folder contains configuration files and results for the three model architectures evaluated.

## Folder structure
experiments/
├── config_eegnet.yaml (optional – hyperparameters)
├── config_transformer.yaml
├── config_gat.yaml
└── results/
├── eegnet/
│ ├── best_model.pth # trained weights
│ └── confusion_matrix.png
├── transformer/
│ ├── best_model.pth
│ └── confusion_matrix.png
└── gat/
├── best_model.pth
└── confusion_matrix.png

## How to use the saved models

See the `examples/inference.py` script for loading a model and running inference on a new EEG file.

## Reproducing results

The training notebooks in `notebooks/` contain the exact code used to generate these results. Use the corresponding notebook for each architecture.
