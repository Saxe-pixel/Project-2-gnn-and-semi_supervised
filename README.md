# GNN Introduction

This project provides an introduction to Graph Neural Networks (GNNs) using PyTorch and PyTorch Geometric on the dataset QM9.

## Dependencies

Core runtime libraries pulled from the codebase and Hydra configs:
- `torch` (install CPU/GPU build for your hardware) and `torchvision`
- `torch-geometric` (and its PyG extras such as `torch_geometric.graphgym`)
- `pytorch-lightning`
- `hydra-core` and `omegaconf`
- `wandb` for experiment logging
- `matplotlib` for plots in the analysis scripts
- `numpy` and `tqdm` for data handling/progress bars

## Installation

To run this project, you need to install the required Python packages. You can install them using pip:

```bash
# It is recommended to install PyTorch first, following the official instructions
# for your specific hardware (CPU or GPU with a specific CUDA version).
# See: https://pytorch.org/get-started/locally/

# For example, for a recent CUDA version:
# pip install torch torchvision torchaudio

# Or for CPU only:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# After installing PyTorch, install PyTorch Geometric.
# The exact command depends on your PyTorch and CUDA versions.
# Please refer to the PyTorch Geometric installation guide:
# https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

# Example for PyTorch 2.7 and CUDA 11.8:
# pip install torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv

# Then, install the other required packages (or just use requirements.txt):
pip install hydra-core omegaconf wandb pytorch-lightning numpy tqdm torchvision matplotlib

# Alternatively, install everything (except the hardware-specific torch/torch-geometric wheels) via:
# pip install -r requirements.txt
```

## How to Run

The main entry point for this project is `src/run.py`. It uses `hydra` for configuration management. Hydra is a broadly used and highly respected so I recommend using it. You can find a guide to it here https://medium.com/@jennytan5522/introduction-to-hydra-configuration-for-python-646e1dd4d1e9.

To run the code, execute the following command from the root of the project:

```bash
python src/run.py
```

You can override the default configuration by passing arguments from the command line. For example, to use a different model configuration:

```bash
python src/run.py model=gcn
```

The configuration files are located in the `configs/` directory.

## Improving the predictive accuracy
There are many ways to improve the GNN. Please try to get the validation error (MSE) as low as possible. I have not implemented the code to run on the test data. That is for you to do, but please wait until you have the final model.
Here are some great resources:
- Try different GNN architectures and layers see (https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html)
- Try different optimizers and schedulers
- Tune hyperparameters (especially learning rate, layers, and hidden units)
- Use advanced regularization techniques such as https://openreview.net/forum?id=xkljKdGe4E#discussion
- You can try changing the generated features of the dataloader
