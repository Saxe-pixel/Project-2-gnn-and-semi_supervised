import os
import pickle
import random
from itertools import chain
from time import time

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR


def sharpen(pseudo, tau=2.0):
    """Sharpening function to encourage more confident predictions."""
    mean = pseudo.mean(dim=0, keepdim=True)
    return mean + tau * (pseudo - mean)


def denorm(pred, mean, std):
    return pred * std + mean


def seed_everything(seed, force_deterministic):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if force_deterministic:
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# OBS (Mikkjo comment): This may actually be done in place on cfg file, just not entirely sure so I
# am returning cfg object just to be sure.
def set_debug_params(cfg):
    torch.autograd.set_detect_anomaly(True)
    # Limit number of cpu threads
    torch.set_num_threads(8)
    cfg.trainer.num_epochs = 1
    cfg.ensemble.num_models = 4
    cfg.trainer.validation_interval = 1
    cfg.logger.group_name = "debug"
    return cfg


def prepare_noise_hook(noise: float):
    def gradient_noise_hook(grad_input, noise):
        noise = torch.randn_like(grad_input) * noise
        return grad_input + noise


def prepare_optimizer(optimizer, models):
    # Assume optimizer is a hydra-configured object (already instantiated)
    # Set parameters from all models
    optimizer.param_groups = []
    params = list(chain(*[model.parameters() for model in models]))
    optimizer.add_param_group({'params': params})
    return optimizer


def prepare_scheduler(optimizer, step_size, gamma):
    return StepLR(optimizer, step_size=step_size, gamma=gamma)


# (Mikkjo comment): This validate model should maybe be included in another object.
# either a trainer or a wrapper for the models. But again, this would require large refactoring.
def validate_models(models, val_loader, device, ensemble=False, classification=False):
    for model in models:
        model.eval()
    if classification:
        correct, total = 0, 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = torch.stack([model(images) for model in models])
            outputs = torch.mean(torch.softmax(outputs, dim=0), dim=0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return 100 * correct / total
    else: 
        # regression
        for molecules, targets in val_loader:
            molecules, targets = molecules.to(device), targets.to(device)
            outputs = torch.stack([model(molecules) for model in models])
            if ensemble:
                outputs = torch.mean(outputs, dim=0)
            else:
                outputs = outputs[0]

        mse = torch.mean((outputs - targets) ** 2)
        return mse.item()


def save_results(results, log_dir):
    # Save the accuracy list
    t = str(int(time()))
    r = torch.randint(low=0, high=2**31, size=(1,)).item()
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/results_{t}_{r}.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to {log_dir}/results_{t}_{r}.pkl")
