from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from augmentations import augment_batch
from utils import denorm, sharpen


class SemiSupervisedEnsemble:
    """
    Simple supervised ensemble trainer used as baseline.
    """

    def __init__(
        self,
        supervised_criterion,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule,
    ):
        self.device = device
        self.models = [m.to(device) for m in models]

        self.supervised_criterion = supervised_criterion
        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)

        self.datamodule = datamodule
        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()

        self.logger = logger

    def validate(self):
        for model in self.models:
            model.eval()

        val_losses = []

        y_mean = self.datamodule.y_mean.to(self.device)
        y_std = self.datamodule.y_std.to(self.device)

        with torch.no_grad():
            for batch, targets in self.val_dataloader:
                batch, targets = batch.to(self.device), targets.to(self.device)

                preds = [model(batch) for model in self.models]
                avg_preds = torch.stack(preds).mean(0)

                avg_preds_denorm = denorm(avg_preds, y_mean, y_std)
                targets_denorm = denorm(targets, y_mean, y_std)

                val_loss = F.mse_loss(avg_preds_denorm, targets_denorm)
                val_losses.append(val_loss.item())

        return {"val_MSE": float(np.mean(val_losses))}

    def train(self, total_epochs, validation_interval):
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for model in self.models:
                model.train()

            supervised_losses_logged = []
            for batch, targets in self.train_dataloader:
                batch, targets = batch.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                supervised_losses = [
                    self.supervised_criterion(model(batch), targets)
                    for model in self.models
                ]
                supervised_loss = sum(supervised_losses)
                supervised_losses_logged.append(
                    supervised_loss.detach().item() / len(self.models)
                )
                supervised_loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            supervised_losses_logged = float(np.mean(supervised_losses_logged))

            summary_dict = {"supervised_loss": supervised_losses_logged}
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)

            self.logger.log_dict(summary_dict, step=epoch)


class MeanTeacherTrainer:


class NCPSTrainer:
    """
    n-way Cross Pseudo Supervision trainer.

    - Clones a base GNN into an ensemble of `num_models`.
    - Supervised loss on labeled data (denormalized).
    - CPS losses on both labeled and unlabeled data with uncertainty masking.
    """

    def __init__(
        self,
        supervised_criterion,
        unsupervised_criterion,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule,
        num_models: int = 3,
        cps_weight: float = 0.0005,
        cps_rampup_epochs: int = 150,
        confidence_threshold: float = 0.1,
        strong_augment_node_noise_std: float = 0.05,
        strong_augment_edge_drop_prob: float = 0.05,
    ):
        self.device = device
        self.logger = logger

        if len(models) == 1 and num_models > 1:
            base = models[0].to(device)
            self.models = [base] + [deepcopy(base).to(device) for _ in range(num_models - 1)]
        else:
            self.models = [m.to(device) for m in models]

        # Add some seed diversity
        for idx, model in enumerate(self.models):
            torch.manual_seed(42 + idx)

        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)

        self.supervised_criterion = supervised_criterion
        self.unsupervised_criterion = unsupervised_criterion

        self.cps_weight = cps_weight
        self.cps_rampup_epochs = cps_rampup_epochs
        self.confidence_threshold = confidence_threshold

        self.datamodule = datamodule
        self.train_labeled = datamodule.train_dataloader()
        self.train_unlabeled = datamodule.unsupervised_train_dataloader()
        self.val_loader = datamodule.val_dataloader()

        self.weak_augment = datamodule.augment_batch
        self.strong_augment = lambda batch: augment_batch(
            batch,
            noise_std=strong_augment_node_noise_std,
            drop_edge_p=strong_augment_edge_drop_prob,
        )

        self.y_mean = datamodule.y_mean.to(device)
        self.y_std = datamodule.y_std.to(device)

    def _cps_factor(self, epoch: int) -> float:
        if epoch >= self.cps_rampup_epochs:
            return self.cps_weight
        return self.cps_weight * epoch / max(1, self.cps_rampup_epochs)

    def validate(self):
        for m in self.models:
            m.eval()

        losses = []
        with torch.no_grad():
            for batch, targets in self.val_loader:
                batch, targets = batch.to(self.device), targets.to(self.device)
                preds = torch.stack([m(batch) for m in self.models], dim=0).mean(0)
                preds_denorm = denorm(preds, self.y_mean, self.y_std)
                targets_denorm = denorm(targets, self.y_mean, self.y_std)
                loss = F.mse_loss(preds_denorm, targets_denorm)
                losses.append(loss.item())

        return {"val_MSE": float(np.mean(losses))}

    def train(self, total_epochs, validation_interval):
        unl_iter = iter(self.train_unlabeled)

        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for m in self.models:
                m.train()

            sup_losses, cps_losses_u, cps_losses_l = [], [], []
            cps_factor = self._cps_factor(epoch)

            for batch_l, y_l in self.train_labeled:
                batch_l, y_l = batch_l.to(self.device), y_l.to(self.device)

                # Unlabeled batch
                try:
                    batch_u, _ = next(unl_iter)
                except StopIteration:
                    unl_iter = iter(self.train_unlabeled)
                    batch_u, _ = next(unl_iter)
                batch_u = batch_u.to(self.device)

                self.optimizer.zero_grad()

                # Supervised loss on labeled data
                sup_ls = []
                for m in self.models:
                    pred_l = m(batch_l)
                    pred_denorm = denorm(pred_l, self.y_mean, self.y_std)
                    target_denorm = denorm(y_l, self.y_mean, self.y_std)
                    sup_ls.append(self.supervised_criterion(pred_denorm, target_denorm))
                sup_loss = torch.stack(sup_ls).mean()

                # CPS on unlabeled data
                with torch.no_grad():
                    weak_preds_u = torch.stack(
                        [m(self.weak_augment(batch_u)) for m in self.models], dim=0
                    )
                    pseudo_u_std = weak_preds_u.std(dim=0)
                    mask = (pseudo_u_std < self.confidence_threshold).float()
                    # For regression, keep pseudo-labels on the natural
                    # (normalized) scale to avoid over-confident targets.
                    pseudo_u = weak_preds_u.mean(dim=0)

                if mask.sum() > 0:
                    strong_preds_u = torch.stack(
                        [m(self.strong_augment(batch_u)) for m in self.models], dim=0
                    )
                    strong_preds_u_denorm = denorm(
                        strong_preds_u, self.y_mean, self.y_std
                    )
                    pseudo_u_denorm = denorm(pseudo_u, self.y_mean, self.y_std)

                    cps_u_all = F.mse_loss(
                        strong_preds_u_denorm,
                        pseudo_u_denorm.unsqueeze(0).expand_as(strong_preds_u_denorm),
                        reduction="none",
                    )
                    cps_u_all = cps_u_all.mean(dim=-1, keepdim=True)

                    mask_reduced = mask
                    if mask_reduced.dim() > 2:
                        mask_reduced = mask_reduced.mean(dim=-1, keepdim=True)
                    mask_broadcast = mask_reduced.unsqueeze(0).expand(
                        strong_preds_u_denorm.size(0), -1, -1
                    )

                    cps_loss_u = (cps_u_all * mask_broadcast).sum() / mask_broadcast.sum()
                else:
                    cps_loss_u = torch.tensor(0.0, device=self.device)

                # CPS on labeled data
                with torch.no_grad():
                    weak_preds_l = torch.stack(
                        [m(self.weak_augment(batch_l)) for m in self.models], dim=0
                    )
                    # Use the ensemble mean directly as pseudo-labels on
                    # labeled data as well to keep the CPS signal stable.
                    pseudo_l = weak_preds_l.mean(dim=0)

                strong_preds_l = torch.stack(
                    [m(self.strong_augment(batch_l)) for m in self.models], dim=0
                )
                strong_preds_l_denorm = denorm(
                    strong_preds_l, self.y_mean, self.y_std
                )
                pseudo_l_denorm = denorm(pseudo_l, self.y_mean, self.y_std)

                cps_l_all = F.mse_loss(
                    strong_preds_l_denorm,
                    pseudo_l_denorm.unsqueeze(0).expand_as(strong_preds_l_denorm),
                    reduction="none",
                )
                cps_l_all = cps_l_all.mean(dim=-1, keepdim=True)
                cps_loss_l = cps_l_all.mean()

                total_loss = sup_loss + cps_factor * (cps_loss_u + cps_loss_l)
                total_loss.backward()
                self.optimizer.step()

                sup_losses.append(sup_loss.item())
                cps_losses_u.append(cps_loss_u.item())
                cps_losses_l.append(cps_loss_l.item())

            self.scheduler.step()

            log = {
                "sup_loss": float(np.mean(sup_losses)),
                "cps_loss_u": float(np.mean(cps_losses_u)) if cps_losses_u else 0.0,
                "cps_loss_l": float(np.mean(cps_losses_l)),
                "epoch": epoch,
                "cps_factor": cps_factor,
            }

            if epoch % validation_interval == 0 or epoch == total_epochs:
                val = self.validate()
                log.update(val)
                pbar.set_postfix(log)

            self.logger.log_dict(log, step=epoch)

        return log
