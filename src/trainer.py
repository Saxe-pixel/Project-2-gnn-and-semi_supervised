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
    """
    Mean Teacher trainer for semi-supervised QM9.

    - One student and one EMA teacher.
    - Strong/weak graph augmentations.
    - Uncertainty-based masking on unlabeled data.
    - Consistency loss on both labeled and unlabeled graphs.
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
        ema_decay: float = 0.99,
        warmup_epochs: int = 7,
        consistency_weight: float = 0.01,
        consistency_rampup_epochs: int = 50,
        confidence_threshold: float = 0.1,
        strong_augment_node_noise_std: float = 0.05,
        strong_augment_edge_drop_prob: float = 0.05,
    ):
        self.device = device

        self.student = models[0].to(device)
        self.teacher = deepcopy(self.student).to(device)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.supervised_criterion = supervised_criterion
        self.unsupervised_criterion = unsupervised_criterion
        self.optimizer = optimizer(params=self.student.parameters())
        self.scheduler = scheduler(optimizer=self.optimizer)

        self.ema_initial = ema_decay
        self.warmup_epochs = warmup_epochs
        self.consistency_weight = consistency_weight
        self.rampup_epochs = consistency_rampup_epochs
        self.confidence_threshold = confidence_threshold

        self.logger = logger

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

    def _ema_decay(self, epoch: int) -> float:
        return min(0.999, self.ema_initial + 0.0005 * max(epoch - 1, 0))

    def _update_teacher(self, epoch: int) -> None:
        decay = self._ema_decay(epoch)
        with torch.no_grad():
            for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
                t_param.data.mul_(decay).add_(s_param.data, alpha=1.0 - decay)

    def _consistency_factor(self, epoch: int) -> float:
        if epoch <= self.warmup_epochs:
            return 0.0
        if epoch >= self.rampup_epochs:
            return self.consistency_weight
        span = max(1, self.rampup_epochs - self.warmup_epochs)
        progress = (epoch - self.warmup_epochs) / span
        return self.consistency_weight * progress

    def validate(self):
        self.teacher.eval()
        losses = []

        with torch.no_grad():
            for batch, targets in self.val_loader:
                batch, targets = batch.to(self.device), targets.to(self.device)
                preds = self.teacher(batch)
                preds_denorm = denorm(preds, self.y_mean, self.y_std)
                targets_denorm = denorm(targets, self.y_mean, self.y_std)
                loss = F.mse_loss(preds_denorm, targets_denorm)
                losses.append(loss.item())

        return {"val_MSE": float(np.mean(losses))}

    def train(self, total_epochs, validation_interval):
        unl_iter = iter(self.train_unlabeled)

        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            self.student.train()
            self.teacher.eval()

            sup_losses, cons_losses_u, cons_losses_l = [], [], []
            cons_factor = self._consistency_factor(epoch)

            for batch_l, y_l in self.train_labeled:
                batch_l, y_l = batch_l.to(self.device), y_l.to(self.device)

                # Unlabeled batch
                try:
                    batch_u, _ = next(unl_iter)
                except StopIteration:
                    unl_iter = iter(self.train_unlabeled)
                    batch_u, _ = next(unl_iter)
                batch_u = batch_u.to(self.device)

                # Labeled path: use clean graphs for supervised signal
                stu_in_l = batch_l
                tea_in_l = batch_l

                # Unlabeled path: apply strong/weak augmentations
                stu_in_u = self.strong_augment(batch_u)
                tea_in_u = self.weak_augment(batch_u)

                self.optimizer.zero_grad()

                # Supervised loss on labeled data (normalized scale)
                student_pred_l = self.student(stu_in_l)
                sup_loss = self.supervised_criterion(student_pred_l, y_l)

                # Teacher predictions and pseudo-labels
                with torch.no_grad():
                    tea_pred_u1 = self.teacher(tea_in_u)
                    tea_pred_u2 = self.teacher(tea_in_u)
                    pseudo_u = (tea_pred_u1 + tea_pred_u2) / 2.0
                    pseudo_u_std = torch.stack([tea_pred_u1, tea_pred_u2], dim=0).std(dim=0)
                    mask = (pseudo_u_std < self.confidence_threshold).float()

                    pseudo_u = sharpen(pseudo_u)
                    tea_pred_l = self.teacher(tea_in_l)

                # Unlabeled consistency loss (masked, normalized scale)
                if mask.sum() > 0:
                    stu_pred_u = self.student(stu_in_u)
                    loss_u_all = F.mse_loss(
                        stu_pred_u,
                        pseudo_u,
                        reduction="none",
                    )
                    loss_u_all = loss_u_all.mean(dim=-1, keepdim=True)

                    mask_reduced = mask
                    if mask_reduced.dim() > 2:
                        mask_reduced = mask_reduced.mean(dim=-1, keepdim=True)

                    cons_loss_u = (loss_u_all * mask_reduced).sum() / mask_reduced.sum()
                else:
                    cons_loss_u = torch.tensor(0.0, device=self.device)

                # Labeled consistency between student and teacher (normalized scale)
                cons_loss_l = F.mse_loss(student_pred_l, tea_pred_l)

                total_loss = sup_loss + cons_factor * (cons_loss_u + cons_loss_l)
                total_loss.backward()
                self.optimizer.step()

                sup_losses.append(sup_loss.item())
                cons_losses_u.append(cons_loss_u.item())
                cons_losses_l.append(cons_loss_l.item())

            # Update teacher and LR scheduler
            self._update_teacher(epoch)
            self.scheduler.step()

            log = {
                "sup_loss": float(np.mean(sup_losses)),
                "cons_loss_u": float(np.mean(cons_losses_u)) if cons_losses_u else 0.0,
                "cons_loss_l": float(np.mean(cons_losses_l)),
                "epoch": epoch,
                "consistency_factor": cons_factor,
            }

            if epoch % validation_interval == 0 or epoch == total_epochs:
                val = self.validate()
                log.update(val)
                pbar.set_postfix(log)

            self.logger.log_dict(log, step=epoch)

        return log


class NCPSTrainer:
    """
    Improved n-way Cross Pseudo Supervision Trainer for regression.

    Key improvements:
    - CPS loss computed in *normalized* space (much more stable)
    - Softer/safer strong augmentations
    - Percentile-based uncertainty masking
    - Better ensemble diversity
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
        cps_rampup_epochs: int = 50,
        mask_percentile: float = 0.05,
        strong_node_noise: float = 0.003,
        strong_edge_drop: float = 0.002,
    ):
        self.device = device
        self.logger = logger

        # -------------------------------
        # Build ensemble with real diversity
        # -------------------------------
        if len(models) == 1:
            base = models[0]
            self.models = []
            for i in range(num_models):
                torch.manual_seed(1234 + i * 17)
                self.models.append(deepcopy(base).to(device))
        else:
            self.models = [m.to(device) for m in models]

        # Optimizer + scheduler over all models jointly
        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)

        self.supervised_criterion = supervised_criterion
        self.unsupervised_criterion = unsupervised_criterion

        self.cps_weight = cps_weight
        self.cps_rampup_epochs = cps_rampup_epochs
        self.mask_percentile = mask_percentile

        # Data
        self.datamodule = datamodule
        self.train_labeled = datamodule.train_dataloader()
        self.train_unlabeled = datamodule.unsupervised_train_dataloader()
        self.val_loader = datamodule.val_dataloader()

        # Weak and strong augmentations
        self.weak_augment = datamodule.augment_batch

        self.strong_augment = lambda batch: augment_batch(
            batch,
            noise_std=strong_node_noise,
            drop_edge_p=strong_edge_drop,
        )

        # Normalization parameters
        self.y_mean = datamodule.y_mean.to(device)
        self.y_std = datamodule.y_std.to(device)

    # -----------------------
    # CPS Factor (ramp-up)
    # -----------------------
    def _cps_factor(self, epoch):
        if epoch >= self.cps_rampup_epochs:
            return self.cps_weight
        return self.cps_weight * epoch / max(1, self.cps_rampup_epochs)

    # -----------------------
    # Validation
    # -----------------------
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

                losses.append(F.mse_loss(preds_denorm, targets_denorm).item())

        return {"val_MSE": float(np.mean(losses))}

    # -----------------------
    # Training loop
    # -----------------------
    def train(self, total_epochs, validation_interval):
        unl_iter = iter(self.train_unlabeled)

        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for m in self.models:
                m.train()

            sup_losses = []
            cps_losses_u = []
            cps_losses_l = []

            cps_factor = self._cps_factor(epoch)

            for batch_l, y_l in self.train_labeled:
                batch_l = batch_l.to(self.device)
                y_l = y_l.to(self.device)

                # Pull unlabeled batch
                try:
                    batch_u, _ = next(unl_iter)
                except StopIteration:
                    unl_iter = iter(self.train_unlabeled)
                    batch_u, _ = next(unl_iter)
                batch_u = batch_u.to(self.device)

                self.optimizer.zero_grad()

                # -----------------------
                # 1. Supervised loss
                # -----------------------
                sup_list = []
                for m in self.models:
                    pred = m(batch_l)
                    pred_denorm = denorm(pred, self.y_mean, self.y_std)
                    y_denorm = denorm(y_l, self.y_mean, self.y_std)
                    sup_list.append(self.supervised_criterion(pred_denorm, y_denorm))

                sup_loss = torch.stack(sup_list).mean()

                # -----------------------
                # 2. CPS on unlabeled data
                # -----------------------
                with torch.no_grad():
                    weak_preds_u = torch.stack(
                        [m(self.weak_augment(batch_u)) for m in self.models], dim=0
                    )
                    pseudo_std_u = weak_preds_u.std(dim=0)

                    # percentile-based masking = more robust
                    threshold = torch.quantile(pseudo_std_u, self.mask_percentile)
                    mask_u = (pseudo_std_u < threshold).float()

                    pseudo_u = weak_preds_u.mean(dim=0)  # still normalized range

                if mask_u.sum() > 0:
                    strong_preds_u = torch.stack(
                        [m(self.strong_augment(batch_u)) for m in self.models], dim=0
                    )

                    # CPS in normalized space (critical fix)
                    diff_u = (strong_preds_u - pseudo_u.unsqueeze(0)) ** 2
                    cps_u_per_model = diff_u.mean(dim=-1, keepdim=True)

                    mask_expanded = mask_u.unsqueeze(0).expand_as(cps_u_per_model)

                    cps_loss_u = (cps_u_per_model * mask_expanded).sum() / mask_expanded.sum()
                else:
                    cps_loss_u = torch.tensor(0.0, device=self.device)

                # -----------------------
                # 3. CPS on labeled data
                # -----------------------
                with torch.no_grad():
                    weak_preds_l = torch.stack(
                        [m(self.weak_augment(batch_l)) for m in self.models], dim=0
                    )
                    pseudo_l = weak_preds_l.mean(dim=0)  # normalized

                strong_preds_l = torch.stack(
                    [m(self.strong_augment(batch_l)) for m in self.models], dim=0
                )

                diff_l = (strong_preds_l - pseudo_l.unsqueeze(0)) ** 2
                cps_loss_l = diff_l.mean()

                # -----------------------
                # 4. Total loss
                # -----------------------
                total_loss = sup_loss + cps_factor * (cps_loss_u + cps_loss_l)
                total_loss.backward()
                self.optimizer.step()

                # Logging
                sup_losses.append(sup_loss.item())
                cps_losses_u.append(cps_loss_u.item())
                cps_losses_l.append(cps_loss_l.item())

            self.scheduler.step()

            log = {
                "sup_loss": float(np.mean(sup_losses)),
                "cps_loss_u": float(np.mean(cps_losses_u)),
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
