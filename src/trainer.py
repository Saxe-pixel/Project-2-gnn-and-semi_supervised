from copy import deepcopy
import torch.nn.functional as F
import numpy as np
import torch
from tqdm import tqdm

class SemiSupervisedEnsemble:
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
        self.models = models

        # Optim related things
        self.supervised_criterion = supervised_criterion
        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)

        # Dataloader setup
        self.datamodule = datamodule
        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()

        # Logging
        self.logger = logger

    def validate(self):
        for model in self.models:
            model.eval()

        val_losses = []

        y_mean = self.datamodule.y_mean.to(self.device)
        y_std = self.datamodule.y_std.to(self.device)

        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)

                # Ensemble prediction
                preds = [model(x) for model in self.models]
                avg_preds = torch.stack(preds).mean(0)

                avg_preds_denorm = avg_preds * y_std + y_mean
                targets_denorm = targets * y_std + y_mean

                val_loss = torch.nn.functional.mse_loss(avg_preds_denorm, targets_denorm)
                val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)
        return {"val_MSE": val_loss}

    def train(self, total_epochs, validation_interval):
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for model in self.models:
                model.train()
            supervised_losses_logged = []
            for x, targets in self.train_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                supervised_losses = [self.supervised_criterion(model(x), targets) for model in self.models]
                supervised_loss = sum(supervised_losses)
                supervised_losses_logged.append(supervised_loss.detach().item() / len(self.models))
                loss = supervised_loss
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            supervised_losses_logged = np.mean(supervised_losses_logged)

            summary_dict = {
                "supervised_loss": supervised_losses_logged,
            }
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)
            self.logger.log_dict(summary_dict, step=epoch)


import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm


def denorm(pred, mean, std):
    return pred * std + mean


class MeanTeacherTrainer:
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
        ema_decay=0.99,
        consistency_weight=0.1,
        consistency_rampup_epochs=10,
        warmup_epochs=5
    ):
        # One student
        self.student = models[0].to(device)
        self.teacher = deepcopy(self.student).to(device)

        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.device = device
        self.supervised_criterion = supervised_criterion
        self.unsupervised_criterion = unsupervised_criterion
        self.optimizer = optimizer(params=self.student.parameters())
        self.scheduler = scheduler(optimizer=self.optimizer)

        self.ema_initial = ema_decay
        self.consistency_weight = consistency_weight
        self.consistency_rampup_epochs = consistency_rampup_epochs
        self.warmup_epochs = warmup_epochs

        self.logger = logger
        self.global_step = 0

        self.datamodule = datamodule
        self.train_labeled = datamodule.train_dataloader()
        self.train_unlabeled = datamodule.unsupervised_train_dataloader()
        self.val_loader = datamodule.val_dataloader()
        self.augment = datamodule.augment_batch

        # stats
        self.y_mean = datamodule.y_mean.to(device)
        self.y_std = datamodule.y_std.to(device)

    # --------------------------------------------------------
    # EMA update with increasing decay schedule
    # --------------------------------------------------------
    def ema_decay(self, epoch):
        return min(0.999, self.ema_initial + 0.0005 * epoch)

    def update_teacher(self, epoch):
        decay = self.ema_decay(epoch)
        with torch.no_grad():
            for t, s in zip(self.teacher.parameters(), self.student.parameters()):
                t.data.mul_(decay).add_(s.data, alpha=1 - decay)

    # --------------------------------------------------------
    def consistency_factor(self, epoch):
        if epoch > self.consistency_rampup_epochs:
            return self.consistency_weight
        return self.consistency_weight * (epoch / self.consistency_rampup_epochs)

    # --------------------------------------------------------
    def validate(self):
        self.teacher.eval()
        losses = []
        with torch.no_grad():
            for batch, y in self.val_loader:
                batch, y = batch.to(self.device), y.to(self.device)
                pred = self.teacher(batch)
                pred = denorm(pred, self.y_mean, self.y_std)
                target = denorm(y, self.y_mean, self.y_std)
                losses.append(torch.nn.functional.mse_loss(pred, target).item())
        return {"val_MSE": float(np.mean(losses))}

    # --------------------------------------------------------
    def train(self, total_epochs, validation_interval):
        unl_iter = iter(self.train_unlabeled)

        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):

            self.student.train()
            self.teacher.eval()

            sup_losses, cons_losses = [], []

            for batch_l, y_l in self.train_labeled:
                batch_l = batch_l.to(self.device)
                y_l = y_l.to(self.device)

                # unlabeled
                try:
                    batch_u, _ = next(unl_iter)
                except StopIteration:
                    unl_iter = iter(self.train_unlabeled)
                    batch_u, _ = next(unl_iter)

                batch_u = batch_u.to(self.device)

                # augment unlabeled batch
                stu_in = self.augment(batch_u)
                tea_in = batch_u

                self.optimizer.zero_grad()

                # supervised
                pred_l = self.student(batch_l)
                sup_loss = self.supervised_criterion(
                    denorm(pred_l, self.y_mean, self.y_std),
                    denorm(y_l, self.y_mean, self.y_std)
                )

                # unsupervised
                with torch.no_grad():
                    tea_pred = self.teacher(tea_in)

                stu_pred = self.student(stu_in)

                cons_loss = self.unsupervised_criterion(
                    denorm(stu_pred, self.y_mean, self.y_std),
                    denorm(tea_pred, self.y_mean, self.y_std)
                )

                # combine
                weight = self.consistency_factor(epoch)
                loss = sup_loss + weight * cons_loss

                loss.backward()
                self.optimizer.step()
                self.update_teacher(epoch)

                sup_losses.append(sup_loss.item())
                cons_losses.append(cons_loss.item())

            self.scheduler.step()

            log = {
                "supervised_loss": float(np.mean(sup_losses)),
                "consistency_loss": float(np.mean(cons_losses)),
                "epoch": epoch,
            }

            if epoch % validation_interval == 0:
                val = self.validate()
                log.update(val)
                pbar.set_postfix(log)

            self.logger.log_dict(log, step=epoch)

        return log



import torch
import numpy as np
from tqdm import tqdm


def sharpen(pseudo, tau=2.0):
    mean = pseudo.mean(dim=0, keepdim=True)
    return mean + tau * (pseudo - mean)


def denorm(pred, mean, std):
    return pred * std + mean


class NCPSTrainer:
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
        num_models: int = 3,      # ← added argument (fix)
        cps_weight=0.1,
        cps_rampup_epochs=5
    ):
        self.device = device
        self.logger = logger

        # ----------------------------------------------------
        # MODEL DIVERSITY / CLONING FOR n-CPS
        # ----------------------------------------------------
        if len(models) == 1 and num_models > 1:
            # Clone the base model (ensures fair comparison to MT/Ensemble)
            base = models[0].to(device)
            self.models = [base] + [
                deepcopy(base).to(device) for _ in range(num_models - 1)
            ]
        else:
            # Already multiple models passed in
            self.models = [m.to(device) for m in models]

        # Add seed diversity
        for i, m in enumerate(self.models):
            torch.manual_seed(42 + i)

        # Optimizer over all model parameters
        self.optimizer = optimizer(
            params=[p for m in self.models for p in m.parameters()]
        )
        self.scheduler = scheduler(optimizer=self.optimizer)

        self.supervised_criterion = supervised_criterion
        self.unsupervised_criterion = unsupervised_criterion

        self.cps_weight = cps_weight
        self.cps_rampup_epochs = cps_rampup_epochs

        # ----------------------------------------------------
        # DATA
        # ----------------------------------------------------
        self.datamodule = datamodule
        self.train_labeled = datamodule.train_dataloader()
        self.train_unlabeled = datamodule.unsupervised_train_dataloader()
        self.val_loader = datamodule.val_dataloader()

        self.augment = datamodule.augment_batch

        # Normalization stats
        self.y_mean = datamodule.y_mean.to(device)
        self.y_std = datamodule.y_std.to(device)

    # --------------------------------------------------------
    def ramp(self, epoch):
        if epoch >= self.cps_rampup_epochs:
            return self.cps_weight
        return self.cps_weight * epoch / self.cps_rampup_epochs

    # --------------------------------------------------------
    def validate(self):
        for m in self.models:
            m.eval()

        losses = []
        with torch.no_grad():
            for batch, y in self.val_loader:
                batch, y = batch.to(self.device), y.to(self.device)

                preds = torch.stack([m(batch) for m in self.models]).mean(0)
                preds = denorm(preds, self.y_mean, self.y_std)
                y = denorm(y, self.y_mean, self.y_std)

                losses.append(torch.nn.functional.mse_loss(preds, y).item())

        return {"val_MSE": float(np.mean(losses))}

    # --------------------------------------------------------
    def train(self, total_epochs, validation_interval):
        unl_iter = iter(self.train_unlabeled)

        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):

            for m in self.models:
                m.train()

            sup_losses, cps_losses = [], []
            cpsw = self.ramp(epoch)

            for batch_l, y_l in self.train_labeled:
                batch_l, y_l = batch_l.to(self.device), y_l.to(self.device)

                # unlabeled batch
                try:
                    batch_u, _ = next(unl_iter)
                except StopIteration:
                    unl_iter = iter(self.train_unlabeled)
                    batch_u, _ = next(unl_iter)

                batch_u = batch_u.to(self.device)

                self.optimizer.zero_grad()

                # --------------------
                # supervised loss
                # --------------------
                sup_ls = []
                for m in self.models:
                    pred_l = m(batch_l)
                    sup_ls.append(
                        self.supervised_criterion(
                            denorm(pred_l, self.y_mean, self.y_std),
                            denorm(y_l, self.y_mean, self.y_std),
                        )
                    )
                sup_loss = torch.stack(sup_ls).mean()

                # --------------------
                # CPS consistency loss
                # --------------------
                weak_pred = torch.stack([m(batch_u) for m in self.models])
                pseudo = sharpen(weak_pred, tau=2.0)

                strong_batch = self.augment(batch_u)
                strong_preds = torch.stack([m(strong_batch) for m in self.models])

                cps_loss = self.unsupervised_criterion(
                    denorm(strong_preds, self.y_mean, self.y_std),
                    denorm(pseudo, self.y_mean, self.y_std),
                )

                total = sup_loss + cpsw * cps_loss
                total.backward()
                self.optimizer.step()

                sup_losses.append(sup_loss.item())
                cps_losses.append(cps_loss.item())

            self.scheduler.step()

            log = {
                "supervised_loss": float(np.mean(sup_losses)),
                "cps_loss": float(np.mean(cps_losses)),
                "epoch": epoch,
            }

            if epoch % validation_interval == 0:
                val = self.validate()
                log.update(val)
                pbar.set_postfix(log)

            self.logger.log_dict(log, step=epoch)

        return log

