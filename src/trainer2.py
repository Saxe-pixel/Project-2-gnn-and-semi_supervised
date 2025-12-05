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

        # History of validation metrics (logged at validation steps)
        self.history = {"epoch": [], "val_MSE": []}

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
                if "val_MSE" in val_metrics:
                    self.history["epoch"].append(epoch)
                    self.history["val_MSE"].append(float(val_metrics["val_MSE"]))
                pbar.set_postfix(summary_dict)
            self.logger.log_dict(summary_dict, step=epoch)


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
        ema_decay: float = 0.99,
        consistency_weight: float = 1.0,
        consistency_rampup_epochs: int = 5,
    ) -> None:
        if len(models) != 1:
            raise ValueError("MeanTeacherTrainer expects exactly one student model")

        self.device = device
        self.student = models[0]
        self.teacher = deepcopy(self.student).to(self.device)
        for param in self.teacher.parameters():
            param.requires_grad_(False)

        self.supervised_criterion = supervised_criterion
        self.unsupervised_criterion = unsupervised_criterion
        self.optimizer = optimizer(params=self.student.parameters())
        self.scheduler = scheduler(optimizer=self.optimizer)
        self.ema_decay = ema_decay
        self.base_consistency_weight = consistency_weight
        self.consistency_rampup_epochs = consistency_rampup_epochs

        self.datamodule = datamodule
        self.train_dataloader = datamodule.train_dataloader()
        self.unlabeled_dataloader = (
            datamodule.unsupervised_train_dataloader()
            if hasattr(datamodule, "unsupervised_train_dataloader")
            else None
        )
        self.val_dataloader = datamodule.val_dataloader()
        self.augmentation_fn = getattr(datamodule, 'augment_batch', None)

        self.logger = logger
        self.global_step = 0

        self.y_mean = self.datamodule.y_mean.to(self.device)
        self.y_std = self.datamodule.y_std.to(self.device)

        # History of validation metrics (teacher and student) per validation step
        self.history = {
            "epoch": [],
            "val_MSE_teacher": [],
            "val_MSE_student": [],
        }

    def _update_teacher(self) -> None:
        with torch.no_grad():
            # EMA for parameters
            for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
                t_param.data.mul_(self.ema_decay).add_(s_param.data, alpha=1 - self.ema_decay)

            # BatchNorm running stats
            for t_buf, s_buf in zip(self.teacher.buffers(), self.student.buffers()):
                try:
                    t_buf.data.mul_(self.ema_decay).add_(s_buf.data, alpha=1 - self.ema_decay)
                except Exception:
                    t_buf.data.copy_(s_buf.data)



    def _current_consistency_weight(self, epoch: int) -> float:
        if self.consistency_rampup_epochs <= 0:
            return self.base_consistency_weight

        steps_per_epoch = max(1, len(self.train_dataloader))
        total_ramp_steps = self.consistency_rampup_epochs * steps_per_epoch

        ramp = min(1.0, self.global_step / total_ramp_steps)
        return self.base_consistency_weight * ramp

    def _augment_batch(self, batch):
        if batch is None or self.augmentation_fn is None:
            return batch
        return self.augmentation_fn(batch)

    def validate(self, use_teacher: bool = True):
        """
        If use_teacher=True: validate EMA teacher (deafault).
        If use_teacher=False: validate the student.
        """
        model = self.teacher if use_teacher else self.student
        model.eval()

        val_losses = []
        with torch.no_grad():
            for batch, targets in self.val_dataloader:
                batch, targets = batch.to(self.device), targets.to(self.device)
                preds = model(batch)
                preds_denorm = preds * self.y_std + self.y_mean
                targets_denorm = targets * self.y_std + self.y_mean
                val_loss = torch.nn.functional.mse_loss(preds_denorm, targets_denorm)
                val_losses.append(val_loss.item())

        key = "val_MSE_teacher" if use_teacher else "val_MSE_student"
        return {key: np.mean(val_losses)}


    def train(self, total_epochs, validation_interval):
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            self.student.train()
            self.teacher.eval()
            unlabeled_iter = iter(self.unlabeled_dataloader) if self.unlabeled_dataloader is not None else None

            supervised_losses = []
            consistency_losses = []
            for labeled_batch, labeled_targets in self.train_dataloader:
                labeled_batch = labeled_batch.to(self.device)
                labeled_targets = labeled_targets.to(self.device)
                if unlabeled_iter is not None:
                    try:
                        unlabeled_batch, _ = next(unlabeled_iter)
                    except StopIteration:
                        unlabeled_iter = iter(self.unlabeled_dataloader)
                        unlabeled_batch, _ = next(unlabeled_iter)
                    unlabeled_batch = unlabeled_batch.to(self.device)
                else:
                    unlabeled_batch = None

                self.optimizer.zero_grad()
                student_preds = self.student(labeled_batch)
                supervised_loss = self.supervised_criterion(student_preds, labeled_targets)

                consistency_loss = torch.tensor(0.0, device=self.device)
                consistency_weight = self._current_consistency_weight(epoch)
                if unlabeled_batch is not None:
                    student_view = self._augment_batch(unlabeled_batch)
                    teacher_view = self._augment_batch(unlabeled_batch)
                    student_unlabeled = self.student(student_view)
                    with torch.no_grad():
                        teacher_unlabeled = self.teacher(teacher_view)
                    consistency_loss = self.unsupervised_criterion(student_unlabeled, teacher_unlabeled)
                    if hasattr(labeled_batch, 'num_graphs') and hasattr(student_view, 'num_graphs'):
                        batch_ratio = labeled_batch.num_graphs / max(1, student_view.num_graphs)
                        consistency_weight = consistency_weight * batch_ratio

                total_loss = supervised_loss + consistency_weight * consistency_loss
                total_loss.backward()
                self.optimizer.step()
                self._update_teacher()
                self.global_step += 1

                supervised_losses.append(supervised_loss.detach().item())
                consistency_losses.append(consistency_loss.detach().item())

            self.scheduler.step()

            summary_dict = {
                "supervised_loss": float(np.mean(supervised_losses)) if supervised_losses else 0.0,
                "consistency_loss": float(np.mean(consistency_losses)) if consistency_losses else 0.0,
                "consistency_weight": self._current_consistency_weight(epoch),
            }

            if epoch % validation_interval == 0 or epoch == total_epochs:
                # teacher metrics
                val_teacher = self.validate(use_teacher=True)
                # student metrics
                val_student = self.validate(use_teacher=False)

                summary_dict.update(val_teacher)
                summary_dict.update(val_student)

                # Store validation history
                self.history["epoch"].append(epoch)
                self.history["val_MSE_teacher"].append(float(val_teacher["val_MSE_teacher"]))
                self.history["val_MSE_student"].append(float(val_student["val_MSE_student"]))

                pbar.set_postfix(summary_dict)

            self.logger.log_dict(summary_dict, step=epoch)

        return summary_dict


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
        num_models: int = 2,        
        cps_weight: float = 1.0,
        cps_rampup_epochs: int = 5,
    ):
        if len(models) == 0:
            raise ValueError("NCPSTrainer received no models")

        self.device = device

        # only one model is passed, clone it num_models-1 times
        if len(models) == 1 and num_models > 1:
            base = models[0].to(device)
            self.models = [base] + [deepcopy(base).to(device) for _ in range(num_models - 1)]
        else:
            assert len(models) >= 2, "n-CPS needs at least 2 models"
            self.models = [m.to(device) for m in models]

        self.supervised_criterion = supervised_criterion
        self.unsupervised_criterion = unsupervised_criterion

        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)

        self.datamodule = datamodule
        self.train_dataloader = datamodule.train_dataloader()
        self.unlabeled_dataloader = (
            datamodule.unsupervised_train_dataloader()
            if hasattr(datamodule, "unsupervised_train_dataloader")
            else None
        )
        self.val_dataloader = datamodule.val_dataloader()
        self.augmentation_fn = getattr(datamodule, "augment_batch", None)

        self.logger = logger

        self.cps_weight = cps_weight
        self.cps_rampup_epochs = cps_rampup_epochs

        # QM9 regression MSE metric
        self.y_mean = datamodule.y_mean.to(device)
        self.y_std = datamodule.y_std.to(device)

        # History of validation metrics per validation step
        self.history = {"epoch": [], "val_MSE": []}

    def _current_cps_weight(self, epoch: int) -> float:
        if self.cps_rampup_epochs <= 0:
            return self.cps_weight
        ramp = min(1.0, epoch / self.cps_rampup_epochs)
        return self.cps_weight * ramp

    def _augment_batch(self, batch):
        if batch is None or self.augmentation_fn is None:
            return batch
        return self.augmentation_fn(batch)

    def validate(self):
        for m in self.models:
            m.eval()

        val_losses = []
        with torch.no_grad():
            for batch, targets in self.val_dataloader:
                batch, targets = batch.to(self.device), targets.to(self.device)

                # ensemble preds
                preds_list = [m(batch) for m in self.models]
                avg_preds = torch.stack(preds_list).mean(0)

                preds_denorm = avg_preds * self.y_std + self.y_mean
                targets_denorm = targets * self.y_std + self.y_mean

                val_loss = F.mse_loss(preds_denorm, targets_denorm)
                val_losses.append(val_loss.item())

        return {"val_MSE": float(np.mean(val_losses))}

    def train(self, total_epochs, validation_interval):
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for m in self.models:
                m.train()

            unlabeled_iter = iter(self.unlabeled_dataloader) if self.unlabeled_dataloader is not None else None
            cps_weight = self._current_cps_weight(epoch)

            supervised_losses_logged = []
            cps_losses_logged = []

            for labeled_batch, labeled_targets in self.train_dataloader:
                labeled_batch = labeled_batch.to(self.device)
                labeled_targets = labeled_targets.to(self.device)

                if unlabeled_iter is not None:
                    try:
                        unlabeled_batch, _ = next(unlabeled_iter)
                    except StopIteration:
                        unlabeled_iter = iter(self.unlabeled_dataloader)
                        unlabeled_batch, _ = next(unlabeled_iter)
                    unlabeled_batch = unlabeled_batch.to(self.device)
                else:
                    unlabeled_batch = None

                self.optimizer.zero_grad()

                # supervised loss on labeled
                sup_losses = []
                for m in self.models:
                    preds = m(labeled_batch)
                    sup_losses.append(self.supervised_criterion(preds, labeled_targets))
                sup_loss = sum(sup_losses) / len(sup_losses)

                # CPS loss on unlabeled
                cps_loss = torch.tensor(0.0, device=self.device)
                if unlabeled_batch is not None and cps_weight > 0.0:
                    weak_view = unlabeled_batch
                    strong_view = self._augment_batch(unlabeled_batch)

                    # predictions of each model on weak view
                    weak_preds = [m(weak_view) for m in self.models]

                    # pseudo-targets for each model 
                    pseudo_targets = []
                    n = len(self.models)
                    with torch.no_grad():
                        for i in range(n):
                            others = [j for j in range(n) if j != i]
                            pseudo = sum(weak_preds[j] for j in others) / len(others)
                            pseudo_targets.append(pseudo)

                    # each model predicts on strong view match pseudo-targets
                    cps_losses = []
                    for m, pseudo in zip(self.models, pseudo_targets):
                        strong_preds = m(strong_view)
                        cps_losses.append(self.unsupervised_criterion(strong_preds, pseudo))

                    cps_loss = sum(cps_losses) / len(cps_losses)

                total_loss = sup_loss + cps_weight * cps_loss
                total_loss.backward()
                self.optimizer.step()

                supervised_losses_logged.append(sup_loss.detach().item())
                cps_losses_logged.append(cps_loss.detach().item())

            self.scheduler.step()

            summary_dict = {
                "supervised_loss": float(np.mean(supervised_losses_logged)) if supervised_losses_logged else 0.0,
                "cps_loss": float(np.mean(cps_losses_logged)) if cps_losses_logged else 0.0,
                "cps_weight": cps_weight,
            }

            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                if "val_MSE" in val_metrics:
                    self.history["epoch"].append(epoch)
                    self.history["val_MSE"].append(float(val_metrics["val_MSE"]))
                pbar.set_postfix(summary_dict)

            self.logger.log_dict(summary_dict, step=epoch)

        return summary_dict
