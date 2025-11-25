import os
import subprocess

import numpy as np
import pytorch_lightning as pl
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from torch_geometric.transforms import BaseTransform
from qm9_utils import DataLoader, GetTarget
from augmentations import augment_batch as augment_fn

class QM9DataModule(pl.LightningDataModule):
    def __init__(
        self,
        target: int = 0,
        data_dir: str = os.path.join('datasets'),
        batch_size_train: int = 32,
        batch_size_inference: int = 32,
        num_workers: int = 0,
        splits: list[int] | list[float] = [0.77, 0.03, 0.1, 0.1],
        seed: int = 0,
        subset_size: int | None = None,
        data_augmentation: bool = False,
        augment_node_noise_std: float = 0.02,
        augment_edge_drop_prob: float = 0.05,
        name: str = 'qm9',
        ood: bool = False,

    ) -> None:
        super().__init__()
        self.target = target
        self.data_dir = data_dir
        self.batch_size_train = batch_size_train
        self.batch_size_inference = batch_size_inference
        self.num_workers = num_workers
        self.splits = splits
        self.seed = seed
        self.subset_size = subset_size
        self.data_augmentation = data_augmentation
        self.augment_node_noise_std = augment_node_noise_std
        self.augment_edge_drop_prob = augment_edge_drop_prob
        self.name = name
        self.ood = ood

        self.data_train_unlabeled = None
        self.data_train_labeled = None
        self.data_val = None
        self.data_test = None
        self.ood_datasets = None

        self.batch_size_train_labeled = None
        self.batch_size_train_unlabeled = None

        self.setup()  # Call setup to initialize the datasets


    def prepare_data(self) -> None:
        # Download data
        QM9(root=self.data_dir)

    def setup(self, stage: str | None = None) -> None:
        dataset = QM9(root=self.data_dir, transform=GetTarget(self.target))

        # Shuffle dataset
        rng = np.random.default_rng(seed=self.seed)
        dataset = dataset[rng.permutation(len(dataset))]

        # Subset dataset
        if self.subset_size is not None:
            dataset = dataset[:self.subset_size]

        # Split dataset
        if all([type(split) == int for split in self.splits]):
            split_sizes = self.splits
        elif all([type(split) == float for split in self.splits]):
            split_sizes = [int(len(dataset) * prop) for prop in self.splits]

        split_idx = np.cumsum(split_sizes)

        self.data_train_unlabeled = dataset[:split_idx[0]]
        self.data_train_labeled = dataset[split_idx[0]:split_idx[1]]
        self.data_val = dataset[split_idx[1]:split_idx[2]]
        self.data_test = dataset[split_idx[2]:]

        # NORMALIZE TARGETS USING TRAIN-LABELED ONLY
        ys = torch.stack([d.y for d in self.data_train_labeled])
        self.y_mean = ys.mean()
        self.y_std = ys.std()

        # normalization function
        def norm_dataset(dset):
            for d in dset:
                d.y = (d.y - self.y_mean) / self.y_std

        # Apply normalization to ALL splits, using train stats
        norm_dataset(self.data_train_labeled)
        norm_dataset(self.data_train_unlabeled)
        norm_dataset(self.data_val)
        norm_dataset(self.data_test)

        print(f"Target normalization (train only): mean={self.y_mean.item():.4f}, std={self.y_std.item():.4f}")


        # Set batch sizes. We want the labeled batch size to be the one given by the user, and the unlabeled one to be so that we have the same number of batches
        self.batch_size_train_labeled = self.batch_size_train
        labeled_batches = max(1, int(np.ceil(len(self.data_train_labeled) / self.batch_size_train_labeled)))
        target_unlabeled_batch = max(1, int(np.ceil(len(self.data_train_unlabeled) / labeled_batches)))
        self.batch_size_train_unlabeled = target_unlabeled_batch

        print(f"QM9 dataset loaded with {len(self.data_train_labeled)} labeled, {len(self.data_train_unlabeled)} unlabeled, "
              f"{len(self.data_val)} validation, and {len(self.data_test)} test samples.")
        print(f"Batch sizes: labeled={self.batch_size_train_labeled}, unlabeled={self.batch_size_train_unlabeled}")

    def augment_batch(self, batch: Data) -> Data:
        if batch is None:
            return batch
        return augment_fn(batch)


    def train_dataloader(self, shuffle=True) -> DataLoader:
        return DataLoader(
            self.data_train_labeled,
            batch_size=self.batch_size_train_labeled,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=True
        )

    def unsupervised_train_dataloader(self, shuffle=True) -> DataLoader:
        return DataLoader(
            self.data_train_unlabeled,
            batch_size=self.batch_size_train_unlabeled,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=True
        )


    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size_inference,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True
        )


    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size_inference,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True
        )


    def ood_dataloaders(self) -> dict[str, DataLoader]:
        return {
            dataset_name: DataLoader(
                dataset,
                batch_size=self.batch_size_inference,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
                persistent_workers=True
            )
            for dataset_name, dataset in self.ood_datasets.items()
        }

    def ood_dataloader(self) -> list[DataLoader]:
        """Returns a list of DataLoader for each OOD dataset."""
        if self.ood_datasets is None:
            return [], []
        else:
            ood_dataloaders = []
            ood_names = []
            #for dm in self.ood_datasets:
            for dataset_name, dataset in self.ood_datasets.items():
                val_dataloader = DataLoader(
                    dataset,
                    batch_size=self.batch_size_inference,
                    num_workers=self.num_workers,
                    shuffle=False,
                    pin_memory=True,
                    persistent_workers=True
                )
                ood_dataloaders.append(val_dataloader)
                ood_names.append(dataset_name)
            return ood_names, ood_dataloaders
