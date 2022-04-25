from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification.accuracy import Accuracy
from torchvision.datasets import ImageFolder
from transformers import ViTFeatureExtractor, ViTForImageClassification

from transformers import pipeline


class ImageClassifierDataModule(pl.LightningDataModule):
    def __init__(
        self,
        images_dir: str,
        feature_extractor,
        batch_size: int,
        num_workers: int,
        train_validation_split: float,
    ):
        super().__init__()
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_validation_split = train_validation_split
        self.feature_extractor = feature_extractor

    def setup(self, stage: Optional[str] = None):
        ds = ImageFolder(self.images_dir)
        label2id = dict((k, str(v)) for k, v, in ds.class_to_idx.items())
        id2label = dict((str(v), k) for k, v in ds.class_to_idx.items())
        train_size = int(len(ds) * self.train_validation_split)
        train_ds, val_ds = random_split(ds, lengths=[train_size, len(ds) - train_size])
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.label2id = label2id
        self.id2label = id2label

    def _collate_fn(self, batch):
        imgs = [x[0] for x in batch]
        return {
            "pixel_values": torch.stack(
                [
                    t
                    for t in self.feature_extractor(
                        imgs, return_tensors="pt"
                    ).pixel_values
                ]
            ),
            "labels": torch.tensor([x[1] for x in batch]),
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
        )


class ImageClassifier(pl.LightningModule):
    def __init__(self, model, learning_rate: float = 3e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def forward(self, pixel_values, labels):
        return self.model(pixel_values=pixel_values, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)

        preds = outputs.logits.argmax(-1)
        targets = batch["labels"]
        acc = self.train_acc(preds, targets)

        self.log("train acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        loss = outputs.loss
        self.log("val_loss", loss)

        preds = outputs.logits.argmax(-1)
        targets = batch["labels"]
        acc = self.val_acc(preds, targets)
        self.log("val acc", acc, prog_bar=True)

        return loss
