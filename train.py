import fire
import pandas as pd
import pytorch_lightning as pl

# import wandb

from model import ImageClassifier, ImageClassifierDataModule

# from huggingface_hub import HfApi, HfFolder, notebook_login, Repository
from pytorch_lightning.loggers import WandbLogger


def main(
    images_dir,
    wandb_project,
    wandb_run_name=None,
    learning_rate=3e-6,
    epochs=20,
    batch_size=32,
    random_seed=1234,
    accelerator="gpu",
    precision=16,
    num_workers=4,
    num_gpus=1,
    model_name="google/vit-base-patch16-224-in21k",
):
    pl.seed_everything(random_seed)

    # wandb.config["images_dir"] = images_dir
    # wandb.config["learning_rate"] = learning_rate
    # wandb.config["batch_size"] = batch_size
    # wandb.config["random_seed"] = random_seed
    # wandb.config["batch_size"] = batch_size
    # wandb.config.update()

    feature_extractor =  ViTFeatureExtractor.from_pretrained(model_name)
    dm = ImageClassifierDataModule(images_dir, feature_extractor, batch_size, num_workers, 0.9)
    dm.setup()

    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=len(dm.label2id),
        label2id=dm.label2id,
        id2label=dict((str(v), k) for k, v in dm.label2id.items()),
    )
    clf = ImageClassifier(model, learning_rate==learning_rate)

    # turn the labels into a pandas dataframe
    df = pd.DataFrame(
        {
            "label": list(dm.label2id.keys()),
            "id": list(dm.label2id.values()),
        }
    )

    wandb_logger = WandbLogger(project=wandb_project, name=wandb_run_name)
    wandb_logger.log_table(key="labels 2 id", dataframe=df)

    trainer = pl.Trainer(
        accelerator=accelerator,
        gpus=num_gpus,
        max_epochs=epochs,
        precision=precision,
        logger=wandb_logger,
    )
    trainer.fit(clf, dm)


if __name__ == "__main__":
    fire.Fire(main)
