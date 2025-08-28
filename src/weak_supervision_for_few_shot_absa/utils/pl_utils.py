"""
Inspiration from: https://github.com/Lightning-AI/lightning/issues/5473#issuecomment-764002682
The idea is to save checkpoints after a given number of training steps
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path


class SavingPointsCheckpoint(ModelCheckpoint):
    def __init__(self, saving_points=None, weights_only=False):
        super().__init__()
        if saving_points:
            self.saving_points = saving_points
        else:
            self.saving_points = set(
                [
                    0,
                    100,
                    200,
                    300,
                    400,
                    500,
                    600,
                    700,
                    800,
                    900,
                    1000,
                    2000,
                    3000,
                    4000,
                    5000,
                    6000,
                    7000,
                    8000,
                    9000,
                    10000,
                ]
            )

        self.weights_only = weights_only

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs
    ):
        if pl_module.global_step in self.saving_points:
            assert self.dirpath is not None
            current = Path(self.dirpath) / f"latest-{pl_module.global_step}.ckpt"
            trainer.save_checkpoint(current, self.weights_only)
