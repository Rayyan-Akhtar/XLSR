from model.network import LightningXLSR
from data.dataloader import Div2kDataloader
from lightning import Trainer
from lightning import Callback
import numpy as np
import os


class LearningRateCallback(Callback):
    def __init__(self):
        super(LearningRateCallback, self).__init__()
        self.lr1 = 5e-5
        self.lr2 = 25e-4
        self.lr3 = 1e-4
        self.first_stage = 1
        self.second_stage = 50 
        self.third_stage = 5000
        self.current_lr = 0
        self.first_slope = (self.second_stage - self.first_stage) / (self.lr2 - self.lr1)
        self.second_slope = (self.third_stage - self.second_stage) / (self.lr3 - self.lr2)

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch <= 1:
            self.current_lr = self.lr1
        if self.first_stage > current_epoch > self.second_stage:
            self.current_lr = self.first_slope * (self.lr2 - self.lr1) + self.first_stage
        if self.second_stage > current_epoch > self.third_stage:
            self.current_lr = self.second_slope * (self.lr3 - self.lr2) + self.second_stage
        trainer.optimizers[0].param_groups[0]["lr"] = self.current_lr

lr_callback = LearningRateCallback()

trainer = Trainer(
    max_epochs=5000,
    enable_progress_bar=True,
    enable_model_summary=True,
    default_root_dir="/teamspace/studios/this_studio/XLSR",
    log_every_n_steps=1,
    callbacks=[lr_callback]
)

trainer.fit(
    model=LightningXLSR(32, 32, lr=5e-5),
    train_dataloaders=Div2kDataloader,
    
)