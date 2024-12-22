from lightning import LightningModule
from model import XLSR
import torch
import numpy as np
from PIL import Image
from data import output_dir
import os


class LightningXLSR(LightningModule):
    def __init__(self, in_ch, out_ch, lr):
        super(LightningXLSR, self).__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.lr = lr
        self.model = XLSR(self.in_ch, self.out_ch)
        self.l1_loss = torch.nn.L1Loss()
        self.chernober_loss = torch.nn.HuberLoss()
        self.train_iteration = 0
        self.val_iteration = 0

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
    
    def training_step(self, batch, batch_idx):
        self.train_iteration += 1
        model_input, target = batch
        output = self.model(model_input)
        loss = self.chernober_loss(output, target)
        self.logger.log_metrics({"xlsr/train/loss": loss}, self.train_iteration)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.val_iteration += 1
        model_input, target = batch
        output = self.model(model_input)
        loss = self.chernober_loss(output, target)
        self.logger.log_metrics({"xlsr/val/loss": loss}, self.val_iteration)
        image = output[0].detach().add(1.0).mul(0.5).clip(0, 1).mul(255).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        in_image = model_input[0].detach().add(1.0).mul(0.5).clip(0, 1).mul(255).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        gt_image = target[0].detach().add(1.0).mul(0.5).clip(0, 1).mul(255).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        outut_image = Image.fromarray(image)
        input_image = Image.fromarray(in_image)
        gtruth_image = Image.fromarray(gt_image)
        out = Image.new("RGB", (512*3, 512), color=(0, 0, 0))
        out.paste(input_image, (192, 192))
        out.paste(outut_image, (512, 0))
        out.paste(gtruth_image, (1024, 0))
        out.save(os.path.join(output_dir, f"{batch_idx}_result.png"))
        return loss
    
    def test_step(self, batch, batch_idx):
        model_input, target = batch
        output = self.model(model_input)
        image = output[0].detach().add(1.0).mul(0.5).clip(0, 1).mul(255).numpy().transpose(1, 2, 0).astype(np.uint8)
        Image.fromarray(image).save(os.path.join(output_dir, f"test_{self.current_epoch}_{batch_idx}.png"))

        