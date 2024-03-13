import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchio as tio
from .WatNet_3D import WatNet3D
from .WatNet_2D import WatNet2D


class WatNet2DModel(pl.LightningModule):
    def __init__(self, params):
        super(WatNet2DModel, self).__init__()
        self.save_hyperparameters()
        self.loss_type = params.model.loss_type
        allowed_loss = ['MAE']
        error_msg = f"Allowed loss types: {allowed_loss}"
        assert self.loss_type in allowed_loss, error_msg

        self.lr = params.model.learning_rate
        self.l2 = params.model.weight_decay
        self.bn = params.data.batch_size
        self.MAE_alpha = params.model.MAE_loss_weight
        self.percep_alpha = params.model.preceptual_loss_weight
        self.criterion_MAE = nn.L1Loss()
        self.basemodel = WatNet2D()

    def forward(self, x):
        x = self.basemodel(x)
        return x

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.l2)
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        img_3t = batch['t1_3T'][tio.DATA].swapaxes(1,4).squeeze()
        img_7t = batch['t1_7T'][tio.DATA].swapaxes(1,4).squeeze()
        mask = batch['mask'][tio.DATA].swapaxes(1,4).squeeze()
        img_s7t = self.forward(img_3t) * mask

        # MAE loss
        mae_loss = self.criterion_MAE(img_s7t.squeeze(), img_7t.squeeze())

        content_loss = self.MAE_alpha * mae_loss

        if self.loss_type == 'MAE':
            total_loss = content_loss

        bn = self.bn
        self.log('train_MAE', mae_loss, on_step=True, on_epoch=True,
                 logger=True, batch_size=bn, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        img_3t = batch['t1_3T'][tio.DATA].swapaxes(1,4).squeeze()
        img_7t = batch['t1_7T'][tio.DATA].swapaxes(1,4).squeeze()
        mask = batch['mask'][tio.DATA].swapaxes(1,4).squeeze()
        img_s7t = self.forward(img_3t) * mask

        # MAE loss
        mae_loss = self.criterion_MAE(img_s7t.squeeze(), img_7t.squeeze())

        content_loss = self.MAE_alpha * mae_loss

        if self.loss_type == 'MAE':
            total_loss = content_loss

        bn = self.bn
        self.log('val_MAE', mae_loss, on_step=True, on_epoch=True,
                 logger=True, batch_size=bn, sync_dist=True)

        return total_loss


class WatNet3DModel(pl.LightningModule):
    def __init__(self, params):
        super(WatNet3DModel, self).__init__()
        self.save_hyperparameters()
        self.loss_type = params.model.loss_type
        allowed_loss = ['MAE']
        error_msg = f"Allowed loss types: {allowed_loss}"
        assert self.loss_type in allowed_loss, error_msg

        self.lr = params.model.learning_rate
        self.l2 = params.model.weight_decay
        self.bn = params.data.batch_size
        self.MAE_alpha = params.model.MAE_loss_weight
        self.percep_alpha = params.model.preceptual_loss_weight
        self.criterion_MAE = nn.L1Loss()
        self.basemodel = WatNet3D()

    def forward(self, x):
        x = self.basemodel(x)
        return x

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.l2)
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        img_3t = batch['t1_3T'][tio.DATA]
        img_7t = batch['t1_7T'][tio.DATA]
        mask = batch['mask'][tio.DATA]
        img_s7t = self.forward(img_3t) * mask

        # MAE loss
        mae_loss = self.criterion_MAE(img_s7t.squeeze(), img_7t.squeeze())

        content_loss = self.MAE_alpha * mae_loss

        if self.loss_type == 'MAE':
            total_loss = content_loss

        bn = self.bn
        self.log('train_MAE', mae_loss, on_step=True, on_epoch=True,
                 logger=True, batch_size=bn, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        img_3t = batch['t1_3T'][tio.DATA]
        img_7t = batch['t1_7T'][tio.DATA]
        mask = batch['mask'][tio.DATA]
        img_s7t = self.forward(img_3t) * mask

        # MAE loss
        mae_loss = self.criterion_MAE(img_s7t.squeeze(), img_7t.squeeze())

        content_loss = self.MAE_alpha * mae_loss

        if self.loss_type == 'MAE':
            total_loss = content_loss

        bn = self.bn
        self.log('val_MAE', mae_loss, on_step=True, on_epoch=True,
                 logger=True, batch_size=bn, sync_dist=True)

        return total_loss
