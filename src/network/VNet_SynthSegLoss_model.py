import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchio as tio
from .VNet import VNet
from SynthSeg_parts import SynthSegModel
from SynthSeg_loss import LSynthSeg


class VNet_SynthSegLoss(pl.LightningModule):
    def __init__(self, params):
        super(VNet_SynthSegLoss, self).__init__()
        self.save_hyperparameters()
        self.loss_type = params.model.loss_type
        allowed_loss = ['SynthSeg3']
        error_msg = f"Allowed loss types: {allowed_loss}"
        assert self.loss_type in allowed_loss, error_msg

        self.lr = params.model.learning_rate
        self.l2 = params.model.weight_decay
        self.bn = params.data.batch_size
        self.MAE_alpha = params.model.MAE_loss_weight
        self.SynthSeg_alpha = params.model.SynthSeg_loss_weight

        SynthSeg_model = SynthSegModel()
        SynthSeg_model.load_state_dict(torch.load(params.model.SynthSeg_weight_path))
        SynthSeg_model.eval()

        self.criterion_MAE = nn.L1Loss()
        self.criterion_SynthSeg_AfterBlock3 = LSynthSeg(4, 6, SynthSeg_model)
        self.basemodel = VNet()

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

        # loss
        MAE_loss = self.criterion_MAE(img_s7t.squeeze(), img_7t.squeeze())
        SynthSeg_loss = self.criterion_SynthSeg_AfterBlock3(img_s7t.squeeze(), img_7t.squeeze())
        total_loss = self.MAE_alpha * MAE_loss + self.SynthSeg_alpha * SynthSeg_loss

        bn = self.bn
        self.log('train_MAE', MAE_loss, on_step=True, on_epoch=True,
                 logger=True, batch_size=bn, sync_dist=True)
        self.log('train_SynthSegLoss', SynthSeg_loss, on_step=True, on_epoch=True,
                 logger=True, batch_size=bn, sync_dist=True)
        self.log('train_Loss', total_loss, on_step=True, on_epoch=True,
                 logger=True, batch_size=bn, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        img_3t = batch['t1_3T'][tio.DATA]
        img_7t = batch['t1_7T'][tio.DATA]
        mask = batch['mask'][tio.DATA]
        img_s7t = self.forward(img_3t) * mask

        # loss
        MAE_loss = self.criterion_MAE(img_s7t.squeeze(), img_7t.squeeze())
        SynthSeg_loss = self.criterion_SynthSeg_AfterBlock3(img_s7t.squeeze(), img_7t.squeeze())
        total_loss = self.MAE_alpha * MAE_loss + self.SynthSeg_alpha * SynthSeg_loss

        bn = self.bn
        self.log('val_MAE', MAE_loss, on_step=True, on_epoch=True,
                 logger=True, batch_size=bn, sync_dist=True)
        self.log('val_SynthSegLoss', SynthSeg_loss, on_step=True, on_epoch=True,
                 logger=True, batch_size=bn, sync_dist=True)
        self.log('val_Loss', total_loss, on_step=True, on_epoch=True,
                 logger=True, batch_size=bn, sync_dist=True)

        return total_loss
