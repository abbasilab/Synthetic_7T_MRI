import torch
import pytorch_lightning as pl
import numpy as np
from box import Box
from clearml import Task
# from torchmetrics.functional import structural_similarity_index_measure as SSIM

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

import sys, os, json
sys.path.append(os.path.join(sys.path[0], '../'))
from dataloader.patch_dataloader import patch_dataloader
from dataloader.data_utils import load_onefold_dataset, load_all  # ,load_MRIs
from network.VNet_model import VNetModel
from network.WatNet_model import WatNet3DModel, WatNet2DModel

script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)

def load_params(param_dir):
    with open(param_dir, 'r') as json_file:
        params = json.load(json_file)
    return Box(params)


params = load_params('../../config/params_WatNet2D.json')
torch.set_float32_matmul_precision(params.training.matmul_precision)
assert params.model.name in ["VNet", "WatNet2D","WatNet3D"], "model name should be VNet or WatNet2D or WatNet3D"

# if fold_ind == 'all', load all folds, else load specified fold
if params.data.fold_ind == "allfolds":
    fold_inds = np.arange(params.data.kfold_num)
if params.data.fold_ind == "finalmodel":
    fold_inds = None
else:
    fold_inds = params.data.fold_ind
    # assert isinstance(fold_ind, int), "fold_ind should be 'all' or an integer"
    # assert fold_ind < params.data.kfold_num, "fold_ind out of bound (k)"
    # fold_inds = [fold_ind]


exp_name = params.training.exp_name

if fold_inds is not None:
    for fold_ind in fold_inds:
        exp_ver = f"{params.model.loss_type}loss_fold{fold_ind}"
        # initialize ClearML
        task = Task.init(project_name=exp_name, task_name=exp_ver)
        
        train_dataset, val_dataset = (
            load_onefold_dataset(params, fold_ind=fold_ind))
        # get patch dataloaders for each dataset
        train_loader, _ = patch_dataloader(train_dataset, params)
        # calib_loader, _ = patch_dataloader(calib_dataset,params)
        val_loader, aggregator = patch_dataloader(val_dataset, params)

        if params.model.name == 'VNet':
            model = VNetModel(params)
        elif params.model.name == 'WatNet2D':
            model = WatNet2DModel(params)
        elif params.model.name == 'WatNet3D':
            model = WatNet3DModel(params)

        checkpoint_dir = os.path.join(params.checkpoint.ckpt_dir, exp_name,
                                    exp_ver, 'k-checkpoints')
        logger = TensorBoardLogger('tensorlog', name=exp_name, version=exp_ver)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_last=True,
            every_n_epochs=params.checkpoint.save_frequency_epoch,
            save_top_k=params.checkpoint.save_topk,
            monitor=params.checkpoint.monitor,
            filename='{epoch}-{step}-{val_MAE:.8f}'
        )
        # ClearML parameter recording
        task.set_parameters_as_dict(params.to_dict())

        trainer = pl.Trainer(
            max_epochs=params.training.num_epochs,
            accelerator='auto',
            strategy='ddp',
            logger=logger,
            precision=params.training.precision,
            log_every_n_steps=params.training.log_every_n_steps,
            # num_sanity_val_steps=0,
            callbacks=[lr_monitor,
                    checkpoint_callback,
                    RichProgressBar()]
        )
        trainer.fit(model, train_loader, val_loader)
        task.close()

else:
    exp_ver = f"{params.model.loss_type}loss_finalmodel"
    task = Task.init(project_name=exp_name, task_name=exp_ver)
    train_dataset = load_all(params)
    train_loader, aggregator = patch_dataloader(train_dataset, params)

    if params.model.name == 'VNet':
            model = BaseModel(params)
    elif params.model.name == 'WatNet2D':
        model = WatNet2DModel(params)
    elif params.model.name == 'WatNet3D':
        model = WatNet3DModel(params)

    checkpoint_dir = os.path.join(params.checkpoint.ckpt_dir, exp_name,
                                exp_ver, 'k-checkpoints')
    logger = TensorBoardLogger('tensorlog', name=exp_name, version=exp_ver)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_last=True,
        every_n_epochs=params.checkpoint.save_frequency_epoch,
        save_top_k=params.checkpoint.save_topk,
        monitor="train_MAE",
        filename='{epoch}-{step}-{train_MAE:.8f}'
    )
    # ClearML parameter recording
    task.set_parameters_as_dict(params.to_dict())

    trainer = pl.Trainer(
        max_epochs=params.training.num_epochs,
        accelerator='auto',
        strategy='ddp',
        # accelerator='gpu',
        # devices=[0],
        logger=logger,
        precision=params.training.precision,
        log_every_n_steps=params.training.log_every_n_steps,
        # num_sanity_val_steps=0,
        callbacks=[lr_monitor,
                checkpoint_callback,
                RichProgressBar()]
    )
    trainer.fit(model, train_loader)

    task.close()
