# %%
import torch
import argparse
import torchio as tio
import numpy as np
import nibabel as nib
from box import Box

import sys, os, json
sys.path.append(os.path.join(sys.path[0], '../'))
from dataloader.patch_dataloader import patch_dataloader
from dataloader.data_utils import load_onefold_dataset, load_all
from network.VNet_model import VNetModel
from network.WatNet_model import WatNet3DModel, WatNet2DModel


script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)
# %%
# parser = argparse.ArgumentParser(description="input checkpoint path and output path")
# parser.add_argument('-ckpt', '--checkpoint', type=str, help='The input checkpoint file')
# parser.add_argument('-out', '--output', type=str, help='The output path')
# args = parser.parse_args()

# ckpt_p = args.checkpoint
# out_p = args.output


def load_params(param_dir):
    with open(param_dir, 'r') as json_file:
        params = json.load(json_file)
    return Box(params)


def mode_update(params, mode):
    if mode == 'org':
        print("original mode")
        pass
    elif mode == 'down2':
        print("down2 mode")
        params.fp.filepaths[0] = params.fp.filepaths[0] + '_down2'
        params.fp.postfixes[0] = '_down2' + params.fp.postfixes[0]
    elif mode == 'down4':
        print("down4 mode")
        params.fp.filepaths[0] = params.fp.filepaths[0] + '_down4'
        params.fp.postfixes[0] = '_down4' + params.fp.postfixes[0]
    else:
        raise ValueError("mode should be 'org' or 'down2' or 'down4'")

    return params


def predict(model, dataset, params):
    model.eval().to('cuda')
    with torch.no_grad():
        subjects = [] 
        for subject in dataset:
            loader, aggregator = patch_dataloader(
                tio.SubjectsDataset([subject]), params)
            if params.model.name == 'VNet':
                for patches_batch in loader:
                    patches = patches_batch['t1_3T'][tio.DATA].to('cuda')
                    locations = patches_batch[tio.LOCATION]
                    patches_preds = model(patches) * patches_batch['mask'][tio.DATA].to('cuda')
                    aggregator.add_batch(patches_preds.cpu(), locations)
                # get images from aggregators
                image = aggregator.get_output_tensor()
                affine = subject.t1_3T.affine  # same affine for images now, this may need change
                image = tio.ScalarImage(tensor=image, affine=affine)
                subject.add_image(image, 'pred')
                subjects.append(subject)

            elif params.model.name == 'WatNet2D':
                for patches_batch in loader:
                    patches = patches_batch['t1_3T'][tio.DATA].swapaxes(1,4).squeeze().to('cuda')
                    locations = patches_batch[tio.LOCATION]
                    patches_preds = model(patches) * patches_batch['mask'][tio.DATA].to('cuda').swapaxes(1,4).squeeze()
                    patches_preds = patches_preds.unsqueeze(1)
                    patches_preds = patches_preds.moveaxis(2,4)
                    aggregator.add_batch(patches_preds.cpu(), locations)
                # get images from aggregators
                image = aggregator.get_output_tensor()
                affine = subject.t1_3T.affine  # same affine for images now, this may need change
                image = tio.ScalarImage(tensor=image, affine=affine)
                subject.add_image(image, 'pred')
                subjects.append(subject)

    return tio.SubjectsDataset(subjects)


def save_preds(dataset, out_p, params):
    for subject in dataset:
        id = subject.id
        print(f"saving images for subject {id}")
        output_path = os.path.join(out_p, f'{id}_synthetic7t{params.fp.postfixes[0]}')
        subject.pred.save(output_path)
# %%


param_path = '../../config/params_VNet_eval.json'
params = load_params(param_path)
modes = params.fp.eval_modes

for mode in modes:
    # reload params 
    params = load_params(param_path)
    params = mode_update(params, mode)
    torch.set_float32_matmul_precision(params.training.matmul_precision)
    assert params.model.name in ["VNet", "WatNet2D","WatNet3D"], "model name should be VNet or WatNet2D or WatNet3D"

    # if fold_ind == 'allfolds', load all folds, else load specified fold
    if params.data.fold_ind == "allfolds":
        fold_inds = np.arange(params.data.kfold_num)
    if params.data.fold_ind == "finalmodel":
        fold_inds = None
        folder_names = [f"{params.model.loss_type}loss_finalmodel"]
    else:
        fold_inds = params.data.fold_ind
        folder_names = [f"{params.model.loss_type}loss_fold{i}" for i in fold_inds]
        # assert isinstance(fold_ind, int), "fold_ind should be 'all' or an integer"
        # assert fold_ind < params.data.kfold_num, "fold_ind out of bound (k)"
        # fold_inds = [fold_ind]

    for folder_name in folder_names:
        if fold_inds is None:
            fold_ind = None
        else:
            fold_ind = fold_inds[folder_names.index(folder_name)]
        ckpt_folder = os.path.join(params.fp.ckpt_dir, folder_name)
        assert len(os.listdir(ckpt_folder)) == 1, "only one checkpoint file should be in the folder"
        ckpt_p = os.path.join(ckpt_folder, os.listdir(ckpt_folder)[0])
        out_p = f"../../preds/{params.model.name}_{mode}_{folder_name}"
        if not os.path.isdir(out_p):
            os.makedirs(out_p)

        if params.model.name == 'VNet':
            model = BaseModel(params).load_from_checkpoint(ckpt_p)
        elif params.model.name == 'WatNet2D':
            model = WatNet2DModel(params).load_from_checkpoint(ckpt_p)
        elif params.model.name == 'WatNet3D':
            model = WatNet3DModel(params).load_from_checkpoint(ckpt_p)

        if fold_ind is None:
            val_dataset = load_all(params)
            print(r"predicting for final model")

        else:
            _, val_dataset = (load_onefold_dataset(params, fold_ind=fold_ind))
            print(f"predicting for fold {fold_ind}")

        val_dataset = predict(model, val_dataset, params)

        save_preds(val_dataset, out_p, params)
# %%
