import torch
import torchvision
torchvision.disable_beta_transforms_warning()
import argparse
import torchio as tio
import numpy as np
import nibabel as nib
from box import Box
import os.path as path
import torch
import sys, os, json

sys.path.append(os.path.join(sys.path[0], '../'))
from dataloader.patch_dataloader import patch_dataloader
from dataloader.data_utils_eval import (load_MRI_from_paths_list,
                                        load_mask_from_paths_list,
                                        get_filelist_from_path,
                                        default_mask_generation,
                                        apply_mask_to_subject)
from network.VNet_model import VNetModel

parser = argparse.ArgumentParser(description='Generate synthetic 7T T1-weighted MRI from 3T input')

parser.add_argument('-i', '--input', '--in', type=str, help="""Path to the input file or directory containing input files.
                    Only processes .nii.gz images""")
parser.add_argument('-m', '--mask', type=str, default=None, help='path to mask file')
parser.add_argument('-c', '--ckpt', type=str, help='path to model file')
parser.add_argument('-o', '--output', '--out', type=str, help="""path to save file to""")
parser.add_argument('--suffix', type=str, default='_pred', help="""suffix to add to output file name""")
parser.add_argument('-bs', '--batch_size', type=int, default=5, help="""batch size for inference""")
parser.add_argument('-ps', '--patch_size', type=int, default=64, help="""patch size for inference""")
parser.add_argument('-po', '--patch_overlap', type=int, default=8, help="""patch overlap for inference""")
parser.add_argument('-ql', '--max_queue_length', type=int, default=1000, help="""max queue length for patch loading""")
parser.add_argument('-nw', '--num_workers', type=int, default=1, help="""number of workers for patch loading,
                    should not exceed number of CPU cores""")

args = parser.parse_args()

input_3T_path = args.input
input_mask_path = args.mask
ckpt_path = args.ckpt
output_path = args.output
suffix = args.suffix
params = {
    "data": {
            "batch_size": args.batch_size,
            "patch_size": args.patch_size,
            "patch_overlap": args.patch_overlap
    },
    "training":{ # this is a naming oversight, these are actually the parameters for the patched dataloader
                "num_workers": args.num_workers,
                "max_queue_length": args.max_queue_length
    }
}
params = Box(params)

if output_path is None:
    raise ValueError('Please provide an output path')

if not os.path.exists(ckpt_path):
    raise ValueError('Checkpoint path does not exist')

if not ckpt_path.endswith('.ckpt'):
    raise ValueError('Please provide a .ckpt file')

input_3T_files = get_filelist_from_path(input_3T_path)
subjects_t1_3T = load_MRI_from_paths_list(input_3T_files, 't1_3T')
ids_subjects = sorted([s.id for s in subjects_t1_3T])
ids_missing_masks = []

if input_mask_path is None:
    print('No mask path provided, using default mask generation for post-processing')
    print('NOTE: this pipeline expects the images to be brain stripped!')
    input_mask_path = None
    subjects_mask = []
    ids_missing_masks = ids_subjects

if input_mask_path is not None:
    # load masks if provided
    input_mask_files = get_filelist_from_path(input_mask_path)
    subjects_mask = load_mask_from_paths_list(input_mask_files)
    ids_masks = sorted([s.id for s in subjects_mask])

    if ids_subjects != ids_masks:
        # checck if all subjects have a corresponding mask
        ids_missing_masks = list(set(ids_subjects) - set(ids_masks))
        print("id of subjects and masks do not match.\n" \
            f"missing masks for subjects {ids_missing_masks}\n"
            "using default mask generation for post-processing of these subjects")

subjects_missing_masks = [s for s in subjects_t1_3T if s.id in ids_missing_masks]
for subject_for_maskgen in subjects_missing_masks:
    generated_mask = default_mask_generation(subject_for_maskgen)
    subjects_mask.append(generated_mask)

subjects_dataset = tio.SubjectsDataset(apply_mask_to_subject(subjects_t1_3T, subjects_mask))

model = VNetModel.load_from_checkpoint(ckpt_path).eval().to('cuda')

with torch.no_grad():
    preds = []
    for subject in subjects_dataset:
        loader, aggregator = patch_dataloader(tio.SubjectsDataset([subject]), params)

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
        preds.append(subject)

if '.nii.gz' in output_path and len(input_3T_files) != 1:
    print('Multiple input files detected, but output path is a file. Saving to output directory instead')
    output_dir = path.dirname(output_path)
    if not path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
elif '.nii.gz' in output_path and len(input_3T_files) == 1:
    preds[0].pred.save(output_path)
    print(f'Saved {output_path}')
else:
    output_dir = output_path
    if not path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
# suffix = '_pred'

for subject in preds:
    output_path_updated = path.join(output_dir, f'{subject.id}'+suffix+'.nii.gz')
    subject.pred.save(output_path_updated)
    print(f'Saved {output_path_updated}')

